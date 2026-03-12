import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# VCP Scanner
# ------------------------------------------------------------
# 목적
# - VCP(Volatility Contraction Pattern) 탐지
# - RS 하드컷 적용
# - 결과 CSV / Summary JSON 저장
# - 텔레그램 상세 결과 전송
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
RESULT_FILE = os.path.join(OUTPUT_DIR, "vcp_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "vcp_summary.json")

BENCHMARK = "SPY"
DOWNLOAD_PERIOD = "2y"

MIN_HISTORY = 260
MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000

# RS
RS_LOOKBACK = 252
RS_PERCENTILE_HARDCUT = 80.0
RS_GRADE_A = 90.0
RS_NEAR_HIGH_MIN_RATIO = 0.80
RS_NEW_HIGH_TOL = 0.995

# VCP
PATTERN_MIN_LEN = 24
PATTERN_MAX_LEN = 90
MIN_STAGES = 3
MAX_STAGES = 5
MIN_STAGE_LEN = 5
MAX_STAGE_LEN = 25

MAX_LAST_STAGE_RANGE = 0.08
NEAR_PIVOT_LOW = -0.03
NEAR_PIVOT_HIGH = 0.03
FIRST_BREAKOUT_HIGH = 0.08

VOL_CONTRACTION_MAX = 0.80  # 최근 10일 거래량 / 최근 50일 거래량
ENTRY_BUFFER = 0.002
STOP_PCT = 0.08
PYRAMID_LEVELS = [0.02, 0.04, 0.06]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_MESSAGE_LEN = 3500
MAX_TELEGRAM_ROWS = 20


@dataclass
class VCPResult:
    ticker: str
    name: str
    as_of_date: str
    state: str
    grade: str
    grade_label: str
    rs_grade: str
    score_total: float
    close: float
    pivot_price: float
    entry_price: float
    stop_price: float
    add_price_1: float
    add_price_2: float
    add_price_3: float
    stages: int
    ranges_text: str
    last_stage_range: float
    vol_ratio_10_50: float
    rs_percentile: float
    rs_current_vs_high: float


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def round_price(x: Optional[float], digits: int = 2) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    return round(float(x), digits)


def format_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.2f}"


def format_pct_ratio(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(text: str) -> None:
    if not telegram_enabled():
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )


def send_telegram_chunked(lines: List[str]) -> None:
    if not telegram_enabled() or not lines:
        return

    chunks: List[str] = []
    current = ""

    for line in lines:
        candidate = f"{current}\n{line}".strip() if current else line
        if len(candidate) > MAX_TELEGRAM_MESSAGE_LEN:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    for chunk in chunks:
        send_telegram(chunk)


def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"유니버스 파일 없음: {UNIVERSE_FILE}")

    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise ValueError("universe.csv에 ticker 컬럼 필요")
    if "name" not in df.columns:
        df["name"] = df["ticker"]

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str)
    return df[["ticker", "name"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def normalize_downloaded(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "date":
            rename_map[c] = "Date"
        elif cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"

    df = df.rename(columns=rename_map)
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[needed].copy()

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna().sort_values("Date").reset_index(drop=True)


def download_history(ticker: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


def grade_label(score: float) -> Tuple[str, str]:
    if score >= 85:
        return "A", "우선 검토"
    return "B", "관찰 후보"


def get_rs_grade(rs_percentile: float, rs_current_vs_high: float) -> str:
    if rs_current_vs_high >= RS_NEW_HIGH_TOL:
        return "S"
    if rs_percentile >= RS_GRADE_A:
        return "A"
    return "B"


def classify_state(close: float, pivot: float) -> Optional[str]:
    dist = close / pivot - 1.0

    if NEAR_PIVOT_LOW <= dist <= NEAR_PIVOT_HIGH:
        return "돌파 임박"
    if NEAR_PIVOT_HIGH < dist <= FIRST_BREAKOUT_HIGH:
        return "1차 돌파"
    return None


def build_trade_plan(pivot: float) -> Tuple[float, float, float, float, float]:
    entry = pivot * (1.0 + ENTRY_BUFFER)
    stop = entry * (1.0 - STOP_PCT)
    add1 = entry * (1.0 + PYRAMID_LEVELS[0])
    add2 = entry * (1.0 + PYRAMID_LEVELS[1])
    add3 = entry * (1.0 + PYRAMID_LEVELS[2])
    return entry, stop, add1, add2, add3


def split_into_stages(seg: pd.DataFrame, stages: int) -> Optional[List[pd.DataFrame]]:
    n = len(seg)
    part = n // stages
    if part < MIN_STAGE_LEN or part > MAX_STAGE_LEN:
        return None

    chunks: List[pd.DataFrame] = []
    for i in range(stages):
        if i == stages - 1:
            sub = seg.iloc[i * part :]
        else:
            sub = seg.iloc[i * part : (i + 1) * part]

        if len(sub) < MIN_STAGE_LEN:
            return None
        chunks.append(sub)

    return chunks


def detect_vcp(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df) < 120:
        return None

    for total_len in range(PATTERN_MIN_LEN, PATTERN_MAX_LEN + 1):
        seg = df.tail(total_len).copy()

        for stages in range(MIN_STAGES, MAX_STAGES + 1):
            chunks = split_into_stages(seg, stages)
            if chunks is None:
                continue

            ranges: List[float] = []
            highs: List[float] = []

            ok = True
            for ch in chunks:
                high = float(ch["High"].max())
                low = float(ch["Low"].min())
                if high <= 0 or low <= 0 or low >= high:
                    ok = False
                    break

                ranges.append((high - low) / high)
                highs.append(high)

            if not ok:
                continue

            contraction_ok = all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1))
            if not contraction_ok:
                continue

            if ranges[-1] > MAX_LAST_STAGE_RANGE:
                continue

            pivot = max(highs)
            close = float(df["Close"].iloc[-1])

            state = classify_state(close, pivot)
            if state is None:
                continue

            vol10 = float(df["Volume"].tail(10).mean())
            vol50 = float(df["Volume"].tail(50).mean())
            if vol50 <= 0:
                continue
            vol_ratio = vol10 / vol50

            if vol_ratio > VOL_CONTRACTION_MAX:
                continue

            return {
                "stages": stages,
                "ranges": ranges,
                "pivot": pivot,
                "state": state,
                "vol_ratio_10_50": vol_ratio,
            }

    return None


def scan_one(ticker: str, name: str, spy_df: pd.DataFrame) -> Optional[VCPResult]:
    df = download_history(ticker)
    if df.empty or len(df) < MIN_HISTORY:
        return None

    close = float(df["Close"].iloc[-1])
    if close < MIN_PRICE:
        return None

    df["dollar_vol_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    dollar_vol_20 = safe_float(df["dollar_vol_20"].iloc[-1])
    if dollar_vol_20 is None or dollar_vol_20 < MIN_DOLLAR_VOL_20:
        return None

    stock = df.copy().set_index("Date")
    spy = spy_df.copy().set_index("Date")
    spy_close = spy["Close"].reindex(stock.index).ffill()

    rs_line = stock["Close"] / spy_close
    if len(rs_line) < RS_LOOKBACK:
        return None

    rs_high = float(rs_line.tail(RS_LOOKBACK).max())
    rs_low = float(rs_line.tail(RS_LOOKBACK).min())
    rs_now = float(rs_line.iloc[-1])

    if rs_high <= 0 or rs_high <= rs_low:
        return None

    rs_current_vs_high = rs_now / rs_high
    rs_percentile = ((rs_now - rs_low) / (rs_high - rs_low)) * 100.0

    if rs_percentile < RS_PERCENTILE_HARDCUT:
        return None
    if rs_current_vs_high < RS_NEAR_HIGH_MIN_RATIO:
        return None

    vcp = detect_vcp(df)
    if vcp is None:
        return None

    score = 60.0
    score += max(0.0, 15.0 - (sum(vcp["ranges"]) * 100.0))
    score += 10.0 if rs_current_vs_high >= RS_NEW_HIGH_TOL else 5.0
    score += 10.0 if rs_percentile >= 90 else 5.0
    score += 5.0 if vcp["vol_ratio_10_50"] <= 0.65 else 2.0
    score = min(100.0, round(score, 1))

    grade, grade_text = grade_label(score)
    rs_grade = get_rs_grade(rs_percentile, rs_current_vs_high)

    entry, stop, add1, add2, add3 = build_trade_plan(vcp["pivot"])
    ranges_text = " → ".join([f"{r * 100:.1f}%" for r in vcp["ranges"]])

    return VCPResult(
        ticker=ticker,
        name=name,
        as_of_date=str(df["Date"].iloc[-1].date()),
        state=vcp["state"],
        grade=grade,
        grade_label=grade_text,
        rs_grade=rs_grade,
        score_total=score,
        close=round(close, 2),
        pivot_price=round(vcp["pivot"], 2),
        entry_price=round(entry, 2),
        stop_price=round(stop, 2),
        add_price_1=round(add1, 2),
        add_price_2=round(add2, 2),
        add_price_3=round(add3, 2),
        stages=vcp["stages"],
        ranges_text=ranges_text,
        last_stage_range=round(vcp["ranges"][-1], 4),
        vol_ratio_10_50=round(vcp["vol_ratio_10_50"], 4),
        rs_percentile=round(rs_percentile, 2),
        rs_current_vs_high=round(rs_current_vs_high, 4),
    )


def save_outputs(results: List[VCPResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = [f.name for f in VCPResult.__dataclass_fields__.values()]

    if not results:
        pd.DataFrame(columns=cols).to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_at": datetime.now().isoformat(),
                    "total": 0,
                    "near_breakout": 0,
                    "first_breakout": 0,
                    "rs_s": 0,
                    "rs_a": 0,
                    "rs_b": 0,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    df = pd.DataFrame([asdict(r) for r in results])

    df["rs_rank"] = df["rs_grade"].map({"S": 0, "A": 1, "B": 2}).fillna(9)
    df["grade_rank"] = df["grade"].map({"A": 0, "B": 1}).fillna(9)
    df["state_rank"] = df["state"].map({"돌파 임박": 0, "1차 돌파": 1}).fillna(9)

    df = df.sort_values(
        ["rs_rank", "grade_rank", "state_rank", "score_total", "ticker"],
        ascending=[True, True, True, False, True],
    ).drop(columns=["rs_rank", "grade_rank", "state_rank"])

    df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "near_breakout": int((df["state"] == "돌파 임박").sum()),
        "first_breakout": int((df["state"] == "1차 돌파").sum()),
        "rs_s": int((df["rs_grade"] == "S").sum()),
        "rs_a": int((df["rs_grade"] == "A").sum()),
        "rs_b": int((df["rs_grade"] == "B").sum()),
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_result_lines(r: VCPResult) -> List[str]:
    return [
        f"- {r.ticker} {r.name}",
        f"  RS: {r.rs_grade} | RS Percentile: {r.rs_percentile:.1f} | RS 고점 거리: {format_pct_ratio(r.rs_current_vs_high - 1.0)}",
        f"  등급: {r.grade} ({r.grade_label}) | 점수: {r.score_total}/100",
        f"  상태: {r.state}",
        f"  종가: {format_price(r.close)}",
        f"  피벗: {format_price(r.pivot_price)}",
        f"  단계 수: {r.stages}",
        f"  수축 범위: {r.ranges_text}",
        f"  마지막 수축폭: {format_pct_ratio(r.last_stage_range)}",
        f"  거래량 비율(10/50): {r.vol_ratio_10_50:.2f}",
        f"  진입가: {format_price(r.entry_price)}",
        f"  손절가: {format_price(r.stop_price)}",
    ]


def notify_all(results: List[VCPResult]) -> None:
    if not telegram_enabled():
        return

    sorted_results = sorted(
        results,
        key=lambda x: (
            {"S": 0, "A": 1, "B": 2}.get(x.rs_grade, 9),
            {"A": 0, "B": 1}.get(x.grade, 9),
            {"돌파 임박": 0, "1차 돌파": 1}.get(x.state, 9),
            -x.score_total,
            x.ticker,
        ),
    )

    lines = ["[VCP 패턴] VCP", f"전체 후보: {len(sorted_results)}"]
    for r in sorted_results[:MAX_TELEGRAM_ROWS]:
        lines.extend(build_result_lines(r))

    send_telegram_chunked(lines)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    universe = load_universe()
    spy = download_history(BENCHMARK)
    if spy.empty or len(spy) < MIN_HISTORY:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    results: List[VCPResult] = []

    for row in universe.itertuples(index=False):
        try:
            r = scan_one(row.ticker, row.name, spy)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"{row.ticker} failed: {type(e).__name__}: {e}")

    save_outputs(results)
    notify_all(results)


if __name__ == "__main__":
    main()
