from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =========================================================
# PATHS / ENV
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
RESULT_FILE = os.path.join(OUTPUT_DIR, "vcp_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "vcp_summary.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MAX_TELEGRAM_MESSAGE_LEN = 3500
MAX_TELEGRAM_ROWS = 25
SEPARATOR = "────────────"


# =========================================================
# CONFIG
# =========================================================
DOWNLOAD_PERIOD = "2y"
MIN_HISTORY = 260

MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000
MIN_MARKET_CAP = 1_000_000_000

REQUIRE_MA_ALIGNMENT = True

RS_LOOKBACK = 252
RS_NEW_HIGH_TOL = 0.995

MAX_BASE_DEPTH = 0.35
MIN_BASE_LENGTH = 25
MAX_BASE_LENGTH = 180

MIN_STAGES = 3
MAX_STAGES = 5

PIVOT_READY_MAX = 0.00
PIVOT_READY_MIN = -0.03
PIVOT_BREAKOUT_MAX = 0.03
LATE_BREAKOUT_MAX = 0.08

MAX_SINGLE_DAY_SPIKE = 0.18
MAX_20D_RETURN = 0.35
MIN_10D_RANGE = 0.018

STATE_PRIORITY = {
    "돌파 임박": 0,
    "1차 돌파": 1,
    "후행 가능": 2,
    "관찰": 9,
}


# =========================================================
# DATACLASS
# =========================================================
@dataclass
class VCPResult:
    ticker: str
    name: str
    as_of_date: str

    state: str
    grade: str
    score_total: float

    close: float
    pivot_price: float
    entry_price: float
    stop_price: float
    pivot_distance_pct: float

    stages: int
    ranges_text: str
    contractions: str

    rs_grade: str
    rs_percentile: float
    rs_current_vs_high: float

    base_length: int
    base_depth: float
    dollar_vol_20: float
    market_cap: float


# =========================================================
# HELPERS
# =========================================================
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_price(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "-"
    return f"{v:,.2f}"


def fmt_pct(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def fmt_cap(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "-"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.0f}M"
    return f"${v:,.0f}"


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


def send_telegram_chunked(blocks: List[str]) -> None:
    if not telegram_enabled() or not blocks:
        return

    chunks: List[str] = []
    current = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) > MAX_TELEGRAM_MESSAGE_LEN:
            if current:
                chunks.append(current)
            current = block
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
    need = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[need].copy()
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


def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def get_market_cap(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(ticker).info
        return safe_float(info.get("marketCap"))
    except Exception:
        return None


def get_rs_metrics(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    stock = stock_df.copy().set_index("Date")
    spy = spy_df.copy().set_index("Date")
    spy_close = spy["Close"].reindex(stock.index).ffill()

    rs_line = stock["Close"] / spy_close
    if len(rs_line) < RS_LOOKBACK:
        return None, None, None

    rs_high = safe_float(rs_line.tail(RS_LOOKBACK).max())
    rs_low = safe_float(rs_line.tail(RS_LOOKBACK).min())
    rs_now = safe_float(rs_line.iloc[-1])

    if rs_high is None or rs_low is None or rs_now is None:
        return None, None, None
    if rs_high <= rs_low:
        return None, None, None

    rs_current_vs_high = rs_now / rs_high
    rs_percentile = ((rs_now - rs_low) / (rs_high - rs_low)) * 100.0

    if rs_current_vs_high >= RS_NEW_HIGH_TOL:
        rs_grade = "S"
    elif rs_percentile >= 90:
        rs_grade = "A"
    else:
        rs_grade = "B"

    return rs_percentile, rs_current_vs_high, rs_grade


def is_event_or_ma_pattern(df: pd.DataFrame) -> bool:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    ret_20d = safe_float(rolling_return(close, 20).iloc[-1])
    if ret_20d is not None and ret_20d > MAX_20D_RETURN:
        return True

    daily_ret = close.pct_change().tail(20)
    if not daily_ret.empty and daily_ret.max() > MAX_SINGLE_DAY_SPIKE:
        return True

    gap_ret = (open_ / close.shift(1) - 1.0).tail(20)
    if not gap_ret.empty and gap_ret.max() > 0.15:
        return True

    recent_high = safe_float(high.tail(10).max())
    recent_low = safe_float(low.tail(10).min())
    current_close = safe_float(close.iloc[-1])

    if recent_high and recent_low and current_close and current_close > 0:
        range_10d = (recent_high - recent_low) / current_close
        if range_10d < MIN_10D_RANGE:
            return True

    return False


# =========================================================
# VCP DETECTION
# =========================================================
def get_local_extrema(close: pd.Series, window: int = 5) -> Tuple[List[int], List[int]]:
    highs: List[int] = []
    lows: List[int] = []

    values = close.values
    n = len(values)

    for i in range(window, n - window):
        segment = values[i - window:i + window + 1]
        if values[i] == np.max(segment):
            highs.append(i)
        if values[i] == np.min(segment):
            lows.append(i)

    return highs, lows


def compute_contractions(df: pd.DataFrame, lookback: int) -> Tuple[List[float], List[int], int, float, float]:
    sub = df.tail(lookback).reset_index(drop=True)
    close = sub["Close"]
    high = sub["High"]
    low = sub["Low"]

    local_highs, local_lows = get_local_extrema(close, window=4)

    if not local_highs or not local_lows:
        return [], [], 0, 0.0, 0.0

    peak_idx = local_highs[-1]
    pivot = float(high.iloc[peak_idx])

    relevant_highs = [i for i in local_highs if i <= peak_idx]
    relevant_lows = [i for i in local_lows if i <= peak_idx]

    contractions: List[float] = []
    anchors: List[int] = []

    for hi in relevant_highs[-MAX_STAGES:]:
        subsequent_lows = [lo for lo in relevant_lows if lo > hi]
        if not subsequent_lows:
            continue
        lo = subsequent_lows[0]
        hi_price = float(high.iloc[hi])
        lo_price = float(low.iloc[lo])
        if hi_price <= 0:
            continue
        drawdown = (hi_price - lo_price) / hi_price
        contractions.append(drawdown)
        anchors.append(hi)

    contractions = contractions[-MAX_STAGES:]
    anchors = anchors[-MAX_STAGES:]

    if len(contractions) < MIN_STAGES:
        return [], [], 0, 0.0, pivot

    shrink_ok = 0
    for i in range(1, len(contractions)):
        if contractions[i] <= contractions[i - 1] * 1.05:
            shrink_ok += 1

    if shrink_ok < len(contractions) - 1:
        return [], [], 0, 0.0, pivot

    base_high = float(high.tail(lookback).max())
    base_low = float(low.tail(lookback).min())
    base_depth = (base_high - base_low) / base_high if base_high > 0 else 0.0

    return contractions, anchors, lookback, base_depth, pivot


def score_vcp(
    contractions: List[float],
    rs_grade: str,
    rs_percentile: float,
    pivot_distance: float,
    base_depth: float,
    base_length: int,
) -> float:
    score = 0.0

    stages = len(contractions)
    if stages >= 4:
        score += 35
    elif stages == 3:
        score += 28

    final_c = contractions[-1]
    if final_c <= 0.04:
        score += 20
    elif final_c <= 0.06:
        score += 15
    elif final_c <= 0.08:
        score += 10

    if rs_grade == "S":
        score += 20
    elif rs_grade == "A":
        score += 14
    else:
        score += 8

    if rs_percentile >= 98:
        score += 10
    elif rs_percentile >= 95:
        score += 6

    if -0.02 <= pivot_distance <= 0.01:
        score += 10
    elif -0.04 <= pivot_distance <= 0.03:
        score += 6

    if 0.08 <= base_depth <= 0.30:
        score += 8
    elif base_depth <= MAX_BASE_DEPTH:
        score += 5

    if 35 <= base_length <= 120:
        score += 7
    else:
        score += 3

    return round(score, 1)


def classify_state(pivot_distance: float) -> str:
    if PIVOT_READY_MIN <= pivot_distance <= PIVOT_READY_MAX:
        return "돌파 임박"
    if 0.0 < pivot_distance <= PIVOT_BREAKOUT_MAX:
        return "1차 돌파"
    if PIVOT_BREAKOUT_MAX < pivot_distance <= LATE_BREAKOUT_MAX:
        return "후행 가능"
    return "관찰"


def grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    return "WATCH"


def scan_one(
    ticker: str,
    name: str,
    spy_df: pd.DataFrame,
) -> Optional[VCPResult]:
    df = download_history(ticker)
    if df.empty or len(df) < MIN_HISTORY:
        return None

    close = safe_float(df["Close"].iloc[-1])
    if close is None or close < MIN_PRICE:
        return None

    df["dollar_vol_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    dollar_vol_20 = safe_float(df["dollar_vol_20"].iloc[-1])
    if dollar_vol_20 is None or dollar_vol_20 < MIN_DOLLAR_VOL_20:
        return None

    market_cap = get_market_cap(ticker)
    if market_cap is None or market_cap < MIN_MARKET_CAP:
        return None

    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma150"] = df["Close"].rolling(150).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    ma50 = safe_float(df["ma50"].iloc[-1])
    ma150 = safe_float(df["ma150"].iloc[-1])
    ma200 = safe_float(df["ma200"].iloc[-1])
    if ma50 is None or ma150 is None or ma200 is None:
        return None

    if REQUIRE_MA_ALIGNMENT and not (close > ma50 > ma150 > ma200):
        return None

    if is_event_or_ma_pattern(df):
        return None

    rs_percentile, rs_current_vs_high, rs_grade = get_rs_metrics(df, spy_df)
    if rs_percentile is None or rs_current_vs_high is None or rs_grade is None:
        return None

    best: Optional[Tuple[List[float], List[int], int, float, float]] = None
    for lookback in [35, 50, 70, 90, 120, 150, 180]:
        contractions, anchors, base_length, base_depth, pivot = compute_contractions(df, lookback)
        if not contractions:
            continue
        if base_depth > MAX_BASE_DEPTH:
            continue
        best = (contractions, anchors, base_length, base_depth, pivot)

    if best is None:
        return None

    contractions, anchors, base_length, base_depth, pivot = best

    pivot_distance = (close / pivot) - 1.0 if pivot > 0 else 0.0
    state = classify_state(pivot_distance)
    if state == "관찰":
        return None

    score_total = score_vcp(
        contractions=contractions,
        rs_grade=rs_grade,
        rs_percentile=rs_percentile,
        pivot_distance=pivot_distance,
        base_depth=base_depth,
        base_length=base_length,
    )
    grade = grade_from_score(score_total)

    entry_price = round(pivot * 1.003, 2)
    final_contraction = contractions[-1]
    stop_price = round(entry_price * (1 - max(final_contraction * 0.9, 0.06)), 2)

    ranges_text = " → ".join(f"{c * 100:.0f}%" for c in contractions)
    contractions_text = "|".join(f"{c:.4f}" for c in contractions)

    return VCPResult(
        ticker=ticker,
        name=name,
        as_of_date=str(df["Date"].iloc[-1].date()),
        state=state,
        grade=grade,
        score_total=score_total,
        close=round(close, 2),
        pivot_price=round(pivot, 2),
        entry_price=entry_price,
        stop_price=stop_price,
        pivot_distance_pct=round(pivot_distance, 4),
        stages=len(contractions),
        ranges_text=ranges_text,
        contractions=contractions_text,
        rs_grade=rs_grade,
        rs_percentile=round(rs_percentile, 2),
        rs_current_vs_high=round(rs_current_vs_high, 4),
        base_length=base_length,
        base_depth=round(base_depth, 4),
        dollar_vol_20=round(dollar_vol_20, 2),
        market_cap=round(market_cap, 2),
    )


# =========================================================
# OUTPUT
# =========================================================
def build_result_block(r: VCPResult) -> str:
    return (
        f"{SEPARATOR}\n"
        f"{r.ticker} | {r.name}\n"
        f"상태: {r.state}\n"
        f"등급: {r.grade} | 점수: {r.score_total:.1f}/100\n"
        f"종가: {fmt_price(r.close)}\n"
        f"피벗: {fmt_price(r.pivot_price)}\n"
        f"피벗 거리: {fmt_pct(r.pivot_distance_pct)}\n"
        f"단계: {r.stages}단\n"
        f"수축폭: {r.ranges_text}\n"
        f"RS: {r.rs_grade} | RS Percentile: {r.rs_percentile:.1f}\n"
        f"베이스 길이: {r.base_length}일 | 베이스 깊이: {fmt_pct(r.base_depth)}\n"
        f"시총: {fmt_cap(r.market_cap)} | 거래대금20D: {fmt_cap(r.dollar_vol_20)}"
    )


def notify_results(results: List[VCPResult]) -> None:
    if not telegram_enabled():
        return

    blocks: List[str] = [f"[VCP 패턴] 전체 후보: {len(results)}"]
    for r in results[:MAX_TELEGRAM_ROWS]:
        blocks.append(build_result_block(r))

    send_telegram_chunked(blocks)


def save_outputs(results: List[VCPResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if results:
        df = pd.DataFrame([asdict(r) for r in results])
        df["state_rank"] = df["state"].map(STATE_PRIORITY).fillna(9)
        df["rs_rank"] = df["rs_grade"].map({"S": 0, "A": 1, "B": 2}).fillna(9)
        df = df.sort_values(
            ["state_rank", "rs_rank", "score_total", "pivot_distance_pct"],
            ascending=[True, True, False, False],
        ).drop(columns=["state_rank", "rs_rank"])
    else:
        df = pd.DataFrame(columns=[f.name for f in VCPResult.__dataclass_fields__.values()])

    df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "near_breakout": int((df["state"] == "돌파 임박").sum()) if not df.empty else 0,
        "first_breakout": int((df["state"] == "1차 돌파").sum()) if not df.empty else 0,
        "late_breakout": int((df["state"] == "후행 가능").sum()) if not df.empty else 0,
        "rs_s": int((df["rs_grade"] == "S").sum()) if not df.empty else 0,
        "rs_a": int((df["rs_grade"] == "A").sum()) if not df.empty else 0,
        "rs_b": int((df["rs_grade"] == "B").sum()) if not df.empty else 0,
        "as_of_date": str(df["as_of_date"].max()) if not df.empty else None,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    universe = load_universe()
    spy_df = download_history("SPY")
    if spy_df.empty or len(spy_df) < MIN_HISTORY:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    results: List[VCPResult] = []

    for row in universe.itertuples(index=False):
        try:
            r = scan_one(row.ticker, row.name, spy_df)
            if r is not None:
                results.append(r)
        except Exception:
            continue

    results = sorted(
        results,
        key=lambda x: (
            STATE_PRIORITY.get(x.state, 9),
            {"S": 0, "A": 1, "B": 2}.get(x.rs_grade, 9),
            -x.score_total,
            -x.pivot_distance_pct,
            x.ticker,
        ),
    )

    save_outputs(results)
    notify_results(results)


if __name__ == "__main__":
    main()
