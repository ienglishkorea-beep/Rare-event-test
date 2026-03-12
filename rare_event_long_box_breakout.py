import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# Long Box Breakout Scanner
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.log")
STATE_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_state.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
BENCHMARK_TICKER = "SPY"

MIN_HISTORY = 320
DOWNLOAD_PERIOD = "2y"
DOWNLOAD_INTERVAL = "1d"

MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000

BOX_LOOKBACKS = [100, 140, 180, 220]
MIN_BOX_LENGTH = 80
MIN_BOX_WIDTH = 0.08
MAX_BOX_WIDTH = 0.32
BOX_HIGH_QUANTILE = 0.97
BOX_LOW_QUANTILE = 0.03

TOUCH_TOLERANCE = 0.03
TOUCH_RESET_PCT = 0.05

NEAR_PIVOT_PCT = 0.03
MAX_BREAKOUT_EXTEND_PCT = 0.03
BREAKOUT_VOL_RATIO = 1.35
NEAR_52W_HIGH_RATIO = 0.92

ENTRY_BUFFER = 0.002
STOP_BUFFER = 0.0025
MAX_INITIAL_STOP_PCT = 0.08
PYRAMID_LEVELS = [0.02, 0.04, 0.06]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# 텔레그램 안전 길이 (4096보다 약간 작게)
TELEGRAM_MAX_CHARS = 3500


@dataclass
class LongBoxResult:
    ticker: str
    name: str
    as_of_date: str
    state: str
    grade: str
    grade_label: str
    close: float
    box_high: float
    box_low: float
    box_width_pct: float
    box_length: int
    touch_count: int
    near_52w_high: bool
    breakout_volume_ratio: Optional[float]
    entry_price: Optional[float]
    stop_price: Optional[float]
    add_price_1: Optional[float]
    add_price_2: Optional[float]
    add_price_3: Optional[float]
    ma50: Optional[float]
    ma150: Optional[float]
    ma200: Optional[float]
    dollar_vol_20: Optional[float]
    rs_6m_excess: Optional[float]
    recovery_ratio: Optional[float]
    min_recovery_ratio: Optional[float]
    recovery_strength_score: Optional[float]
    recovery_speed_score: Optional[float]
    touch_score: Optional[float]
    length_score: Optional[float]
    width_score: Optional[float]
    volume_score: Optional[float]
    total_score: Optional[float]
    reason: str


def setup_logging() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


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


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def grade_label(score: float) -> Tuple[str, str]:
    if score >= 80:
        return "A", "우선 검토"
    return "B", "관찰 후보"


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("universe.csv에는 ticker 컬럼이 필요")
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
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def add_indicators(df: pd.DataFrame, benchmark: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma150"] = out["Close"].rolling(150).mean()
    out["ma200"] = out["Close"].rolling(200).mean()
    out["vol_ma20"] = out["Volume"].rolling(20).mean()
    out["dollar_vol_20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    out["high_252"] = out["High"].rolling(252).max()
    out["ret_6m"] = rolling_return(out["Close"], 126)
    out["rs_6m_excess"] = np.nan

    if benchmark is not None and len(benchmark) > 0:
        b = benchmark[["Date", "Close"]].rename(columns={"Close": "benchmark_close"})
        x = out.merge(b, on="Date", how="left")
        x["benchmark_close"] = x["benchmark_close"].ffill()
        x["benchmark_ret_6m"] = rolling_return(x["benchmark_close"], 126)
        x["rs_6m_excess"] = x["ret_6m"] - x["benchmark_ret_6m"]
        out = x

    return out


def count_distinct_box_touches(seg: pd.DataFrame, box_high: float) -> int:
    threshold = box_high * (1.0 - TOUCH_TOLERANCE)
    reset_level = box_high * (1.0 - TOUCH_TOLERANCE - TOUCH_RESET_PCT)
    touches = 0
    in_touch = False
    reset_done = True

    for _, row in seg.iterrows():
        high = float(row["High"])
        low = float(row["Low"])

        if high >= threshold and (not in_touch) and reset_done:
            touches += 1
            in_touch = True
            reset_done = False

        if in_touch and low <= reset_level:
            in_touch = False
            reset_done = True

    return touches


def min_recovery_ratio_by_depth(depth: float) -> float:
    if depth <= 0.12:
        return 0.85
    elif depth <= 0.22:
        return 0.72
    return 0.60


def recovery_strength_score(recovery_ratio: float, depth: float) -> float:
    min_req = min_recovery_ratio_by_depth(depth)
    if recovery_ratio < min_req:
        return 0.0
    excess = recovery_ratio - min_req
    if excess >= 0.20:
        return 1.2
    elif excess >= 0.10:
        return 1.0
    elif excess >= 0.05:
        return 0.7
    return 0.4


def recovery_speed_score(right_side_len: int, box_len: int) -> float:
    if box_len <= 0:
        return 0.0
    speed_ratio = right_side_len / box_len
    if 0.20 <= speed_ratio <= 0.40:
        return 1.2
    elif 0.15 <= speed_ratio <= 0.50:
        return 1.0
    return 0.6


def touch_score(touches: int) -> float:
    if touches <= 1:
        return 0.5
    elif touches == 2:
        return 1.0
    elif touches == 3:
        return 1.2
    elif touches == 4:
        return 0.9
    return 0.7


def length_score(box_len: int) -> float:
    if box_len >= 180:
        return 1.2
    elif box_len >= 140:
        return 1.0
    elif box_len >= 100:
        return 0.8
    return 0.5


def width_score(box_width: float) -> float:
    if 0.10 <= box_width <= 0.20:
        return 1.2
    elif 0.08 <= box_width <= 0.25:
        return 1.0
    elif 0.25 < box_width <= MAX_BOX_WIDTH:
        return 0.6
    return 0.0


def breakout_volume_score(vol_ratio: Optional[float]) -> float:
    if vol_ratio is None or pd.isna(vol_ratio):
        return 0.0
    if vol_ratio >= 2.0:
        return 1.2
    elif vol_ratio >= 1.6:
        return 1.0
    elif vol_ratio >= BREAKOUT_VOL_RATIO:
        return 0.8
    return 0.0


def find_best_long_box(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df) < max(BOX_LOOKBACKS):
        return None

    best = None
    best_score = -1e18
    end_idx = len(df) - 1

    for lookback in BOX_LOOKBACKS:
        start_idx = end_idx - lookback + 1
        if start_idx < 0:
            continue

        seg = df.iloc[start_idx:end_idx + 1].copy()
        if len(seg) < MIN_BOX_LENGTH:
            continue

        box_high = float(seg["High"].quantile(BOX_HIGH_QUANTILE))
        box_low = float(seg["Low"].quantile(BOX_LOW_QUANTILE))
        if box_high <= 0 or box_low <= 0 or box_low >= box_high:
            continue

        box_width = (box_high - box_low) / box_high
        if box_width < MIN_BOX_WIDTH or box_width > MAX_BOX_WIDTH:
            continue

        current_close = float(seg.iloc[-1]["Close"])
        recovery_ratio = (current_close - box_low) / max(box_high - box_low, 1e-9)
        min_recovery = min_recovery_ratio_by_depth(box_width)
        if recovery_ratio < min_recovery:
            continue

        low_idx_local = int(np.argmin(seg["Low"].values))
        right_side_len = len(seg) - 1 - low_idx_local
        if right_side_len <= 0:
            continue

        touches = count_distinct_box_touches(seg, box_high)
        s_len = length_score(len(seg))
        s_width = width_score(box_width)
        s_rec_strength = recovery_strength_score(recovery_ratio, box_width)
        s_rec_speed = recovery_speed_score(right_side_len, len(seg))
        s_touch = touch_score(touches)

        score = (
            s_len * 18
            + s_width * 20
            + s_rec_strength * 24
            + s_rec_speed * 18
            + s_touch * 20
        )

        if score > best_score:
            best_score = score
            best = {
                "box_high": box_high,
                "box_low": box_low,
                "box_width_pct": box_width,
                "touch_count": touches,
                "recovery_ratio": recovery_ratio,
                "min_recovery": min_recovery,
                "recovery_strength_score": s_rec_strength,
                "recovery_speed_score": s_rec_speed,
                "touch_score": s_touch,
                "length_score": s_len,
                "width_score": s_width,
                "base_score": score,
                "box_length": len(seg),
            }

    return best


def classify_event(
    df: pd.DataFrame,
    box: Dict[str, Any],
) -> Tuple[
    Optional[str],
    str,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    row = df.iloc[-1]
    close = float(row["Close"])
    box_high = float(box["box_high"])
    box_low = float(box["box_low"])

    vol_ratio = None
    if pd.notna(row["vol_ma20"]) and row["vol_ma20"] > 0:
        vol_ratio = float(row["Volume"] / row["vol_ma20"])

    near_52w = bool(pd.notna(row["high_252"]) and close >= row["high_252"] * NEAR_52W_HIGH_RATIO)
    if not near_52w:
        return None, "52주 고점과 거리 멀음", None, None, None, None, None, vol_ratio

    breakout = close >= box_high
    extended = close > box_high * (1.0 + MAX_BREAKOUT_EXTEND_PCT)
    near_pivot = (close < box_high) and (close >= box_high * (1.0 - NEAR_PIVOT_PCT))

    if breakout and extended:
        return None, "추격 구간", None, None, None, None, None, vol_ratio

    entry_price = box_high * (1.0 + ENTRY_BUFFER)
    structural_stop = box_low * (1.0 - STOP_BUFFER)
    pct_stop = entry_price * (1.0 - MAX_INITIAL_STOP_PCT)
    stop_price = max(structural_stop, pct_stop)
    add_1 = entry_price * (1.0 + PYRAMID_LEVELS[0])
    add_2 = entry_price * (1.0 + PYRAMID_LEVELS[1])
    add_3 = entry_price * (1.0 + PYRAMID_LEVELS[2])

    if breakout:
        if vol_ratio is None or vol_ratio < BREAKOUT_VOL_RATIO:
            return None, "돌파 거래량 부족", None, None, None, None, None, vol_ratio
        return "돌파", "장기 박스 상단 돌파", entry_price, stop_price, add_1, add_2, add_3, vol_ratio

    if near_pivot:
        return "돌파 임박", "장기 박스 상단 근접", entry_price, stop_price, add_1, add_2, add_3, vol_ratio

    return None, "피벗 구간 아님", None, None, None, None, None, vol_ratio


def calculate_total_score(
    box: Dict[str, Any],
    vol_ratio: Optional[float],
    state: str,
    rs_6m_excess: Optional[float],
) -> float:
    s_vol = breakout_volume_score(vol_ratio)

    s_rs = 0.2
    if rs_6m_excess is not None and not pd.isna(rs_6m_excess):
        if rs_6m_excess >= 0.20:
            s_rs = 1.2
        elif rs_6m_excess >= 0.10:
            s_rs = 1.0
        elif rs_6m_excess >= 0.00:
            s_rs = 0.7

    s_state = 1.0 if state == "돌파" else 0.7

    score = (
        box["length_score"] * 15
        + box["width_score"] * 15
        + box["recovery_strength_score"] * 20
        + box["recovery_speed_score"] * 12
        + box["touch_score"] * 14
        + s_vol * 14
        + s_rs * 10
        + s_state * 10
    )
    return round(score, 2)


def scan_one_ticker(ticker: str, name: str, benchmark: pd.DataFrame) -> Optional[LongBoxResult]:
    price_df = download_history(ticker)
    if price_df.empty or len(price_df) < MIN_HISTORY:
        return None

    df = add_indicators(price_df, benchmark)
    row = df.iloc[-1]

    if pd.isna(row["Close"]) or row["Close"] < MIN_PRICE:
        return None
    if pd.isna(row["dollar_vol_20"]) or row["dollar_vol_20"] < MIN_DOLLAR_VOL_20:
        return None

    box = find_best_long_box(df)
    if box is None:
        return None

    state, reason, entry_price, stop_price, add_1, add_2, add_3, vol_ratio = classify_event(df, box)
    if state is None:
        return None

    total_score = calculate_total_score(box, vol_ratio, state, safe_float(row["rs_6m_excess"]))
    grade, grade_text = grade_label(total_score)

    return LongBoxResult(
        ticker=ticker,
        name=name,
        as_of_date=str(pd.to_datetime(row["Date"]).date()),
        state=state,
        grade=grade,
        grade_label=grade_text,
        close=round(float(row["Close"]), 2),
        box_high=round(float(box["box_high"]), 2),
        box_low=round(float(box["box_low"]), 2),
        box_width_pct=round(float(box["box_width_pct"]), 4),
        box_length=int(box["box_length"]),
        touch_count=int(box["touch_count"]),
        near_52w_high=bool(pd.notna(row["high_252"]) and row["Close"] >= row["high_252"] * NEAR_52W_HIGH_RATIO),
        breakout_volume_ratio=round(vol_ratio, 2) if vol_ratio is not None else None,
        entry_price=round_price(entry_price),
        stop_price=round_price(stop_price),
        add_price_1=round_price(add_1),
        add_price_2=round_price(add_2),
        add_price_3=round_price(add_3),
        ma50=round_price(row["ma50"]),
        ma150=round_price(row["ma150"]),
        ma200=round_price(row["ma200"]),
        dollar_vol_20=round_price(row["dollar_vol_20"], 0),
        rs_6m_excess=safe_float(row["rs_6m_excess"]),
        recovery_ratio=round(float(box["recovery_ratio"]), 4),
        min_recovery_ratio=round(float(box["min_recovery"]), 4),
        recovery_strength_score=round(float(box["recovery_strength_score"]), 2),
        recovery_speed_score=round(float(box["recovery_speed_score"]), 2),
        touch_score=round(float(box["touch_score"]), 2),
        length_score=round(float(box["length_score"]), 2),
        width_score=round(float(box["width_score"]), 2),
        volume_score=round(breakout_volume_score(vol_ratio), 2),
        total_score=total_score,
        reason=reason,
    )


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str) -> None:
    if not telegram_enabled():
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, data=payload, timeout=20)
    except Exception:
        pass


def send_telegram_chunked(title: str, lines: List[str], max_chars: int = TELEGRAM_MAX_CHARS) -> None:
    """
    텔레그램 4096자 제한 대응.
    lines 전체를 여러 메시지로 자동 분할 전송.
    """
    if not telegram_enabled():
        return

    if not lines:
        send_telegram_message(title)
        return

    chunks: List[str] = []
    current = title

    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars:
            chunks.append(current)
            current = f"{title}\n{line}"
        else:
            current = candidate

    if current:
        chunks.append(current)

    total = len(chunks)

    if total <= 1:
        send_telegram_message(chunks[0])
        return

    for idx, chunk in enumerate(chunks, start=1):
        header = f"{title} ({idx}/{total})"
        body_lines = chunk.split("\n")[1:] if "\n" in chunk else []
        body = "\n".join(body_lines)
        send_telegram_message(f"{header}\n{body}")


def load_state() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, Dict[str, Any]]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def build_result_lines(r: LongBoxResult, rank: int) -> List[str]:
    return [
        f"{rank}. {r.ticker} {r.name}",
        f"등급: {r.grade} ({r.grade_label}) | 점수: {r.total_score}",
        f"상태: {r.state}",
        f"종가: {format_price(r.close)}",
        f"박스 상단: {format_price(r.box_high)}",
        f"박스 하단: {format_price(r.box_low)}",
        f"박스 길이: {r.box_length}일",
        f"박스 폭: {format_pct(r.box_width_pct)}",
        f"상단 테스트: {r.touch_count}회",
        f"진입가: {format_price(r.entry_price)}",
        f"손절가: {format_price(r.stop_price)}",
        f"1차 추가: {format_price(r.add_price_1)}",
        f"2차 추가: {format_price(r.add_price_2)}",
        f"3차 추가: {format_price(r.add_price_3)}",
        "",
    ]


def sort_results(results: List[LongBoxResult]) -> List[LongBoxResult]:
    state_rank = {
        "돌파": 0,
        "돌파 임박": 1,
    }
    return sorted(
        results,
        key=lambda r: (
            state_rank.get(r.state, 9),
            -(r.total_score if r.total_score is not None else -1e18),
            r.ticker,
        ),
    )


def notify_changes(results: List[LongBoxResult]) -> None:
    """
    요구사항 반영:
    - 알파벳 순 금지
    - 점수순 출력
    - 돌파 / 돌파 임박 분리
    - 종목 수가 많아도 전부 전송
    - 텔레그램 길이 초과 시 자동 분할
    """
    if not telegram_enabled():
        return

    sorted_results = sort_results(results)

    prev_state = load_state()
    new_state: Dict[str, Dict[str, Any]] = {}

    for r in sorted_results:
        new_state[r.ticker] = {
            "state": r.state,
            "grade": r.grade,
            "as_of_date": r.as_of_date,
            "total_score": r.total_score,
        }

    # 매일 전체 후보를 전송
    breakout_results = [r for r in sorted_results if r.state == "돌파"]
    near_results = [r for r in sorted_results if r.state == "돌파 임박"]

    if breakout_results:
        breakout_lines: List[str] = []
        for idx, r in enumerate(breakout_results, start=1):
            breakout_lines.extend(build_result_lines(r, idx))
        send_telegram_chunked("[박스 돌파] 점수순", breakout_lines)

    if near_results:
        near_lines: List[str] = []
        for idx, r in enumerate(near_results, start=1):
            near_lines.extend(build_result_lines(r, idx))
        send_telegram_chunked("[박스 돌파 임박] 점수순", near_lines)

    # 상태 저장
    save_state(new_state)


def save_outputs(results: List[LongBoxResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not results:
        empty_cols = [f.name for f in LongBoxResult.__dataclass_fields__.values()]
        pd.DataFrame(columns=empty_cols).to_csv(
            os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        with open(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_at": datetime.now().isoformat(),
                    "total": 0,
                    "breakout": 0,
                    "near_breakout": 0,
                    "A": 0,
                    "B": 0,
                    "as_of_date": None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    df = pd.DataFrame([asdict(r) for r in sort_results(results)])
    df.to_csv(
        os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "breakout": int((df["state"] == "돌파").sum()),
        "near_breakout": int((df["state"] == "돌파 임박").sum()),
        "A": int((df["grade"] == "A").sum()),
        "B": int((df["grade"] == "B").sum()),
        "as_of_date": str(df["as_of_date"].max()),
    }
    with open(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_logging()
    universe = load_universe()

    benchmark = download_history(BENCHMARK_TICKER)
    if benchmark.empty:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    results: List[LongBoxResult] = []

    for row in universe.itertuples(index=False):
        try:
            result = scan_one_ticker(row.ticker, row.name, benchmark)
            if result is not None:
                results.append(result)
        except Exception as e:
            logging.exception("%s failed: %s", row.ticker, e)

    results = sort_results(results)

    save_outputs(results)
    notify_changes(results)

    if telegram_enabled():
        breakout = sum(r.state == "돌파" for r in results)
        near_breakout = sum(r.state == "돌파 임박" for r in results)
        count_a = sum(r.grade == "A" for r in results)
        count_b = sum(r.grade == "B" for r in results)
        msg = (
            f"[요약] 박스 돌파\n"
            f"날짜: {results[0].as_of_date if results else 'N/A'}\n"
            f"유니버스 수: {len(universe)}\n"
            f"돌파: {breakout}\n"
            f"돌파 임박: {near_breakout}\n"
            f"A: {count_a}\n"
            f"B: {count_b}\n"
            f"총 후보: {len(results)}"
        )
        send_telegram_message(msg)


if __name__ == "__main__":
    main()
