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
MAX_TELEGRAM_ROWS_PER_BUCKET = 20
SEPARATOR = "────────────"


# =========================================================
# CONFIG
# =========================================================
DOWNLOAD_PERIOD = "2y"
MIN_HISTORY = 260
BENCHMARK = "SPY"

MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000
MIN_MARKET_CAP = 1_000_000_000

REQUIRE_MA_ALIGNMENT = True

RS_LOOKBACK = 252
RS_NEW_HIGH_TOL = 0.995

# prior advance
PRIOR_ADVANCE_MIN = 0.25
PRIOR_LOOKBACK_CANDIDATES = [35, 50, 70, 90, 120, 150]

# base / pattern
BASE_LOOKBACK_CANDIDATES = [20, 25, 30, 35, 40, 50, 60, 70, 90]
MAX_BASE_DEPTH = 0.25
MIN_BASE_LENGTH = 15
MAX_BASE_LENGTH = 90

# bucket rules
PULLBACK_MIN_DEPTH = 0.02
PULLBACK_MAX_DEPTH = 0.25

TIGHT_SIDEWAYS_MAX_DEPTH = 0.05
DECELERATING_RISE_MAX_DEPTH = 0.08

PULLBACK_52W_MAX_DIST = -0.30
OTHER_52W_MAX_DIST = -0.08

# pivot state
PIVOT_READY_MAX = 0.00
PIVOT_READY_MIN = -0.03
PIVOT_BREAKOUT_MAX = 0.03
LATE_BREAKOUT_MAX = 0.08

# event / weird pattern filters (kept light)
MAX_SINGLE_DAY_SPIKE = 0.18
MAX_20D_RETURN = 0.45
MIN_10D_RANGE = 0.012

STATE_PRIORITY = {
    "돌파 임박": 0,
    "1차 돌파": 1,
    "후행 가능": 2,
    "관찰": 9,
}

BUCKET_PRIORITY = {
    "조정형": 0,
    "타이트 횡보형": 1,
    "감속 상승형": 2,
}


# =========================================================
# DATACLASS
# =========================================================
@dataclass
class VCPResult:
    ticker: str
    name: str
    as_of_date: str

    bucket: str
    state: str
    grade: str
    score_total: float

    close: float
    pivot_price: float
    entry_price: float
    stop_price: float
    pivot_distance_pct: float

    prior_advance_pct: float
    base_length: int
    base_depth: float
    base_depth_label: str

    range_1: float
    range_2: float
    range_3: float
    range_contract_label: str

    vol_1: float
    vol_2: float
    vol_3: float
    volume_dryup_label: str

    right_side_label: str
    slope_now: float
    slope_prev: float

    rs_grade: str
    rs_percentile: float
    rs_current_vs_high: float

    high_52w_dist: float
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
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
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


def split_three(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    a = max(1, n // 3)
    b = max(a + 1, (2 * n) // 3)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def avg_range_ratio(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    v = ((df["High"] - df["Low"]) / df["Close"]).replace([np.inf, -np.inf], np.nan)
    return float(v.mean())


def avg_volume(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df["Volume"].mean())


def slope_ratio(close: pd.Series) -> float:
    if len(close) < 2:
        return 0.0
    first = safe_float(close.iloc[0])
    last = safe_float(close.iloc[-1])
    if first is None or last is None or first <= 0:
        return 0.0
    return (last / first) - 1.0


def classify_state(pivot_distance: float) -> str:
    if PIVOT_READY_MIN <= pivot_distance <= PIVOT_READY_MAX:
        return "돌파 임박"
    if 0.0 < pivot_distance <= PIVOT_BREAKOUT_MAX:
        return "1차 돌파"
    if PIVOT_BREAKOUT_MAX < pivot_distance <= LATE_BREAKOUT_MAX:
        return "후행 가능"
    return "관찰"


def classify_depth_label(depth: float) -> str:
    if depth < 0.08:
        return "얕은 조정"
    if depth < 0.15:
        return "중간 조정"
    return "깊은 조정"


def classify_strength_monotonic(a: float, b: float, c: float, reverse: bool = False) -> str:
    vals = [a, b, c]
    if any(pd.isna(x) for x in vals):
        return "약"
    if reverse:
        # decreasing preferred
        if a > b > c:
            return "강"
        if a > c:
            return "중"
        return "약"
    if a < b < c:
        return "강"
    if a < c:
        return "중"
    return "약"


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
# CORE DETECTION
# =========================================================
def find_prior_advance_and_base(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Finds:
    - a meaningful prior advance
    - a base starting after that advance peak
    """
    best: Optional[Dict[str, Any]] = None
    n = len(df)

    for lookback in PRIOR_LOOKBACK_CANDIDATES:
        if n < lookback + MIN_BASE_LENGTH:
            continue

        sub = df.tail(lookback).reset_index(drop=True)

        # find low then later high
        best_adv = -1.0
        best_pair: Optional[Tuple[int, int]] = None

        for i in range(0, len(sub) - MIN_BASE_LENGTH - 3):
            low_price = safe_float(sub["Low"].iloc[i])
            if low_price is None or low_price <= 0:
                continue

            later_high = sub["High"].iloc[i + 3 :].max()
            later_high = safe_float(later_high)
            if later_high is None:
                continue

            adv = (later_high / low_price) - 1.0
            if adv > best_adv:
                peak_rel = int(sub["High"].iloc[i + 3 :].idxmax())
                best_adv = adv
                best_pair = (i, peak_rel)

        if best_pair is None or best_adv < PRIOR_ADVANCE_MIN:
            continue

        low_rel, peak_rel = best_pair
        peak_abs = n - lookback + peak_rel

        # base begins after peak
        base_len = n - peak_abs
        if base_len < MIN_BASE_LENGTH or base_len > MAX_BASE_LENGTH:
            continue

        base_df = df.iloc[peak_abs:].copy().reset_index(drop=True)
        base_high = safe_float(base_df["High"].max())
        base_low = safe_float(base_df["Low"].min())
        if base_high is None or base_low is None or base_high <= 0:
            continue

        base_depth = (base_high - base_low) / base_high

        candidate = {
            "prior_advance_pct": best_adv,
            "peak_abs": peak_abs,
            "base_df": base_df,
            "base_length": len(base_df),
            "base_high": base_high,
            "base_low": base_low,
            "base_depth": base_depth,
        }

        # prioritize stronger prior advance, then more recent base
        if best is None:
            best = candidate
        else:
            if candidate["prior_advance_pct"] > best["prior_advance_pct"]:
                best = candidate
            elif candidate["prior_advance_pct"] == best["prior_advance_pct"] and candidate["base_length"] < best["base_length"]:
                best = candidate

    return best


def classify_bucket(base_df: pd.DataFrame, base_depth: float) -> Optional[str]:
    """
    3 buckets:
    1) 조정형: real pullback, 2%~25%
    2) 타이트 횡보형: 0%~5%, flat/high hold
    3) 감속 상승형: 0%~8%, still rising but slowing
    """
    if len(base_df) < MIN_BASE_LENGTH:
        return None

    first, mid, last = split_three(base_df)
    close = base_df["Close"]

    slope_total = slope_ratio(close)
    slope_prev = slope_ratio(close.iloc[: max(2, len(close) // 2)])
    slope_now = slope_ratio(close.iloc[max(0, len(close) // 2):])

    range_1 = avg_range_ratio(first)
    range_2 = avg_range_ratio(mid)
    range_3 = avg_range_ratio(last)
    vol_contract_ok = range_1 > range_3

    vol_1 = avg_volume(first)
    vol_2 = avg_volume(mid)
    vol_3 = avg_volume(last)
    volume_dryup_ok = vol_1 > vol_3

    left_low = safe_float(first["Low"].min())
    right_low = safe_float(last["Low"].min())
    right_side_ok = (
        left_low is not None
        and right_low is not None
        and right_low >= left_low * 0.99
    )

    if not vol_contract_ok or not volume_dryup_ok:
        return None

    # 1) classic pullback
    if PULLBACK_MIN_DEPTH <= base_depth <= PULLBACK_MAX_DEPTH and right_side_ok:
        return "조정형"

    # 2) tight sideways
    if 0.0 <= base_depth <= TIGHT_SIDEWAYS_MAX_DEPTH and right_side_ok:
        # sideways means slope not strongly down and not strongly rising
        if abs(slope_total) <= 0.08:
            return "타이트 횡보형"

    # 3) decelerating rise
    if 0.0 <= base_depth <= DECELERATING_RISE_MAX_DEPTH:
        if slope_total > 0 and slope_now >= 0 and slope_now < slope_prev:
            return "감속 상승형"

    return None


def score_pattern(
    bucket: str,
    prior_advance_pct: float,
    pivot_distance: float,
    base_depth: float,
    range_1: float,
    range_2: float,
    range_3: float,
    vol_1: float,
    vol_2: float,
    vol_3: float,
    rs_grade: str,
    rs_percentile: float,
    high_52w_dist: float,
    right_side_ok: bool,
) -> float:
    score = 0.0

    # prior advance
    if prior_advance_pct >= 0.60:
        score += 18
    elif prior_advance_pct >= 0.40:
        score += 14
    else:
        score += 10

    # bucket preference
    if bucket == "조정형":
        score += 18
    elif bucket == "타이트 횡보형":
        score += 15
    elif bucket == "감속 상승형":
        score += 15

    # volatility contraction
    if range_1 > range_2 > range_3:
        score += 15
    elif range_1 > range_3:
        score += 10

    # volume dry-up
    if vol_1 > vol_2 > vol_3:
        score += 15
    elif vol_1 > vol_3:
        score += 10

    # right side
    if right_side_ok:
        score += 10

    # depth
    if bucket == "조정형":
        if 0.04 <= base_depth <= 0.15:
            score += 10
        elif base_depth <= 0.25:
            score += 6
    else:
        if base_depth <= 0.05:
            score += 10
        else:
            score += 6

    # pivot distance
    if -0.02 <= pivot_distance <= 0.01:
        score += 10
    elif -0.04 <= pivot_distance <= 0.03:
        score += 6

    # RS
    if rs_grade == "S":
        score += 12
    elif rs_grade == "A":
        score += 8
    else:
        score += 4

    if rs_percentile >= 98:
        score += 7
    elif rs_percentile >= 95:
        score += 4

    # 52w proximity as helper only
    if bucket == "조정형":
        if high_52w_dist >= -0.15:
            score += 5
        elif high_52w_dist >= -0.30:
            score += 2
    else:
        if high_52w_dist >= -0.05:
            score += 5
        elif high_52w_dist >= -0.08:
            score += 2

    return round(score, 1)


def grade_from_score(score: float) -> str:
    if score >= 82:
        return "A"
    if score >= 68:
        return "B"
    return "WATCH"


def scan_one(ticker: str, name: str, spy_df: pd.DataFrame) -> Optional[VCPResult]:
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

    core = find_prior_advance_and_base(df)
    if core is None:
        return None

    prior_advance_pct = float(core["prior_advance_pct"])
    base_df = core["base_df"]
    base_length = int(core["base_length"])
    base_depth = float(core["base_depth"])
    base_high = float(core["base_high"])

    bucket = classify_bucket(base_df, base_depth)
    if bucket is None:
        return None

    # 52w filter by bucket
    high_52w = safe_float(df["High"].rolling(252).max().iloc[-1])
    if high_52w is None or high_52w <= 0:
        return None
    high_52w_dist = (close / high_52w) - 1.0

    if bucket == "조정형":
        if high_52w_dist < PULLBACK_52W_MAX_DIST:
            return None
    else:
        if high_52w_dist < OTHER_52W_MAX_DIST:
            return None

    # current base high / pivot
    pivot = base_high
    pivot_distance = (close / pivot) - 1.0 if pivot > 0 else 0.0
    state = classify_state(pivot_distance)
    if state == "관찰":
        return None

    first, mid, last = split_three(base_df)
    range_1 = avg_range_ratio(first)
    range_2 = avg_range_ratio(mid)
    range_3 = avg_range_ratio(last)
    vol_1 = avg_volume(first)
    vol_2 = avg_volume(mid)
    vol_3 = avg_volume(last)

    left_low = safe_float(first["Low"].min())
    right_low = safe_float(last["Low"].min())
    right_side_ok = (
        left_low is not None
        and right_low is not None
        and right_low >= left_low * 0.99
    )

    score_total = score_pattern(
        bucket=bucket,
        prior_advance_pct=prior_advance_pct,
        pivot_distance=pivot_distance,
        base_depth=base_depth,
        range_1=range_1,
        range_2=range_2,
        range_3=range_3,
        vol_1=vol_1,
        vol_2=vol_2,
        vol_3=vol_3,
        rs_grade=rs_grade,
        rs_percentile=rs_percentile,
        high_52w_dist=high_52w_dist,
        right_side_ok=right_side_ok,
    )
    grade = grade_from_score(score_total)

    entry_price = round(pivot * 1.003, 2)
    stop_basis = max(base_depth * 0.7, 0.06)
    stop_price = round(entry_price * (1 - stop_basis), 2)

    slope_prev = slope_ratio(base_df["Close"].iloc[: max(2, len(base_df) // 2)])
    slope_now = slope_ratio(base_df["Close"].iloc[max(0, len(base_df) // 2):])

    return VCPResult(
        ticker=ticker,
        name=name,
        as_of_date=str(df["Date"].iloc[-1].date()),
        bucket=bucket,
        state=state,
        grade=grade,
        score_total=score_total,
        close=round(close, 2),
        pivot_price=round(pivot, 2),
        entry_price=entry_price,
        stop_price=stop_price,
        pivot_distance_pct=round(pivot_distance, 4),
        prior_advance_pct=round(prior_advance_pct, 4),
        base_length=base_length,
        base_depth=round(base_depth, 4),
        base_depth_label=classify_depth_label(base_depth) if bucket == "조정형" else "시간 소화",
        range_1=round(range_1, 4),
        range_2=round(range_2, 4),
        range_3=round(range_3, 4),
        range_contract_label=classify_strength_monotonic(range_1, range_2, range_3, reverse=True),
        vol_1=round(vol_1, 2),
        vol_2=round(vol_2, 2),
        vol_3=round(vol_3, 2),
        volume_dryup_label=classify_strength_monotonic(vol_1, vol_2, vol_3, reverse=True),
        right_side_label="YES" if right_side_ok else "NO",
        slope_now=round(slope_now, 4),
        slope_prev=round(slope_prev, 4),
        rs_grade=rs_grade,
        rs_percentile=round(rs_percentile, 2),
        rs_current_vs_high=round(rs_current_vs_high, 4),
        high_52w_dist=round(high_52w_dist, 4),
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
        f"유형: {r.bucket}\n"
        f"상태: {r.state}\n"
        f"등급: {r.grade} | 점수: {r.score_total:.1f}/100\n"
        f"종가: {fmt_price(r.close)}\n"
        f"피벗: {fmt_price(r.pivot_price)}\n"
        f"피벗 거리: {fmt_pct(r.pivot_distance_pct)}\n"
        f"선행 상승: {fmt_pct(r.prior_advance_pct)}\n"
        f"베이스 길이: {r.base_length}일\n"
        f"베이스 깊이: {fmt_pct(r.base_depth)} | {r.base_depth_label}\n"
        f"변동성: {fmt_pct(r.range_1)} → {fmt_pct(r.range_2)} → {fmt_pct(r.range_3)} | {r.range_contract_label}\n"
        f"거래량: {fmt_cap(r.vol_1)} → {fmt_cap(r.vol_2)} → {fmt_cap(r.vol_3)} | {r.volume_dryup_label}\n"
        f"오른쪽 구조: {r.right_side_label}\n"
        f"기울기: 이전 {fmt_pct(r.slope_prev)} | 최근 {fmt_pct(r.slope_now)}\n"
        f"RS: {r.rs_grade} | RS Percentile: {r.rs_percentile:.1f}\n"
        f"52주 고점 거리: {fmt_pct(r.high_52w_dist)}\n"
        f"시총: {fmt_cap(r.market_cap)} | 거래대금20D: {fmt_cap(r.dollar_vol_20)}"
    )


def notify_bucket(results: List[VCPResult], bucket: str) -> None:
    if not telegram_enabled() or not results:
        return

    title = f"[VCP {bucket}] 전체 후보: {len(results)}"
    blocks: List[str] = [title]

    for r in results[:MAX_TELEGRAM_ROWS_PER_BUCKET]:
        blocks.append(build_result_block(r))

    send_telegram_chunked(blocks)


def save_outputs(results: List[VCPResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if results:
        df = pd.DataFrame([asdict(r) for r in results])
        df["bucket_rank"] = df["bucket"].map(BUCKET_PRIORITY).fillna(9)
        df["state_rank"] = df["state"].map(STATE_PRIORITY).fillna(9)
        df["rs_rank"] = df["rs_grade"].map({"S": 0, "A": 1, "B": 2}).fillna(9)

        df = df.sort_values(
            ["bucket_rank", "state_rank", "rs_rank", "score_total", "pivot_distance_pct"],
            ascending=[True, True, True, False, False],
        ).drop(columns=["bucket_rank", "state_rank", "rs_rank"])
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
        "pullback_count": int((df["bucket"] == "조정형").sum()) if not df.empty else 0,
        "tight_sideways_count": int((df["bucket"] == "타이트 횡보형").sum()) if not df.empty else 0,
        "decelerating_rise_count": int((df["bucket"] == "감속 상승형").sum()) if not df.empty else 0,
        "as_of_date": str(df["as_of_date"].max()) if not df.empty else None,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    universe = load_universe()
    spy_df = download_history(BENCHMARK)
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
            BUCKET_PRIORITY.get(x.bucket, 9),
            STATE_PRIORITY.get(x.state, 9),
            {"S": 0, "A": 1, "B": 2}.get(x.rs_grade, 9),
            -x.score_total,
            -x.pivot_distance_pct,
            x.ticker,
        ),
    )

    save_outputs(results)

    pullback = [r for r in results if r.bucket == "조정형"]
    tight_sideways = [r for r in results if r.bucket == "타이트 횡보형"]
    decelerating_rise = [r for r in results if r.bucket == "감속 상승형"]

    notify_bucket(pullback, "조정형")
    notify_bucket(tight_sideways, "타이트 횡보형")
    notify_bucket(decelerating_rise, "감속 상승형")


if __name__ == "__main__":
    main()
