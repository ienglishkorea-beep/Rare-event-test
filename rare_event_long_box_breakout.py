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
# Long Box Breakout Scanner - RS Hard Cut + Box Shape Filters
# ------------------------------------------------------------
# 핵심
# 1) RS Percentile >= 80 하드컷
# 2) RS 등급 표기
#    - S: RS line 신고가
#    - A: RS Percentile >= 90
#    - B: RS Percentile >= 80
# 3) 매 실행마다 전체 후보를 텔레그램으로 전송
# 4) 박스 로직
#    - pivot 고정
#    - 긴 박스 우선
#    - depth <= 25%
# 5) 박스 형태 필터 추가
#    - 완만한 상승 채널 제거
#    - 사이클성 우상향 구간 제거
# ============================================================


# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.log")
STATE_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_state.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
BENCHMARK_TICKER = "SPY"


# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
MIN_HISTORY = 320
DOWNLOAD_PERIOD = "2y"
DOWNLOAD_INTERVAL = "1d"

MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000

# 긴 박스 우선
BOX_LOOKBACKS = [220, 180, 140, 100]
MIN_BOX_LENGTH = 80

# 박스 폭 제한
MIN_BOX_WIDTH = 0.08
MAX_BOX_WIDTH = 0.25

# pivot 계산
BOX_HIGH_QUANTILE = 0.97
BOX_LOW_QUANTILE = 0.03

# 상단 테스트
TOUCH_TOLERANCE = 0.025
TOUCH_RESET_PCT = 0.05

# 상태 구분 (pivot 기준 고정)
NEAR_BREAKOUT_LOW = -0.03
NEAR_BREAKOUT_HIGH = 0.03
FIRST_BREAKOUT_HIGH = 0.08
LATE_BREAKOUT_HIGH = 0.10

# 돌파 품질
BREAKOUT_VOL_RATIO = 1.35
NEAR_52W_HIGH_RATIO = 0.92

# 실행 계획
ENTRY_BUFFER = 0.002
STOP_BUFFER = 0.0025
MAX_INITIAL_STOP_PCT = 0.08
PYRAMID_LEVELS = [0.02, 0.04, 0.06]

# RS 설정
RS_LOOKBACK = 252
RS_PERCENTILE_HARD_CUT = 80.0
RS_GRADE_A = 90.0
RS_NEAR_HIGH_MIN_RATIO = 0.80
RS_NEW_HIGH_TOLERANCE = 0.995

# 박스 형태 필터
# 1) 박스 시작점 대비 pivot 상승률 제한
TREND_SLOPE_MAX = 0.18
# 2) 박스 구간 종가 회귀 기울기 제한 (일평균 상대 기울기)
REGRESSION_SLOPE_MAX = 0.0025
# 3) 상단/하단 동시 상승 채널 제거
CHANNEL_SLOPE_MAX = 0.0020

# 텔레그램
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_MESSAGE_LEN = 3500
MAX_TELEGRAM_ROWS = 200


# ------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------
@dataclass
class LongBoxResult:
    ticker: str
    name: str
    as_of_date: str

    state: str
    grade: str
    grade_label: str
    rs_grade: str

    total_score: float

    close: float
    pivot_price: float
    box_low: float
    box_width_pct: float
    box_length: int
    pivot_distance_pct: float

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
    rs_percentile: Optional[float]
    rs_current_vs_high: Optional[float]
    rs_new_high: bool

    recovery_ratio: Optional[float]
    min_recovery_ratio: Optional[float]

    score_length: float
    score_width: float
    score_touch: float
    score_recovery_strength: float
    score_recovery_speed: float
    score_volume: float
    score_rs: float
    score_state: float

    reason: str


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------
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


def rs_grade_rank(rs_grade: str) -> int:
    return {"S": 0, "A": 1, "B": 2}.get(rs_grade, 9)


def grade_rank(grade: str) -> int:
    return {"A": 0, "B": 1}.get(grade, 9)


def state_rank(state: str) -> int:
    return {
        "돌파 임박": 0,
        "1차 돌파": 1,
        "후행 가능": 2,
    }.get(state, 9)


def linear_slope_ratio(values: np.ndarray) -> float:
    """
    종가/상단/하단 시계열의 회귀 기울기를
    평균값 대비 상대 기울기로 변환
    """
    if len(values) < 10:
        return 0.0

    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)

    if np.any(np.isnan(y)):
        return 0.0

    mean_y = float(np.mean(y))
    if mean_y <= 0:
        return 0.0

    slope = np.polyfit(x, y, 1)[0]
    return float(slope / mean_y)


# ------------------------------------------------------------
# 로드
# ------------------------------------------------------------
def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"유니버스 파일 없음: {UNIVERSE_FILE}")

    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise ValueError("universe.csv에는 ticker 컬럼이 필요")
    if "name" not in df.columns:
        df["name"] = df["ticker"]

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str)

    return df[["ticker", "name"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


# ------------------------------------------------------------
# 다운로드 / 정규화
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 지표
# ------------------------------------------------------------
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
    out["rs_line"] = np.nan
    out["rs_percentile"] = np.nan
    out["rs_current_vs_high"] = np.nan
    out["rs_new_high"] = False

    if benchmark is not None and len(benchmark) > 0:
        b = benchmark[["Date", "Close"]].rename(columns={"Close": "benchmark_close"})
        x = out.merge(b, on="Date", how="left")
        x["benchmark_close"] = x["benchmark_close"].ffill()
        x["benchmark_ret_6m"] = rolling_return(x["benchmark_close"], 126)
        x["rs_6m_excess"] = x["ret_6m"] - x["benchmark_ret_6m"]

        x["rs_line"] = x["Close"] / x["benchmark_close"]
        x["rs_high_252"] = x["rs_line"].rolling(RS_LOOKBACK).max()
        x["rs_current_vs_high"] = x["rs_line"] / x["rs_high_252"]
        x["rs_new_high"] = x["rs_current_vs_high"] >= RS_NEW_HIGH_TOLERANCE

        rs_min_252 = x["rs_line"].rolling(RS_LOOKBACK).min()
        denom = (x["rs_high_252"] - rs_min_252).replace(0, np.nan)
        x["rs_percentile"] = ((x["rs_line"] - rs_min_252) / denom) * 100.0

        out = x

    return out


# ------------------------------------------------------------
# 박스 평가
# ------------------------------------------------------------
def count_distinct_box_touches(seg: pd.DataFrame, pivot_price: float) -> int:
    threshold = pivot_price * (1.0 - TOUCH_TOLERANCE)
    reset_level = pivot_price * (1.0 - TOUCH_TOLERANCE - TOUCH_RESET_PCT)

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
    elif depth <= 0.18:
        return 0.75
    return 0.65


def recovery_strength_score(recovery_ratio: float, depth: float) -> float:
    min_req = min_recovery_ratio_by_depth(depth)
    if recovery_ratio < min_req:
        return 0.0

    excess = recovery_ratio - min_req
    if excess >= 0.20:
        return 1.2
    elif excess >= 0.12:
        return 1.0
    elif excess >= 0.06:
        return 0.8
    return 0.5


def recovery_speed_score(right_side_len: int, box_len: int) -> float:
    if box_len <= 0:
        return 0.0
    speed_ratio = right_side_len / box_len
    if 0.18 <= speed_ratio <= 0.38:
        return 1.2
    elif 0.15 <= speed_ratio <= 0.45:
        return 1.0
    return 0.7


def touch_score(touches: int) -> float:
    if touches <= 1:
        return 0.4
    elif touches == 2:
        return 0.8
    elif touches == 3:
        return 1.2
    elif touches == 4:
        return 1.0
    return 0.8


def length_score(box_len: int) -> float:
    if box_len >= 180:
        return 1.2
    elif box_len >= 140:
        return 1.0
    elif box_len >= 100:
        return 0.8
    return 0.5


def width_score(box_width: float) -> float:
    if 0.10 <= box_width <= 0.18:
        return 1.2
    elif 0.08 <= box_width <= 0.22:
        return 1.0
    elif 0.22 < box_width <= MAX_BOX_WIDTH:
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
    return 0.4


def rs_score(rs_percentile: Optional[float], rs_current_vs_high: Optional[float], rs_new_high: bool) -> float:
    if rs_percentile is None or pd.isna(rs_percentile):
        return 0.0

    if rs_new_high:
        return 1.2
    if rs_percentile >= 90:
        return 1.0
    if rs_percentile >= 80 and rs_current_vs_high is not None and not pd.isna(rs_current_vs_high) and rs_current_vs_high >= RS_NEAR_HIGH_MIN_RATIO:
        return 0.7
    return 0.0


def state_score(state: str) -> float:
    if state == "돌파 임박":
        return 1.2
    elif state == "1차 돌파":
        return 1.0
    elif state == "후행 가능":
        return 0.7
    return 0.0


def box_shape_filters_pass(seg: pd.DataFrame, pivot_price: float, box_low: float) -> bool:
    """
    완만한 상승 채널을 박스로 오인하는 경우 제거
    """
    if len(seg) < 40:
        return False

    start_price = float(seg["Close"].iloc[0])
    if start_price <= 0:
        return False

    # 1) 박스 시작점 대비 pivot 상승률 제한
    trend_slope = (pivot_price / start_price) - 1.0
    if trend_slope > TREND_SLOPE_MAX:
        return False

    # 2) 종가 회귀 기울기 제한
    close_slope_ratio = linear_slope_ratio(seg["Close"].values)
    if close_slope_ratio > REGRESSION_SLOPE_MAX:
        return False

    # 3) 상단/하단이 같이 우상향하는 채널 제거
    roll_high = seg["High"].rolling(10).max().dropna()
    roll_low = seg["Low"].rolling(10).min().dropna()

    if len(roll_high) >= 20 and len(roll_low) >= 20:
        upper_slope_ratio = linear_slope_ratio(roll_high.values)
        lower_slope_ratio = linear_slope_ratio(roll_low.values)

        if upper_slope_ratio > CHANNEL_SLOPE_MAX and lower_slope_ratio > CHANNEL_SLOPE_MAX:
            return False

    return True


def find_selected_box(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    end_idx = len(df) - 1

    for lookback in BOX_LOOKBACKS:
        start_idx = end_idx - lookback + 1
        if start_idx < 0:
            continue

        seg = df.iloc[start_idx:end_idx + 1].copy()
        if len(seg) < MIN_BOX_LENGTH:
            continue

        pivot_price = float(seg["High"].quantile(BOX_HIGH_QUANTILE))
        box_low = float(seg["Low"].quantile(BOX_LOW_QUANTILE))

        if pivot_price <= 0 or box_low <= 0 or box_low >= pivot_price:
            continue

        depth = (pivot_price - box_low) / pivot_price
        if depth < MIN_BOX_WIDTH or depth > MAX_BOX_WIDTH:
            continue

        if not box_shape_filters_pass(seg=seg, pivot_price=pivot_price, box_low=box_low):
            continue

        current_close = float(seg.iloc[-1]["Close"])
        recovery_ratio = (current_close - box_low) / max(pivot_price - box_low, 1e-9)
        min_recovery = min_recovery_ratio_by_depth(depth)
        if recovery_ratio < min_recovery:
            continue

        low_idx_local = int(np.argmin(seg["Low"].values))
        right_side_len = len(seg) - 1 - low_idx_local
        if right_side_len <= 0:
            continue

        touches = count_distinct_box_touches(seg, pivot_price)

        return {
            "pivot_price": pivot_price,
            "box_low": box_low,
            "box_width_pct": depth,
            "box_length": len(seg),
            "touch_count": touches,
            "recovery_ratio": recovery_ratio,
            "min_recovery_ratio": min_recovery,
            "score_length": length_score(len(seg)),
            "score_width": width_score(depth),
            "score_touch": touch_score(touches),
            "score_recovery_strength": recovery_strength_score(recovery_ratio, depth),
            "score_recovery_speed": recovery_speed_score(right_side_len, len(seg)),
        }

    return None


def classify_state(close: float, pivot_price: float) -> Tuple[Optional[str], float, str]:
    distance = close / pivot_price - 1.0

    if distance < NEAR_BREAKOUT_LOW:
        return None, distance, "pivot 아래 너무 멀음"

    if NEAR_BREAKOUT_LOW <= distance <= NEAR_BREAKOUT_HIGH:
        return "돌파 임박", distance, "pivot -3% ~ +3%"

    if NEAR_BREAKOUT_HIGH < distance <= FIRST_BREAKOUT_HIGH:
        return "1차 돌파", distance, "pivot +3% ~ +8%"

    if FIRST_BREAKOUT_HIGH < distance <= LATE_BREAKOUT_HIGH:
        return "후행 가능", distance, "pivot +8% ~ +10%"

    return None, distance, "pivot +10% 초과"


def build_trade_plan(pivot_price: float, box_low: float) -> Tuple[float, float, float, float, float]:
    entry_price = pivot_price * (1.0 + ENTRY_BUFFER)
    structural_stop = box_low * (1.0 - STOP_BUFFER)
    pct_stop = entry_price * (1.0 - MAX_INITIAL_STOP_PCT)
    stop_price = max(structural_stop, pct_stop)
    add_1 = entry_price * (1.0 + PYRAMID_LEVELS[0])
    add_2 = entry_price * (1.0 + PYRAMID_LEVELS[1])
    add_3 = entry_price * (1.0 + PYRAMID_LEVELS[2])
    return entry_price, stop_price, add_1, add_2, add_3


def get_rs_grade(rs_percentile: float, rs_new_high: bool) -> str:
    if rs_new_high:
        return "S"
    if rs_percentile >= RS_GRADE_A:
        return "A"
    return "B"


def calculate_total_score(
    box: Dict[str, Any],
    state: str,
    vol_ratio: Optional[float],
    rs_percentile: Optional[float],
    rs_current_vs_high: Optional[float],
    rs_new_high: bool,
) -> float:
    total = (
        box["score_length"] * 15
        + box["score_width"] * 15
        + box["score_touch"] * 14
        + box["score_recovery_strength"] * 20
        + box["score_recovery_speed"] * 12
        + breakout_volume_score(vol_ratio) * 14
        + rs_score(rs_percentile, rs_current_vs_high, rs_new_high) * 10
        + state_score(state) * 10
    )
    return round(total, 2)


# ------------------------------------------------------------
# 스캔
# ------------------------------------------------------------
def scan_one_ticker(ticker: str, name: str, benchmark: pd.DataFrame) -> Optional[LongBoxResult]:
    price_df = download_history(ticker)
    if price_df.empty or len(price_df) < MIN_HISTORY:
        return None

    df = add_indicators(price_df, benchmark)
    row = df.iloc[-1]

    close = float(row["Close"])
    if close < MIN_PRICE:
        return None

    dollar_vol_20 = safe_float(row["dollar_vol_20"])
    if dollar_vol_20 is None or dollar_vol_20 < MIN_DOLLAR_VOL_20:
        return None

    # RS 하드컷
    rs_percentile = safe_float(row["rs_percentile"])
    rs_current_vs_high = safe_float(row["rs_current_vs_high"])
    rs_new_high = bool(row["rs_new_high"]) if "rs_new_high" in row.index else False

    if rs_percentile is None or rs_percentile < RS_PERCENTILE_HARD_CUT:
        return None

    if rs_current_vs_high is None or rs_current_vs_high < RS_NEAR_HIGH_MIN_RATIO:
        return None

    box = find_selected_box(df)
    if box is None:
        return None

    pivot_price = float(box["pivot_price"])
    state, distance, state_reason = classify_state(close, pivot_price)
    if state is None:
        return None

    near_52w_high = bool(pd.notna(row["high_252"]) and close >= row["high_252"] * NEAR_52W_HIGH_RATIO)
    if not near_52w_high:
        return None

    vol_ratio = None
    if pd.notna(row["vol_ma20"]) and row["vol_ma20"] > 0:
        vol_ratio = float(row["Volume"] / row["vol_ma20"])

    entry_price, stop_price, add_1, add_2, add_3 = build_trade_plan(
        pivot_price=pivot_price,
        box_low=float(box["box_low"]),
    )

    total_score = calculate_total_score(
        box=box,
        state=state,
        vol_ratio=vol_ratio,
        rs_percentile=rs_percentile,
        rs_current_vs_high=rs_current_vs_high,
        rs_new_high=rs_new_high,
    )
    grade, grade_text = grade_label(total_score)
    rs_grade = get_rs_grade(rs_percentile=rs_percentile, rs_new_high=rs_new_high)

    return LongBoxResult(
        ticker=ticker,
        name=name,
        as_of_date=str(pd.to_datetime(row["Date"]).date()),
        state=state,
        grade=grade,
        grade_label=grade_text,
        rs_grade=rs_grade,
        total_score=total_score,
        close=round(close, 2),
        pivot_price=round(pivot_price, 2),
        box_low=round(float(box["box_low"]), 2),
        box_width_pct=round(float(box["box_width_pct"]), 4),
        box_length=int(box["box_length"]),
        pivot_distance_pct=round(distance, 4),
        touch_count=int(box["touch_count"]),
        near_52w_high=near_52w_high,
        breakout_volume_ratio=round(vol_ratio, 2) if vol_ratio is not None else None,
        entry_price=round_price(entry_price),
        stop_price=round_price(stop_price),
        add_price_1=round_price(add_1),
        add_price_2=round_price(add_2),
        add_price_3=round_price(add_3),
        ma50=round_price(safe_float(row["ma50"])),
        ma150=round_price(safe_float(row["ma150"])),
        ma200=round_price(safe_float(row["ma200"])),
        dollar_vol_20=round_price(dollar_vol_20, 0),
        rs_6m_excess=safe_float(row["rs_6m_excess"]),
        rs_percentile=round(rs_percentile, 2) if rs_percentile is not None else None,
        rs_current_vs_high=round(rs_current_vs_high, 4) if rs_current_vs_high is not None else None,
        rs_new_high=rs_new_high,
        recovery_ratio=round(float(box["recovery_ratio"]), 4),
        min_recovery_ratio=round(float(box["min_recovery_ratio"]), 4),
        score_length=round(float(box["score_length"]), 2),
        score_width=round(float(box["score_width"]), 2),
        score_touch=round(float(box["score_touch"]), 2),
        score_recovery_strength=round(float(box["score_recovery_strength"]), 2),
        score_recovery_speed=round(float(box["score_recovery_speed"]), 2),
        score_volume=round(breakout_volume_score(vol_ratio), 2),
        score_rs=round(rs_score(rs_percentile, rs_current_vs_high, rs_new_high), 2),
        score_state=round(state_score(state), 2),
        reason=state_reason,
    )


# ------------------------------------------------------------
# 텔레그램 / 상태
# ------------------------------------------------------------
def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str) -> None:
    if not telegram_enabled():
        return
    try:
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
    except Exception:
        pass


def send_telegram_message_chunked(lines: List[str]) -> None:
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
        send_telegram_message(chunk)


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


def build_result_lines(r: LongBoxResult) -> List[str]:
    rs_line = f"{r.rs_percentile:.1f}" if r.rs_percentile is not None else "-"
    rs_high_line = format_pct(r.rs_current_vs_high - 1.0) if r.rs_current_vs_high is not None else "-"
    return [
        f"- {r.ticker} {r.name}",
        f"  RS: {r.rs_grade} | RS Percentile: {rs_line} | RS 고점 거리: {rs_high_line}",
        f"  등급: {r.grade} ({r.grade_label}) | 점수: {r.total_score}/100",
        f"  상태: {r.state}",
        f"  종가: {format_price(r.close)}",
        f"  고정 피벗: {format_price(r.pivot_price)}",
        f"  피벗 거리: {format_pct(r.pivot_distance_pct)}",
        f"  박스 길이: {r.box_length}일",
        f"  박스 폭: {format_pct(r.box_width_pct)}",
        f"  상단 테스트: {r.touch_count}회",
        f"  진입가: {format_price(r.entry_price)}",
        f"  손절가: {format_price(r.stop_price)}",
        f"  1차 추가: {format_price(r.add_price_1)}",
        f"  2차 추가: {format_price(r.add_price_2)}",
        f"  3차 추가: {format_price(r.add_price_3)}",
    ]


def notify_all_results(results: List[LongBoxResult]) -> None:
    if not telegram_enabled():
        return

    sorted_results = sorted(
        results,
        key=lambda x: (
            rs_grade_rank(x.rs_grade),
            grade_rank(x.grade),
            state_rank(x.state),
            -x.total_score,
            x.ticker,
        ),
    )

    header = ["[박스 돌파] Long Box Breakout", f"전체 후보: {len(sorted_results)}"]
    all_lines: List[str] = header[:]

    for r in sorted_results[:MAX_TELEGRAM_ROWS]:
        all_lines.extend(build_result_lines(r))

    send_telegram_message_chunked(all_lines)

    new_state: Dict[str, Dict[str, Any]] = {}
    for r in sorted_results:
        new_state[r.ticker] = {
            "state": r.state,
            "grade": r.grade,
            "rs_grade": r.rs_grade,
            "as_of_date": r.as_of_date,
        }
    save_state(new_state)


# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
def save_outputs(results: List[LongBoxResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = [f.name for f in LongBoxResult.__dataclass_fields__.values()]

    if not results:
        pd.DataFrame(columns=cols).to_csv(
            os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        with open(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_at": datetime.now().isoformat(),
                    "total": 0,
                    "near_breakout": 0,
                    "first_breakout": 0,
                    "late_breakout": 0,
                    "rs_s": 0,
                    "rs_a": 0,
                    "rs_b": 0,
                    "A": 0,
                    "B": 0,
                    "as_of_date": None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    df = pd.DataFrame([asdict(r) for r in results])

    df["rs_rank"] = df["rs_grade"].map({"S": 0, "A": 1, "B": 2}).fillna(9)
    df["grade_rank"] = df["grade"].map({"A": 0, "B": 1}).fillna(9)
    df["state_rank"] = df["state"].map({"돌파 임박": 0, "1차 돌파": 1, "후행 가능": 2}).fillna(9)

    df = df.sort_values(
        ["rs_rank", "grade_rank", "state_rank", "total_score", "ticker"],
        ascending=[True, True, True, False, True],
    ).drop(columns=["rs_rank", "grade_rank", "state_rank"])

    df.to_csv(
        os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "near_breakout": int((df["state"] == "돌파 임박").sum()),
        "first_breakout": int((df["state"] == "1차 돌파").sum()),
        "late_breakout": int((df["state"] == "후행 가능").sum()),
        "rs_s": int((df["rs_grade"] == "S").sum()),
        "rs_a": int((df["rs_grade"] == "A").sum()),
        "rs_b": int((df["rs_grade"] == "B").sum()),
        "A": int((df["grade"] == "A").sum()),
        "B": int((df["grade"] == "B").sum()),
        "as_of_date": str(df["as_of_date"].max()),
    }

    with open(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    setup_logging()
    universe = load_universe()

    benchmark = download_history(BENCHMARK_TICKER)
    if benchmark.empty:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    results: List[LongBoxResult] = []

    for row in universe.itertuples(index=False):
        try:
            result = scan_one_ticker(
                ticker=row.ticker,
                name=row.name,
                benchmark=benchmark,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            logging.exception("%s failed: %s", row.ticker, e)

    save_outputs(results)
    notify_all_results(results)

    if telegram_enabled():
        msg = (
            f"[요약] 박스 돌파\n"
            f"날짜: {results[0].as_of_date if results else 'N/A'}\n"
            f"유니버스 수: {len(universe)}\n"
            f"총 후보: {len(results)}\n"
            f"돌파 임박: {sum(r.state == '돌파 임박' for r in results)}\n"
            f"1차 돌파: {sum(r.state == '1차 돌파' for r in results)}\n"
            f"후행 가능: {sum(r.state == '후행 가능' for r in results)}\n"
            f"RS S: {sum(r.rs_grade == 'S' for r in results)}\n"
            f"RS A: {sum(r.rs_grade == 'A' for r in results)}\n"
            f"RS B: {sum(r.rs_grade == 'B' for r in results)}\n"
            f"A: {sum(r.grade == 'A' for r in results)}\n"
            f"B: {sum(r.grade == 'B' for r in results)}"
        )
        send_telegram_message(msg)


if __name__ == "__main__":
    main()
