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
# Rare Event Scanner - Long Box Breakout
# ------------------------------------------------------------
# 목적
# - 희귀 이벤트성 "장기 박스 돌파" 후보 탐지
# - 기존 Minervini/VCP 스캐너와 분리 운용
# - 구조 필터 + 품질 점수 + A/B 등급
#
# 특징
# - 박스 상단/하단을 max/min 대신 quantile로 정의
# - 우측 회복은 필수, 회복 강도/속도는 점수화
# - 상단 테스트 횟수는 하드컷이 아니라 점수 요소
# - 텔레그램 알림 전부 한글화
#
# 준비
# pip install pandas numpy requests yfinance openpyxl
#
# 필수 파일
# data/universe.csv
# 컬럼:
# ticker,name
# 선택:
# market_cap
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
ENFORCE_MARKET_CAP = False
MIN_MARKET_CAP = 1_000_000_000

# 시장 레짐
USE_MARKET_REGIME = True
MIN_BREADTH_50MA = 0.40  # 유니버스 중 50MA 위 비율 최소 40%

# 장기 박스 탐지
BOX_LOOKBACKS = [100, 140, 180, 220]   # 장기 박스 후보 길이
MIN_BOX_LENGTH = 80                    # 하드컷 최소 길이
MIN_BOX_WIDTH = 0.08                   # 너무 좁은 박스 제외
MAX_BOX_WIDTH = 0.32                   # 너무 넓은 박스 제외
BOX_HIGH_QUANTILE = 0.97               # 상단 extreme wick 제거
BOX_LOW_QUANTILE = 0.03                # 하단 extreme wick 제거

# 상단 테스트
TOUCH_TOLERANCE = 0.03                 # 상단 3% 이내면 상단 테스트
TOUCH_RESET_PCT = 0.05                 # 한 번 터치 후 5% 이상 밀려야 다음 독립 터치 인정

# 돌파 / 근접 / 추격 제한
BREAKOUT_CONFIRM_PCT = 0.000           # 종가가 박스 상단 넘으면 돌파로 인정
NEAR_PIVOT_PCT = 0.03                  # 아직 돌파 전이라도 상단 3% 이내면 돌파 임박
MAX_BREAKOUT_EXTEND_PCT = 0.03         # 박스 상단 대비 3% 넘게 확장되면 추격으로 제외
BREAKOUT_VOL_RATIO = 1.35              # 돌파 시 거래량 최소 1.35배

# 52주 신고가 근접
NEAR_52W_HIGH_RATIO = 0.92             # 52주 고점의 92% 이상

# 진입 / 손절 / 피라미딩
ENTRY_BUFFER = 0.002                   # 피벗 상단 0.2% 위 진입가
STOP_BUFFER = 0.0025                   # 구조 손절 버퍼 0.25%
MAX_INITIAL_STOP_PCT = 0.08            # 초기 손절 최대폭 8%
PYRAMID_LEVELS = [0.02, 0.04, 0.06]    # 1차/2차/3차 추가

# 텔레그램
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_ROWS = 15


# ------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------
@dataclass
class LongBoxResult:
    ticker: str
    name: str
    as_of_date: str
    상태: str
    등급: str
    등급_설명: str
    종가: float
    박스상단: float
    박스하단: float
    박스폭_pct: float
    박스길이: int
    상단테스트횟수: int
    52주고점근접: bool
    돌파거래량배수: Optional[float]
    진입가: Optional[float]
    손절가: Optional[float]
    1차추가가: Optional[float]
    2차추가가: Optional[float]
    3차추가가: Optional[float]
    ma50: Optional[float]
    ma150: Optional[float]
    ma200: Optional[float]
    거래대금20일평균: Optional[float]
    상대강도6개월초과수익: Optional[float]
    우측회복비율: Optional[float]
    우측회복최소기준: Optional[float]
    우측회복강도점수: Optional[float]
    우측회복속도점수: Optional[float]
    상단테스트점수: Optional[float]
    길이점수: Optional[float]
    폭점수: Optional[float]
    거래량점수: Optional[float]
    총점: Optional[float]
    사유: str


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
    if score >= 60:
        return "B", "관찰 후보"
    return "WATCH", "관찰 유지"


# ------------------------------------------------------------
# 유니버스 로드
# ------------------------------------------------------------
def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"유니버스 파일 없음: {UNIVERSE_FILE}")

    meta = pd.read_csv(UNIVERSE_FILE)
    meta.columns = [str(c).strip().lower() for c in meta.columns]

    if "ticker" not in meta.columns:
        raise ValueError("universe.csv에는 ticker 컬럼이 필요")
    if "name" not in meta.columns:
        meta["name"] = meta["ticker"]
    if "market_cap" not in meta.columns:
        meta["market_cap"] = np.nan

    meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
    meta["name"] = meta["name"].astype(str)
    meta["market_cap"] = pd.to_numeric(meta["market_cap"], errors="coerce")

    return meta[["ticker", "name", "market_cap"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


# ------------------------------------------------------------
# 데이터 다운로드
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

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).sort_values("Date").reset_index(drop=True)
    return df


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


def add_indicators(df: pd.DataFrame, benchmark: Optional[pd.DataFrame], market_cap_from_universe: Optional[float]) -> pd.DataFrame:
    out = df.copy()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma150"] = out["Close"].rolling(150).mean()
    out["ma200"] = out["Close"].rolling(200).mean()
    out["vol_ma20"] = out["Volume"].rolling(20).mean()
    out["dollar_vol"] = out["Close"] * out["Volume"]
    out["dollar_vol_20"] = out["dollar_vol"].rolling(20).mean()
    out["high_252"] = out["High"].rolling(252).max()
    out["ret_6m"] = rolling_return(out["Close"], 126)
    out["market_cap"] = market_cap_from_universe
    out["rs_6m_excess"] = np.nan

    if benchmark is not None and len(benchmark) > 0:
        b = benchmark[["Date", "Close"]].copy().rename(columns={"Close": "benchmark_close"})
        x = out.merge(b, on="Date", how="left")
        x["benchmark_close"] = x["benchmark_close"].ffill()
        x["benchmark_ret_6m"] = rolling_return(x["benchmark_close"], 126)
        x["rs_6m_excess"] = x["ret_6m"] - x["benchmark_ret_6m"]
        out = x

    return out


# ------------------------------------------------------------
# 시장 breadth / 레짐
# ------------------------------------------------------------
def compute_universe_breadth(dfs: List[pd.DataFrame]) -> float:
    valid = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        last = df.iloc[-1]
        if pd.notna(last.get("ma50")):
            valid.append(bool(last["Close"] > last["ma50"]))
    if not valid:
        return 0.0
    return float(np.mean(valid))


def market_regime_ok(benchmark_df: pd.DataFrame, breadth: float) -> bool:
    if len(benchmark_df) < 220:
        return False
    row = benchmark_df.iloc[-1]
    if pd.isna(row["ma50"]) or pd.isna(row["ma200"]):
        return False
    cond1 = row["Close"] > row["ma200"]
    cond2 = row["ma50"] > row["ma200"]
    cond3 = breadth >= MIN_BREADTH_50MA
    return bool(cond1 and cond2 and cond3)


# ------------------------------------------------------------
# 박스 / 점수 로직
# ------------------------------------------------------------
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
    else:
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
    else:
        return 0.4


def recovery_speed_score(right_side_len: int, box_len: int) -> float:
    if box_len <= 0:
        return 0.0

    speed_ratio = right_side_len / box_len

    if 0.20 <= speed_ratio <= 0.40:
        return 1.2
    elif 0.15 <= speed_ratio <= 0.50:
        return 1.0
    else:
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
    else:
        return 0.7


def length_score(box_len: int) -> float:
    if box_len >= 180:
        return 1.2
    elif box_len >= 140:
        return 1.0
    elif box_len >= 100:
        return 0.8
    else:
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


def find_box_low_index(seg: pd.DataFrame) -> int:
    lows = seg["Low"].values
    if len(lows) == 0:
        return 0
    return int(np.argmin(lows))


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

        # quantile 기반 박스 정의
        box_high = float(seg["High"].quantile(BOX_HIGH_QUANTILE))
        box_low = float(seg["Low"].quantile(BOX_LOW_QUANTILE))

        if box_high <= 0 or box_low <= 0 or box_low >= box_high:
            continue

        box_width = (box_high - box_low) / box_high
        if box_width < MIN_BOX_WIDTH or box_width > MAX_BOX_WIDTH:
            continue

        current_close = float(seg.iloc[-1]["Close"])
        depth = box_width

        # 우측 회복 비율
        recovery_ratio = (current_close - box_low) / max(box_high - box_low, 1e-9)
        min_recovery = min_recovery_ratio_by_depth(depth)
        if recovery_ratio < min_recovery:
            continue

        # 저점 위치 이후 회복 구간 길이
        low_idx_local = find_box_low_index(seg)
        right_side_len = len(seg) - 1 - low_idx_local
        if right_side_len <= 0:
            continue

        # 상단 독립 테스트 횟수
        touches = count_distinct_box_touches(seg, box_high)

        # 근본 점수
        s_len = length_score(len(seg))
        s_width = width_score(box_width)
        s_recovery_strength = recovery_strength_score(recovery_ratio, depth)
        s_recovery_speed = recovery_speed_score(right_side_len, len(seg))
        s_touch = touch_score(touches)

        score = (
            s_len * 18
            + s_width * 20
            + s_recovery_strength * 24
            + s_recovery_speed * 18
            + s_touch * 20
        )

        if score > best_score:
            best_score = score
            best = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "lookback": lookback,
                "box_high": box_high,
                "box_low": box_low,
                "box_width_pct": box_width,
                "touch_count": touches,
                "recovery_ratio": recovery_ratio,
                "min_recovery": min_recovery,
                "recovery_strength_score": s_recovery_strength,
                "recovery_speed_score": s_recovery_speed,
                "touch_score": s_touch,
                "length_score": s_len,
                "width_score": s_width,
                "right_side_len": right_side_len,
                "base_score": score,
            }

    return best


def classify_long_box_event(df: pd.DataFrame, box: Dict[str, Any]) -> Tuple[Optional[str], str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    row = df.iloc[-1]
    close = float(row["Close"])
    box_high = float(box["box_high"])
    box_low = float(box["box_low"])

    breakout_vol_ratio = None
    if pd.notna(row["vol_ma20"]) and row["vol_ma20"] > 0:
        breakout_vol_ratio = float(row["Volume"] / row["vol_ma20"])

    near_52w = bool(pd.notna(row["high_252"]) and close >= row["high_252"] * NEAR_52W_HIGH_RATIO)
    if not near_52w:
        return None, "52주 고점과 거리 멀음", None, None, None, None, None, None, breakout_vol_ratio

    breakout = close >= box_high * (1.0 + BREAKOUT_CONFIRM_PCT)
    extended = close > box_high * (1.0 + MAX_BREAKOUT_EXTEND_PCT)
    near_pivot = (close < box_high) and (close >= box_high * (1.0 - NEAR_PIVOT_PCT))

    if breakout and extended:
        return None, "돌파는 했지만 추격 구간", None, None, None, None, None, None, breakout_vol_ratio

    entry_price = box_high * (1.0 + ENTRY_BUFFER)

    # 구조 손절
    structural_stop = box_low * (1.0 - STOP_BUFFER)
    # -8% 손절
    pct_stop = entry_price * (1.0 - MAX_INITIAL_STOP_PCT)
    # 더 촘촘한 쪽 사용
    stop_price = max(structural_stop, pct_stop)

    add_1 = entry_price * (1.0 + PYRAMID_LEVELS[0])
    add_2 = entry_price * (1.0 + PYRAMID_LEVELS[1])
    add_3 = entry_price * (1.0 + PYRAMID_LEVELS[2])

    if breakout:
        if breakout_vol_ratio is None or breakout_vol_ratio < BREAKOUT_VOL_RATIO:
            return None, "돌파 거래량 부족", None, None, None, None, None, None, breakout_vol_ratio
        return "돌파", "장기 박스 상단 돌파", entry_price, stop_price, add_1, add_2, add_3, entry_price, breakout_vol_ratio

    if near_pivot:
        return "돌파 임박", "장기 박스 상단 근접", entry_price, stop_price, add_1, add_2, add_3, entry_price, breakout_vol_ratio

    return None, "피벗 구간 아님", None, None, None, None, None, None, breakout_vol_ratio


def calculate_total_score(box: Dict[str, Any], breakout_vol_ratio: Optional[float], event_state: str, rs_6m_excess: Optional[float]) -> float:
    s_len = float(box["length_score"])
    s_width = float(box["width_score"])
    s_rec_strength = float(box["recovery_strength_score"])
    s_rec_speed = float(box["recovery_speed_score"])
    s_touch = float(box["touch_score"])
    s_vol = breakout_volume_score(breakout_vol_ratio)

    s_rs = 0.0
    if rs_6m_excess is not None and not pd.isna(rs_6m_excess):
        if rs_6m_excess >= 0.20:
            s_rs = 1.2
        elif rs_6m_excess >= 0.10:
            s_rs = 1.0
        elif rs_6m_excess >= 0.00:
            s_rs = 0.7
        else:
            s_rs = 0.2

    s_state = 1.0 if event_state == "돌파" else 0.7

    score = (
        s_len * 15
        + s_width * 15
        + s_rec_strength * 20
        + s_rec_speed * 12
        + s_touch * 14
        + s_vol * 14
        + s_rs * 10
        + s_state * 10
    )
    return round(score, 2)


# ------------------------------------------------------------
# 스캔
# ------------------------------------------------------------
def scan_one_ticker(
    ticker: str,
    name: str,
    market_cap: Optional[float],
    benchmark: pd.DataFrame,
) -> Optional[LongBoxResult]:
    price_df = download_history(ticker)
    if price_df.empty or len(price_df) < MIN_HISTORY:
        return None

    df = add_indicators(price_df, benchmark, market_cap)
    row = df.iloc[-1]

    if pd.isna(row["Close"]) or row["Close"] < MIN_PRICE:
        return None
    if pd.isna(row["dollar_vol_20"]) or row["dollar_vol_20"] < MIN_DOLLAR_VOL_20:
        return None
    if ENFORCE_MARKET_CAP and (pd.isna(row["market_cap"]) or row["market_cap"] < MIN_MARKET_CAP):
        return None

    box = find_best_long_box(df)
    if box is None:
        return None

    상태, 사유, 진입가, 손절가, add_1, add_2, add_3, _, 돌파거래량배수 = classify_long_box_event(df, box)
    if 상태 is None:
        return None

    총점 = calculate_total_score(
        box=box,
        breakout_vol_ratio=돌파거래량배수,
        event_state=상태,
        rs_6m_excess=safe_float(row["rs_6m_excess"]),
    )
    등급, 등급_설명 = grade_label(총점)

    near_52w = bool(pd.notna(row["high_252"]) and row["Close"] >= row["high_252"] * NEAR_52W_HIGH_RATIO)

    return LongBoxResult(
        ticker=ticker,
        name=name,
        as_of_date=str(pd.to_datetime(row["Date"]).date()),
        상태=상태,
        등급=등급,
        등급_설명=등급_설명,
        종가=round(float(row["Close"]), 2),
        박스상단=round(float(box["box_high"]), 2),
        박스하단=round(float(box["box_low"]), 2),
        박스폭_pct=round(float(box["box_width_pct"]), 4),
        박스길이=int(box["lookback"]),
        상단테스트횟수=int(box["touch_count"]),
        52주고점근접=near_52w,
        돌파거래량배수=round(돌파거래량배수, 2) if 돌파거래량배수 is not None else None,
        진입가=round_price(진입가),
        손절가=round_price(손절가),
        **{
            "1차추가가": round_price(add_1),
            "2차추가가": round_price(add_2),
            "3차추가가": round_price(add_3),
        },
        ma50=round_price(row["ma50"]),
        ma150=round_price(row["ma150"]),
        ma200=round_price(row["ma200"]),
        거래대금20일평균=round_price(row["dollar_vol_20"], 0),
        상대강도6개월초과수익=safe_float(row["rs_6m_excess"]),
        우측회복비율=round(float(box["recovery_ratio"]), 4),
        우측회복최소기준=round(float(box["min_recovery"]), 4),
        우측회복강도점수=round(float(box["recovery_strength_score"]), 2),
        우측회복속도점수=round(float(box["recovery_speed_score"]), 2),
        상단테스트점수=round(float(box["touch_score"]), 2),
        길이점수=round(float(box["length_score"]), 2),
        폭점수=round(float(box["width_score"]), 2),
        거래량점수=round(breakout_volume_score(돌파거래량배수), 2),
        총점=총점,
        사유=사유,
    )


# ------------------------------------------------------------
# 텔레그램 / 상태
# ------------------------------------------------------------
def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str) -> None:
    if not telegram_enabled():
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code != 200:
            logging.error("Telegram send failed: %s | %s", r.status_code, r.text)
    except Exception as e:
        logging.exception("Telegram send exception: %s", e)


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
    return [
        f"- {r.ticker} {r.name}",
        f"  등급: {r.등급} ({r.등급_설명})",
        f"  상태: {r.상태}",
        f"  종가: {format_price(r.종가)}",
        f"  박스 상단: {format_price(r.박스상단)}",
        f"  박스 하단: {format_price(r.박스하단)}",
        f"  박스 길이: {r.박스길이}일",
        f"  박스 폭: {format_pct(r.박스폭_pct)}",
        f"  상단 테스트: {r.상단테스트횟수}회",
        f"  우측 회복: {format_pct(r.우측회복비율)} / 최소 {format_pct(r.우측회복최소기준)}",
        f"  거래량: {r.돌파거래량배수 if r.돌파거래량배수 is not None else '-'}배 (20일 평균 대비)",
        f"  진입가: {format_price(r.진입가)}",
        f"  손절가: {format_price(r.손절가)}",
        f"  1차 추가: {format_price(r.__dict__['1차추가가'])}",
        f"  2차 추가: {format_price(r.__dict__['2차추가가'])}",
        f"  3차 추가: {format_price(r.__dict__['3차추가가'])}",
        f"  총점: {r.총점}",
    ]


def notify_changes(results: List[LongBoxResult]) -> None:
    if not telegram_enabled():
        return

    prev_state = load_state()
    new_state: Dict[str, Dict[str, Any]] = {}

    new_alerts: List[LongBoxResult] = []
    for r in results:
        old = prev_state.get(r.ticker, {})
        old_state = str(old.get("상태", ""))
        old_date = str(old.get("as_of_date", ""))
        old_grade = str(old.get("등급", ""))

        if not (old_state == r.상태 and old_date == r.as_of_date and old_grade == r.등급):
            new_alerts.append(r)

        new_state[r.ticker] = {
            "상태": r.상태,
            "등급": r.등급,
            "as_of_date": r.as_of_date,
        }

    if new_alerts:
        lines = ["[희귀 이벤트] 장기 박스 돌파"]
        for r in new_alerts[:MAX_TELEGRAM_ROWS]:
            lines.extend(build_result_lines(r))
        send_telegram_message("\n".join(lines))

    save_state(new_state)


# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
def save_outputs(results: List[LongBoxResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not results:
        pd.DataFrame(
            columns=[f.name for f in LongBoxResult.__dataclass_fields__.values()]
        ).to_csv(
            os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        with open(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_at": datetime.now().isoformat(),
                    "total": 0,
                    "돌파": 0,
                    "돌파임박": 0,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    df = pd.DataFrame([asdict(r) for r in results]).sort_values(
        ["등급", "상태", "ticker"],
        ascending=[True, True, True],
    )

    df.to_csv(
        os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "돌파": int((df["상태"] == "돌파").sum()),
        "돌파임박": int((df["상태"] == "돌파 임박").sum()),
        "A": int((df["등급"] == "A").sum()),
        "B": int((df["등급"] == "B").sum()),
        "WATCH": int((df["등급"] == "WATCH").sum()),
        "as_of_date": str(df["as_of_date"].max()),
    }
    with open(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    setup_logging()
    logging.info("Rare event scanner start")

    universe = load_universe()

    benchmark_raw = download_history(BENCHMARK_TICKER)
    if benchmark_raw.empty:
        raise RuntimeError("SPY 데이터 다운로드 실패")
    benchmark = add_indicators(benchmark_raw, None, None)

    # breadth 계산용 사전 적재
    enriched_list = []
    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).upper().strip()
        market_cap = safe_float(row.market_cap)
        try:
            raw_df = download_history(ticker)
            if raw_df.empty or len(raw_df) < MIN_HISTORY:
                continue
            enriched = add_indicators(raw_df, benchmark_raw, market_cap)
            enriched_list.append(enriched)
        except Exception as e:
            logging.exception("%s preload failed: %s", ticker, e)

    breadth = compute_universe_breadth(enriched_list)
    regime_ok = market_regime_ok(benchmark, breadth)

    logging.info("Market regime | breadth=%.2f | regime_ok=%s", breadth, regime_ok)

    if USE_MARKET_REGIME and not regime_ok:
        send_telegram_message(
            f"[시장 레짐 OFF] 장기 박스 돌파\n"
            f"SPY: {benchmark.iloc[-1]['Close']:.2f}\n"
            f"SPY 50일선: {benchmark.iloc[-1]['ma50']:.2f}\n"
            f"SPY 200일선: {benchmark.iloc[-1]['ma200']:.2f}\n"
            f"시장 폭(Breadth): {breadth * 100:.1f}%\n"
            f"신규 알림 중단"
        )
        save_outputs([])
        return

    results: List[LongBoxResult] = []
    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).upper().strip()
        name = str(row.name)
        market_cap = safe_float(row.market_cap)

        try:
            result = scan_one_ticker(
                ticker=ticker,
                name=name,
                market_cap=market_cap,
                benchmark=benchmark_raw,
            )
            if result is not None:
                results.append(result)
                logging.info("%s | %s | %s | %s", ticker, result.상태, result.등급, result.사유)
        except Exception as e:
            logging.exception("%s failed: %s", ticker, e)

    save_outputs(results)
    notify_changes(results)

    if telegram_enabled():
        count_breakout = sum(r.상태 == "돌파" for r in results)
        count_near = sum(r.상태 == "돌파 임박" for r in results)
        count_a = sum(r.등급 == "A" for r in results)
        count_b = sum(r.등급 == "B" for r in results)
        count_watch = sum(r.등급 == "WATCH" for r in results)

        msg = (
            f"[요약] 장기 박스 돌파\n"
            f"날짜: {results[0].as_of_date if results else 'N/A'}\n"
            f"유니버스 수: {len(universe)}\n"
            f"시장 폭(Breadth): {breadth * 100:.1f}%\n"
            f"돌파: {count_breakout}\n"
            f"돌파 임박: {count_near}\n"
            f"A (우선 검토): {count_a}\n"
            f"B (관찰 후보): {count_b}\n"
            f"WATCH (관찰 유지): {count_watch}\n"
            f"총 알림 수: {len(results)}"
        )
        send_telegram_message(msg)

    logging.info("Rare event scanner done | total=%d", len(results))


if __name__ == "__main__":
    main()
