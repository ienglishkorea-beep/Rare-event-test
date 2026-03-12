import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# 10-Day Tight Scanner - Event Freeze Rejection Version
# ------------------------------------------------------------
# 목적
# - 진짜 10-Day Tight만 남기고
# - M&A / 뉴스 급등 / 이벤트 가격 고정 패턴 제거
#
# 핵심 변경
# 1) 최근 15일 단일 급등봉 제거
# 2) 최근 15일 대형 갭 제거
# 3) spike 후 freeze 패턴 제거
# 4) spike 후 거래량 붕괴 + 변동성 붕괴 제거
# 5) 기존 RS / 추세 / 위치 필터 유지
# ============================================================


# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_scanner.log")
STATE_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_state.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
BENCHMARK = "SPY"


# ------------------------------------------------------------
# 텔레그램
# ------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_ROWS = 20
MAX_TELEGRAM_MESSAGE_LEN = 3500


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
DOWNLOAD_PERIOD = "2y"
MIN_HISTORY = 280

MIN_PRICE = 10.0
MIN_DOLLAR_VOL = 10_000_000

# Tight 기본
TIGHT_WINDOW = 10
MAX_CLOSE_TIGHT = 0.055
MAX_RANGE_TIGHT = 0.075

# 모멘텀 조건
LOOKBACK_8W = 40
LOOKBACK_4W = 20
MIN_RET_8W = 0.18
MIN_RET_4W = 0.06

# 위치 조건
NEAR_BREAKOUT = 0.03
MAX_DISTANCE_52W = -0.08
MIN_8W_RANGE_POSITION = 0.70

# RS 필터
RS_HIGH_LOOKBACK = 252
RS_NEAR_HIGH_MIN_RATIO = 0.80
REQUIRE_RS_UPTREND = True

# MA 정렬
REQUIRE_PRICE_ABOVE_MA20 = True
REQUIRE_MA_ALIGNMENT = True

# 점수
A_GRADE_MIN_SCORE = 82
B_GRADE_MIN_SCORE = 68

# 실행 계획
ENTRY_BUFFER = 0.002
STOP_PCT = 0.08
PYRAMID_LEVELS = [0.02, 0.04, 0.06]

# ------------------------------------------------------------
# 이벤트 / M&A / 가격고정 제거 필터
# ------------------------------------------------------------
EVENT_LOOKBACK = 15

# 단일 일봉 급등
MAX_SINGLE_DAY_RETURN = 0.12       # 최근 15일 중 하루 +12% 초과면 제거

# 갭 급등
MAX_UP_GAP = 0.10                  # 최근 15일 중 갭상승 +10% 초과면 제거

# spike + freeze 패턴
FREEZE_LOOKBACK_AFTER_SPIKE = 5
FREEZE_RANGE_MAX = 0.025           # spike 후 5일 range 2.5% 이하면 freeze 의심
FREEZE_CLOSE_DIST_MAX = 0.02       # spike 종가 대비 2% 내 머물면 freeze 의심

# 거래량 이벤트
SPIKE_VOLUME_RATIO_MIN = 3.0       # spike day 거래량 / 20일 평균
POST_SPIKE_VOL_COLLAPSE_MAX = 0.75 # 이후 5일 평균 거래량 / spike day 거래량
POST_SPIKE_RANGE_MAX = 0.03        # 이후 5일 전체 range

# 추가 필터
MIN_10D_PROGRESS_FROM_SPIKE = 0.01 # spike 후 10일간 추가 진행이 1% 미만이면 제거


# ------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------
@dataclass
class Result:
    ticker: str
    name: str
    date: str
    state: str
    grade: str

    score_total: float
    score_close: float
    score_range: float
    score_momentum: float
    score_rs: float
    score_volume: float
    score_atr: float
    score_52w: float

    close: float
    high_10: float
    low_10: float
    close_tight: float
    range_tight: float

    ret_8w: float
    ret_4w: float
    rs_excess_3m: float
    rs_current_vs_high: float
    high_52w_distance: float
    range_position_8w: float

    vol_ratio_10_30: float
    atr_ratio_10_30: float

    entry: float
    stop: float
    add1: float
    add2: float
    add3: float


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


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def format_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.2f}"


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(msg: str) -> None:
    if not telegram_enabled():
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
    except Exception:
        pass


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


def download(ticker: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


# ------------------------------------------------------------
# 지표
# ------------------------------------------------------------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n).mean()


def grade_from_score(score_total: float) -> str:
    if score_total >= A_GRADE_MIN_SCORE:
        return "A"
    return "B"


def score_close_compression(v: float) -> int:
    if v <= 0.035:
        return 25
    if v <= 0.045:
        return 22
    if v <= 0.055:
        return 18
    return 0


def score_range_compression(v: float) -> int:
    if v <= 0.05:
        return 20
    if v <= 0.06:
        return 17
    if v <= 0.075:
        return 13
    return 0


def score_momentum(r8: float, r4: float, range_pos_8w: float) -> int:
    s = 0

    if r8 >= 0.30:
        s += 9
    elif r8 >= 0.18:
        s += 6

    if r4 >= 0.10:
        s += 6
    elif r4 >= 0.06:
        s += 4

    if range_pos_8w >= 0.90:
        s += 5
    elif range_pos_8w >= 0.80:
        s += 3

    return s


def score_rs(rs_excess_3m: float, rs_current_vs_high: float) -> int:
    s = 0

    if rs_excess_3m >= 0.15:
        s += 8
    elif rs_excess_3m >= 0.05:
        s += 5

    if rs_current_vs_high >= 0.98:
        s += 12
    elif rs_current_vs_high >= 0.90:
        s += 8
    elif rs_current_vs_high >= 0.80:
        s += 5

    return s


def score_volume(vratio: float) -> int:
    if vratio <= 0.75:
        return 10
    if vratio <= 0.90:
        return 7
    if vratio <= 1.05:
        return 4
    return 0


def score_atr(atr_ratio: float) -> int:
    if atr_ratio <= 0.80:
        return 5
    if atr_ratio <= 0.92:
        return 3
    return 0


def score_52w(distance_52w: float) -> int:
    if distance_52w >= -0.03:
        return 5
    if distance_52w >= -0.06:
        return 4
    if distance_52w >= -0.08:
        return 2
    return 0


# ------------------------------------------------------------
# 하드 필터
# ------------------------------------------------------------
def rs_filters_pass(
    rs_line: pd.Series,
    rs_ma20: pd.Series,
    rs_ma50: pd.Series,
) -> bool:
    if len(rs_line) < 60:
        return False

    current_rs = safe_float(rs_line.iloc[-1])
    current_rs_ma20 = safe_float(rs_ma20.iloc[-1])
    current_rs_ma50 = safe_float(rs_ma50.iloc[-1])

    if current_rs is None or current_rs_ma20 is None or current_rs_ma50 is None:
        return False

    rs_high_1y = safe_float(rs_line.tail(RS_HIGH_LOOKBACK).max())
    if rs_high_1y is None or rs_high_1y <= 0:
        return False

    current_vs_high = current_rs / rs_high_1y
    if current_vs_high < RS_NEAR_HIGH_MIN_RATIO:
        return False

    if REQUIRE_RS_UPTREND:
        if not (current_rs >= current_rs_ma20 and current_rs_ma20 >= current_rs_ma50):
            return False

    return True


def trend_filters_pass(
    close: float,
    ma20: float,
    ma50: float,
    ma200: float,
) -> bool:
    if REQUIRE_PRICE_ABOVE_MA20 and close <= ma20:
        return False

    if REQUIRE_MA_ALIGNMENT:
        if not (close > ma20 > ma50 > ma200):
            return False

    return True


# ------------------------------------------------------------
# 이벤트 패턴 제거
# ------------------------------------------------------------
def event_freeze_filter(df: pd.DataFrame) -> bool:
    """
    최근 15일 안에
    1) 단일 급등봉
    2) 대형 갭
    3) spike 후 가격고정
    4) spike 후 거래량붕괴 + range collapse
    패턴이 있으면 제거
    """
    if len(df) < EVENT_LOOKBACK + FREEZE_LOOKBACK_AFTER_SPIKE + 5:
        return False

    x = df.copy()
    x["daily_ret"] = x["Close"].pct_change()
    x["gap_up"] = (x["Open"] / x["Close"].shift(1)) - 1.0
    x["vol_ma20"] = x["Volume"].rolling(20).mean()

    recent = x.tail(EVENT_LOOKBACK).copy()
    if recent.empty:
        return False

    # 1) 단일 급등봉
    spike_ret = recent["daily_ret"].max()
    if pd.notna(spike_ret) and spike_ret > MAX_SINGLE_DAY_RETURN:
        spike_idx = int(recent["daily_ret"].idxmax())
        spike_pos = x.index.get_loc(spike_idx)
        if spike_pos < len(x) - 1:
            spike_close = float(x.loc[spike_idx, "Close"])

            after = x.iloc[spike_pos + 1: spike_pos + 1 + FREEZE_LOOKBACK_AFTER_SPIKE].copy()
            if len(after) >= 3:
                after_high = float(after["High"].max())
                after_low = float(after["Low"].min())
                after_range = (after_high - after_low) / spike_close
                after_dist = abs(float(after["Close"].iloc[-1]) / spike_close - 1.0)

                if after_range <= FREEZE_RANGE_MAX and after_dist <= FREEZE_CLOSE_DIST_MAX:
                    return True

    # 2) 대형 갭
    max_gap = recent["gap_up"].max()
    if pd.notna(max_gap) and max_gap > MAX_UP_GAP:
        gap_idx = int(recent["gap_up"].idxmax())
        gap_pos = x.index.get_loc(gap_idx)
        if gap_pos < len(x) - 1:
            gap_close = float(x.loc[gap_idx, "Close"])
            after = x.iloc[gap_pos + 1: gap_pos + 1 + FREEZE_LOOKBACK_AFTER_SPIKE].copy()
            if len(after) >= 3:
                after_high = float(after["High"].max())
                after_low = float(after["Low"].min())
                after_range = (after_high - after_low) / gap_close
                after_dist = abs(float(after["Close"].iloc[-1]) / gap_close - 1.0)

                if after_range <= FREEZE_RANGE_MAX and after_dist <= FREEZE_CLOSE_DIST_MAX:
                    return True

    # 3) spike volume + freeze
    recent = recent.reset_index(drop=True)
    for i in range(len(recent)):
        row = recent.iloc[i]
        daily_ret = safe_float(row["daily_ret"])
        gap_up = safe_float(row["gap_up"])
        vol = safe_float(row["Volume"])
        vol_ma20 = safe_float(row["vol_ma20"])
        close = safe_float(row["Close"])

        if close is None or vol is None or vol_ma20 is None or vol_ma20 <= 0:
            continue

        vol_ratio = vol / vol_ma20
        is_spike = (
            (daily_ret is not None and daily_ret > 0.08)
            or (gap_up is not None and gap_up > 0.06)
        )

        if not is_spike:
            continue

        real_pos = len(x) - len(recent) + i
        after = x.iloc[real_pos + 1: real_pos + 1 + FREEZE_LOOKBACK_AFTER_SPIKE].copy()
        if len(after) < 3:
            continue

        after_range = (float(after["High"].max()) - float(after["Low"].min())) / close
        after_vol_mean = float(after["Volume"].mean())
        vol_collapse = after_vol_mean / vol if vol > 0 else 999

        if (
            vol_ratio >= SPIKE_VOLUME_RATIO_MIN
            and after_range <= POST_SPIKE_RANGE_MAX
            and vol_collapse <= POST_SPIKE_VOL_COLLAPSE_MAX
        ):
            return True

    # 4) 최근 10일 내 큰 spike 후 추가 진행이 거의 없으면 제거
    recent10 = x.tail(10).copy()
    if len(recent10) >= 5:
        max_ret_10 = recent10["daily_ret"].max()
        if pd.notna(max_ret_10) and max_ret_10 > 0.08:
            spike_idx = int(recent10["daily_ret"].idxmax())
            spike_close = float(x.loc[spike_idx, "Close"])
            latest_close = float(x["Close"].iloc[-1])
            extra_progress = latest_close / spike_close - 1.0
            if extra_progress < MIN_10D_PROGRESS_FROM_SPIKE:
                return True

    return False


# ------------------------------------------------------------
# 스캔
# ------------------------------------------------------------
def scan_one(ticker: str, name: str, spy_df: pd.DataFrame) -> Optional[Result]:
    df = download(ticker)
    if df.empty or len(df) < MIN_HISTORY:
        return None

    close = float(df["Close"].iloc[-1])

    # 기본 유동성 필터
    df["dollar"] = df["Close"] * df["Volume"]
    dollar_vol_20 = float(df["dollar"].rolling(20).mean().iloc[-1])

    if close < MIN_PRICE or dollar_vol_20 < MIN_DOLLAR_VOL:
        return None

    # 이벤트 / 가격고정 제거
    if event_freeze_filter(df):
        return None

    # 이동평균
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    ma20 = safe_float(df["ma20"].iloc[-1])
    ma50 = safe_float(df["ma50"].iloc[-1])
    ma200 = safe_float(df["ma200"].iloc[-1])

    if ma20 is None or ma50 is None or ma200 is None:
        return None

    if not trend_filters_pass(close=close, ma20=ma20, ma50=ma50, ma200=ma200):
        return None

    # 최근 10일 tight 계산
    high10 = float(df["High"].tail(TIGHT_WINDOW).max())
    low10 = float(df["Low"].tail(TIGHT_WINDOW).min())
    close_high = float(df["Close"].tail(TIGHT_WINDOW).max())
    close_low = float(df["Close"].tail(TIGHT_WINDOW).min())

    close_tight = (close_high - close_low) / close_high
    range_tight = (high10 - low10) / high10

    if close_tight > MAX_CLOSE_TIGHT or range_tight > MAX_RANGE_TIGHT:
        return None

    # 최근 수익률
    r8 = float(df["Close"].iloc[-1] / df["Close"].iloc[-LOOKBACK_8W] - 1.0)
    r4 = float(df["Close"].iloc[-1] / df["Close"].iloc[-LOOKBACK_4W] - 1.0)

    if r8 < MIN_RET_8W or r4 < MIN_RET_4W:
        return None

    # 최근 8주 위치
    high_8w = float(df["High"].tail(LOOKBACK_8W).max())
    low_8w = float(df["Low"].tail(LOOKBACK_8W).min())
    if high_8w <= low_8w:
        return None

    range_pos_8w = (close - low_8w) / (high_8w - low_8w)
    if range_pos_8w < MIN_8W_RANGE_POSITION:
        return None

    # 52주 고점 근접
    high_52w = float(df["High"].rolling(252).max().iloc[-1]) if len(df) >= 252 else float(df["High"].max())
    if high_52w <= 0:
        return None

    high_52w_distance = close / high_52w - 1.0
    if high_52w_distance < MAX_DISTANCE_52W:
        return None

    # 고점 근접 여부
    if close < high10 * (1.0 - NEAR_BREAKOUT) and close < high10:
        return None

    state = "돌파" if close >= high10 else "돌파 임박"

    # SPY 기준 RS
    spy = spy_df.copy().set_index("Date")
    stock = df.copy().set_index("Date")
    spy_close = spy["Close"].reindex(stock.index).ffill()

    rs_line = stock["Close"] / spy_close
    rs_ma20 = rs_line.rolling(20).mean()
    rs_ma50 = rs_line.rolling(50).mean()

    if not rs_filters_pass(rs_line=rs_line, rs_ma20=rs_ma20, rs_ma50=rs_ma50):
        return None

    rs_high_1y = float(rs_line.tail(RS_HIGH_LOOKBACK).max())
    rs_current_vs_high = float(rs_line.iloc[-1] / rs_high_1y)

    spy_r3 = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-63] - 1.0)
    stock_r3 = float(stock["Close"].iloc[-1] / stock["Close"].iloc[-63] - 1.0)
    rs_excess_3m = stock_r3 - spy_r3

    # 거래량 / ATR 수축
    v10 = float(df["Volume"].tail(10).mean())
    v30 = float(df["Volume"].tail(30).mean())
    vol_ratio = v10 / v30 if v30 > 0 else 999

    df["atr"] = atr(df)
    atr10 = float(df["atr"].tail(10).mean())
    atr30 = float(df["atr"].tail(30).mean())
    atr_ratio = atr10 / atr30 if atr30 > 0 else 999

    # 점수
    s_close = score_close_compression(close_tight)
    s_range = score_range_compression(range_tight)
    s_momo = score_momentum(r8, r4, range_pos_8w)
    s_rs = score_rs(rs_excess_3m, rs_current_vs_high)
    s_vol = score_volume(vol_ratio)
    s_atr = score_atr(atr_ratio)
    s_52w = score_52w(high_52w_distance)

    total = float(s_close + s_range + s_momo + s_rs + s_vol + s_atr + s_52w)

    if total < B_GRADE_MIN_SCORE:
        return None

    grade = grade_from_score(total)

    # 실행 계획
    entry = high10 * (1.0 + ENTRY_BUFFER)
    stop = entry * (1.0 - STOP_PCT)
    add1 = entry * (1.0 + PYRAMID_LEVELS[0])
    add2 = entry * (1.0 + PYRAMID_LEVELS[1])
    add3 = entry * (1.0 + PYRAMID_LEVELS[2])

    return Result(
        ticker=ticker,
        name=name,
        date=str(df["Date"].iloc[-1].date()),
        state=state,
        grade=grade,
        score_total=round(total, 2),
        score_close=float(s_close),
        score_range=float(s_range),
        score_momentum=float(s_momo),
        score_rs=float(s_rs),
        score_volume=float(s_vol),
        score_atr=float(s_atr),
        score_52w=float(s_52w),
        close=round(close, 2),
        high_10=round(high10, 2),
        low_10=round(low10, 2),
        close_tight=round(close_tight, 4),
        range_tight=round(range_tight, 4),
        ret_8w=round(r8, 4),
        ret_4w=round(r4, 4),
        rs_excess_3m=round(rs_excess_3m, 4),
        rs_current_vs_high=round(rs_current_vs_high, 4),
        high_52w_distance=round(high_52w_distance, 4),
        range_position_8w=round(range_pos_8w, 4),
        vol_ratio_10_30=round(vol_ratio, 4),
        atr_ratio_10_30=round(atr_ratio, 4),
        entry=round(entry, 2),
        stop=round(stop, 2),
        add1=round(add1, 2),
        add2=round(add2, 2),
        add3=round(add3, 2),
    )


# ------------------------------------------------------------
# 저장 / 출력
# ------------------------------------------------------------
def save_outputs(results: List[Result]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = [f.name for f in Result.__dataclass_fields__.values()]

    if not results:
        pd.DataFrame(columns=cols).to_csv(
            os.path.join(OUTPUT_DIR, "ten_day_tight_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        with open(os.path.join(OUTPUT_DIR, "ten_day_tight_summary.json"), "w", encoding="utf-8") as f:
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

    df = pd.DataFrame([asdict(x) for x in results])

    df["grade_rank"] = df["grade"].map({"A": 0, "B": 1}).fillna(9)
    df["state_rank"] = df["state"].map({"돌파 임박": 0, "돌파": 1}).fillna(9)

    df = df.sort_values(
        ["grade_rank", "state_rank", "score_total", "ticker"],
        ascending=[True, True, False, True],
    ).drop(columns=["grade_rank", "state_rank"])

    df.to_csv(
        os.path.join(OUTPUT_DIR, "ten_day_tight_results.csv"),
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
        "as_of_date": str(df["date"].max()),
    }

    with open(os.path.join(OUTPUT_DIR, "ten_day_tight_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_result_lines(r: Result) -> List[str]:
    rs_high_text = f"{r.rs_current_vs_high * 100:.1f}%"
    return [
        f"- {r.ticker} {r.name}",
        f"  등급: {r.grade} | 점수: {r.score_total}/100",
        f"  상태: {r.state}",
        f"  종가: {format_price(r.close)}",
        f"  종가 압축: {format_pct(r.close_tight)}",
        f"  전체 변동폭: {format_pct(r.range_tight)}",
        f"  8주 수익률: {format_pct(r.ret_8w)}",
        f"  4주 수익률: {format_pct(r.ret_4w)}",
        f"  RS 3M 초과수익: {format_pct(r.rs_excess_3m)}",
        f"  RS/1년RS고점: {rs_high_text}",
        f"  52주 고점 거리: {format_pct(r.high_52w_distance)}",
        f"  8주 위치: {r.range_position_8w * 100:.1f}%",
        f"  진입가: {format_price(r.entry)}",
        f"  손절가: {format_price(r.stop)}",
    ]


def notify_all(results: List[Result]) -> None:
    if not telegram_enabled():
        return

    sorted_results = sorted(
        results,
        key=lambda x: (
            0 if x.grade == "A" else 1,
            0 if x.state == "돌파 임박" else 1,
            -x.score_total,
            x.ticker,
        ),
    )

    lines = ["[압축 패턴] 10-Day Tight", f"전체 후보: {len(sorted_results)}"]
    for r in sorted_results[:MAX_TELEGRAM_ROWS]:
        lines.extend(build_result_lines(r))

    send_telegram_chunked(lines)

    new_state: Dict[str, Dict[str, Any]] = {}
    for r in sorted_results:
        new_state[r.ticker] = {
            "state": r.state,
            "grade": r.grade,
            "date": r.date,
        }
    save_state(new_state)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    setup_logging()
    universe = load_universe()

    spy = download(BENCHMARK)
    if spy.empty or len(spy) < MIN_HISTORY:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    results: List[Result] = []

    for row in universe.itertuples(index=False):
        try:
            res = scan_one(ticker=row.ticker, name=row.name, spy_df=spy)
            if res is not None:
                results.append(res)
        except Exception as e:
            logging.exception("%s failed: %s", row.ticker, e)

    save_outputs(results)
    notify_all(results)

    if telegram_enabled():
        msg = (
            f"[요약] 10-Day Tight\n"
            f"날짜: {results[0].date if results else 'N/A'}\n"
            f"유니버스 수: {len(universe)}\n"
            f"총 후보: {len(results)}\n"
            f"돌파: {sum(r.state == '돌파' for r in results)}\n"
            f"돌파 임박: {sum(r.state == '돌파 임박' for r in results)}\n"
            f"A: {sum(r.grade == 'A' for r in results)}\n"
            f"B: {sum(r.grade == 'B' for r in results)}"
        )
        send_telegram(msg)


if __name__ == "__main__":
    main()
