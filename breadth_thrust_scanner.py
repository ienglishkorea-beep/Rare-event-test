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
# Breadth Thrust Scanner
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "breadth_thrust_scanner.log")
STATE_FILE = os.path.join(OUTPUT_DIR, "breadth_thrust_state.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")

SPY_TICKER = "SPY"
SECTOR_ETFS = {
    "기술": "XLK",
    "금융": "XLF",
    "헬스케어": "XLV",
    "산업재": "XLI",
    "임의소비재": "XLY",
    "필수소비재": "XLP",
    "에너지": "XLE",
    "소재": "XLB",
    "통신": "XLC",
    "유틸리티": "XLU",
    "부동산": "XLRE",
}

DOWNLOAD_PERIOD = "1y"
MIN_HISTORY = 120
THRUST_LOW = 0.40
THRUST_HIGH = 0.615
THRUST_WINDOW = 10
BREADTH_EMA_SPAN = 10
BREADTH_50MA_A = 0.60
BREADTH_50MA_B = 0.55
SPY_20D_RET_A = 0.05
SPY_20D_RET_B = 0.03
SECTOR_UP_LOOKBACK = 10
SECTOR_UP_COUNT_A = 7
SECTOR_UP_COUNT_B = 6
MIN_VALID_UNIVERSE = 50

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


@dataclass
class BreadthResult:
    as_of_date: str
    signal: bool
    grade: Optional[str]
    grade_label: Optional[str]
    zweig_triggered: bool
    zweig_low: Optional[float]
    zweig_current_ema10: Optional[float]
    breadth_50ma: Optional[float]
    spy_close: Optional[float]
    spy_ma50: Optional[float]
    spy_ma200: Optional[float]
    spy_ret20: Optional[float]
    sector_up_count: int
    sector_up_names: str
    valid_universe_count: int
    comment: str


def setup_logging() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
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


def load_universe() -> List[str]:
    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("universe.csv에는 ticker 컬럼이 필요")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df["ticker"].drop_duplicates().tolist()


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


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


def download_history(ticker: str, period: str = DOWNLOAD_PERIOD) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma200"] = out["Close"].rolling(200).mean()
    out["ret20"] = out["Close"] / out["Close"].shift(20) - 1.0
    return out


def build_universe_price_matrix(tickers: List[str]) -> pd.DataFrame:
    close_frames = []
    success = 0

    for ticker in tickers:
        try:
            df = download_history(ticker)
            if df.empty or len(df) < MIN_HISTORY:
                continue
            s = df[["Date", "Close"]].copy().rename(columns={"Close": ticker})
            close_frames.append(s)
            success += 1
        except Exception:
            continue

    if not close_frames:
        return pd.DataFrame()

    merged = close_frames[0]
    for frame in close_frames[1:]:
        merged = merged.merge(frame, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)
    logging.info("Valid downloaded symbols: %d", success)
    return merged


def calculate_daily_breadth(close_matrix: pd.DataFrame) -> pd.Series:
    if close_matrix.empty:
        return pd.Series(dtype=float)

    prices = close_matrix.set_index("Date")
    changes = prices.diff()
    adv = (changes > 0).astype(int)
    dec = (changes < 0).astype(int)

    adv_count = adv.sum(axis=1)
    dec_count = dec.sum(axis=1)
    denom = (adv_count + dec_count).replace(0, np.nan)
    breadth = adv_count / denom
    return breadth.ffill().dropna()


def detect_zweig_breadth_thrust(breadth: pd.Series) -> Tuple[bool, Optional[float], Optional[float]]:
    if breadth.empty or len(breadth) < THRUST_WINDOW + 5:
        return False, None, None

    ema10 = breadth.ewm(span=BREADTH_EMA_SPAN, adjust=False).mean()
    for i in range(THRUST_WINDOW, len(ema10)):
        past_window = ema10.iloc[i - THRUST_WINDOW:i]
        past_low = float(past_window.min())
        current = float(ema10.iloc[i])
        if past_low <= THRUST_LOW and current >= THRUST_HIGH:
            return True, past_low, current

    return False, float(ema10.iloc[-THRUST_WINDOW:].min()), float(ema10.iloc[-1])


def calculate_breadth_50ma(close_matrix: pd.DataFrame) -> Optional[float]:
    if close_matrix.empty:
        return None

    prices = close_matrix.set_index("Date")
    ma50 = prices.rolling(50).mean()
    latest_prices = prices.iloc[-1]
    latest_ma50 = ma50.iloc[-1]
    valid = latest_ma50.notna()
    if valid.sum() == 0:
        return None
    return float((latest_prices[valid] > latest_ma50[valid]).mean())


def get_spy_confirmation() -> Dict[str, Optional[float]]:
    df = download_history(SPY_TICKER)
    if df.empty or len(df) < MIN_HISTORY:
        return {"close": None, "ma50": None, "ma200": None, "ret20": None}
    x = add_indicators(df)
    row = x.iloc[-1]
    return {
        "close": safe_float(row["Close"]),
        "ma50": safe_float(row["ma50"]),
        "ma200": safe_float(row["ma200"]),
        "ret20": safe_float(row["ret20"]),
    }


def count_sector_participation() -> Tuple[int, List[str]]:
    up_names: List[str] = []
    for name, ticker in SECTOR_ETFS.items():
        try:
            df = download_history(ticker)
            if df.empty or len(df) < SECTOR_UP_LOOKBACK + 5:
                continue
            ret = df["Close"].iloc[-1] / df["Close"].iloc[-1 - SECTOR_UP_LOOKBACK] - 1.0
            if ret > 0:
                up_names.append(name)
        except Exception:
            continue
    return len(up_names), up_names


def classify_grade(zweig_ok: bool, breadth_50ma: Optional[float], spy_info: Dict[str, Optional[float]], sector_up_count: int) -> Tuple[Optional[str], Optional[str], str]:
    if not zweig_ok:
        return None, None, "Zweig Breadth Thrust 미발생"

    confirms = 0
    notes: List[str] = []

    if breadth_50ma is not None and breadth_50ma >= BREADTH_50MA_B:
        confirms += 1
        notes.append("50일선 위 종목 비율 양호")

    if (
        spy_info["close"] is not None
        and spy_info["ma50"] is not None
        and spy_info["ma200"] is not None
        and spy_info["ret20"] is not None
        and spy_info["close"] > spy_info["ma50"]
        and spy_info["ma50"] > spy_info["ma200"]
        and spy_info["ret20"] >= SPY_20D_RET_B
    ):
        confirms += 1
        notes.append("SPY 추세 양호")

    if sector_up_count >= SECTOR_UP_COUNT_B:
        confirms += 1
        notes.append("섹터 확산 양호")

    if confirms >= 3:
        return "A", "공격 강도 상향 검토", " | ".join(notes)
    if confirms >= 2:
        return "B", "환경 개선 신호", " | ".join(notes)
    return None, None, "Zweig 발생은 했지만 확인 신호 부족"


def save_output(result: BreadthResult) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame([asdict(result)]).to_csv(
        os.path.join(OUTPUT_DIR, "breadth_thrust_result.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with open(os.path.join(OUTPUT_DIR, "breadth_thrust_result.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=20)
    except Exception:
        pass


def build_message(result: BreadthResult) -> str:
    return (
        f"[시장 환경] Breadth Thrust\n\n"
        f"등급: {result.grade} ({result.grade_label})\n"
        f"날짜: {result.as_of_date}\n"
        f"Zweig: {'발생' if result.zweig_triggered else '미발생'}\n"
        f"최근 저점: {format_pct(result.zweig_low)}\n"
        f"현재 EMA10: {format_pct(result.zweig_current_ema10)}\n"
        f"50MA Breadth: {format_pct(result.breadth_50ma)}\n"
        f"SPY 종가: {format_price(result.spy_close)}\n"
        f"SPY 50일선: {format_price(result.spy_ma50)}\n"
        f"SPY 200일선: {format_price(result.spy_ma200)}\n"
        f"SPY 20일 수익률: {format_pct(result.spy_ret20)}\n"
        f"상승 섹터 수: {result.sector_up_count}\n"
        f"상승 섹터: {result.sector_up_names or '-'}\n"
        f"유효 유니버스 수: {result.valid_universe_count}\n"
        f"해석: {result.comment}"
    )


def main() -> None:
    setup_logging()

    tickers = load_universe()
    close_matrix = build_universe_price_matrix(tickers)

    valid_count = max(0, len(close_matrix.columns) - 1) if not close_matrix.empty else 0
    if valid_count < MIN_VALID_UNIVERSE:
        result = BreadthResult(
            as_of_date=str(datetime.now().date()),
            signal=False,
            grade=None,
            grade_label=None,
            zweig_triggered=False,
            zweig_low=None,
            zweig_current_ema10=None,
            breadth_50ma=None,
            spy_close=None,
            spy_ma50=None,
            spy_ma200=None,
            spy_ret20=None,
            sector_up_count=0,
            sector_up_names="",
            valid_universe_count=valid_count,
            comment="유효 유니버스 부족",
        )
        save_output(result)
        return

    breadth = calculate_daily_breadth(close_matrix)
    zweig_ok, zweig_low, zweig_current = detect_zweig_breadth_thrust(breadth)
    breadth_50ma = calculate_breadth_50ma(close_matrix)
    spy_info = get_spy_confirmation()
    sector_up_count, sector_up_names = count_sector_participation()
    grade, grade_label, comment = classify_grade(zweig_ok, breadth_50ma, spy_info, sector_up_count)

    result = BreadthResult(
        as_of_date=str(breadth.index[-1].date()) if len(breadth) > 0 else str(datetime.now().date()),
        signal=grade is not None,
        grade=grade,
        grade_label=grade_label,
        zweig_triggered=zweig_ok,
        zweig_low=zweig_low,
        zweig_current_ema10=zweig_current,
        breadth_50ma=breadth_50ma,
        spy_close=spy_info.get("close"),
        spy_ma50=spy_info.get("ma50"),
        spy_ma200=spy_info.get("ma200"),
        spy_ret20=spy_info.get("ret20"),
        sector_up_count=sector_up_count,
        sector_up_names=", ".join(sector_up_names),
        valid_universe_count=valid_count,
        comment=comment,
    )

    save_output(result)

    prev_state = load_state()
    should_notify = result.signal and not (
        str(prev_state.get("as_of_date")) == result.as_of_date and str(prev_state.get("grade")) == str(result.grade)
    )

    if should_notify:
        send_telegram_message(build_message(result))

    save_state({"as_of_date": result.as_of_date, "grade": result.grade})


if __name__ == "__main__":
    main()
