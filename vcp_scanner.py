import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


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

VOL_CONTRACTION_MAX = 0.80
ENTRY_BUFFER = 0.002
STOP_PCT = 0.08
PYRAMID_LEVELS = [0.02, 0.04, 0.06]

# ---------- NEW FILTERS (M&A 제거) ----------

EVENT_SPIKE_THRESHOLD = 0.40      # 15일 40% 급등 제거
FLAT_RANGE_THRESHOLD = 0.015      # 10일 변동폭 1.5% 이하 제거
SINGLE_BAR_SPIKE = 0.20           # 단일 20% 급등 제거

# -------------------------------------------

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


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(text: str) -> None:
    if not telegram_enabled():
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)


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
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

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


# -----------------------------------------------------------
# EVENT / M&A FILTER
# -----------------------------------------------------------

def event_pattern_filter(df: pd.DataFrame) -> bool:

    close = df["Close"]

    # 15일 급등
    if len(close) > 15:
        spike = close.iloc[-1] / close.iloc[-15] - 1
        if spike > EVENT_SPIKE_THRESHOLD:
            return False

    # flat price anchor
    high10 = df["High"].tail(10).max()
    low10 = df["Low"].tail(10).min()
    range10 = (high10 - low10) / close.iloc[-1]

    if range10 < FLAT_RANGE_THRESHOLD:
        return False

    # single bar spike
    daily = close.pct_change().abs().tail(20)
    if daily.max() > SINGLE_BAR_SPIKE:
        return False

    return True


def detect_vcp(df: pd.DataFrame) -> Optional[Dict[str, Any]]:

    if not event_pattern_filter(df):
        return None

    if len(df) < 120:
        return None

    for total_len in range(PATTERN_MIN_LEN, PATTERN_MAX_LEN + 1):
        seg = df.tail(total_len)

        for stages in range(MIN_STAGES, MAX_STAGES + 1):

            n = len(seg)
            part = n // stages
            if part < MIN_STAGE_LEN or part > MAX_STAGE_LEN:
                continue

            ranges = []
            highs = []

            ok = True

            for i in range(stages):

                if i == stages - 1:
                    ch = seg.iloc[i * part :]
                else:
                    ch = seg.iloc[i * part : (i + 1) * part]

                high = float(ch["High"].max())
                low = float(ch["Low"].min())

                if low >= high:
                    ok = False
                    break

                ranges.append((high - low) / high)
                highs.append(high)

            if not ok:
                continue

            contraction = all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1))
            if not contraction:
                continue

            if ranges[-1] > MAX_LAST_STAGE_RANGE:
                continue

            pivot = max(highs)
            close = float(df["Close"].iloc[-1])

            dist = close / pivot - 1

            if not (NEAR_PIVOT_LOW <= dist <= FIRST_BREAKOUT_HIGH):
                continue

            vol10 = df["Volume"].tail(10).mean()
            vol50 = df["Volume"].tail(50).mean()

            if vol50 <= 0:
                continue

            vol_ratio = vol10 / vol50

            if vol_ratio > VOL_CONTRACTION_MAX:
                continue

            return {
                "stages": stages,
                "ranges": ranges,
                "pivot": pivot,
                "vol_ratio": vol_ratio,
            }

    return None


def build_trade_plan(pivot: float):

    entry = pivot * (1 + ENTRY_BUFFER)
    stop = entry * (1 - STOP_PCT)

    add1 = entry * (1 + PYRAMID_LEVELS[0])
    add2 = entry * (1 + PYRAMID_LEVELS[1])
    add3 = entry * (1 + PYRAMID_LEVELS[2])

    return entry, stop, add1, add2, add3


def load_universe():

    df = pd.read_csv(UNIVERSE_FILE)

    if "name" not in df.columns:
        df["name"] = df["ticker"]

    return df[["ticker", "name"]]


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    universe = load_universe()
    spy = download_history(BENCHMARK)

    results: List[VCPResult] = []

    for row in universe.itertuples(index=False):

        try:

            df = download_history(row.ticker)

            if len(df) < MIN_HISTORY:
                continue

            vcp = detect_vcp(df)

            if vcp is None:
                continue

            pivot = vcp["pivot"]

            entry, stop, add1, add2, add3 = build_trade_plan(pivot)

            ranges_text = " → ".join([f"{r*100:.1f}%" for r in vcp["ranges"]])

            r = VCPResult(
                ticker=row.ticker,
                name=row.name,
                as_of_date=str(df["Date"].iloc[-1].date()),
                state="돌파 임박",
                grade="A",
                grade_label="우선 검토",
                rs_grade="A",
                score_total=90,
                close=float(df["Close"].iloc[-1]),
                pivot_price=pivot,
                entry_price=entry,
                stop_price=stop,
                add_price_1=add1,
                add_price_2=add2,
                add_price_3=add3,
                stages=vcp["stages"],
                ranges_text=ranges_text,
                last_stage_range=vcp["ranges"][-1],
                vol_ratio_10_50=vcp["vol_ratio"],
                rs_percentile=90,
                rs_current_vs_high=1,
            )

            results.append(r)

        except Exception:
            continue

    pd.DataFrame([asdict(x) for x in results]).to_csv(
        RESULT_FILE, index=False, encoding="utf-8-sig"
    )

    if telegram_enabled():
        send_telegram(f"[VCP 패턴] {len(results)} 종목")


if __name__ == "__main__":
    main()
