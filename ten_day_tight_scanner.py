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
# 10-Day Tight Scanner
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_scanner.log")
STATE_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_state.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
BENCHMARK = "SPY"

MIN_HISTORY = 260
DOWNLOAD_PERIOD = "2y"

MIN_PRICE = 10
MIN_DOLLAR_VOL = 10_000_000

TIGHT_WINDOW = 10
MAX_CLOSE_TIGHT = 0.06
MAX_RANGE_TIGHT = 0.08
LOOKBACK_8W = 40
LOOKBACK_4W = 20
MIN_RET_8W = 0.20
MIN_RET_4W = 0.08
NEAR_BREAKOUT = 0.03

ENTRY_BUFFER = 0.002
STOP_PCT = 0.08
PYRAMID_LEVELS = [0.02, 0.04, 0.06]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_ROWS = 15


@dataclass
class Result:
    ticker: str
    name: str
    date: str
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
    rs_excess: float
    entry: float
    stop: float
    add1: float
    add2: float
    add3: float
    state: str


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


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def score_close_compression(v: float) -> int:
    if v <= 0.04:
        return 25
    if v <= 0.06:
        return 18
    return 0


def score_range_compression(v: float) -> int:
    if v <= 0.06:
        return 20
    if v <= 0.08:
        return 14
    return 0


def score_momentum(r8: float, r4: float) -> int:
    s = 0
    if r8 >= 0.30:
        s += 12
    elif r8 >= 0.20:
        s += 8

    if r4 >= 0.12:
        s += 8
    elif r4 >= 0.08:
        s += 5
    return s


def score_rs(excess: float, rs_new_high: bool) -> int:
    if rs_new_high:
        return 15
    if excess >= 0.10:
        return 10
    if excess >= 0:
        return 6
    return 0


def score_volume(r: float) -> int:
    if r <= 0.8:
        return 10
    if r <= 0.95:
        return 6
    return 0


def score_atr(r: float) -> int:
    if r <= 0.85:
        return 5
    if r <= 0.95:
        return 3
    return 0


def score_52w(close: float, high: float) -> int:
    if high <= 0 or pd.isna(high):
        return 0
    ratio = close / high
    if ratio >= 0.95:
        return 5
    if ratio >= 0.90:
        return 3
    return 0


def send_telegram(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=20)
    except Exception:
        pass


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


def scan_one(ticker: str, name: str, spy: pd.DataFrame) -> Optional[Result]:
    df = download(ticker)
    if df.empty or len(df) < MIN_HISTORY:
        return None

    close = float(df["Close"].iloc[-1])
    if close < MIN_PRICE:
        return None

    df["dollar"] = df["Close"] * df["Volume"]
    if df["dollar"].rolling(20).mean().iloc[-1] < MIN_DOLLAR_VOL:
        return None

    high10 = float(df["High"].tail(10).max())
    low10 = float(df["Low"].tail(10).min())
    close_high = float(df["Close"].tail(10).max())
    close_low = float(df["Close"].tail(10).min())

    close_tight = (close_high - close_low) / close_high
    range_tight = (high10 - low10) / high10

    if close_tight > MAX_CLOSE_TIGHT or range_tight > MAX_RANGE_TIGHT:
        return None

    r8 = float(df["Close"].iloc[-1] / df["Close"].iloc[-LOOKBACK_8W] - 1)
    r4 = float(df["Close"].iloc[-1] / df["Close"].iloc[-LOOKBACK_4W] - 1)
    if r8 < MIN_RET_8W or r4 < MIN_RET_4W:
        return None

    if close < high10 * (1 - NEAR_BREAKOUT) and close < high10:
        return None

    state = "돌파" if close >= high10 else "돌파 임박"

    spy = spy.copy()
    spy = spy.set_index("Date")
    stock = df.set_index("Date")
    spy_close = spy["Close"].reindex(stock.index).ffill()

    rs_line = stock["Close"] / spy_close
    rs_new_high = bool(rs_line.iloc[-1] >= rs_line.rolling(252, min_periods=30).max().iloc[-1] * 0.995)
    spy_r8 = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-LOOKBACK_8W] - 1)
    rs_excess = r8 - spy_r8

    v10 = float(df["Volume"].tail(10).mean())
    v30 = float(df["Volume"].tail(30).mean())
    vol_ratio = v10 / v30 if v30 > 0 else 999

    df["atr"] = atr(df)
    atr10 = float(df["atr"].tail(10).mean())
    atr30 = float(df["atr"].tail(30).mean())
    atr_ratio = atr10 / atr30 if atr30 > 0 else 999

    high252 = float(df["High"].rolling(252).max().iloc[-1]) if len(df) >= 252 else np.nan

    s1 = score_close_compression(close_tight)
    s2 = score_range_compression(range_tight)
    s3 = score_momentum(r8, r4)
    s4 = score_rs(rs_excess, rs_new_high)
    s5 = score_volume(vol_ratio)
    s6 = score_atr(atr_ratio)
    s7 = score_52w(close, high252)

    total = s1 + s2 + s3 + s4 + s5 + s6 + s7
    grade = "A" if total >= 80 else "B"

    entry = high10 * 1.002
    stop = entry * 0.92
    add1 = entry * (1 + PYRAMID_LEVELS[0])
    add2 = entry * (1 + PYRAMID_LEVELS[1])
    add3 = entry * (1 + PYRAMID_LEVELS[2])

    return Result(
        ticker=ticker,
        name=name,
        date=str(df["Date"].iloc[-1].date()),
        grade=grade,
        score_total=total,
        score_close=s1,
        score_range=s2,
        score_momentum=s3,
        score_rs=s4,
        score_volume=s5,
        score_atr=s6,
        score_52w=s7,
        close=close,
        high_10=high10,
        low_10=low10,
        close_tight=close_tight,
        range_tight=range_tight,
        ret_8w=r8,
        ret_4w=r4,
        rs_excess=rs_excess,
        entry=entry,
        stop=stop,
        add1=add1,
        add2=add2,
        add3=add3,
        state=state,
    )


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
                {"run_at": datetime.now().isoformat(), "total": 0, "breakout": 0, "near_breakout": 0, "A": 0, "B": 0, "as_of_date": None},
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    df = pd.DataFrame([asdict(x) for x in results])
    df = df.sort_values("score_total", ascending=False)
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


def notify(results: List[Result]) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    prev_state = load_state()
    new_state: Dict[str, Dict[str, Any]] = {}
    new_alerts: List[Result] = []

    for r in results:
        old = prev_state.get(r.ticker, {})
        if not (str(old.get("state")) == r.state and str(old.get("date")) == r.date and str(old.get("grade")) == r.grade):
            new_alerts.append(r)
        new_state[r.ticker] = {"state": r.state, "grade": r.grade, "date": r.date}

    if new_alerts:
        lines = ["[압축 패턴] 10-Day Tight"]
        for r in new_alerts[:MAX_TELEGRAM_ROWS]:
            lines.extend(
                [
                    f"- {r.ticker} {r.name}",
                    f"  등급: {r.grade} | 점수: {r.score_total}/100",
                    f"  상태: {r.state}",
                    f"  종가 압축: {r.score_close}/25",
                    f"  전체 변동폭: {r.score_range}/20",
                    f"  상승: {r.score_momentum}/20",
                    f"  RS: {r.score_rs}/15",
                    f"  거래량: {r.score_volume}/10",
                    f"  ATR: {r.score_atr}/5",
                    f"  52주: {r.score_52w}/5",
                    f"  진입가: {r.entry:.2f}",
                    f"  손절가: {r.stop:.2f}",
                ]
            )
        send_telegram("\n".join(lines))

    save_state(new_state)


def main() -> None:
    setup_logging()
    universe = load_universe()
    spy = download(BENCHMARK)
    if spy.empty or len(spy) < MIN_HISTORY:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    results: List[Result] = []
    for row in universe.itertuples(index=False):
        try:
            res = scan_one(row.ticker, row.name, spy)
            if res:
                results.append(res)
        except Exception as e:
            logging.exception("%s failed: %s", row.ticker, e)

    save_outputs(results)
    notify(results)

    send_telegram(
        f"[요약] 10-Day Tight\n"
        f"날짜: {results[0].date if results else 'N/A'}\n"
        f"유니버스 수: {len(universe)}\n"
        f"총 후보: {len(results)}\n"
        f"A: {sum(r.grade == 'A' for r in results)}\n"
        f"B: {sum(r.grade == 'B' for r in results)}"
    )


if __name__ == "__main__":
    main()
