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
# 10일 타이트 스캐너 (최종 버전)
# ------------------------------------------------------------
# 구조
# 1. 하드컷 (패턴 존재 확인)
# 2. 100점 점수화
# 3. A / B 분류
#
# 특징
# - 하드컷 통과 = 반드시 A 또는 B
# - WATCH 없음
# - 세부 항목별 점수 출력
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
# 하드컷 조건
# ------------------------------------------------------------
MIN_HISTORY = 260

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


# ------------------------------------------------------------
# 점수 체계 (100점)
# ------------------------------------------------------------
WEIGHT_CLOSE_COMPRESSION = 25
WEIGHT_RANGE_COMPRESSION = 20
WEIGHT_MOMENTUM = 20
WEIGHT_RS = 15
WEIGHT_VOLUME = 10
WEIGHT_ATR = 5
WEIGHT_52W = 5


# ------------------------------------------------------------
# 텔레그램
# ------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------
def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def pct(a, b):
    if b == 0:
        return None
    return (a - b) / b


def load_universe():
    df = pd.read_csv(UNIVERSE_FILE)
    df["ticker"] = df["ticker"].str.upper()
    return df


# ------------------------------------------------------------
# 다운로드
# ------------------------------------------------------------
def download(ticker):
    df = yf.download(ticker, period="2y", progress=False)

    if df is None or len(df) == 0:
        return None

    df = df.reset_index()

    return df


# ------------------------------------------------------------
# ATR
# ------------------------------------------------------------
def atr(df, n=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    ], axis=1).max(axis=1)

    return tr.rolling(n).mean()


# ------------------------------------------------------------
# 스코어 계산
# ------------------------------------------------------------
def score_close_compression(v):

    if v <= 0.04:
        return 25
    if v <= 0.06:
        return 18

    return 0


def score_range_compression(v):

    if v <= 0.06:
        return 20
    if v <= 0.08:
        return 14

    return 0


def score_momentum(r8, r4):

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


def score_rs(excess, rs_new_high):

    if rs_new_high:
        return 15

    if excess >= 0.10:
        return 10

    if excess >= 0:
        return 6

    return 0


def score_volume(r):

    if r <= 0.8:
        return 10
    if r <= 0.95:
        return 6

    return 0


def score_atr(r):

    if r <= 0.85:
        return 5
    if r <= 0.95:
        return 3

    return 0


def score_52w(close, high):

    if high == 0:
        return 0

    r = close / high

    if r >= 0.95:
        return 5
    if r >= 0.90:
        return 3

    return 0


# ------------------------------------------------------------
# 텔레그램
# ------------------------------------------------------------
def send_telegram(msg):

    if TELEGRAM_BOT_TOKEN == "":
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg
    }

    requests.post(url, data=data)


# ------------------------------------------------------------
# 스캔
# ------------------------------------------------------------
def scan_one(ticker, name, spy):

    df = download(ticker)

    if df is None or len(df) < MIN_HISTORY:
        return None

    close = df["Close"].iloc[-1]

    if close < MIN_PRICE:
        return None

    df["dollar"] = df["Close"] * df["Volume"]

    if df["dollar"].rolling(20).mean().iloc[-1] < MIN_DOLLAR_VOL:
        return None

    high10 = df["High"].tail(10).max()
    low10 = df["Low"].tail(10).min()

    close_high = df["Close"].tail(10).max()
    close_low = df["Close"].tail(10).min()

    close_tight = (close_high - close_low) / close_high
    range_tight = (high10 - low10) / high10

    if close_tight > MAX_CLOSE_TIGHT:
        return None

    if range_tight > MAX_RANGE_TIGHT:
        return None

    r8 = df["Close"].iloc[-1] / df["Close"].iloc[-LOOKBACK_8W] - 1
    r4 = df["Close"].iloc[-1] / df["Close"].iloc[-LOOKBACK_4W] - 1

    if r8 < MIN_RET_8W:
        return None

    if r4 < MIN_RET_4W:
        return None

    # 상태
    if close < high10 * (1 - NEAR_BREAKOUT) and close < high10:
        return None

    # RS
    spy_close = spy["Close"].reindex(df["Date"]).ffill()

    rs_line = df["Close"] / spy_close

    rs_excess = r8 - (spy["Close"].iloc[-1] / spy["Close"].iloc[-LOOKBACK_8W] - 1)

    rs_new_high = rs_line.iloc[-1] >= rs_line.rolling(252).max().iloc[-1] * 0.995

    # 거래량
    v10 = df["Volume"].tail(10).mean()
    v30 = df["Volume"].tail(30).mean()

    vol_ratio = v10 / v30

    # ATR
    df["atr"] = atr(df)

    atr10 = df["atr"].tail(10).mean()
    atr30 = df["atr"].tail(30).mean()

    atr_ratio = atr10 / atr30

    # 52주
    high252 = df["High"].rolling(252).max().iloc[-1]

    # 점수
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

    return Result(
        ticker,
        name,
        str(df["Date"].iloc[-1].date()),
        grade,
        total,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        close,
        high10,
        low10,
        close_tight,
        range_tight,
        r8,
        r4,
        rs_excess,
        entry,
        stop
    )


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main():

    setup_logging()

    universe = load_universe()

    spy = download(BENCHMARK)

    results = []

    for r in universe.itertuples():

        try:

            res = scan_one(r.ticker, r.name, spy)

            if res:
                results.append(res)

        except Exception as e:
            logging.info(f"{r.ticker} error")

    df = pd.DataFrame([asdict(x) for x in results])

    df = df.sort_values("score_total", ascending=False)

    df.to_csv(
        os.path.join(OUTPUT_DIR, "ten_day_tight_results.csv"),
        index=False
    )

    # 텔레그램
    msg = "10일 타이트 결과\n\n"

    for r in results[:10]:

        msg += (
            f"{r.ticker} {r.name}\n"
            f"등급 {r.grade} | 점수 {r.score_total}/100\n"
            f"압축 {r.score_close}/25\n"
            f"변동폭 {r.score_range}/20\n"
            f"상승 {r.score_momentum}/20\n"
            f"RS {r.score_rs}/15\n"
            f"거래량 {r.score_volume}/10\n"
            f"ATR {r.score_atr}/5\n"
            f"52주 {r.score_52w}/5\n"
            f"\n"
        )

    send_telegram(msg)

    logging.info(f"완료 {len(results)} 종목")


if __name__ == "__main__":
    main()
