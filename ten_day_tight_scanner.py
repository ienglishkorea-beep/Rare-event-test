import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

DATA_DIR = "data"
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")

MIN_PRICE = 10
MIN_DOLLAR_VOL = 10_000_000

TIGHT_DAYS = 10
UPTREND_LOOKBACK = 40

A_TIGHT = 0.05
B_TIGHT = 0.08

ENTRY_BUFFER = 1.002
STOP_LOSS = 0.92

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def download_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if df is None or len(df) < 60:
        return None
    df = df.reset_index()
    return df


def calc_tight_range(df):
    recent = df.tail(TIGHT_DAYS)

    high = recent["High"].max()
    low = recent["Low"].min()

    tight_range = (high - low) / high

    return tight_range, high, low


def calc_uptrend(df):
    past = df.iloc[-UPTREND_LOOKBACK]
    last = df.iloc[-1]

    rise = (last["Close"] - past["Close"]) / past["Close"]

    return rise


def calc_dollar_vol(df):
    df["dollar"] = df["Close"] * df["Volume"]
    return df["dollar"].rolling(20).mean().iloc[-1]


def classify(tight_range):

    if tight_range <= A_TIGHT:
        return "A", "우선 검토"

    if tight_range <= B_TIGHT:
        return "B", "관찰 후보"

    return None, None


def build_trade_levels(pivot):

    entry = pivot * ENTRY_BUFFER
    stop = entry * STOP_LOSS

    add1 = entry * 1.02
    add2 = entry * 1.04
    add3 = entry * 1.06

    return entry, stop, add1, add2, add3


def send_telegram(msg):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg
        }
    )


def format_stock_message(row):

    ticker = row["ticker"]
    name = row["name"]

    tight_pct = row["tight"] * 100
    rise_pct = row["rise"] * 100

    msg = f"""
[10일 타이트 압축]

- {ticker} {name}

등급: {row['grade']} ({row['grade_label']})

종가: {row['close']:.2f}

10일 변동폭: {tight_pct:.1f}%
(설명: 최근 10거래일 최고가와 최저가 차이 / 최고가)

8주 상승: {rise_pct:.1f}%
(설명: 약 40거래일 전 가격 대비 현재 상승률)

거래대금: ${row['dollar_vol']:,.0f}
(설명: 최근 20일 평균 거래대금)

진입가: {row['entry']:.2f}
손절가: {row['stop']:.2f}

1차 추가: {row['add1']:.2f}
2차 추가: {row['add2']:.2f}
3차 추가: {row['add3']:.2f}
"""

    return msg


def scan():

    universe = pd.read_csv(UNIVERSE_FILE)

    results = []

    for _, r in universe.iterrows():

        ticker = r["ticker"]
        name = r["name"]

        try:

            df = download_data(ticker)

            if df is None:
                continue

            price = df["Close"].iloc[-1]

            if price < MIN_PRICE:
                continue

            dollar_vol = calc_dollar_vol(df)

            if dollar_vol < MIN_DOLLAR_VOL:
                continue

            tight, high, low = calc_tight_range(df)

            grade, label = classify(tight)

            if grade is None:
                continue

            rise = calc_uptrend(df)

            entry, stop, add1, add2, add3 = build_trade_levels(high)

            results.append({
                "ticker": ticker,
                "name": name,
                "grade": grade,
                "grade_label": label,
                "tight": tight,
                "rise": rise,
                "close": price,
                "entry": entry,
                "stop": stop,
                "add1": add1,
                "add2": add2,
                "add3": add3,
                "dollar_vol": dollar_vol
            })

        except:
            continue

    return results


def main():

    res = scan()

    if not res:
        print("No signals")
        return

    for r in res:

        msg = format_stock_message(r)

        print(msg)

        send_telegram(msg)


if __name__ == "__main__":
    main()
