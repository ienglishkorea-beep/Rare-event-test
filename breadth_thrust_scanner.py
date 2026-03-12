import os
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests


# ============================================================
# Breadth Thrust Scanner
# ------------------------------------------------------------
# 목적
# 시장 내부 참여도가 급격히 확대되는 순간 탐지
#
# 핵심 신호
# Zweig Breadth Thrust
#
# 정의
# 10일 EMA breadth가
# 0.40 이하 → 0.615 이상
# 10일 이내에 발생
#
# 의미
# 강력한 상승 추세 시작 가능성
# ============================================================


# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "breadth_thrust.log")

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")


# ------------------------------------------------------------
# 텔레그램
# ------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ------------------------------------------------------------
# Zweig 기준
# ------------------------------------------------------------
THRUST_LOW = 0.40
THRUST_HIGH = 0.615
WINDOW = 10


# ------------------------------------------------------------
# 로깅
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
# 유니버스
# ------------------------------------------------------------
def load_universe():

    df = pd.read_csv(UNIVERSE_FILE)

    df["ticker"] = df["ticker"].str.upper()

    return df["ticker"].tolist()


# ------------------------------------------------------------
# 데이터 다운로드
# ------------------------------------------------------------
def download_prices(tickers: List[str]):

    data = yf.download(
        tickers,
        period="6mo",
        group_by="ticker",
        progress=False
    )

    return data


# ------------------------------------------------------------
# breadth 계산
# ------------------------------------------------------------
def calculate_breadth(data, tickers):

    advances = []
    declines = []

    dates = None

    for t in tickers:

        try:

            df = data[t]

            if dates is None:
                dates = df.index

            change = df["Close"].diff()

            adv = change > 0
            dec = change < 0

            advances.append(adv)
            declines.append(dec)

        except:
            continue

    adv_df = pd.concat(advances, axis=1)
    dec_df = pd.concat(declines, axis=1)

    adv_count = adv_df.sum(axis=1)
    dec_count = dec_df.sum(axis=1)

    breadth = adv_count / (adv_count + dec_count)

    return breadth


# ------------------------------------------------------------
# thrust 탐지
# ------------------------------------------------------------
def detect_thrust(breadth):

    ema10 = breadth.ewm(span=10).mean()

    for i in range(WINDOW, len(ema10)):

        past = ema10.iloc[i-WINDOW:i]

        if past.min() <= THRUST_LOW:

            if ema10.iloc[i] >= THRUST_HIGH:

                return True, ema10.iloc[i], ema10.iloc[i-WINDOW:i].min()

    return False, ema10.iloc[-1], ema10.iloc[-WINDOW:].min()


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main():

    setup_logging()

    tickers = load_universe()

    logging.info(f"Universe {len(tickers)}")

    data = download_prices(tickers)

    breadth = calculate_breadth(data, tickers)

    thrust, current, recent_low = detect_thrust(breadth)

    if thrust:

        msg = (
            "Breadth Thrust 발생\n\n"
            f"최근 breadth 저점: {recent_low:.3f}\n"
            f"현재 breadth EMA10: {current:.3f}\n\n"
            "시장 내부 상승 참여도 급증\n"
            "강세 레짐 전환 가능"
        )

        logging.info("Breadth Thrust DETECTED")

        send_telegram(msg)

    else:

        logging.info("No thrust")

    df = pd.DataFrame({
        "breadth": breadth,
        "ema10": breadth.ewm(span=10).mean()
    })

    df.to_csv(os.path.join(OUTPUT_DIR, "breadth_series.csv"))


if __name__ == "__main__":
    main()
