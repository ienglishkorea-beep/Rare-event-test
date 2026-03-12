import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")

RESULT_FILE = os.path.join(OUTPUT_DIR, "vcp_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "vcp_summary.json")

MIN_STAGE = 3
MAX_STAGE = 5
MIN_PATTERN = 20
MAX_PATTERN = 80

NEAR_PIVOT = 0.97
VOLUME_CONTRACTION = 0.75

RS_HARDCUT = 0.80


def load_universe():
    if not os.path.exists(UNIVERSE_FILE):
        return []

    df = pd.read_csv(UNIVERSE_FILE)
    return df["ticker"].dropna().tolist()


def download_data(ticker):
    try:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def compute_rs(df, spy):
    rs = df["Close"] / spy["Close"]
    rs_high = rs.rolling(252).max().iloc[-1]
    rs_now = rs.iloc[-1]
    ratio = rs_now / rs_high if rs_high else 0
    return rs_now, rs_high, ratio


def rs_grade(ratio):
    if ratio >= 1.0:
        return "S"
    if ratio >= 0.9:
        return "A"
    if ratio >= 0.8:
        return "B"
    return None


def detect_vcp(df):
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    vols = df["Volume"].values

    for window in range(MIN_PATTERN, MAX_PATTERN):
        segment = df.iloc[-window:]

        ranges = []

        step = MIN_STAGE
        part = len(segment) // step

        for i in range(step):
            sub = segment.iloc[i * part : (i + 1) * part]

            high = sub["High"].max()
            low = sub["Low"].min()

            r = (high - low) / high
            ranges.append(r)

        contraction = True
        for i in range(len(ranges) - 1):
            if ranges[i] <= ranges[i + 1]:
                contraction = False
                break

        if not contraction:
            continue

        pivot = segment["High"].max()

        vol10 = segment["Volume"].tail(10).mean()
        vol50 = df["Volume"].tail(50).mean()

        if vol10 / vol50 > VOLUME_CONTRACTION:
            continue

        close = df["Close"].iloc[-1]

        if close < pivot * NEAR_PIVOT:
            continue

        return pivot, ranges

    return None, None


def score_pattern(ranges):
    base = 60
    contraction = sum(ranges) * 100
    score = base + max(0, 30 - contraction)
    return min(100, score)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    universe = load_universe()

    spy = yf.download("SPY", period="1y", interval="1d", progress=False)

    rows = []

    for ticker in universe:

        df = download_data(ticker)

        if df is None or len(df) < 200:
            continue

        rs_now, rs_high, rs_ratio = compute_rs(df, spy)

        if rs_ratio < RS_HARDCUT:
            continue

        pivot, ranges = detect_vcp(df)

        if pivot is None:
            continue

        close = df["Close"].iloc[-1]

        entry = pivot * 1.01
        stop = pivot * 0.92

        score = score_pattern(ranges)

        grade = "A" if score >= 85 else "B"

        rows.append(
            {
                "ticker": ticker,
                "name": ticker,
                "grade": grade,
                "state": "돌파 임박",
                "score_total": score,
                "close": close,
                "pivot_price": pivot,
                "entry_price": entry,
                "stop_price": stop,
                "stages": len(ranges),
                "ranges_text": " → ".join([f"{r*100:.1f}%" for r in ranges]),
                "rs_grade": rs_grade(rs_ratio),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df.to_csv(RESULT_FILE, index=False)

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total": int(len(df)),
        "near_breakout": int(len(df)),
        "first_breakout": 0,
        "rs_s": int((df["rs_grade"] == "S").sum()) if not df.empty else 0,
        "rs_a": int((df["rs_grade"] == "A").sum()) if not df.empty else 0,
        "rs_b": int((df["rs_grade"] == "B").sum()) if not df.empty else 0,
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("VCP scan completed")
    print("Candidates:", len(df))


if __name__ == "__main__":
    main()
