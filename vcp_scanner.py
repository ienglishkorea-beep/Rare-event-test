import pandas as pd
import numpy as np

MIN_STAGE = 3
MAX_STAGE = 5
MIN_STAGE_LEN = 5
MAX_STAGE_LEN = 20
MIN_PATTERN_LEN = 20
MAX_PATTERN_LEN = 80

VOLUME_CONTRACTION = 0.75
NEAR_PIVOT = 0.97

def detect_vcp(df):
    if len(df) < 120:
        return None

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    vols = df["Volume"].values

    for window in range(MIN_PATTERN_LEN, MAX_PATTERN_LEN):
        segment = df.iloc[-window:]

        ranges = []
        pivots = []

        # 단계 분할
        for step in range(MIN_STAGE, MAX_STAGE + 1):

            part = len(segment) // step
            stage_ranges = []

            for i in range(step):
                sub = segment.iloc[i*part:(i+1)*part]
                if len(sub) < MIN_STAGE_LEN:
                    continue

                high = sub["High"].max()
                low = sub["Low"].min()

                r = (high - low) / high
                stage_ranges.append(r)

            if len(stage_ranges) < MIN_STAGE:
                continue

            # 수축 확인
            if all(stage_ranges[i] > stage_ranges[i+1] for i in range(len(stage_ranges)-1)):
                pivot = segment["High"].max()

                vol10 = segment["Volume"].tail(10).mean()
                vol50 = df["Volume"].tail(50).mean()

                if vol10 / vol50 > VOLUME_CONTRACTION:
                    continue

                close = df["Close"].iloc[-1]

                if close < pivot * NEAR_PIVOT:
                    continue

                return {
                    "stages": step,
                    "ranges": stage_ranges,
                    "pivot": pivot
                }

    return None
