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
# Breadth Thrust Scanner - Upgraded Final
# ------------------------------------------------------------
# 목적
# - 시장 내부 참여 폭 확장을 탐지
# - 개별 종목 신호보다 상위의 "시장 환경 신호"로 사용
# - 다른 모멘텀 / 베이스 / 압축 전략의 공격 강도 조절용
#
# 구조
# 1) Zweig Breadth Thrust 탐지
# 2) 50MA Breadth 확인
# 3) SPY 추세 확인
# 4) 섹터 확산 확인
# 5) 최종 A / B 분류
#
# 해석
# - A: 공격 강도 상향 검토 가능
# - B: 환경 개선 신호, 부분적 적극 운용 가능
# - 신호 없음: 보수적으로 유지
#
# 준비
# pip install pandas numpy requests yfinance openpyxl
#
# 필수 파일
# data/universe.csv
# 컬럼:
# ticker,name
# ============================================================


# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "breadth_thrust_scanner.log")
STATE_FILE = os.path.join(OUTPUT_DIR, "breadth_thrust_state.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")

SPY_TICKER = "SPY"

# 섹터 ETF
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


# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
DOWNLOAD_PERIOD = "1y"
MIN_HISTORY = 120

# Zweig Breadth Thrust
THRUST_LOW = 0.40
THRUST_HIGH = 0.615
THRUST_WINDOW = 10
BREADTH_EMA_SPAN = 10

# 50MA Breadth
BREADTH_50MA_A = 0.60
BREADTH_50MA_B = 0.55

# SPY 추세
SPY_20D_RET_A = 0.05
SPY_20D_RET_B = 0.03

# 섹터 확산
SECTOR_UP_LOOKBACK = 10
SECTOR_UP_COUNT_A = 7
SECTOR_UP_COUNT_B = 6

# 최소 유니버스
MIN_VALID_UNIVERSE = 300

# 텔레그램
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# ------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------
@dataclass
class BreadthResult:
    as_of_date: str
    신호발생: bool
    등급: Optional[str]
    등급_설명: Optional[str]
    Zweig_발생: bool
    Zweig_저점: Optional[float]
    Zweig_현재EMA10: Optional[float]
    Breadth50MA: Optional[float]
    SPY_종가: Optional[float]
    SPY_50일선: Optional[float]
    SPY_200일선: Optional[float]
    SPY_20일수익률: Optional[float]
    섹터상승개수: int
    섹터상승목록: str
    유니버스유효종목수: int
    설명: str


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


# ------------------------------------------------------------
# 파일 / 상태
# ------------------------------------------------------------
def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"유니버스 파일 없음: {UNIVERSE_FILE}")

    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise ValueError("universe.csv에는 ticker 컬럼이 필요")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df[["ticker"]].drop_duplicates().reset_index(drop=True)


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

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).sort_values("Date").reset_index(drop=True)
    return df


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


# ------------------------------------------------------------
# 지표
# ------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma200"] = out["Close"].rolling(200).mean()
    out["ret20"] = out["Close"] / out["Close"].shift(20) - 1.0
    return out


# ------------------------------------------------------------
# Breadth 계산
# ------------------------------------------------------------
def build_universe_price_matrix(tickers: List[str]) -> pd.DataFrame:
    close_frames = []

    for ticker in tickers:
        try:
            df = download_history(ticker)
            if df.empty or len(df) < MIN_HISTORY:
                continue
            s = df[["Date", "Close"]].copy()
            s = s.rename(columns={"Close": ticker})
            close_frames.append(s)
        except Exception as e:
            logging.exception("%s download failed: %s", ticker, e)

    if not close_frames:
        return pd.DataFrame()

    merged = close_frames[0]
    for frame in close_frames[1:]:
        merged = merged.merge(frame, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)
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

    denom = adv_count + dec_count
    breadth = adv_count / denom.replace(0, np.nan)
    breadth = breadth.fillna(method="ffill")
    return breadth


def detect_zweig_breadth_thrust(breadth: pd.Series) -> Tuple[bool, Optional[float], Optional[float]]:
    if breadth.empty or len(breadth) < THRUST_WINDOW + 5:
        return False, None, None

    ema10 = breadth.ewm(span=BREADTH_EMA_SPAN, adjust=False).mean()

    for i in range(THRUST_WINDOW, len(ema10)):
        window = ema10.iloc[i - THRUST_WINDOW:i]
        past_low = float(window.min())
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

    above = latest_prices[valid] > latest_ma50[valid]
    return float(above.mean())


# ------------------------------------------------------------
# SPY / 섹터 확인
# ------------------------------------------------------------
def get_spy_confirmation() -> Dict[str, Optional[float]]:
    df = download_history(SPY_TICKER)
    if df.empty or len(df) < MIN_HISTORY:
        return {
            "close": None,
            "ma50": None,
            "ma200": None,
            "ret20": None,
        }

    x = add_indicators(df)
    row = x.iloc[-1]
    return {
        "close": safe_float(row["Close"]),
        "ma50": safe_float(row["ma50"]),
        "ma200": safe_float(row["ma200"]),
        "ret20": safe_float(row["ret20"]),
    }


def count_sector_participation() -> Tuple[int, List[str]]:
    up_names = []

    for name, ticker in SECTOR_ETFS.items():
        try:
            df = download_history(ticker)
            if df.empty or len(df) < SECTOR_UP_LOOKBACK + 5:
                continue

            ret = df["Close"].iloc[-1] / df["Close"].iloc[-1 - SECTOR_UP_LOOKBACK] - 1.0
            if ret > 0:
                up_names.append(name)
        except Exception as e:
            logging.exception("%s sector failed: %s", ticker, e)

    return len(up_names), up_names


# ------------------------------------------------------------
# 등급 분류
# ------------------------------------------------------------
def classify_grade(
    zweig_ok: bool,
    breadth_50ma: Optional[float],
    spy_info: Dict[str, Optional[float]],
    sector_up_count: int,
) -> Tuple[Optional[str], Optional[str], str]:
    if not zweig_ok:
        return None, None, "Zweig Breadth Thrust 미발생"

    confirm_count = 0
    detail = []

    # 50MA Breadth
    if breadth_50ma is not None:
        if breadth_50ma >= BREADTH_50MA_B:
            confirm_count += 1
            detail.append("50일선 위 종목 비율 양호")
        if breadth_50ma >= BREADTH_50MA_A:
            detail.append("50일선 위 종목 비율 강함")

    # SPY 추세
    spy_close = spy_info.get("close")
    spy_ma50 = spy_info.get("ma50")
    spy_ma200 = spy_info.get("ma200")
    spy_ret20 = spy_info.get("ret20")

    spy_ok = False
    if (
        spy_close is not None
        and spy_ma50 is not None
        and spy_ma200 is not None
        and spy_ret20 is not None
    ):
        if spy_close > spy_ma50 and spy_ma50 > spy_ma200 and spy_ret20 >= SPY_20D_RET_B:
            confirm_count += 1
            spy_ok = True
            detail.append("SPY 추세 양호")
        if spy_ret20 >= SPY_20D_RET_A:
            detail.append("SPY 단기 상승 강함")

    # 섹터 확산
    if sector_up_count >= SECTOR_UP_COUNT_B:
        confirm_count += 1
        detail.append("섹터 확산 양호")
    if sector_up_count >= SECTOR_UP_COUNT_A:
        detail.append("섹터 확산 강함")

    if confirm_count >= 3:
        return "A", "공격 강도 상향 검토", " | ".join(detail) if detail else "강한 환경 신호"
    if confirm_count >= 2:
        return "B", "환경 개선 신호", " | ".join(detail) if detail else "개선된 환경 신호"

    return None, None, "Zweig 발생은 했지만 확인 신호 부족"


# ------------------------------------------------------------
# 출력
# ------------------------------------------------------------
def build_message(result: BreadthResult) -> str:
    return (
        f"[시장 환경 신호] Breadth Thrust\n\n"
        f"등급: {result.등급} ({result.등급_설명})\n"
        f"날짜: {result.as_of_date}\n\n"
        f"Zweig Thrust: {'발생' if result.Zweig_발생 else '미발생'}\n"
        f"최근 저점: {format_pct(result.Zweig_저점)}\n"
        f"(설명: 10일 EMA Breadth가 얼마나 낮은 곳에서 시작했는지)\n"
        f"현재 EMA10: {format_pct(result.Zweig_현재EMA10)}\n"
        f"(설명: 현재 시장 상승 참여 비율의 10일 EMA)\n\n"
        f"50일선 위 종목 비율: {format_pct(result.Breadth50MA)}\n"
        f"(설명: 유니버스 중 현재 가격이 50일선 위에 있는 종목 비율)\n\n"
        f"SPY 종가: {format_price(result.SPY_종가)}\n"
        f"SPY 50일선: {format_price(result.SPY_50일선)}\n"
        f"SPY 200일선: {format_price(result.SPY_200일선)}\n"
        f"SPY 20일 수익률: {format_pct(result.SPY_20일수익률)}\n"
        f"(설명: 시장 가격 추세 확인용)\n\n"
        f"상승 섹터 수: {result.섹터상승개수}개\n"
        f"상승 섹터 목록: {result.섹터상승목록 if result.섹터상승목록 else '-'}\n"
        f"(설명: 최근 10일 기준 상승 중인 주요 섹터 ETF 수)\n\n"
        f"유효 유니버스 종목 수: {result.유니버스유효종목수}\n\n"
        f"해석: {result.설명}"
    )


# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
def save_output(result: BreadthResult) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame([asdict(result)])
    df.to_csv(
        os.path.join(OUTPUT_DIR, "breadth_thrust_result.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(OUTPUT_DIR, "breadth_thrust_result.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    setup_logging()
    logging.info("Breadth thrust scanner start")

    universe = load_universe()
    tickers = universe["ticker"].tolist()

    close_matrix = build_universe_price_matrix(tickers)
    if close_matrix.empty:
        raise RuntimeError("유니버스 가격 데이터 생성 실패")

    valid_count = len(close_matrix.columns) - 1
    logging.info("Valid universe count: %d", valid_count)

    if valid_count < MIN_VALID_UNIVERSE:
        logging.warning("유효 유니버스 종목 수 부족: %d", valid_count)

    breadth = calculate_daily_breadth(close_matrix)
    zweig_ok, zweig_low, zweig_current = detect_zweig_breadth_thrust(breadth)
    breadth_50ma = calculate_breadth_50ma(close_matrix)
    spy_info = get_spy_confirmation()
    sector_up_count, sector_up_names = count_sector_participation()

    grade, grade_desc, explanation = classify_grade(
        zweig_ok=zweig_ok,
        breadth_50ma=breadth_50ma,
        spy_info=spy_info,
        sector_up_count=sector_up_count,
    )

    result = BreadthResult(
        as_of_date=str(breadth.index[-1].date()) if len(breadth) > 0 else str(datetime.now().date()),
        신호발생=grade is not None,
        등급=grade,
        등급_설명=grade_desc,
        Zweig_발생=zweig_ok,
        Zweig_저점=zweig_low,
        Zweig_현재EMA10=zweig_current,
        Breadth50MA=breadth_50ma,
        SPY_종가=spy_info.get("close"),
        SPY_50일선=spy_info.get("ma50"),
        SPY_200일선=spy_info.get("ma200"),
        SPY_20일수익률=spy_info.get("ret20"),
        섹터상승개수=sector_up_count,
        섹터상승목록=", ".join(sector_up_names),
        유니버스유효종목수=valid_count,
        설명=explanation,
    )

    save_output(result)

    prev_state = load_state()
    prev_date = str(prev_state.get("as_of_date", ""))
    prev_grade = str(prev_state.get("등급", ""))

    # 같은 날 같은 등급이면 중복 방지
    should_notify = result.신호발생 and not (
        prev_date == result.as_of_date and prev_grade == str(result.등급)
    )

    if should_notify:
        send_telegram_message(build_message(result))

    save_state({"as_of_date": result.as_of_date, "등급": result.등급})

    logging.info(
        "Done | signal=%s | grade=%s | breadth50=%.3f | sector_up=%d",
        result.신호발생,
        result.등급,
        result.Breadth50MA if result.Breadth50MA is not None else -1,
        result.섹터상승개수,
    )


if __name__ == "__main__":
    main()
