from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =========================================================
# PATHS / ENV
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
RESULT_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_summary.json")
PROFILE_CACHE_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_profile_cache.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MAX_TELEGRAM_MESSAGE_LEN = 3500
MAX_TELEGRAM_ROWS = 25
SEPARATOR = "────────────"


# =========================================================
# CONFIG
# =========================================================
DOWNLOAD_PERIOD = "2y"
MIN_HISTORY = 260
BENCHMARK = "SPY"

MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000
MIN_MARKET_CAP = 1_000_000_000

# prior strength
MIN_8W_RETURN = 0.20
MIN_4W_RETURN = 0.05
MIN_RS_EXCESS_3M = 0.05
MIN_RS_PERCENTILE = 85.0

# 10-day tight
TIGHT_WINDOW = 10
MAX_CLOSE_TIGHT = 0.035          # 최근 10일 종가 압축
MAX_RANGE_TIGHT = 0.07           # 최근 10일 전체 변동폭
MIN_POSITION_IN_8W = 0.80        # 최근 8주 범위 상단 쪽
ENTRY_BUFFER = 0.003             # 돌파가 소폭 상단
STOP_FLOOR = 0.08                # 최소 손절 8%
STOP_CEIL = 0.12                 # 최대 손절 12%

# event / M&A trap is kept light on purpose
MAX_SINGLE_DAY_SPIKE = 0.20
MAX_20D_RETURN = 0.60
MIN_10D_RANGE_FLOOR = 0.002      # 완전 고정형만 제거

REQUIRE_MA_ALIGNMENT = True

# remove useless sectors / industries / names
BLOCKED_KEYWORDS = [
    # real estate / homebuilding
    "reit",
    "real estate investment trust",
    "real estate",
    "residential construction",
    "homebuilding",
    "home builders",
    "property management",
    "mortgage reit",
    "real estate development",
    "real estate services",
    "real estate - development",
    # energy / materials / cyclical commodity
    "energy",
    "oil",
    "gas",
    "midstream",
    "upstream",
    "downstream",
    "coal",
    "metals",
    "mining",
    "materials",
    "basic materials",
    "steel",
    "aluminum",
    "precious metals",
    "gold",
    "silver",
    "uranium",
    "solar",
    # utilities / insurance / finance noise
    "utilities",
    "utility",
    "insurance",
    "specialty finance",
    "mortgage finance",
    "regional banks",
    # biotech / pharma
    "biotech",
    "biotechnology",
    "drug manufacturers",
    "pharmaceutical",
    "pharma",
    # food / tobacco / agri
    "food",
    "beverage",
    "tobacco",
    "farm products",
    "agricultural inputs",
    "agricultural products",
    # shipping / boring cyclicals user dislikes
    "marine shipping",
    "shipping",
    "water transportation",
    "airlines",
    "air freight",
    "trucking",
    "broadcasting",
    "traditional media",
    # obvious structure noise
    "holdings",
    "acquisition",
    "royalty",
    "partnership",
    "l.p.",
    "lp",
    "adr",
    "ads",
    "depositary",
    "trust",
]

STATE_PRIORITY = {
    "돌파 임박": 0,
    "1차 돌파": 1,
    "후행 가능": 2,
    "관찰": 9,
}


# =========================================================
# DATACLASS
# =========================================================
@dataclass
class TightResult:
    ticker: str
    name: str
    as_of_date: str

    grade: str
    score_total: float
    state: str

    close: float
    entry: float
    stop: float

    close_tight: float
    range_tight: float

    ret_8w: float
    ret_4w: float
    rs_excess_3m: float
    rs_percentile: float
    rs_current_vs_high: float

    position_8w: float
    market_cap: float
    dollar_vol_20: float

    sector: str
    industry: str
    country: str


# =========================================================
# HELPERS
# =========================================================
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_price(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "-"
    return f"{v:,.2f}"


def fmt_pct(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def fmt_cap(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "-"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.0f}M"
    return f"${v:,.0f}"


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(text: str) -> None:
    if not telegram_enabled():
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
        timeout=20,
    )


def send_telegram_chunked(blocks: List[str]) -> None:
    if not telegram_enabled() or not blocks:
        return

    chunks: List[str] = []
    current = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) > MAX_TELEGRAM_MESSAGE_LEN:
            if current:
                chunks.append(current)
            current = block
        else:
            current = candidate

    if current:
        chunks.append(current)

    for chunk in chunks:
        send_telegram(chunk)


def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"유니버스 파일 없음: {UNIVERSE_FILE}")

    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise ValueError("universe.csv에 ticker 컬럼 필요")
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
    need = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[need].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    for c in need[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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


def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def load_profile_cache() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(PROFILE_CACHE_FILE):
        return {}
    try:
        with open(PROFILE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_profile_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROFILE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_profile(ticker: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if ticker in cache:
        return cache[ticker]

    profile = {
        "sector": "",
        "industry": "",
        "country": "",
        "quoteType": "",
        "marketCap": None,
    }

    try:
        info = yf.Ticker(ticker).info
        profile["sector"] = str(info.get("sector", "") or "")
        profile["industry"] = str(info.get("industry", "") or "")
        profile["country"] = str(info.get("country", "") or "")
        profile["quoteType"] = str(info.get("quoteType", "") or "")
        profile["marketCap"] = safe_float(info.get("marketCap"))
    except Exception:
        pass

    cache[ticker] = profile
    return profile


def is_blocked_profile(profile: Dict[str, Any], company_name: str) -> bool:
    quote_type = str(profile.get("quoteType", "") or "").lower()
    country = str(profile.get("country", "") or "").lower()
    sector = str(profile.get("sector", "") or "")
    industry = str(profile.get("industry", "") or "")
    company_name = str(company_name or "")

    text = f"{sector} {industry} {company_name}".lower().strip()

    if quote_type and quote_type != "equity":
        return True

    if country and "united states" not in country and "usa" not in country:
        return True

    return any(keyword in text for keyword in BLOCKED_KEYWORDS)


def get_rs_metrics(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    stock = stock_df.copy().set_index("Date")
    spy = spy_df.copy().set_index("Date")
    spy_close = spy["Close"].reindex(stock.index).ffill()

    rs_line = stock["Close"] / spy_close
    if len(rs_line) < RS_LOOKBACK:
        return None, None, None

    rs_high = safe_float(rs_line.tail(RS_LOOKBACK).max())
    rs_low = safe_float(rs_line.tail(RS_LOOKBACK).min())
    rs_now = safe_float(rs_line.iloc[-1])

    if rs_high is None or rs_low is None or rs_now is None or rs_high <= rs_low:
        return None, None, None

    rs_percentile = ((rs_now - rs_low) / (rs_high - rs_low)) * 100.0
    rs_current_vs_high = rs_now / rs_high
    spy_ret_3m = safe_float(rolling_return(spy["Close"], 63).iloc[-1])
    stock_ret_3m = safe_float(rolling_return(stock["Close"], 63).iloc[-1])

    if spy_ret_3m is None or stock_ret_3m is None:
        return None, None, None

    rs_excess_3m = stock_ret_3m - spy_ret_3m
    return rs_percentile, rs_current_vs_high, rs_excess_3m


def is_event_like(df: pd.DataFrame) -> bool:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    ret_20d = safe_float(rolling_return(close, 20).iloc[-1])
    if ret_20d is not None and ret_20d > MAX_20D_RETURN:
        return True

    daily_ret = close.pct_change().tail(20)
    if not daily_ret.empty and daily_ret.max() > MAX_SINGLE_DAY_SPIKE:
        return True

    gap_ret = (open_ / close.shift(1) - 1.0).tail(20)
    if not gap_ret.empty and gap_ret.max() > 0.15:
        return True

    recent_high = safe_float(high.tail(TIGHT_WINDOW).max())
    recent_low = safe_float(low.tail(TIGHT_WINDOW).min())
    current_close = safe_float(close.iloc[-1])

    if recent_high and recent_low and current_close and current_close > 0:
        floor_range = (recent_high - recent_low) / current_close
        if floor_range < MIN_10D_RANGE_FLOOR:
            return True

    return False


def classify_state(close: float, entry: float) -> str:
    if entry <= 0:
        return "관찰"

    d = close / entry - 1.0
    if -0.03 <= d <= 0.0:
        return "돌파 임박"
    if 0.0 < d <= 0.03:
        return "1차 돌파"
    if 0.03 < d <= 0.08:
        return "후행 가능"
    return "관찰"


def score_pattern(
    close_tight: float,
    range_tight: float,
    ret_8w: float,
    ret_4w: float,
    rs_excess_3m: float,
    rs_percentile: float,
    rs_current_vs_high: float,
    position_8w: float,
) -> float:
    score = 0.0

    if close_tight <= 0.01:
        score += 24
    elif close_tight <= 0.02:
        score += 20
    elif close_tight <= 0.03:
        score += 16
    else:
        score += 10

    if range_tight <= 0.03:
        score += 20
    elif range_tight <= 0.05:
        score += 16
    elif range_tight <= 0.07:
        score += 10

    if ret_8w >= 0.50:
        score += 18
    elif ret_8w >= 0.30:
        score += 14
    elif ret_8w >= 0.20:
        score += 10

    if ret_4w >= 0.20:
        score += 10
    elif ret_4w >= 0.10:
        score += 7
    elif ret_4w >= 0.05:
        score += 4

    if rs_excess_3m >= 0.30:
        score += 10
    elif rs_excess_3m >= 0.15:
        score += 7
    elif rs_excess_3m >= 0.05:
        score += 4

    if rs_percentile >= 98:
        score += 8
    elif rs_percentile >= 92:
        score += 6
    elif rs_percentile >= 85:
        score += 4

    if rs_current_vs_high >= 0.995:
        score += 6
    elif rs_current_vs_high >= 0.97:
        score += 4

    if position_8w >= 0.95:
        score += 4
    elif position_8w >= 0.85:
        score += 2

    return round(score, 1)


def grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    return "WATCH"


def scan_one(ticker: str, name: str, spy_df: pd.DataFrame, profile_cache: Dict[str, Dict[str, Any]]) -> Optional[TightResult]:
    df = download_history(ticker)
    if df.empty or len(df) < MIN_HISTORY:
        return None

    close = safe_float(df["Close"].iloc[-1])
    if close is None or close < MIN_PRICE:
        return None

    df["dollar_vol_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    dollar_vol_20 = safe_float(df["dollar_vol_20"].iloc[-1])
    if dollar_vol_20 is None or dollar_vol_20 < MIN_DOLLAR_VOL_20:
        return None

    profile = get_profile(ticker, profile_cache)
    market_cap = safe_float(profile.get("marketCap"))
    if market_cap is None or market_cap < MIN_MARKET_CAP:
        return None

    if is_blocked_profile(profile, name):
        return None

    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma150"] = df["Close"].rolling(150).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    ma50 = safe_float(df["ma50"].iloc[-1])
    ma150 = safe_float(df["ma150"].iloc[-1])
    ma200 = safe_float(df["ma200"].iloc[-1])
    if ma50 is None or ma150 is None or ma200 is None:
        return None

    if REQUIRE_MA_ALIGNMENT and not (close > ma50 > ma150 > ma200):
        return None

    if is_event_like(df):
        return None

    ret_8w = safe_float(rolling_return(df["Close"], 40).iloc[-1])
    ret_4w = safe_float(rolling_return(df["Close"], 20).iloc[-1])

    if ret_8w is None or ret_4w is None:
        return None
    if ret_8w < MIN_8W_RETURN or ret_4w < MIN_4W_RETURN:
        return None

    rs_percentile, rs_current_vs_high, rs_excess_3m = get_rs_metrics(df, spy_df)
    if rs_percentile is None or rs_current_vs_high is None or rs_excess_3m is None:
        return None
    if rs_percentile < MIN_RS_PERCENTILE or rs_excess_3m < MIN_RS_EXCESS_3M:
        return None

    tight_df = df.tail(TIGHT_WINDOW).copy()
    closes = tight_df["Close"]
    highs = tight_df["High"]
    lows = tight_df["Low"]

    close_tight = (closes.max() - closes.min()) / close
    range_tight = (highs.max() - lows.min()) / close

    if close_tight > MAX_CLOSE_TIGHT or range_tight > MAX_RANGE_TIGHT:
        return None

    recent_8w = df.tail(40).copy()
    high_8w = safe_float(recent_8w["High"].max())
    low_8w = safe_float(recent_8w["Low"].min())
    if high_8w is None or low_8w is None or high_8w <= low_8w:
        return None

    position_8w = (close - low_8w) / (high_8w - low_8w)
    if position_8w < MIN_POSITION_IN_8W:
        return None

    pivot = safe_float(highs.max())
    if pivot is None or pivot <= 0:
        return None

    entry = round(pivot * (1 + ENTRY_BUFFER), 2)
    stop_by_window = lows.min()
    stop_pct = min(max((entry - stop_by_window) / entry, STOP_FLOOR), STOP_CEIL)
    stop = round(entry * (1 - stop_pct), 2)

    state = classify_state(close, entry)
    if state == "관찰":
        return None

    score_total = score_pattern(
        close_tight=close_tight,
        range_tight=range_tight,
        ret_8w=ret_8w,
        ret_4w=ret_4w,
        rs_excess_3m=rs_excess_3m,
        rs_percentile=rs_percentile,
        rs_current_vs_high=rs_current_vs_high,
        position_8w=position_8w,
    )
    grade = grade_from_score(score_total)

    return TightResult(
        ticker=ticker,
        name=name,
        as_of_date=str(df["Date"].iloc[-1].date()),
        grade=grade,
        score_total=score_total,
        state=state,
        close=round(close, 2),
        entry=entry,
        stop=stop,
        close_tight=round(close_tight, 4),
        range_tight=round(range_tight, 4),
        ret_8w=round(ret_8w, 4),
        ret_4w=round(ret_4w, 4),
        rs_excess_3m=round(rs_excess_3m, 4),
        rs_percentile=round(rs_percentile, 2),
        rs_current_vs_high=round(rs_current_vs_high, 4),
        position_8w=round(position_8w, 4),
        market_cap=round(market_cap, 2),
        dollar_vol_20=round(dollar_vol_20, 2),
        sector=str(profile.get("sector", "") or ""),
        industry=str(profile.get("industry", "") or ""),
        country=str(profile.get("country", "") or ""),
    )


# =========================================================
# OUTPUT
# =========================================================
def build_result_block(r: TightResult) -> str:
    return (
        f"{SEPARATOR}\n"
        f"{r.ticker} | {r.name}\n"
        f"등급: {r.grade} | 점수: {r.score_total:.1f}/100\n"
        f"상태: {r.state}\n"
        f"종가: {fmt_price(r.close)}\n"
        f"종가 압축: {fmt_pct(r.close_tight)}\n"
        f"전체 변동폭: {fmt_pct(r.range_tight)}\n"
        f"8주 수익률: {fmt_pct(r.ret_8w)}\n"
        f"4주 수익률: {fmt_pct(r.ret_4w)}\n"
        f"RS 3M 초과수익: {fmt_pct(r.rs_excess_3m)}\n"
        f"RS Percentile: {r.rs_percentile:.1f}\n"
        f"RS/1년RS고점: {fmt_pct(r.rs_current_vs_high)}\n"
        f"8주 위치: {fmt_pct(r.position_8w)}\n"
        f"진입가: {fmt_price(r.entry)}\n"
        f"손절가: {fmt_price(r.stop)}\n"
        f"시총: {fmt_cap(r.market_cap)} | 거래대금20D: {fmt_cap(r.dollar_vol_20)}\n"
        f"섹터: {r.sector or '-'}\n"
        f"산업: {r.industry or '-'}"
    )


def notify_results(results: List[TightResult]) -> None:
    if not telegram_enabled():
        return

    blocks: List[str] = [f"[압축 패턴] 10-Day Tight\n전체 후보: {len(results)}"]
    for r in results[:MAX_TELEGRAM_ROWS]:
        blocks.append(build_result_block(r))

    send_telegram_chunked(blocks)


def save_outputs(results: List[TightResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if results:
        df = pd.DataFrame([asdict(r) for r in results])
        df["state_rank"] = df["state"].map(STATE_PRIORITY).fillna(9)
        df = df.sort_values(
            ["state_rank", "score_total", "rs_percentile", "close_tight"],
            ascending=[True, False, False, True],
        ).drop(columns=["state_rank"])
    else:
        df = pd.DataFrame(columns=[f.name for f in TightResult.__dataclass_fields__.values()])

    df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "near_breakout": int((df["state"] == "돌파 임박").sum()) if not df.empty else 0,
        "breakout": int((df["state"] == "1차 돌파").sum()) if not df.empty else 0,
        "late_breakout": int((df["state"] == "후행 가능").sum()) if not df.empty else 0,
        "A": int((df["grade"] == "A").sum()) if not df.empty else 0,
        "B": int((df["grade"] == "B").sum()) if not df.empty else 0,
        "WATCH": int((df["grade"] == "WATCH").sum()) if not df.empty else 0,
        "as_of_date": str(df["as_of_date"].max()) if not df.empty else None,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    universe = load_universe()
    spy_df = download_history(BENCHMARK)
    if spy_df.empty or len(spy_df) < MIN_HISTORY:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    profile_cache = load_profile_cache()
    results: List[TightResult] = []

    for row in universe.itertuples(index=False):
        try:
            r = scan_one(row.ticker, row.name, spy_df, profile_cache)
            if r is not None:
                results.append(r)
        except Exception:
            continue

    save_profile_cache(profile_cache)

    results = sorted(
        results,
        key=lambda x: (
            STATE_PRIORITY.get(x.state, 9),
            -x.score_total,
            -x.rs_percentile,
            x.close_tight,
            x.ticker,
        ),
    )

    save_outputs(results)
    notify_results(results)


if __name__ == "__main__":
    main()
