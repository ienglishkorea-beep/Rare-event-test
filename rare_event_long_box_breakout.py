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
# Rare Event Scanner - Long Box Breakout
# ------------------------------------------------------------
# 목적
# - 6개월~1년 장기 박스 돌파 후보 탐지
# - 드물게 오는 이벤트성 기회만 별도 알림
# - 기존 Minervini/VCP 스캐너와 분리 운용
#
# 준비
# pip install pandas numpy requests yfinance openpyxl
#
# 필수 파일
# data/universe.csv
# 컬럼:
# ticker,name
# 선택:
# market_cap
# ============================================================


# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.log")
STATE_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_state.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
BENCHMARK_TICKER = "SPY"


# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
MIN_HISTORY = 320
DOWNLOAD_PERIOD = "2y"
DOWNLOAD_INTERVAL = "1d"

MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000
ENFORCE_MARKET_CAP = False
MIN_MARKET_CAP = 1_000_000_000

# 시장 레짐
USE_MARKET_REGIME = True
MIN_BREADTH_50MA = 0.40  # 전체 유니버스 중 50MA 상회 비율 최소 40%

# 장기 박스
BOX_LOOKBACKS = [120, 160, 200, 250]   # 약 6개월 ~ 1년
MAX_BOX_WIDTH = 0.32                    # 장기 박스라 넓이를 너무 타이트하게 잡지 않음
MIN_BOX_WIDTH = 0.08                    # 너무 좁으면 의미 없는 박스 제외
MIN_BOX_TOUCHES = 3                     # 박스 상단 근접 터치 횟수
TOUCH_TOLERANCE = 0.04                  # 상단 4% 이내면 터치로 간주
MIN_BOX_LENGTH = 100                    # 최소 100거래일

# 돌파 / 근접
BREAKOUT_CONFIRM_PCT = 0.00             # 종가가 박스 상단만 넘어도 돌파로 인정
NEAR_PIVOT_PCT = 0.03                   # 아직 돌파 전이라도 3% 이내면 후보
MAX_BREAKOUT_EXTEND_PCT = 0.03          # 돌파 후 3% 이상 확장된 건 chase로 제외
BREAKOUT_VOL_RATIO = 1.30               # 장기 박스 돌파는 거래량 확인이 중요

# 52주 관련
NEAR_52W_HIGH_RATIO = 0.92              # 52주 신고가 92% 이상

# 텔레그램
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_ROWS = 15


# ------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------
@dataclass
class LongBoxResult:
    ticker: str
    name: str
    as_of_date: str
    event_type: str              # BREAKOUT / NEAR_BREAKOUT
    close: float
    box_high: float
    box_low: float
    box_width_pct: float
    box_length: int
    touch_count: int
    near_52w_high: bool
    breakout_volume_ratio: Optional[float]
    entry_price: Optional[float]
    stop_price: Optional[float]
    ma50: Optional[float]
    ma150: Optional[float]
    ma200: Optional[float]
    dollar_vol_20: Optional[float]
    rs_6m_excess: Optional[float]
    reason: str


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


def round_price(x: Optional[float], digits: int = 2) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    return round(float(x), digits)


def format_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.2f}"


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


# ------------------------------------------------------------
# 유니버스 로드
# ------------------------------------------------------------
def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"유니버스 파일 없음: {UNIVERSE_FILE}")

    meta = pd.read_csv(UNIVERSE_FILE)
    meta.columns = [str(c).strip().lower() for c in meta.columns]

    if "ticker" not in meta.columns:
        raise ValueError("universe.csv에는 ticker 컬럼이 필요")
    if "name" not in meta.columns:
        meta["name"] = meta["ticker"]
    if "market_cap" not in meta.columns:
        meta["market_cap"] = np.nan

    meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
    meta["name"] = meta["name"].astype(str)
    meta["market_cap"] = pd.to_numeric(meta["market_cap"], errors="coerce")

    return meta[["ticker", "name", "market_cap"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


# ------------------------------------------------------------
# 데이터 다운로드
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


def download_history(ticker: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=DOWNLOAD_PERIOD,
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


# ------------------------------------------------------------
# 지표
# ------------------------------------------------------------
def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def add_indicators(df: pd.DataFrame, benchmark: Optional[pd.DataFrame], market_cap_from_universe: Optional[float]) -> pd.DataFrame:
    out = df.copy()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma150"] = out["Close"].rolling(150).mean()
    out["ma200"] = out["Close"].rolling(200).mean()
    out["vol_ma20"] = out["Volume"].rolling(20).mean()
    out["dollar_vol"] = out["Close"] * out["Volume"]
    out["dollar_vol_20"] = out["dollar_vol"].rolling(20).mean()
    out["high_252"] = out["High"].rolling(252).max()
    out["ret_6m"] = rolling_return(out["Close"], 126)
    out["market_cap"] = market_cap_from_universe
    out["rs_6m_excess"] = np.nan

    if benchmark is not None and len(benchmark) > 0:
        b = benchmark[["Date", "Close"]].copy().rename(columns={"Close": "benchmark_close"})
        x = out.merge(b, on="Date", how="left")
        x["benchmark_close"] = x["benchmark_close"].ffill()
        x["benchmark_ret_6m"] = rolling_return(x["benchmark_close"], 126)
        x["rs_6m_excess"] = x["ret_6m"] - x["benchmark_ret_6m"]
        out = x

    return out


# ------------------------------------------------------------
# 시장 breadth
# ------------------------------------------------------------
def compute_universe_breadth(dfs: List[pd.DataFrame]) -> float:
    valid = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        last = df.iloc[-1]
        if pd.notna(last.get("ma50")):
            valid.append(bool(last["Close"] > last["ma50"]))
    if not valid:
        return 0.0
    return float(np.mean(valid))


def market_regime_ok(benchmark_df: pd.DataFrame, breadth: float) -> bool:
    if len(benchmark_df) < 220:
        return False
    row = benchmark_df.iloc[-1]
    if pd.isna(row["ma50"]) or pd.isna(row["ma200"]):
        return False
    cond1 = row["Close"] > row["ma200"]
    cond2 = row["ma50"] > row["ma200"]
    cond3 = breadth >= MIN_BREADTH_50MA
    return bool(cond1 and cond2 and cond3)


# ------------------------------------------------------------
# 장기 박스 탐지
# ------------------------------------------------------------
def count_box_touches(seg: pd.DataFrame, box_high: float) -> int:
    threshold = box_high * (1.0 - TOUCH_TOLERANCE)
    return int((seg["High"] >= threshold).sum())


def find_best_long_box(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df) < max(BOX_LOOKBACKS):
        return None

    best = None
    best_score = -1e18
    end_idx = len(df) - 1

    for lookback in BOX_LOOKBACKS:
        start_idx = end_idx - lookback + 1
        if start_idx < 0:
            continue

        seg = df.iloc[start_idx:end_idx + 1].copy()
        if len(seg) < MIN_BOX_LENGTH:
            continue

        box_high = float(seg["High"].max())
        box_low = float(seg["Low"].min())
        if box_high <= 0 or box_low <= 0 or box_low >= box_high:
            continue

        box_width = (box_high - box_low) / box_high
        if box_width < MIN_BOX_WIDTH or box_width > MAX_BOX_WIDTH:
            continue

        touches = count_box_touches(seg, box_high)
        if touches < MIN_BOX_TOUCHES:
            continue

        current_close = float(seg.iloc[-1]["Close"])
        right_side_ratio = current_close / box_high

        # 장기 박스는 우측 회복 중요
        if right_side_ratio < 0.85:
            continue

        score = 0.0
        # 박스 길이가 길수록 가산
        score += lookback * 0.1
        # 박스 폭은 너무 넓지 않을수록 가산
        score += (1.0 - box_width) * 100.0
        # 상단 터치 횟수 가산
        score += touches * 5.0
        # 현재가가 상단 가까울수록 가산
        score += right_side_ratio * 50.0

        if score > best_score:
            best_score = score
            best = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "lookback": lookback,
                "box_high": box_high,
                "box_low": box_low,
                "box_width_pct": box_width,
                "touch_count": touches,
                "right_side_ratio": right_side_ratio,
            }

    return best


def classify_long_box_event(df: pd.DataFrame, box: Dict[str, Any]) -> Tuple[Optional[str], str, Optional[float], Optional[float], Optional[float]]:
    row = df.iloc[-1]
    close = float(row["Close"])
    box_high = float(box["box_high"])
    box_low = float(box["box_low"])

    breakout_vol_ratio = None
    if pd.notna(row["vol_ma20"]) and row["vol_ma20"] > 0:
        breakout_vol_ratio = float(row["Volume"] / row["vol_ma20"])

    near_52w = bool(pd.notna(row["high_252"]) and close >= row["high_252"] * NEAR_52W_HIGH_RATIO)
    if not near_52w:
        return None, "not_near_52w_high", None, None, breakout_vol_ratio

    # 돌파
    breakout = close >= box_high * (1.0 + BREAKOUT_CONFIRM_PCT)
    extended = close > box_high * (1.0 + MAX_BREAKOUT_EXTEND_PCT)
    near_pivot = (close < box_high) and (close >= box_high * (1.0 - NEAR_PIVOT_PCT))

    if breakout and not extended:
        if breakout_vol_ratio is None or breakout_vol_ratio < BREAKOUT_VOL_RATIO:
            return None, "breakout_but_no_volume", None, None, breakout_vol_ratio

        entry_price = box_high * 1.001
        stop_price = box_low * 0.9975
        return "BREAKOUT", "long_box_breakout", entry_price, stop_price, breakout_vol_ratio

    if near_pivot:
        entry_price = box_high * 1.001
        stop_price = box_low * 0.9975
        return "NEAR_BREAKOUT", "near_long_box_pivot", entry_price, stop_price, breakout_vol_ratio

    return None, "not_breakout_zone", None, None, breakout_vol_ratio


# ------------------------------------------------------------
# 스캔
# ------------------------------------------------------------
def scan_one_ticker(
    ticker: str,
    name: str,
    market_cap: Optional[float],
    benchmark: pd.DataFrame,
) -> Optional[LongBoxResult]:
    price_df = download_history(ticker)
    if price_df.empty or len(price_df) < MIN_HISTORY:
        return None

    df = add_indicators(price_df, benchmark, market_cap)
    row = df.iloc[-1]

    if pd.isna(row["Close"]) or row["Close"] < MIN_PRICE:
        return None
    if pd.isna(row["dollar_vol_20"]) or row["dollar_vol_20"] < MIN_DOLLAR_VOL_20:
        return None
    if ENFORCE_MARKET_CAP and (pd.isna(row["market_cap"]) or row["market_cap"] < MIN_MARKET_CAP):
        return None

    box = find_best_long_box(df)
    if box is None:
        return None

    event_type, reason, entry_price, stop_price, breakout_vol_ratio = classify_long_box_event(df, box)
    if event_type is None:
        return None

    near_52w = bool(pd.notna(row["high_252"]) and row["Close"] >= row["high_252"] * NEAR_52W_HIGH_RATIO)

    return LongBoxResult(
        ticker=ticker,
        name=name,
        as_of_date=str(pd.to_datetime(row["Date"]).date()),
        event_type=event_type,
        close=round(float(row["Close"]), 2),
        box_high=round(float(box["box_high"]), 2),
        box_low=round(float(box["box_low"]), 2),
        box_width_pct=round(float(box["box_width_pct"]), 4),
        box_length=int(box["lookback"]),
        touch_count=int(box["touch_count"]),
        near_52w_high=near_52w,
        breakout_volume_ratio=round(breakout_vol_ratio, 2) if breakout_vol_ratio is not None else None,
        entry_price=round_price(entry_price),
        stop_price=round_price(stop_price),
        ma50=round_price(row["ma50"]),
        ma150=round_price(row["ma150"]),
        ma200=round_price(row["ma200"]),
        dollar_vol_20=round_price(row["dollar_vol_20"], 0),
        rs_6m_excess=safe_float(row["rs_6m_excess"]),
        reason=reason,
    )


# ------------------------------------------------------------
# 텔레그램 / 상태
# ------------------------------------------------------------
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


def notify_changes(results: List[LongBoxResult]) -> None:
    if not telegram_enabled():
        return

    prev_state = load_state()
    new_state: Dict[str, Dict[str, Any]] = {}

    new_alerts: List[LongBoxResult] = []
    for r in results:
        old = prev_state.get(r.ticker, {})
        old_event = str(old.get("event_type", ""))
        old_date = str(old.get("as_of_date", ""))

        # 같은 날짜, 같은 이벤트는 중복 방지
        if not (old_event == r.event_type and old_date == r.as_of_date):
            new_alerts.append(r)

        new_state[r.ticker] = {
            "event_type": r.event_type,
            "as_of_date": r.as_of_date,
        }

    if new_alerts:
        lines = ["[RARE EVENT] LONG BOX BREAKOUT"]
        for r in new_alerts[:MAX_TELEGRAM_ROWS]:
            lines.append(
                f"- {r.ticker} {r.name} | {r.event_type} | 종가 {format_price(r.close)} | "
                f"박스 {format_price(r.box_low)}~{format_price(r.box_high)} | "
                f"폭 {format_pct(r.box_width_pct)} | "
                f"진입 {format_price(r.entry_price)} | 손절 {format_price(r.stop_price)} | "
                f"거래량 {r.breakout_volume_ratio if r.breakout_volume_ratio is not None else '-'}배"
            )
        send_telegram_message("\n".join(lines))

    save_state(new_state)


# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
def save_outputs(results: List[LongBoxResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not results:
        pd.DataFrame(
            columns=[f.name for f in LongBoxResult.__dataclass_fields__.values()]
        ).to_csv(
            os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        return

    df = pd.DataFrame([asdict(r) for r in results]).sort_values(
        ["event_type", "ticker"],
        ascending=[True, True],
    )
    df.to_csv(
        os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "breakout": int((df["event_type"] == "BREAKOUT").sum()),
        "near_breakout": int((df["event_type"] == "NEAR_BREAKOUT").sum()),
        "as_of_date": str(df["as_of_date"].max()),
    }
    with open(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    setup_logging()
    logging.info("Rare event scanner start")

    universe = load_universe()
    benchmark_raw = download_history(BENCHMARK_TICKER)
    if benchmark_raw.empty:
        raise RuntimeError("SPY 데이터 다운로드 실패")
    benchmark = add_indicators(benchmark_raw, None, None)

    # breadth 계산용
    enriched_list = []
    staged: List[Tuple[str, str, Optional[float], pd.DataFrame]] = []

    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).upper().strip()
        name = str(row.name)
        market_cap = safe_float(row.market_cap)

        try:
            raw_df = download_history(ticker)
            if raw_df.empty or len(raw_df) < MIN_HISTORY:
                continue
            enriched = add_indicators(raw_df, benchmark_raw, market_cap)
            enriched_list.append(enriched)
            staged.append((ticker, name, market_cap, raw_df))
        except Exception as e:
            logging.exception("%s preload failed: %s", ticker, e)

    breadth = compute_universe_breadth(enriched_list)
    regime_ok = market_regime_ok(benchmark, breadth)

    logging.info("Market regime | breadth=%.2f | regime_ok=%s", breadth, regime_ok)

    if USE_MARKET_REGIME and not regime_ok:
        send_telegram_message(
            f"[RARE EVENT] LONG BOX BREAKOUT\n"
            f"시장 레짐 OFF\n"
            f"SPY {benchmark.iloc[-1]['Close']:.2f} | "
            f"50MA {benchmark.iloc[-1]['ma50']:.2f} | "
            f"200MA {benchmark.iloc[-1]['ma200']:.2f} | "
            f"Breadth {breadth * 100:.1f}%\n"
            f"신규 이벤트 알림 중단"
        )
        save_outputs([])
        return

    results: List[LongBoxResult] = []
    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).upper().strip()
        name = str(row.name)
        market_cap = safe_float(row.market_cap)

        try:
            result = scan_one_ticker(
                ticker=ticker,
                name=name,
                market_cap=market_cap,
                benchmark=benchmark_raw,
            )
            if result is not None:
                results.append(result)
                logging.info("%s | %s | %s", ticker, result.event_type, result.reason)
        except Exception as e:
            logging.exception("%s failed: %s", ticker, e)

    save_outputs(results)
    notify_changes(results)

    if telegram_enabled():
        breakout_count = sum(r.event_type == "BREAKOUT" for r in results)
        near_count = sum(r.event_type == "NEAR_BREAKOUT" for r in results)
        msg = (
            f"[RARE EVENT SUMMARY] LONG BOX BREAKOUT\n"
            f"Date: {results[0].as_of_date if results else 'N/A'}\n"
            f"Universe scanned: {len(universe)}\n"
            f"Breadth: {breadth * 100:.1f}%\n"
            f"Breakout: {breakout_count}\n"
            f"Near breakout: {near_count}\n"
            f"Total alerts: {len(results)}"
        )
        send_telegram_message(msg)

    logging.info("Rare event scanner done | total=%d", len(results))


if __name__ == "__main__":
    main()
