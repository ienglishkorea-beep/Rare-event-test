import os
import re
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests


"""
실행용 유니버스 빌더

목적
- Russell 3000급의 넓은 미국 주식 모체 유니버스를 먼저 확보
- 그 위에서 최소 실행 필터 + 명확한 제외 필터 적용
- breadth / long box / 10-day tight 스캐너가 공통 사용 가능한 universe.csv 생성

출력
- data/universe.csv

기본 철학
1) 큰 유니버스 먼저
2) ETF / SPAC / ADR / 잡주 제거
3) 최소 실행 가능성만 남김

필요 라이브러리
pip install pandas requests
"""


# ============================================================
# 경로
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "universe.csv")


# ============================================================
# 최소 실행 필터
# ============================================================
MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 15_000_000
MIN_MARKET_CAP = 300_000_000


# ============================================================
# 외부 소스
# ------------------------------------------------------------
# 1) SEC company_tickers_exchange.json
#    - 미국 상장사 기본 목록 / 거래소 / 회사명
# 2) Nasdaq screener API
#    - 가격 / 시총 / 거래량 등 숫자 필드
#
# Russell 3000 원본 구성종목을 무료/안정적으로 바로 받기 어려운 경우가 많아,
# 실제 구현은 "미국 주요 거래소 상장 보통주 전체"를 넓게 가져온 뒤
# 제거 필터를 통해 Russell 3000급 실행 유니버스로 근사한다.
# ============================================================
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}


# ============================================================
# 제외 키워드
# ============================================================
EXCLUDE_NAME_KEYWORDS = {
    "etf",
    "fund",
    "trust",
    "spdr",
    "ishares",
    "vanguard",
    "invesco",
    "etn",
    "closed-end",
    "income fund",
    "yield shares",
    "acquisition",
    "spac",
    "blank check",
    "warrant",
    "rights",
    "right",
    "unit",
    "depositary",
    "adr",
    "ads",
    "preferred",
    "pref ",
    " preference ",
    "note",
    "bond",
    "income shares",
    "royalty trust",
}

EXCLUDE_SECTOR_KEYWORDS = {
    "biotech",
    "biotechnology",
    "pharma",
    "pharmaceutical",
    "drug",
    "therapeutic",
    "genomics",
    "diagnostic",
    "bioscience",
    "biosciences",
}

EXCLUDE_SYMBOL_PATTERNS = [
    r"\.",        # BRK.B 같은 특수형 제거
    r"\^",        # 지수/특수기호 제거
    r"/",         # 특수 ticker 제거
]


# ============================================================
# 공통 유틸
# ============================================================
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        if isinstance(x, str):
            x = x.strip().replace("$", "").replace(",", "")
            if x in {"", "N/A", "nan", "None"}:
                return None
        return float(x)
    except Exception:
        return None


def normalize_text(x: Any) -> str:
    return str(x or "").strip()


def symbol_has_excluded_pattern(symbol: str) -> bool:
    s = normalize_text(symbol).upper()
    for pattern in EXCLUDE_SYMBOL_PATTERNS:
        if re.search(pattern, s):
            return True
    return False


def contains_any_keyword(text: str, keywords: Set[str]) -> bool:
    low = normalize_text(text).lower()
    return any(k in low for k in keywords)


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# SEC 기본 유니버스
# ============================================================
def fetch_sec_universe() -> pd.DataFrame:
    r = requests.get(SEC_TICKERS_URL, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and "data" in data:
        rows = data["data"]
        fields = data.get("fields", [])
        df = pd.DataFrame(rows, columns=fields)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise RuntimeError("SEC 유니버스 파싱 실패")

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise RuntimeError("SEC 유니버스에 ticker 컬럼 없음")

    if "title" not in df.columns:
        df["title"] = ""
    if "exchange" not in df.columns:
        df["exchange"] = ""

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()

    # 미국 주요 거래소만
    allowed_exchanges = {"Nasdaq", "NYSE", "NYSE American"}
    df = df[df["exchange"].isin(allowed_exchanges)].copy()

    # 명백히 특수 ticker 제거
    df = df[~df["ticker"].apply(symbol_has_excluded_pattern)].copy()

    # 이름 기반 제외
    df = df[~df["title"].apply(lambda x: contains_any_keyword(x, EXCLUDE_NAME_KEYWORDS))].copy()

    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    return df[["ticker", "title", "exchange"]]


# ============================================================
# Nasdaq screener 숫자 데이터
# ============================================================
def fetch_nasdaq_screener() -> pd.DataFrame:
    r = requests.get(NASDAQ_SCREENER_URL, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = data.get("data", {}).get("rows", [])
    if not rows:
        raise RuntimeError("Nasdaq screener rows 비어 있음")

    df = pd.DataFrame(rows)

    rename_map = {
        "symbol": "ticker",
        "name": "name",
        "lastsale": "price",
        "marketCap": "market_cap",
        "volume": "volume",
        "country": "country",
        "sector": "sector",
        "industry": "industry",
        "ipoyear": "ipo_year",
        "exchange": "exchange",
    }
    df = df.rename(columns=rename_map)

    for col in ["ticker", "name", "country", "sector", "industry", "exchange"]:
        if col not in df.columns:
            df[col] = ""

    if "price" not in df.columns:
        df["price"] = None
    if "market_cap" not in df.columns:
        df["market_cap"] = None
    if "volume" not in df.columns:
        df["volume"] = None

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"N/A": None, "": None})
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["dollar_volume"] = df["price"] * df["volume"]

    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    return df[[
        "ticker",
        "name",
        "price",
        "market_cap",
        "volume",
        "dollar_volume",
        "country",
        "sector",
        "industry",
        "exchange",
    ]]


# ============================================================
# 병합 / 정제
# ============================================================
def merge_sources(sec_df: pd.DataFrame, nasdaq_df: pd.DataFrame) -> pd.DataFrame:
    merged = sec_df.merge(nasdaq_df, on="ticker", how="left", suffixes=("_sec", "_nasdaq"))

    merged["name"] = merged["name"].fillna("").replace("", pd.NA)
    merged["title"] = merged["title"].fillna("").replace("", pd.NA)
    merged["final_name"] = merged["name"].fillna(merged["title"]).fillna(merged["ticker"])

    merged["final_exchange"] = merged["exchange_nasdaq"].fillna("").replace("", pd.NA)
    merged["final_exchange"] = merged["final_exchange"].fillna(merged["exchange_sec"]).fillna("")

    return merged


def exclude_foreign_and_special(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 외국주/예탁증서 제거
    foreign_country_mask = (~out["country"].isin(["", "United States"])) & out["country"].notna()
    adr_name_mask = out["final_name"].astype(str).str.lower().str.contains(
        "adr|ads|depositary", regex=True, na=False
    )
    out = out[~foreign_country_mask].copy()
    out = out[~adr_name_mask].copy()

    # 이름 기반 특수 증권 제거
    out = out[~out["final_name"].apply(lambda x: contains_any_keyword(x, EXCLUDE_NAME_KEYWORDS))].copy()

    # 섹터/산업 기반 바이오 제거
    sector_mask = out["sector"].apply(lambda x: contains_any_keyword(x, EXCLUDE_SECTOR_KEYWORDS))
    industry_mask = out["industry"].apply(lambda x: contains_any_keyword(x, EXCLUDE_SECTOR_KEYWORDS))
    out = out[~sector_mask].copy()
    out = out[~industry_mask].copy()

    # ticker 패턴 기반 제거
    out = out[~out["ticker"].apply(symbol_has_excluded_pattern)].copy()

    return out


def apply_minimum_execution_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out[out["price"].notna()].copy()
    out = out[out["market_cap"].notna()].copy()
    out = out[out["dollar_volume"].notna()].copy()

    out = out[out["price"] >= MIN_PRICE].copy()
    out = out[out["market_cap"] >= MIN_MARKET_CAP].copy()
    out = out[out["dollar_volume"] >= MIN_DOLLAR_VOLUME].copy()

    return out


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["name"] = out["final_name"].astype(str).str.strip()

    keep_cols = [
        "ticker",
        "name",
        "price",
        "market_cap",
        "volume",
        "dollar_volume",
        "sector",
        "industry",
        "country",
        "final_exchange",
    ]
    out = out[keep_cols].copy()

    out = out.rename(columns={"final_exchange": "exchange"})

    out = out.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    return out


# ============================================================
# 메인
# ============================================================
def main() -> None:
    ensure_data_dir()

    print("1) SEC 기본 유니버스 다운로드 중...")
    sec_df = fetch_sec_universe()
    print(f"   SEC 기본 유니버스: {len(sec_df):,}")

    print("2) Nasdaq screener 숫자 데이터 다운로드 중...")
    nasdaq_df = fetch_nasdaq_screener()
    print(f"   Nasdaq screener rows: {len(nasdaq_df):,}")

    print("3) 소스 병합 중...")
    merged = merge_sources(sec_df, nasdaq_df)
    print(f"   병합 후: {len(merged):,}")

    print("4) 외국주 / 특수증권 / 바이오 제거 중...")
    filtered = exclude_foreign_and_special(merged)
    print(f"   제거 후: {len(filtered):,}")

    print("5) 최소 실행 필터 적용 중...")
    filtered = apply_minimum_execution_filters(filtered)
    print(f"   실행 필터 후: {len(filtered):,}")

    print("6) 최종 정리 중...")
    final_df = finalize(filtered)

    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n완료")
    print(f"최종 유니버스 수: {len(final_df):,}")
    print(f"저장 위치: {OUTPUT_FILE}")

    print("\n샘플:")
    print(final_df.head(10).to_string(index=False))

    # 간단 통계
    print("\n요약 통계:")
    print(f"- 가격 최소: ${final_df['price'].min():.2f}")
    print(f"- 가격 중앙값: ${final_df['price'].median():.2f}")
    print(f"- 거래대금 중앙값: ${final_df['dollar_volume'].median():,.0f}")
    print(f"- 시총 중앙값: ${final_df['market_cap'].median():,.0f}")


if __name__ == "__main__":
    main()
