import os
import re
from typing import Any, Optional, Set, List

import pandas as pd
import requests


"""
실행용 유니버스 빌더 (미국 전체 근사 유니버스 안정판)

목적
- GitHub Actions에서 안정적으로 실행
- breadth / long box / 10-day tight / market regime 공통 사용
- Nasdaq API 응답 구조가 바뀌어도 최대한 견고하게 처리
- 거래소(exchange) 파싱 실패 문제 해결
- 최종적으로 미국 상장 보통주 근사 유니버스 생성

출력
- data/universe.csv

철학
1) 원천은 Nasdaq Screener API 하나로 단순화
2) ETF / ETN / SPAC / ADR / 바이오 / 특수증권 제거
3) REIT 제거
4) 최소 실행 가능성 필터 적용
5) exchange 컬럼을 최대한 복구해서 NASDAQ / NYSE / AMEX 분포 확인 가능하게 함
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
# 소스
# ============================================================
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nasdaq.com/",
    "Origin": "https://www.nasdaq.com",
}


# ============================================================
# 제외 키워드
# ============================================================
EXCLUDE_NAME_KEYWORDS: Set[str] = {
    "etf",
    "etn",
    "fund",
    "spdr",
    "ishares",
    "vanguard",
    "invesco",
    "closed-end",
    "income fund",
    "yield shares",
    "acquisition",
    "blank check",
    "spac",
    "warrant",
    "rights",
    "right",
    "unit",
    "depositary",
    "adr",
    "ads",
    "preferred",
    "preference",
    "royalty trust",
    "note",
    "bond",
    "reit",
    "realty",
    "properties",
}

EXCLUDE_SECTOR_KEYWORDS: Set[str] = {
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
    r"\.",   # BRK.B 같은 특수형 제거
    r"\^",   # 지수/특수 심볼 제거
    r"/",    # 특수 ticker 제거
]


# ============================================================
# 유틸
# ============================================================
def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def normalize_text(x: Any) -> str:
    return str(x or "").strip()


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


def contains_any_keyword(text: str, keywords: Set[str]) -> bool:
    low = normalize_text(text).lower()
    return any(k in low for k in keywords)


def symbol_has_excluded_pattern(symbol: str) -> bool:
    s = normalize_text(symbol).upper()
    for pattern in EXCLUDE_SYMBOL_PATTERNS:
        if re.search(pattern, s):
            return True
    return False


def first_existing_value(row: pd.Series, keys: List[str], default: Any = "") -> Any:
    for key in keys:
        if key in row.index:
            value = row[key]
            if value is not None and not pd.isna(value) and str(value).strip() != "":
                return value
    return default


def standardize_exchange_name(x: Any) -> str:
    s = normalize_text(x).upper()

    if s in {"NASDAQ", "NASDAQGS", "NASDAQGM", "NASDAQCM", "Q", "NMS"}:
        return "NASDAQ"

    if s in {"NYSE", "N", "NYQ"}:
        return "NYSE"

    if s in {"AMEX", "NYSEAMERICAN", "NYSE AMERICAN", "A", "ASE"}:
        return "AMEX"

    if s in {"BATS", "CBOE", "CBOEBZX", "BZX"}:
        return "BATS"

    if s in {"OTC", "OTCQX", "OTCQB", "PINK", "PNK"}:
        return "OTC"

    if s == "":
        return ""

    return s


# ============================================================
# 다운로드
# ============================================================
def fetch_nasdaq_screener() -> pd.DataFrame:
    r = requests.get(NASDAQ_SCREENER_URL, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = data.get("data", {}).get("rows", [])
    if not rows:
        raise RuntimeError("Nasdaq screener rows 비어 있음")

    df = pd.DataFrame(rows)

    # 응답 구조가 바뀌어도 최대한 견고하게 필드 복구
    if "ticker" not in df.columns:
        df["ticker"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["symbol", "ticker"],
                "",
            ),
            axis=1,
        )

    if "name" not in df.columns:
        df["name"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["name", "securityName"],
                "",
            ),
            axis=1,
        )

    if "price" not in df.columns:
        df["price"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["lastsale", "lastSalePrice", "price", "last"],
                None,
            ),
            axis=1,
        )

    if "market_cap" not in df.columns:
        df["market_cap"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["marketCap", "marketcap", "market_cap"],
                None,
            ),
            axis=1,
        )

    if "volume" not in df.columns:
        df["volume"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["volume", "shareVolume"],
                None,
            ),
            axis=1,
        )

    if "country" not in df.columns:
        df["country"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["country"],
                "",
            ),
            axis=1,
        )

    if "sector" not in df.columns:
        df["sector"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["sector"],
                "",
            ),
            axis=1,
        )

    if "industry" not in df.columns:
        df["industry"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["industry"],
                "",
            ),
            axis=1,
        )

    # 핵심 수정: exchange 필드 복구
    if "exchange" not in df.columns:
        df["exchange"] = df.apply(
            lambda row: first_existing_value(
                row,
                [
                    "exchange",
                    "exchangeShortName",
                    "exchangeName",
                    "market",
                    "marketName",
                ],
                "",
            ),
            axis=1,
        )

    if "ipo_year" not in df.columns:
        df["ipo_year"] = df.apply(
            lambda row: first_existing_value(
                row,
                ["ipoyear", "ipoYear"],
                "",
            ),
            axis=1,
        )

    # 기본 문자열 정리
    for col in ["ticker", "name", "country", "sector", "industry", "exchange"]:
        df[col] = df[col].astype(str).str.strip()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # 숫자 정리
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"N/A": None, "": None, "None": None})
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # exchange 표준화
    df["exchange"] = df["exchange"].apply(standardize_exchange_name)

    df["dollar_volume"] = df["price"] * df["volume"]

    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df


# ============================================================
# 정제
# ============================================================
def filter_us_common_stocks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 미국 외 국가 제거
    out = out[out["country"].isin(["", "United States"])].copy()

    # 미국 주요 거래소만 허용
    # exchange가 비어 있어도 현재 API 특성상 일부 케이스가 있을 수 있어 일단 유지
    # 다만 OTC는 명시적 제거
    out = out[out["exchange"] != "OTC"].copy()

    # ticker 특수형 제거
    out = out[~out["ticker"].apply(symbol_has_excluded_pattern)].copy()

    # 이름 기반 특수증권 제거
    out = out[~out["name"].apply(lambda x: contains_any_keyword(x, EXCLUDE_NAME_KEYWORDS))].copy()

    # 바이오/개발형 제약 제거
    sector_mask = out["sector"].apply(lambda x: contains_any_keyword(x, EXCLUDE_SECTOR_KEYWORDS))
    industry_mask = out["industry"].apply(lambda x: contains_any_keyword(x, EXCLUDE_SECTOR_KEYWORDS))
    out = out[~sector_mask].copy()
    out = out[~industry_mask].copy()

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
        "exchange",
    ]
    out = out[keep_cols].copy()

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["name"] = out["name"].astype(str).str.strip()
    out["exchange"] = out["exchange"].astype(str).str.strip()

    out = out.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return out


# ============================================================
# 진단 출력
# ============================================================
def print_exchange_distribution(df: pd.DataFrame, title: str) -> None:
    print(f"\n[{title}] 거래소 분포")
    if df.empty:
        print("비어 있음")
        return

    counts = df["exchange"].fillna("").replace("", "UNKNOWN").value_counts(dropna=False)
    total = int(len(df))

    for exchange, count in counts.items():
        pct = (count / total) * 100 if total else 0
        print(f"- {exchange}: {count:,} ({pct:.1f}%)")


def print_exchange_marketcap_median(df: pd.DataFrame, title: str) -> None:
    print(f"\n[{title}] 거래소별 시총 중앙값")
    if df.empty:
        print("비어 있음")
        return

    temp = df.copy()
    temp["exchange"] = temp["exchange"].fillna("").replace("", "UNKNOWN")

    grouped = (
        temp.groupby("exchange", dropna=False)["market_cap"]
        .median()
        .sort_values(ascending=False)
    )

    for exchange, value in grouped.items():
        if pd.isna(value):
            print(f"- {exchange}: N/A")
        else:
            print(f"- {exchange}: ${value:,.0f}")


def print_exchange_dollar_volume_median(df: pd.DataFrame, title: str) -> None:
    print(f"\n[{title}] 거래소별 거래대금 중앙값")
    if df.empty:
        print("비어 있음")
        return

    temp = df.copy()
    temp["exchange"] = temp["exchange"].fillna("").replace("", "UNKNOWN")

    grouped = (
        temp.groupby("exchange", dropna=False)["dollar_volume"]
        .median()
        .sort_values(ascending=False)
    )

    for exchange, value in grouped.items():
        if pd.isna(value):
            print(f"- {exchange}: N/A")
        else:
            print(f"- {exchange}: ${value:,.0f}")


# ============================================================
# 메인
# ============================================================
def main() -> None:
    ensure_data_dir()

    print("1) Nasdaq Screener 다운로드 중...")
    raw_df = fetch_nasdaq_screener()
    print(f"   원시 rows: {len(raw_df):,}")
    print_exchange_distribution(raw_df, "원시 데이터")

    print("\n2) 미국 보통주 근사 유니버스 정제 중...")
    filtered = filter_us_common_stocks(raw_df)
    print(f"   특수증권 / 외국주 / 바이오 / REIT 제거 후: {len(filtered):,}")
    print_exchange_distribution(filtered, "정제 후")

    print("\n3) 최소 실행 필터 적용 중...")
    filtered = apply_minimum_execution_filters(filtered)
    print(f"   실행 필터 후: {len(filtered):,}")
    print_exchange_distribution(filtered, "실행 필터 후")

    print("\n4) 최종 정리 중...")
    final_df = finalize(filtered)

    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n완료")
    print(f"최종 유니버스 수: {len(final_df):,}")
    print(f"저장 위치: {OUTPUT_FILE}")

    print_exchange_distribution(final_df, "최종 유니버스")
    print_exchange_marketcap_median(final_df, "최종 유니버스")
    print_exchange_dollar_volume_median(final_df, "최종 유니버스")

    if len(final_df) > 0:
        print("\n샘플:")
        print(final_df.head(10).to_string(index=False))

        print("\n요약 통계:")
        print(f"- 가격 최소: ${final_df['price'].min():.2f}")
        print(f"- 가격 중앙값: ${final_df['price'].median():.2f}")
        print(f"- 거래대금 중앙값: ${final_df['dollar_volume'].median():,.0f}")
        print(f"- 시총 중앙값: ${final_df['market_cap'].median():,.0f}")


if __name__ == "__main__":
    main()
