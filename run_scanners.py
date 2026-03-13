import json
import os
import runpy
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BREADTH_SCRIPT = os.path.join(BASE_DIR, "breadth_thrust_scanner.py")
BOX_SCRIPT = os.path.join(BASE_DIR, "rare_event_long_box_breakout.py")
TIGHT_SCRIPT = os.path.join(BASE_DIR, "ten_day_tight_scanner.py")
VCP_SCRIPT = os.path.join(BASE_DIR, "vcp_scanner.py")

BREADTH_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "breadth_summary.json")
BOX_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json")
TIGHT_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_summary.json")
VCP_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "vcp_summary.json")

BOX_RESULTS_FILE = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout.csv")
TIGHT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "ten_day_tight_results.csv")
VCP_RESULTS_FILE = os.path.join(OUTPUT_DIR, "vcp_results.csv")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_MESSAGE_LEN = 3500


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(text: str) -> None:
    if not telegram_enabled():
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )


def send_telegram_chunked(lines: List[str]) -> None:
    if not telegram_enabled() or not lines:
        return

    chunks: List[str] = []
    current = ""

    for line in lines:
        candidate = f"{current}\n{line}".strip() if current else line
        if len(candidate) > MAX_TELEGRAM_MESSAGE_LEN:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    for chunk in chunks:
        send_telegram(chunk)


def read_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_str(x: Any) -> str:
    if x is None:
        return "-"
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        pass
    return str(x)


def safe_float(x: Any) -> float:
    try:
        if x is None or pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def fmt_price(x: Any) -> str:
    v = safe_float(x)
    if pd.isna(v):
        return "-"
    return f"{v:,.2f}"


def fmt_pct_from_ratio(x: Any) -> str:
    v = safe_float(x)
    if pd.isna(v):
        return "-"
    return f"{v * 100:.1f}%"


def run_script(script_path: str) -> str:
    try:
        runpy.run_path(script_path, run_name="__main__")
        return "성공"
    except Exception as e:
        return f"실패: {type(e).__name__}: {e}"


def load_box_candidates() -> Dict[str, Dict[str, Any]]:
    df = read_csv_file(BOX_RESULTS_FILE)
    if df.empty:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        ticker = safe_str(row.get("ticker")).upper()
        if not ticker or ticker == "-":
            continue
        out[ticker] = {
            "pattern": "BOX",
            "ticker": ticker,
            "name": safe_str(row.get("name")),
            "rs_grade": safe_str(row.get("rs_grade")),
            "grade": safe_str(row.get("grade")),
            "state": safe_str(row.get("state")),
            "score": safe_float(row.get("total_score")),
            "close": safe_float(row.get("close")),
            "pivot": safe_float(row.get("pivot_price")),
            "entry": safe_float(row.get("entry_price")),
            "stop": safe_float(row.get("stop_price")),
            "box_length": safe_str(row.get("box_length")),
            "box_width_pct": safe_float(row.get("box_width_pct")),
            "pivot_distance_pct": safe_float(row.get("pivot_distance_pct")),
        }
    return out


def load_tight_candidates() -> Dict[str, Dict[str, Any]]:
    df = read_csv_file(TIGHT_RESULTS_FILE)
    if df.empty:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        ticker = safe_str(row.get("ticker")).upper()
        if not ticker or ticker == "-":
            continue
        out[ticker] = {
            "pattern": "TIGHT",
            "ticker": ticker,
            "name": safe_str(row.get("name")),
            "grade": safe_str(row.get("grade")),
            "state": safe_str(row.get("state")),
            "score": safe_float(row.get("score_total")),
            "close": safe_float(row.get("close")),
            "entry": safe_float(row.get("entry")),
            "stop": safe_float(row.get("stop")),
            "close_tight": safe_float(row.get("close_tight")),
            "range_tight": safe_float(row.get("range_tight")),
            "rs_current_vs_high": safe_float(row.get("rs_current_vs_high")),
        }
    return out


def load_vcp_candidates() -> Dict[str, Dict[str, Any]]:
    df = read_csv_file(VCP_RESULTS_FILE)
    if df.empty:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        ticker = safe_str(row.get("ticker")).upper()
        if not ticker or ticker == "-":
            continue
        out[ticker] = {
            "pattern": "VCP",
            "ticker": ticker,
            "name": safe_str(row.get("name")),
            "grade": safe_str(row.get("grade")),
            "state": safe_str(row.get("state")),
            "score": safe_float(row.get("score_total")),
            "close": safe_float(row.get("close")),
            "pivot": safe_float(row.get("pivot_price")),
            "entry": safe_float(row.get("entry_price")),
            "stop": safe_float(row.get("stop_price")),
            "stages": safe_str(row.get("stages")),
            "ranges_text": safe_str(row.get("ranges_text")),
            "rs_grade": safe_str(row.get("rs_grade")),
        }
    return out


def build_confluence_rows(
    box_map: Dict[str, Dict[str, Any]],
    tight_map: Dict[str, Dict[str, Any]],
    vcp_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for source_name, source_map in [
        ("BOX", box_map),
        ("TIGHT", tight_map),
        ("VCP", vcp_map),
    ]:
        for ticker, item in source_map.items():
            if "ticker" not in merged[ticker]:
                merged[ticker]["ticker"] = ticker
                merged[ticker]["name"] = item.get("name", "-")
                merged[ticker]["patterns"] = []
                merged[ticker]["scores"] = []
                merged[ticker]["entry_candidates"] = []
                merged[ticker]["stop_candidates"] = []

            merged[ticker]["patterns"].append(source_name)

            score = safe_float(item.get("score"))
            if not pd.isna(score):
                merged[ticker]["scores"].append(score)

            entry = safe_float(item.get("entry"))
            if not pd.isna(entry):
                merged[ticker]["entry_candidates"].append(entry)

            stop = safe_float(item.get("stop"))
            if not pd.isna(stop):
                merged[ticker]["stop_candidates"].append(stop)

            merged[ticker][source_name] = item

    rows: List[Dict[str, Any]] = []
    for ticker, item in merged.items():
        patterns = item["patterns"]
        if len(patterns) < 2:
            continue

        avg_score = float(np.mean(item["scores"])) if item["scores"] else float("nan")
        entry = min(item["entry_candidates"]) if item["entry_candidates"] else float("nan")
        stop = max(item["stop_candidates"]) if item["stop_candidates"] else float("nan")

        rows.append(
            {
                "ticker": ticker,
                "name": item.get("name", "-"),
                "patterns": sorted(patterns),
                "pattern_count": len(patterns),
                "avg_score": avg_score,
                "entry": entry,
                "stop": stop,
                "BOX": item.get("BOX"),
                "TIGHT": item.get("TIGHT"),
                "VCP": item.get("VCP"),
            }
        )

    rows = sorted(
        rows,
        key=lambda x: (
            -x["pattern_count"],
            -(x["avg_score"] if not pd.isna(x["avg_score"]) else -999),
            x["ticker"],
        ),
    )
    return rows


def build_confluence_message(rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.append("[합류 신호] CONFLUENCE")
    lines.append(f"전체 후보: {len(rows)}")

    for row in rows:
        lines.append(f"- {row['ticker']} {row['name']}")
        lines.append(f"  패턴 수: {row['pattern_count']} | 패턴: {', '.join(row['patterns'])}")
        lines.append(f"  평균 점수: {row['avg_score']:.1f}/100" if not pd.isna(row["avg_score"]) else "  평균 점수: -")
        lines.append(f"  보수 진입가: {fmt_price(row['entry'])}")
        lines.append(f"  보수 손절가: {fmt_price(row['stop'])}")

        if row.get("BOX"):
            b = row["BOX"]
            lines.append(
                f"  BOX | 상태: {safe_str(b.get('state'))} | RS: {safe_str(b.get('rs_grade'))} | "
                f"박스길이: {safe_str(b.get('box_length'))}일 | 박스폭: {fmt_pct_from_ratio(b.get('box_width_pct'))}"
            )

        if row.get("TIGHT"):
            t = row["TIGHT"]
            lines.append(
                f"  TIGHT | 상태: {safe_str(t.get('state'))} | "
                f"종가압축: {fmt_pct_from_ratio(t.get('close_tight'))} | "
                f"전체변동폭: {fmt_pct_from_ratio(t.get('range_tight'))}"
            )

        if row.get("VCP"):
            v = row["VCP"]
            lines.append(
                f"  VCP | 상태: {safe_str(v.get('state'))} | 단계: {safe_str(v.get('stages'))} | "
                f"범위축소: {safe_str(v.get('ranges_text'))}"
            )

    return lines


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_status: Dict[str, str] = {}

    if os.path.exists(BREADTH_SCRIPT):
        run_status["Breadth"] = run_script(BREADTH_SCRIPT)
    else:
        run_status["Breadth"] = "스킵(파일 없음)"

    if os.path.exists(BOX_SCRIPT):
        run_status["Long Box"] = run_script(BOX_SCRIPT)
    else:
        run_status["Long Box"] = "스킵(파일 없음)"

    if os.path.exists(TIGHT_SCRIPT):
        run_status["10-Day Tight"] = run_script(TIGHT_SCRIPT)
    else:
        run_status["10-Day Tight"] = "스킵(파일 없음)"

    if os.path.exists(VCP_SCRIPT):
        run_status["VCP"] = run_script(VCP_SCRIPT)
    else:
        run_status["VCP"] = "스킵(파일 없음)"

    box_map = load_box_candidates()
    tight_map = load_tight_candidates()
    vcp_map = load_vcp_candidates()

    confluence_rows = build_confluence_rows(box_map, tight_map, vcp_map)

    if confluence_rows:
        send_telegram_chunked(build_confluence_message(confluence_rows))
    else:
        send_telegram("[합류 신호] CONFLUENCE\n전체 후보: 0")

    breadth_summary = read_json_file(BREADTH_SUMMARY_FILE)
    box_summary = read_json_file(BOX_SUMMARY_FILE)
    tight_summary = read_json_file(TIGHT_SUMMARY_FILE)
    vcp_summary = read_json_file(VCP_SUMMARY_FILE)

    lines = []
    lines.append("[실행 엔진 요약]")
    lines.append(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("1) 실행 상태")
    lines.append(f"- Breadth: {run_status.get('Breadth', '-')}")
    lines.append(f"- Long Box: {run_status.get('Long Box', '-')}")
    lines.append(f"- 10-Day Tight: {run_status.get('10-Day Tight', '-')}")
    lines.append(f"- VCP: {run_status.get('VCP', '-')}")
    lines.append("")
    lines.append("2) 시장 환경 (참고용)")
    if breadth_summary:
        lines.append(f"- Breadth 신호 발생: {safe_str(breadth_summary.get('zweig_signal', '-'))}")
        lines.append(f"- 등급: {safe_str(breadth_summary.get('grade', '-'))}")
        lines.append(f"- Breadth50MA: {safe_str(breadth_summary.get('breadth_50ma', '-'))}")
        lines.append(f"- 상승 섹터 수: {safe_str(breadth_summary.get('positive_sector_count', '-'))}")
        lines.append(f"- 유효 유니버스 수: {safe_str(breadth_summary.get('effective_universe', '-'))}")
        lines.append(f"- 해석: {safe_str(breadth_summary.get('commentary', '-'))}")
    else:
        lines.append("- 데이터 없음")
    lines.append("")
    lines.append("3) 장기 박스 돌파")
    if box_summary:
        lines.append(f"- 총 후보: {safe_str(box_summary.get('total', 0))}")
        lines.append(f"- 돌파 임박: {safe_str(box_summary.get('near_breakout', 0))}")
        lines.append(f"- 1차 돌파: {safe_str(box_summary.get('first_breakout', 0))}")
        lines.append(f"- 후행 가능: {safe_str(box_summary.get('late_breakout', 0))}")
        lines.append(f"- RS S: {safe_str(box_summary.get('rs_s', 0))}")
        lines.append(f"- RS A: {safe_str(box_summary.get('rs_a', 0))}")
        lines.append(f"- RS B: {safe_str(box_summary.get('rs_b', 0))}")
    else:
        lines.append("- 데이터 없음")
    lines.append("")
    lines.append("4) 10일 타이트")
    if tight_summary:
        lines.append(f"- 총 후보: {safe_str(tight_summary.get('total', 0))}")
        lines.append(f"- 돌파: {safe_str(tight_summary.get('breakout', 0))}")
        lines.append(f"- 돌파 임박: {safe_str(tight_summary.get('near_breakout', 0))}")
        lines.append(f"- A: {safe_str(tight_summary.get('A', 0))}")
        lines.append(f"- B: {safe_str(tight_summary.get('B', 0))}")
    else:
        lines.append("- 데이터 없음")
    lines.append("")
    lines.append("5) VCP")
    if vcp_summary:
        lines.append(f"- 총 후보: {safe_str(vcp_summary.get('total', 0))}")
        lines.append(f"- 돌파 임박: {safe_str(vcp_summary.get('near_breakout', 0))}")
        lines.append(f"- 1차 돌파: {safe_str(vcp_summary.get('first_breakout', 0))}")
        lines.append(f"- 후행 가능: {safe_str(vcp_summary.get('late_breakout', 0))}")
        lines.append(f"- RS S: {safe_str(vcp_summary.get('rs_s', 0))}")
        lines.append(f"- RS A: {safe_str(vcp_summary.get('rs_a', 0))}")
        lines.append(f"- RS B: {safe_str(vcp_summary.get('rs_b', 0))}")
    else:
        lines.append("- 데이터 없음")
    lines.append("")
    lines.append("6) 합류 신호")
    lines.append(f"- 2개 이상 패턴 중복 종목 수: {len(confluence_rows)}")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
