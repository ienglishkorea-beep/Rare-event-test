import importlib.util
import json
import os
import runpy
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


# ============================================================
# 실행 엔진
# ------------------------------------------------------------
# 목적
# - breadth_thrust_scanner.py
# - rare_event_long_box_breakout.py
# - ten_day_tight_scanner.py
# 세 파일을 순서대로 실행
#
# 원칙
# - Breadth는 참고용 상위 환경 신호
# - Box / Tight는 독립 신호
# - 실행 엔진은 "실행 순서 + 로그 + 요약"만 담당
#
# 출력
# - outputs/run_engine_summary.json
# - outputs/run_engine_summary.txt
# ============================================================


# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BREADTH_FILE = os.path.join(BASE_DIR, "breadth_thrust_scanner.py")
BOX_FILE = os.path.join(BASE_DIR, "rare_event_long_box_breakout.py")
TIGHT_FILE = os.path.join(BASE_DIR, "ten_day_tight_scanner.py")

RUN_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "run_engine_summary.json")
RUN_SUMMARY_TXT = os.path.join(OUTPUT_DIR, "run_engine_summary.txt")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------
def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        import requests

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        }
        requests.post(url, data=payload, timeout=20)
    except Exception:
        pass


def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not file_exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "-") -> str:
    if x is None:
        return default
    return str(x)


# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
def run_script(script_path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "script": os.path.basename(script_path),
        "path": script_path,
        "success": False,
        "error": None,
        "started_at": now_str(),
        "finished_at": None,
    }

    if not file_exists(script_path):
        result["error"] = "파일 없음"
        result["finished_at"] = now_str()
        return result

    try:
        runpy.run_path(script_path, run_name="__main__")
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc(limit=8)
    finally:
        result["finished_at"] = now_str()

    return result


# ------------------------------------------------------------
# 결과 수집
# ------------------------------------------------------------
def collect_breadth_summary() -> Dict[str, Any]:
    path_json = os.path.join(OUTPUT_DIR, "breadth_thrust_result.json")
    data = load_json(path_json) or {}

    return {
        "신호발생": bool(data.get("신호발생", False)),
        "등급": safe_str(data.get("등급"), "-"),
        "등급_설명": safe_str(data.get("등급_설명"), "-"),
        "설명": safe_str(data.get("설명"), "-"),
        "Breadth50MA": data.get("Breadth50MA"),
        "섹터상승개수": safe_int(data.get("섹터상승개수"), 0),
        "유니버스유효종목수": safe_int(data.get("유니버스유효종목수"), 0),
    }


def collect_box_summary() -> Dict[str, Any]:
    path_json = os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json")
    data = load_json(path_json) or {}

    return {
        "total": safe_int(data.get("total"), 0),
        "breakout": safe_int(data.get("돌파") or data.get("breakout"), 0),
        "near_breakout": safe_int(data.get("돌파임박") or data.get("near_breakout"), 0),
        "A": safe_int(data.get("A"), 0),
        "B": safe_int(data.get("B"), 0),
        "WATCH": safe_int(data.get("WATCH"), 0),
        "as_of_date": safe_str(data.get("as_of_date"), "-"),
    }


def collect_tight_summary() -> Dict[str, Any]:
    path_json = os.path.join(OUTPUT_DIR, "ten_day_tight_summary.json")
    data = load_json(path_json) or {}

    return {
        "total": safe_int(data.get("total"), 0),
        "breakout": safe_int(data.get("돌파") or data.get("breakout"), 0),
        "near_breakout": safe_int(data.get("돌파임박") or data.get("near_breakout"), 0),
        "A": safe_int(data.get("A"), 0),
        "B": safe_int(data.get("B"), 0),
        "WATCH": safe_int(data.get("WATCH"), 0),
        "as_of_date": safe_str(data.get("as_of_date"), "-"),
    }


# ------------------------------------------------------------
# 요약 생성
# ------------------------------------------------------------
def build_summary_text(
    breadth_exec: Dict[str, Any],
    box_exec: Dict[str, Any],
    tight_exec: Dict[str, Any],
    breadth_summary: Dict[str, Any],
    box_summary: Dict[str, Any],
    tight_summary: Dict[str, Any],
) -> str:
    lines = [
        "[실행 엔진 요약]",
        f"실행 시각: {now_str()}",
        "",
        "1) 실행 상태",
        f"- Breadth: {'성공' if breadth_exec['success'] else '실패'}",
        f"- Long Box: {'성공' if box_exec['success'] else '실패'}",
        f"- 10-Day Tight: {'성공' if tight_exec['success'] else '실패'}",
        "",
        "2) 시장 환경 (참고용)",
        f"- Breadth 신호 발생: {'예' if breadth_summary['신호발생'] else '아니오'}",
        f"- 등급: {breadth_summary['등급']} ({breadth_summary['등급_설명']})",
        f"- Breadth50MA: {breadth_summary['Breadth50MA'] if breadth_summary['Breadth50MA'] is not None else '-'}",
        f"- 상승 섹터 수: {breadth_summary['섹터상승개수']}",
        f"- 유효 유니버스 수: {breadth_summary['유니버스유효종목수']}",
        f"- 해석: {breadth_summary['설명']}",
        "",
        "3) 장기 박스 돌파",
        f"- 총 후보: {box_summary['total']}",
        f"- 돌파: {box_summary['breakout']}",
        f"- 돌파 임박: {box_summary['near_breakout']}",
        f"- A: {box_summary['A']}",
        f"- B: {box_summary['B']}",
        f"- WATCH: {box_summary['WATCH']}",
        "",
        "4) 10일 타이트",
        f"- 총 후보: {tight_summary['total']}",
        f"- 돌파: {tight_summary['breakout']}",
        f"- 돌파 임박: {tight_summary['near_breakout']}",
        f"- A: {tight_summary['A']}",
        f"- B: {tight_summary['B']}",
        f"- WATCH: {tight_summary['WATCH']}",
    ]

    if not breadth_exec["success"]:
        lines += ["", "[Breadth 오류]", safe_str(breadth_exec["error"], "-")]
    if not box_exec["success"]:
        lines += ["", "[Long Box 오류]", safe_str(box_exec["error"], "-")]
    if not tight_exec["success"]:
        lines += ["", "[10-Day Tight 오류]", safe_str(tight_exec["error"], "-")]

    return "\n".join(lines)


# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
def save_outputs(summary: Dict[str, Any], summary_text: str) -> None:
    with open(RUN_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(RUN_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    ensure_dirs()

    breadth_exec = run_script(BREADTH_FILE)
    box_exec = run_script(BOX_FILE)
    tight_exec = run_script(TIGHT_FILE)

    breadth_summary = collect_breadth_summary()
    box_summary = collect_box_summary()
    tight_summary = collect_tight_summary()

    summary: Dict[str, Any] = {
        "run_at": now_str(),
        "executions": {
            "breadth": breadth_exec,
            "long_box": box_exec,
            "ten_day_tight": tight_exec,
        },
        "market_environment": breadth_summary,
        "long_box_summary": box_summary,
        "ten_day_tight_summary": tight_summary,
    }

    summary_text = build_summary_text(
        breadth_exec=breadth_exec,
        box_exec=box_exec,
        tight_exec=tight_exec,
        breadth_summary=breadth_summary,
        box_summary=box_summary,
        tight_summary=tight_summary,
    )

    save_outputs(summary, summary_text)

    send_telegram_message(summary_text)


if __name__ == "__main__":
    main()
