import json
import os
import runpy
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BREADTH_FILE = os.path.join(BASE_DIR, "breadth_thrust_scanner.py")
BOX_FILE = os.path.join(BASE_DIR, "rare_event_long_box_breakout.py")
TIGHT_FILE = os.path.join(BASE_DIR, "ten_day_tight_scanner.py")

RUN_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "run_engine_summary.json")
RUN_SUMMARY_TXT = os.path.join(OUTPUT_DIR, "run_engine_summary.txt")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


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
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=20)
    except Exception:
        pass


def short_error(tb: str) -> str:
    lines = [x for x in tb.strip().splitlines() if x.strip()]
    if not lines:
        return "알 수 없는 오류"
    # 마지막 2줄만
    tail = lines[-2:] if len(lines) >= 2 else lines
    return " | ".join(tail)


def run_script(script_path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "script": os.path.basename(script_path),
        "success": False,
        "error": None,
        "started_at": now_str(),
        "finished_at": None,
    }
    try:
        runpy.run_path(script_path, run_name="__main__")
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["finished_at"] = now_str()
    return result


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "-") -> str:
    return default if x is None else str(x)


def collect_breadth_summary() -> Dict[str, Any]:
    data = load_json(os.path.join(OUTPUT_DIR, "breadth_thrust_result.json")) or {}
    return {
        "signal": bool(data.get("signal", False)),
        "grade": safe_str(data.get("grade"), "-"),
        "grade_label": safe_str(data.get("grade_label"), "-"),
        "comment": safe_str(data.get("comment"), "-"),
        "breadth_50ma": data.get("breadth_50ma"),
        "sector_up_count": safe_int(data.get("sector_up_count"), 0),
        "valid_universe_count": safe_int(data.get("valid_universe_count"), 0),
    }


def collect_box_summary() -> Dict[str, Any]:
    data = load_json(os.path.join(OUTPUT_DIR, "rare_event_long_box_breakout_summary.json")) or {}
    return {
        "total": safe_int(data.get("total"), 0),
        "breakout": safe_int(data.get("breakout"), 0),
        "near_breakout": safe_int(data.get("near_breakout"), 0),
        "A": safe_int(data.get("A"), 0),
        "B": safe_int(data.get("B"), 0),
    }


def collect_tight_summary() -> Dict[str, Any]:
    data = load_json(os.path.join(OUTPUT_DIR, "ten_day_tight_summary.json")) or {}
    return {
        "total": safe_int(data.get("total"), 0),
        "breakout": safe_int(data.get("breakout"), 0),
        "near_breakout": safe_int(data.get("near_breakout"), 0),
        "A": safe_int(data.get("A"), 0),
        "B": safe_int(data.get("B"), 0),
    }


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
        f"- Breadth 신호 발생: {'예' if breadth_summary['signal'] else '아니오'}",
        f"- 등급: {breadth_summary['grade']} ({breadth_summary['grade_label']})",
        f"- Breadth50MA: {breadth_summary['breadth_50ma'] if breadth_summary['breadth_50ma'] is not None else '-'}",
        f"- 상승 섹터 수: {breadth_summary['sector_up_count']}",
        f"- 유효 유니버스 수: {breadth_summary['valid_universe_count']}",
        f"- 해석: {breadth_summary['comment']}",
        "",
        "3) 박스 돌파",
        f"- 총 후보: {box_summary['total']}",
        f"- 돌파: {box_summary['breakout']}",
        f"- 돌파 임박: {box_summary['near_breakout']}",
        f"- A: {box_summary['A']}",
        f"- B: {box_summary['B']}",
        "",
        "4) 10일 타이트",
        f"- 총 후보: {tight_summary['total']}",
        f"- 돌파: {tight_summary['breakout']}",
        f"- 돌파 임박: {tight_summary['near_breakout']}",
        f"- A: {tight_summary['A']}",
        f"- B: {tight_summary['B']}",
    ]

    if not breadth_exec["success"]:
        lines += ["", "[Breadth 오류]", short_error(breadth_exec["error"] or "")]
    if not box_exec["success"]:
        lines += ["", "[Long Box 오류]", short_error(box_exec["error"] or "")]
    if not tight_exec["success"]:
        lines += ["", "[10-Day Tight 오류]", short_error(tight_exec["error"] or "")]

    return "\n".join(lines)


def save_outputs(summary: Dict[str, Any], summary_text: str) -> None:
    with open(RUN_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(RUN_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)


def main() -> None:
    ensure_dirs()

    breadth_exec = run_script(BREADTH_FILE)
    box_exec = run_script(BOX_FILE)
    tight_exec = run_script(TIGHT_FILE)

    breadth_summary = collect_breadth_summary()
    box_summary = collect_box_summary()
    tight_summary = collect_tight_summary()

    summary = {
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
