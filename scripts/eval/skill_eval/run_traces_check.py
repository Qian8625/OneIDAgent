from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXAMPLE_BBOX_PATTERNS = (
    "(10, 20, 50, 80), ship, score 0.73",
    "(12, 22, 52, 82), ship, score 0.86",
)


def load_traces(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _remember(examples: Dict[str, List[Dict[str, Any]]], key: str, trace: Dict[str, Any], detail: Any) -> None:
    if len(examples[key]) < 10:
        examples[key].append({"sample_id": trace.get("sample_id"), "mode": trace.get("mode"), "detail": detail})


def analyze(path: Path) -> Dict[str, Any]:
    counters = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    failed_signatures = Counter()

    for trace in load_traces(path):
        selected = [item.get("skill_name") for item in trace.get("selected_skills", []) if isinstance(item, dict)]
        gold_skill = (trace.get("eval_meta") or {}).get("gold_skill") if isinstance(trace.get("eval_meta"), dict) else None
        called = [item.get("skill_name") for item in trace.get("skill_calls", []) if isinstance(item, dict)]
        if trace.get("mode") == "auto_skill" and gold_skill and (not selected or selected[0] != gold_skill):
            counters["router_miss_count"] += 1
            _remember(examples, "router_miss", trace, {"gold_skill": gold_skill, "selected": selected})
        if trace.get("mode") in {"oracle_skill", "oracle_forced_skill"} and gold_skill and (not called or called[0] != gold_skill):
            counters["oracle_not_called_count"] += 1
            _remember(examples, "oracle_not_called", trace, {"gold_skill": gold_skill, "called": called})

        for image in trace.get("images", []) or []:
            if isinstance(image, str) and image and not os.path.isabs(image):
                counters["relative_path_count"] += 1
                _remember(examples, "relative_path", trace, image)
            if isinstance(image, str) and os.path.isabs(image) and not os.path.exists(image):
                counters["nonexistent_absolute_path_count"] += 1
                _remember(examples, "nonexistent_absolute_path", trace, image)

        if "invalid_dependency" in trace.get("error_tags", []):
            counters["invalid_dependency_count"] += 1
            _remember(examples, "invalid_dependency", trace, trace.get("error_tags"))
        if "ungrounded_final_answer" in trace.get("error_tags", []):
            counters["ungrounded_final_answer_count"] += 1
            _remember(examples, "ungrounded_final_answer", trace, trace.get("final_answer"))
        if trace.get("duplicate_call_blocked") or "duplicate_call_blocked" in trace.get("error_tags", []):
            counters["duplicate_call_blocked_count"] += 1
            _remember(examples, "duplicate_call_blocked", trace, trace.get("duplicate_call_blocked"))

        text_blob = json.dumps(trace.get("actions", []), ensure_ascii=False)
        if any(pattern in text_blob for pattern in EXAMPLE_BBOX_PATTERNS):
            counters["example_bbox_leakage_count"] += 1
            _remember(examples, "example_bbox_leakage", trace, "prompt-example bbox found in actions")
        if "img_1" in text_blob or "img_2" in text_blob:
            counters["img_placeholder_args_count"] += 1
            _remember(examples, "img_placeholder_args", trace, text_blob[:500])

        for response_item in trace.get("tool_responses", []) or []:
            response = response_item.get("response") if isinstance(response_item, dict) else {}
            api_name = response_item.get("api_name")
            if not isinstance(response, dict):
                continue
            text = str(response.get("text", ""))
            if "image not found" in text.lower() or "not found:" in text.lower():
                counters["image_not_found_count"] += 1
                _remember(examples, "image_not_found", trace, {"api_name": api_name, "text": text[:300]})
            if "unsupported task_type" in text.lower():
                counters["unsupported_task_type_count"] += 1
                _remember(examples, "unsupported_task_type", trace, {"api_name": api_name, "text": text[:300]})
            if "chat() timed out" in text.lower():
                counters["tool_timeout_count"] += 1
                _remember(examples, "tool_timeout", trace, {"api_name": api_name, "text": text[:300]})
            if "proj.db" in text.lower() or "no crs" in text.lower() or "transformer from crs" in text.lower():
                counters["proj_crs_error_count"] += 1
                _remember(examples, "proj_crs_error", trace, {"api_name": api_name, "text": text[:300]})
            if response.get("error_code") not in (None, 0):
                if api_name == "SkillExecutor":
                    counters["skill_failure_count"] += 1
                else:
                    counters["raw_tool_failure_count"] += 1
                signature = json.dumps({"api_name": api_name, "arguments": response_item.get("arguments")}, sort_keys=True, default=str)
                failed_signatures[(trace.get("sample_id"), signature)] += 1
            else:
                if api_name == "SkillExecutor":
                    counters["successful_skill_call_count"] += 1
                internal = response.get("internal_tool_calls") or []
                counters["successful_internal_tool_call_count"] += sum(1 for item in internal if isinstance(item, dict) and item.get("success") is True)

    counters["duplicate_call_count"] = sum(1 for count in failed_signatures.values() if count > 1)
    return {"counters": dict(counters), "examples": dict(examples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check skill-eval run traces for grounding/runtime issues.")
    parser.add_argument("trace_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
