#!/usr/bin/env python3
"""CrossModalInterpret_worker.py — OneID 在线跨模态推理工具（薄壳）。

职责边界
--------
本文件只做四件事：

1. 把仓库根目录挂进 ``sys.path``，让 ``DS_VI/`` 以包的形式被导入；
2. 参数校验与 mode 分派；
3. ``shared_identity`` 交给 :mod:`ds_vi_adapter`（唯一新增实现），
   ``legacy_hard`` / ``candidate_graph`` 原样转给上游 ``DS_VI`` 的
   ``CrossModalInterpreter``；
4. 统一出口：补齐 ``structured``、剔掉超预算的大字段、记录 ``payload_bytes``。

上游实现（``DS_VI/models/cross_modal/interpreter.py``，876 行）保持 1:1
vendor，本文件不再自带任何匹配/证据算法。

mode 语义
---------
* ``shared_identity``（在线默认）—— D-S 配对 → 共享身份证据 → cascade 装配；
* ``vi_only``              —— 跳过 D-S 交接，只用共享身份图；
* ``candidate_graph``      —— 诊断档：只看候选图与物理先验，不跑推断；
* ``legacy_hard``          —— 旧几何贪心，作降级/对照，输出与历史逐字段一致。

能力边界（必须对模型声明，不得默认宣称）
----------------------------------------
在线链路没有属性工作簿，因此共享身份只有 ``category`` 证据；``time`` 族
（static/moved）不进在线链路；配准固定 ``registration=identity``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

worker_id = str(uuid.uuid4())[:6]

# --------------------------------------------------------------------------- #
# D1：DS_VI 是仓库根下的包，worker 需要用 4 行引导 sys.path。
# 直接 `python online_workers/CrossModalInterpret_worker.py` 时 sys.path[0]
# 是 online_workers/，仓库根不在其中。
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
    from tool_server.utils.server_utils import build_logger
except Exception:  # pragma: no cover - 允许脱离服务单跑
    BaseToolWorker = object

    def build_logger(*args, **kwargs):
        class DummyLogger:
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def exception(self, *a, **k): pass
        return DummyLogger()


logger = build_logger(__file__, f"CrossModalInterpret_worker_{worker_id}.log")

# 上游模型实现（1:1 vendor）。导入失败属于环境问题，必须显式失败。
from DS_VI.models.cross_modal.interpreter import CrossModalInterpreter  # noqa: E402

from tool_server.tool_workers.online_workers import ds_vi_adapter  # noqa: E402

#: 在线默认 mode。``ONEID_DEFAULT_MODE=legacy_hard`` 可一键回退。
DEFAULT_MODE = os.environ.get("ONEID_DEFAULT_MODE", CrossModalInterpreter.MODE_SHARED_IDENTITY)

#: Ablation override.  When set, the chain behind CrossModalInterpret is forced
#: and any ``mode`` the model puts in the tool arguments is ignored, so an arm is
#: selected by environment only and the prompt / schema / checkpoint stay fixed.
#: Unset -> identical behaviour to before this patch.
FORCE_MODE = (os.environ.get("ONEID_FORCE_MODE") or "").strip() or None

#: Optional jsonl path: one record per tool call.  Used to measure how often the
#: module actually runs (exposure rate) and whether a call silently degraded.
ARM_LOG = (os.environ.get("ONEID_ARM_LOG") or "").strip() or None

ARM_LOG_FIELDS = (
    "requested_mode", "effective_mode", "degraded", "degrade_reason",
    "status", "task", "target", "count", "payload_bytes",
)


def _log_arm_call(params: Dict[str, Any], response: Dict[str, Any]) -> None:
    """Append one line per CrossModalInterpret call.  Never raises."""
    if not ARM_LOG:
        return
    try:
        import time

        _parent = os.path.dirname(ARM_LOG)
        if _parent:
            os.makedirs(_parent, exist_ok=True)
        structured = response.get("structured")
        structured = structured if isinstance(structured, dict) else {}
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "requested_mode": params.get("mode"),
            "effective_mode": response.get("mode"),
            "degraded": bool(response.get("degraded")),
            "degrade_reason": response.get("degrade_reason"),
            "status": response.get("status"),
            "task": params.get("task"),
            "target": params.get("target"),
            "count": structured.get("count"),
            "payload_bytes": structured.get("payload_bytes"),
        }
        with open(ARM_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover - logging must never break the tool
        pass

#: 单次推理预算，供适配器在阶段边界降级。
DEFAULT_BUDGET_SECONDS = float(os.environ.get("ONEID_DSVI_BUDGET_SECONDS", "120") or 120)

#: structured 里允许透传的字段（与上游同名同义，向后兼容）。
STRUCTURED_KEYS = (
    "dedup_count", "source_counts", "distance_pair", "distance_m", "preview",
    "rgb_detections", "sar_detections", "matched", "matched_pairs", "rgb_only",
    "sar_only", "dedup_detections", "count", "artifacts", "result_id",
    "scene_description", "primitive_observations", "claims", "object_hypotheses",
    "candidate_edges", "physical_prior", "evidence_summary", "inference",
    "legacy_match", "evidence_layer_error",
    "source_mode", "family", "dataset_id", "pair_id", "replay_provenance", "replay_diagnostics", "objects", "attribute_claims", "observation_context", "vi_objects",
)

#: 在线能力边界：对模型和下游都必须可见。
CAPABILITY = {
    "attributes": ["category"],
    "time_family": False,
    "registration": "identity",
    "truth_data": False,
}


def _payload_bytes(payload: Dict[str, Any]) -> int:
    body = {key: value for key, value in payload.items() if key != "structured"}
    return len(json.dumps(body, ensure_ascii=False, default=str))


class OneIDCrossModalEngine:
    """参数校验 + mode 分派 + 统一出口。"""

    def __init__(self, budget_seconds: Optional[float] = None):
        self.interpreter = CrossModalInterpreter()
        self.budget_seconds = DEFAULT_BUDGET_SECONDS if budget_seconds is None else budget_seconds

    # -- 兼容层：所有对旧 API 的适配集中在这里 --------------------------

    @property
    def FINAL_TARGETS(self):
        return self.interpreter.FINAL_TARGETS

    @property
    def TARGET_ALIASES(self):
        return self.interpreter.TARGET_ALIASES

    @property
    def TASK_ALIASES(self):
        return self.interpreter.TASK_ALIASES

    @property
    def MATCH_THRESHOLD(self):
        return self.interpreter.MATCH_THRESHOLD

    @property
    def DEFAULT_DETECTION_CONFIDENCE(self):
        return self.interpreter.DEFAULT_DETECTION_CONFIDENCE

    def canonical_mode(self, params: Dict[str, Any]) -> str:
        if FORCE_MODE:
            return self.interpreter.canonical_mode(FORCE_MODE)
        requested = params.get("mode") or params.get("reasoning_mode") or DEFAULT_MODE
        return self.interpreter.canonical_mode(requested)

    def canonical_task(self, value: Any) -> Optional[str]:
        task = self.interpreter.canonical_task(value)
        return task

    def canonical_target(self, value: Any) -> str:
        return self.interpreter.canonical_target(value)

    def normalize_label(self, value: Any) -> str:
        return self.interpreter.normalize_label(value)

    def compute_pair_score(self, rgb_det: Dict[str, Any], sar_det: Dict[str, Any]) -> Dict[str, Any]:
        return self.interpreter.compute_pair_score(rgb_det, sar_det)

    def match_detections(self, *args, **kwargs):
        return self.interpreter.match_detections(*args, **kwargs)

    # -- 校验 ------------------------------------------------------------

    def _preflight(self, params: Dict[str, Any]):
        """镜像上游 ``generate`` 的前置校验，保证两种 mode 的报错一致。"""
        interp = self.interpreter
        provenance = params.get("evidence_provenance")
        allow_unprovenanced = interp.to_bool(params.get("allow_unprovenanced_text"), False)
        if (params.get("rgb_text") or params.get("sar_text")) and not provenance and not allow_unprovenanced:
            return None, interp.error_response(
                "Missing valid RGB/SAR detection evidence from successful previous tools.",
                2, "invalid_dependency", "missing evidence_provenance")
        text_blob = json.dumps(
            {"rgb_text": params.get("rgb_text"), "sar_text": params.get("sar_text")},
            ensure_ascii=False)
        if not provenance and (
            "(10, 20, 50, 80), ship, score 0.73" in text_blob
            or "(12, 22, 52, 82), ship, score 0.86" in text_blob
        ):
            return None, interp.error_response(
                "Prompt example detections are not valid evidence.",
                2, "example_leakage_detected", "prompt example bbox leakage")
        raw_target = str(params.get("target") or "").strip()
        if not raw_target:
            return None, interp.error_response(
                "Missing required parameter: target", 2, "invalid_request", "missing target")
        target = interp.canonical_target(raw_target)
        if target not in interp.FINAL_TARGETS:
            return None, interp.error_response(
                f"Unsupported target: {raw_target}. Supported targets are: {sorted(interp.FINAL_TARGETS)}",
                2, "invalid_request", "unsupported target")
        task = interp.canonical_task(params.get("task"))
        if task is None:
            return None, interp.error_response(
                f"Unsupported task: {params.get('task')}. Supported tasks are: "
                f"{sorted(set(interp.TASK_ALIASES.values()))}",
                2, "invalid_request", "unsupported task")
        try:
            float(params.get("match_threshold", interp.MATCH_THRESHOLD))
            float(params.get("pixel_size_m", params.get("gsd_m_per_pixel", 1.0)))
            if params.get("max_center_distance") not in (None, ""):
                float(params.get("max_center_distance"))
        except (TypeError, ValueError) as exc:
            return None, interp.error_response(
                f"Invalid numeric parameter: {exc}", 2, "invalid_request", str(exc))
        return {"target": target, "task": task, "raw_target": raw_target}, None

    # -- 出口 ------------------------------------------------------------

    def _finalize(self, response: Dict[str, Any], params: Dict[str, Any], mode: str) -> Dict[str, Any]:
        response.setdefault("mode", mode)
        response.setdefault("degraded", False)
        response.setdefault("degrade_reason", None)
        output_mode = str(response.get("output_mode") or params.get("output_mode") or "compact").strip().lower()
        if output_mode not in {"summary", "compact", "full"}:
            output_mode = "compact"
        response["output_mode"] = output_mode

        if int(response.get("error_code") or 0) == 0:
            # G1：compact/summary 不得把证据层大字段塞进上下文。
            ds_vi_adapter.strip_big_fields(response, output_mode)
            structured = dict(response.get("structured") or {})
            structured.update({
                "type": "CrossModalInterpretation",
                "schema_version": "1.0",
                "status": "ok",
                "mode": response["mode"],
                "degraded": bool(response["degraded"]),
                "degrade_reason": response["degrade_reason"],
                "task": response.get("task"),
                "target": response.get("target"),
                "output_mode": output_mode,
                "summary": response.get("summary"),
                "conclusion_type": response.get("conclusion_type"),
                "final_bbox": response.get("final_bbox"),
                "selected_targets": response.get("selected_targets"),
                "capability": dict(response.get("capability") or CAPABILITY),
            })
            for key in STRUCTURED_KEYS:
                if key in response:
                    structured[key] = response[key]
            response["structured"] = structured
        else:
            structured = dict(response.get("structured") or {})
            structured.setdefault("type", "CrossModalInterpretation")
            structured.setdefault("schema_version", "1.0")
            structured.setdefault("status", response.get("status") or "error")
            structured["mode"] = response["mode"]
            structured["degraded"] = bool(response["degraded"])
            structured["degrade_reason"] = response["degrade_reason"]
            structured.setdefault("capability", dict(CAPABILITY))
            response["structured"] = structured

        response["structured"]["payload_bytes"] = _payload_bytes(response)
        return response

    def _legacy_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        legacy = dict(params)
        legacy["mode"] = CrossModalInterpreter.MODE_LEGACY_HARD
        legacy.pop("reasoning_mode", None)
        return legacy

    # -- 入口 ------------------------------------------------------------

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if FORCE_MODE:
            params = dict(params)
            params["mode"] = self.canonical_mode(params)
            params.pop("reasoning_mode", None)
        response = self._dispatch(params)
        _log_arm_call(params, response)
        return response

    def _dispatch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("source_mode") == "workbook_replay":
            from tool_server.tool_workers.online_workers import workbook_replay_adapter
            try:
                response = workbook_replay_adapter.interpret(self.interpreter, params)
            except Exception as exc:
                response = self.interpreter.error_response(str(exc), 2, "replay_error", str(exc))
                response.update(source_mode="workbook_replay", degraded=False)
            result = self._finalize(response, params, "shared_identity")
            if response.get("capability"):
                result["structured"]["capability"] = response["capability"]
            return result
        if params.get("source_mode") not in (None, "online"):
            return self._finalize(self.interpreter.error_response("Unsupported source_mode", 2, "invalid_request", "source_mode"), params, "shared_identity")
        mode = self.canonical_mode(params)

        if mode != CrossModalInterpreter.MODE_SHARED_IDENTITY:
            # legacy_hard / candidate_graph：完全交给上游，输出逐字段不变。
            return self._finalize(self.interpreter.generate(params), params, mode)

        context, error = self._preflight(params)
        if error is not None:
            return self._finalize(error, params, mode)

        try:
            rgb_raw = self.interpreter.load_detections(
                params, "rgb_detections", "rgb_detection_set", "rgb_text")
            sar_raw = self.interpreter.load_detections(
                params, "sar_detections", "sar_detection_set", "sar_text")
        except Exception as exc:  # pragma: no cover - 解析异常兜底
            logger.exception(f"CrossModalInterpret detection parsing failed: {exc}")
            return self._finalize(self.interpreter.error_response(
                f"CrossModalInterpret failed to parse detections: {exc}", 1, "error", str(exc)),
                params, mode)

        try:
            response = ds_vi_adapter.interpret(
                self.interpreter, params, context["target"], rgb_raw, sar_raw,
                budget_seconds=self.budget_seconds)
            # 适配器可能内部降级到 vi_only，实际 mode 优先于请求 mode。
            response.setdefault("mode", mode)
            response["requested_mode"] = mode
        except ds_vi_adapter.AdapterError as exc:
            # 第三级降级：共享身份与 vi_only 都不可用时回退旧贪心，并显式标注。
            logger.warning(f"DS+VI shared-identity chain unavailable ({exc.stage}): {exc}")
            fallback = self.interpreter.generate(self._legacy_params(params))
            fallback["mode"] = CrossModalInterpreter.MODE_LEGACY_HARD
            fallback["degraded"] = True
            fallback["degrade_reason"] = f"adapter_failed:{exc.stage}:{exc}"
            return self._finalize(fallback, params, CrossModalInterpreter.MODE_LEGACY_HARD)
        return self._finalize(response, params, mode)


class CrossModalInterpretWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="CrossModalInterpret",
        host="0.0.0.0",
        port=None,
        limit_model_concurrency=2,
        model_semaphore=None,
        wait_timeout=180.0,
        task_timeout=120.0,
    ):
        super().__init__(
            controller_addr, worker_addr, worker_id, no_register,
            None, None, model_name, False, False, "cpu",
            limit_model_concurrency, host, port, model_semaphore,
            wait_timeout, task_timeout,
        )
        self.engine = OneIDCrossModalEngine()

    def init_model(self):
        logger.info("CrossModalInterpretWorker does not need a model. Ready to run.")

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.generate(params)

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "CrossModalInterpret",
                "description": (
                    "Task-oriented RGB/SAR cross-modal interpretation with shared-identity reasoning. "
                    "It filters RGB and SAR detections by one target, pairs them, reasons about whether "
                    "the two views describe the same object, and returns compact structured output for a "
                    "requested task. Capability boundary: only category-level evidence is available online "
                    "(no colour/size/orientation attributes), the temporal (static/moved) family is not "
                    "part of this tool, and registration is identity. Default mode is shared_identity; "
                    "legacy_hard exists only as a comparison/fallback baseline."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "vehicle, tank, ship, aircraft, bridge, or harbor."},
                        "source_mode": {"type": "string", "enum": ["online", "workbook_replay"], "description": "Explicit server-enabled replay: live detections plus the target's attributes from the frozen property workbook. Default online is unchanged."},
                        "family": {"type": "string", "enum": ["match", "time"]},
                        "dataset_id": {"type": "string", "enum": ["task4", "task5"]},
                        "task": {"type": "string", "description": "crosscheck_exists, select_largest, select_highest_score, joint_scene, or distance_largest_smallest; time workbook replay supports temporal_changes or joint_scene."},
                        "rgb_detections": {"type": "array", "description": "Structured RGB detections from ObjectDetection or similar."},
                        "sar_detections": {"type": "array", "description": "Structured SAR detections from DenoDetV2 or similar."},
                        "rgb_text": {"type": "string", "description": "RGB detection text (requires evidence_provenance)."},
                        "sar_text": {"type": "string", "description": "SAR detection text (requires evidence_provenance)."},
                        "rgb_detection_set": {"type": "array", "description": "Alias of rgb_detections for tool-output payloads."},
                        "sar_detection_set": {"type": "array", "description": "Alias of sar_detections for tool-output payloads."},
                        "evidence_provenance": {"type": "object", "description": "Registry record proving the detections came from successful previous tools."},
                        "allow_unprovenanced_text": {"type": "boolean", "description": "Explicitly accept text detections without provenance (debug only)."},
                        "mode": {"type": "string", "description": "shared_identity (default), vi_only, candidate_graph, or legacy_hard."},
                        "rgb_image": {"type": "string", "description": "RGB image reference, used for provenance and audit only."},
                        "sar_image": {"type": "string", "description": "SAR image reference, used for provenance and audit only."},
                        "pair_id": {"type": "string", "description": "Optional stable id for this RGB/SAR pair."},
                        "output_mode": {"type": "string", "description": "summary, compact, or full. Default compact."},
                        "preview_top_k": {"type": "integer", "description": "Preview count in compact mode."},
                        "category_match_mode": {"type": "string", "description": "exact or substring. Default exact."},
                        "match_threshold": {"type": "number", "description": "Legacy hybrid match threshold. Default 0.45."},
                        "max_center_distance": {"type": "number", "description": "Optional hard center-distance gate in pixels (legacy path)."},
                        "pixel_size_m": {"type": "number", "description": "Pixel size in meters for distance calculation."},
                        "gsd_m_per_pixel": {"type": "number", "description": "Alias of pixel_size_m."},
                        "vi_config_path": {"type": "string", "description": "Optional override of the shared-identity YAML."},
                        "ds_config_path": {"type": "string", "description": "Optional override of the online D-S JSON."},
                        "return_artifact": {"type": "boolean", "description": "Write full debug JSON to tools_output."},
                        "scene_description": {"type": "string", "description": "Optional scene text for joint_scene."},
                    },
                    "required": ["target"],
                },
            },
        }


# --------------------------------------------------------------------------- #
# 兼容层：旧调用方按模块级函数导入时仍然可用
# --------------------------------------------------------------------------- #

_DEFAULT_ENGINE = OneIDCrossModalEngine()


def generate(params: Dict[str, Any]) -> Dict[str, Any]:
    return _DEFAULT_ENGINE.generate(params)


def normalize_label(value: Any) -> str:
    return CrossModalInterpreter.normalize_label(value)


def canonical_target(value: Any) -> str:
    return _DEFAULT_ENGINE.canonical_target(value)


def canonical_task(value: Any) -> Optional[str]:
    return _DEFAULT_ENGINE.canonical_task(value)


def compute_pair_score(rgb_det: Dict[str, Any], sar_det: Dict[str, Any]) -> Dict[str, Any]:
    return _DEFAULT_ENGINE.compute_pair_score(rgb_det, sar_det)


def match_detections(*args, **kwargs):
    return _DEFAULT_ENGINE.match_detections(*args, **kwargs)


FINAL_TARGETS = CrossModalInterpreter.FINAL_TARGETS
TARGET_ALIASES = CrossModalInterpreter.TARGET_ALIASES
TASK_ALIASES = CrossModalInterpreter.TASK_ALIASES
MATCH_THRESHOLD = CrossModalInterpreter.MATCH_THRESHOLD
DEFAULT_DETECTION_CONFIDENCE = CrossModalInterpreter.DEFAULT_DETECTION_CONFIDENCE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20035)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="CrossModalInterpret")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--test-json", type=Path, default=None,
                        help="Run one local generate() call from a params JSON file.")
    args = parser.parse_args()

    if args.test_json:
        engine = OneIDCrossModalEngine(budget_seconds=args.budget_seconds)
        with args.test_json.open("r", encoding="utf-8") as handle:
            params = json.load(handle)
        print(json.dumps(engine.generate(params), ensure_ascii=False, indent=2))
        return 0

    logger.info(f"args: {args}")
    worker = CrossModalInterpretWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
