"""task4/task5 数据视图：切子集、渲染中性问法、解析 GT 与预测。

两个关键事实：

* ``test.jsonl`` 是扁平的一份（task1-5 混在一起），没有按任务切分的入口，所以本模块
  按首个图片路径前缀（``match_images/`` / ``time_images/``）筛出 181 + 536 条；
* 原始问法 100% 写着 "Use workbook replay for pair ..."，即这套题是围绕 OneID 工具
  构造的。直接把原文喂给"无 OneID"臂等于给它一条不可执行的指令，因此默认用
  ``render_question(..., mode="neutral")`` 渲染成不含任何工具名的中性问法，两臂逐字相同。
  想看原始口径时用 ``--question-mode original``，但两臂仍然共用同一份文本。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FAMILIES = ("task4", "task5")

TASK4_COUNT_KEYS = ("matched", "rgb_only", "sar_only")
TASK5_COUNT_KEYS = ("appeared", "disappeared", "static", "moved")
ATTR_KEYS = ("subtype", "color", "size_bin", "orientation")

_TASK4_GT = re.compile(
    r"(\d+)\s+matched[,\s]+(\d+)\s+RGB-only\s+and\s+(\d+)\s+SAR-only", re.IGNORECASE
)
_TASK5_GT = re.compile(
    r"(\d+)\s+appeared[,\s]+(\d+)\s+disappeared[,\s]+(\d+)\s+static\s+and\s+(\d+)\s+moved",
    re.IGNORECASE,
)
_DISTINCT_GT = re.compile(r"(\d+)\s+distinct\s+\w+", re.IGNORECASE)
_TARGET_A = re.compile(r"to cross-validate\s+(\w+)\s+objects", re.IGNORECASE)
_TARGET_B = re.compile(r"to compare\s+(\w+)\s+objects", re.IGNORECASE)
_TARGET_FALLBACK = re.compile(r"largest\s+(?:RGB|time1)\s+(\w+)\s+detection", re.IGNORECASE)
_ATTR_PAIR = re.compile(
    r"(subtype|color|size_bin|orientation|visibility)\s*[=:]\s*"
    r"([A-Za-z_][A-Za-z0-9_/-]*(?:\s+[A-Za-z_][A-Za-z0-9_/-]*){0,3})",
    re.IGNORECASE,
)

#: 这些"值"等价于"没有记录"，不能当成属性写错来扣分。
_NULL_ATTRS = {
    "n", "na", "n_a", "n.a", "n/a", "n.a.", "none", "null", "nil", "unknown", "unclear", "unspecified",
    "not_available", "not_recorded", "not_applicable", "unavailable", "not_detected",
    "not_determined", "not_known", "no_data",
}


def _norm_attr(token: str) -> Optional[str]:
    """小写、空格转下划线、去掉尾随标点；空值等价物统一成 None。"""
    text = re.sub(r"\s+", "_", token.strip().lower()).strip("_.,;:")
    if not text or text in _NULL_ATTRS or not re.search(r"[a-z]", text):
        return None
    return text
_NO_ATTR = re.compile(r"no recorded\s+subtype", re.IGNORECASE)


def _count_keys(family: str) -> tuple:
    return TASK5_COUNT_KEYS if family == "task5" else TASK4_COUNT_KEYS


def _terminate_answer(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        try:
            payload = json.loads(msg.get("content", ""))
        except Exception:
            continue
        actions = payload.get("actions") or []
        if actions and isinstance(actions[0], dict) and actions[0].get("name") == "Terminate":
            ans = (actions[0].get("arguments") or {}).get("ans")
            return ans.strip() if isinstance(ans, str) else None
    return None


def parse_gt(label: str, family: str) -> Dict[str, Any]:
    """从 GT 的自然语言答案里解析出可比较字段。解析不出的字段置 None（表示"没有记录"）。"""
    counts: Dict[str, Optional[int]] = {key: None for key in _count_keys(family)}
    pattern = _TASK5_GT if family == "task5" else _TASK4_GT
    hit = pattern.search(label or "")
    if hit:
        for key, value in zip(_count_keys(family), hit.groups()):
            counts[key] = int(value)

    distinct = None
    distinct_hit = _DISTINCT_GT.search(label or "")
    if distinct_hit:
        distinct = int(distinct_hit.group(1))

    attributes: Dict[str, Optional[str]] = {key: None for key in ATTR_KEYS}
    if not _NO_ATTR.search(label or ""):
        for key, value in _ATTR_PAIR.findall(label or ""):
            key = key.lower()
            if key in attributes:
                attributes[key] = _norm_attr(value)
    return {"counts": counts, "distinct": distinct, "attributes": attributes}


_PRED_COUNT = {
    key: re.compile(r"(\d+)\s*[^0-9\n]{0,24}?" + key.replace("_", "[ _-]"), re.IGNORECASE)
    for key in TASK4_COUNT_KEYS + TASK5_COUNT_KEYS
}


def parse_prediction(text: str, family: str) -> Dict[str, Any]:
    """从最终答案里尽力解析同样字段；解析不出即 None（判为未给出）。"""
    text = text or ""
    counts: Dict[str, Optional[int]] = {}
    for key in _count_keys(family):
        hit = _PRED_COUNT[key].search(text)
        counts[key] = int(hit.group(1)) if hit else None
    attributes: Dict[str, Optional[str]] = {key: None for key in ATTR_KEYS}
    if not _NO_ATTR.search(text):
        for key, value in _ATTR_PAIR.findall(text):
            key = key.lower()
            if key in attributes:
                attributes[key] = _norm_attr(value)
    return {"counts": counts, "attributes": attributes}


def render_question(sample: Dict[str, Any], mode: str = "neutral") -> str:
    """两臂共用的问法。neutral 不出现任何工具名。"""
    if mode == "original":
        return sample["question_original"]

    pair, target = sample["pair_id"], sample["target"]
    if sample["family"] == "task4":
        return (
            f"Compare the optical (RGB) and SAR images of pair {pair}. "
            f"Cross-validate {target} objects across the two views and report "
            f"(a) the number of distinct {target} objects after cross-view deduplication, "
            f"together with the matched, RGB-only and SAR-only counts, and "
            f"(b) the subtype, color, size_bin and orientation of the largest RGB {target} detection."
        )
    return (
        f"Compare the time1 and time2 images of pair {pair}. "
        f"Analyse {target} objects across the two acquisitions and report "
        f"(a) how many {target} objects appeared, disappeared, stayed static and moved, and "
        f"(b) the subtype, color, size_bin and orientation of the largest time1 {target} detection."
    )


def _family_of(images: Iterable[str]) -> Optional[str]:
    first = next(iter(images), "") or ""
    if "match_images/" in first:
        return "task4"
    if "time_images/" in first:
        return "task5"
    return None


def _target_of(question: str) -> str:
    for pattern in (_TARGET_A, _TARGET_B, _TARGET_FALLBACK):
        hit = pattern.search(question or "")
        if hit:
            return hit.group(1).lower()
    return "object"


def load_samples(path: str | Path, families: Iterable[str] = FAMILIES, limit: Optional[int] = None,
                 question_mode: str = "neutral", per_family: Optional[int] = None) -> List[Dict[str, Any]]:
    """按 task4/task5 切子集。顺序即文件顺序，保证四臂看到同一序列。

    ``per_family`` 用于小样本冒烟：两类各取 N 条，避免"前 N 条全是 task4"。
    """
    wanted = tuple(families)
    samples: List[Dict[str, Any]] = []
    taken: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            images = [str(p) for p in (item.get("images") or [])]
            family = _family_of(images)
            if family not in wanted or len(images) < 2:
                continue
            messages = item.get("messages") or []
            question = ""
            for msg in messages:
                if msg.get("role") == "user":
                    question = (msg.get("content") or "").replace("<image>", "").strip()
                    break
            label = _terminate_answer(messages)
            if not label:
                continue
            pair_id = Path(images[0]).name.split("_")[0] + "_" + Path(images[0]).name.split("_")[1]
            sample = {
                "idx": len(samples),
                "line_no": line_no,
                "sample_id": f"S{len(samples):05d}_{pair_id}_{_target_of(question)}",
                "family": family,
                "pair_id": pair_id,
                "target": _target_of(question),
                "images": images,
                "question_original": question,
                "label": label,
                "gt": parse_gt(label, family),
            }
            sample["question"] = render_question(sample, question_mode)
            if per_family is not None and taken.get(family, 0) >= per_family:
                continue
            taken[family] = taken.get(family, 0) + 1
            sample["idx"] = len(samples)
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def fingerprint(samples: List[Dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample["sample_id"].encode())
        digest.update(b"\0")
        digest.update(sample["question"].encode())
        digest.update(b"\0")
        digest.update(sample["label"].encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, int] = {}
    by_target: Dict[str, int] = {}
    with_attr = 0
    for sample in samples:
        by_family[sample["family"]] = by_family.get(sample["family"], 0) + 1
        by_target[sample["target"]] = by_target.get(sample["target"], 0) + 1
        if any(v is not None for v in sample["gt"]["attributes"].values()):
            with_attr += 1
    return {"total": len(samples), "by_family": by_family, "by_target": by_target,
            "with_attributes": with_attr}
