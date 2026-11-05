"""GT 协议冻结测试。

两个断言方向：
1. ``gt_protocol`` 不得依赖 interpreter / worker / DS_VI（只读、纯函数）；
2. 它此刻的数值必须与 legacy 实现逐位一致——这就是"冻结"的含义，
   将来任何一侧改动都会在这里先炸。
"""

from pathlib import Path

from scripts.eval import gt_protocol
from tool_server.tool_workers.online_workers.CrossModalInterpret_worker import (
    CrossModalInterpreter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GT_SOURCE = REPO_ROOT / "scripts" / "eval" / "gt_protocol.py"

BOXES = [
    ([100, 100, 200, 200], [102, 101, 201, 199]),
    ([0, 0, 100, 100], [400, 400, 500, 500]),
    ([10, 10, 60, 60], [12, 11, 61, 59]),
    ([500, 500, 560, 560], [0, 0, 40, 40]),
]


def test_gt_protocol_has_no_pipeline_dependencies():
    """只校验 import 面：文档里提到 legacy 实现是解释性的，不算依赖。"""
    imports = [
        line.strip() for line in GT_SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for line in imports:
        for forbidden in ("CrossModalInterpreter", "DS_VI", "tool_server", "online_workers"):
            assert forbidden not in line, (forbidden, line)
    assert gt_protocol.GT_PROTOCOL_VERSION


def test_frozen_score_matches_legacy_interpreter():
    engine = CrossModalInterpreter()
    for left, right in BOXES:
        frozen = gt_protocol.pair_score(left, right, 0.9, 0.8)
        legacy = engine.compute_pair_score(
            {"bbox": left, "score": 0.9}, {"bbox": right, "score": 0.8})
        for key in ("match_score", "iou", "center_distance", "center_score",
                    "area_score", "confidence_score"):
            assert frozen[key] == legacy[key], (key, left, right, frozen[key], legacy[key])


def test_match_items_hybrid_is_one_to_one():
    left = [{"bbox": [100, 100, 200, 200], "score": 0.9},
            {"bbox": [300, 300, 400, 400], "score": 0.8}]
    right = [{"bbox": [102, 101, 201, 199], "score": 0.85}]
    matched, left_only, right_only = gt_protocol.match_items(
        left, right, mode="hybrid", hybrid_threshold=0.45)
    assert len(matched) == 1
    assert len(left_only) == 1
    assert right_only == []
    assert matched[0]["left_index"] == 0 and matched[0]["right_index"] == 0


def test_match_items_iou_mode_respects_threshold():
    left = [{"bbox": [0, 0, 100, 100]}]
    right = [{"bbox": [400, 400, 500, 500]}]
    matched, left_only, right_only = gt_protocol.match_items(
        left, right, mode="iou", iou_threshold=0.5)
    assert matched == [] and len(left_only) == 1 and len(right_only) == 1


def test_canonical_target_aliases():
    assert gt_protocol.canonical_target("airplane") == "aircraft"
    assert gt_protocol.canonical_target("Cars") == "vehicle"
    assert gt_protocol.canonical_target("dock") == "dock"
