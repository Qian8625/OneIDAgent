"""Object-level matching: the documented Jaccard floor and the exact-pair counter."""
from __future__ import annotations

from DS_VI.experiments.metrics.property_metrics import identity_metrics


def _instance(instance_id: str, view: str) -> dict:
    return {
        "instance_id": instance_id,
        "view": view,
        "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0,
        "category": "ship", "state": "static", "size_bin": "small", "subtype": "cargo_ship",
    }


def _object(object_id: str, a, b) -> dict:
    return {
        "object_id": object_id,
        "instance_a": a,
        "instance_b": b,
        "category": "ship", "state": "static", "size_bin": "small", "subtype": "cargo_ship",
    }


TRUTH_INSTANCES = {
    "t_a": _instance("t_a", "rgb"),
    "t_b": _instance("t_b", "sar"),
    "t_c": _instance("t_c", "rgb"),
    "t_d": _instance("t_d", "sar"),
}
TRUTH_OBJECTS = [_object("gt1", "t_a", "t_b"), _object("gt2", "t_c", "t_d")]


def _metrics(predicted, bridge, jaccard_floor: float = 0.0) -> dict:
    tool_instances = {}
    for obj in predicted:
        for column in ("instance_a", "instance_b"):
            instance_id = obj.get(column)
            if instance_id:
                tool_instances[instance_id] = _instance(instance_id, "rgb")
    return identity_metrics(
        predicted, TRUTH_OBJECTS, bridge,
        tool_instances, TRUTH_INSTANCES, "match",
        jaccard_floor=jaccard_floor,
    )["identity"]


def test_exact_pair_is_counted_and_reported():
    predicted = [_object("p1", "u_a", "u_b")]
    block = _metrics(predicted, {"u_a": "t_a", "u_b": "t_b"})["gt_double_view"]
    assert block["total"] == 2
    assert block["predicted_double_view"] == 1
    assert block["predicted_exact"] == 1
    assert block["association_recall"] == 0.5
    assert block["association_recall_exact"] == 0.5


def test_wrong_partner_is_credited_only_without_the_floor():
    # u_b bridges to t_d: the predicted pair spans two different GT objects, so the
    # member-set Jaccard with gt1 is 1/3.
    predicted = [_object("p1", "u_a", "u_b")]
    bridge = {"u_a": "t_a", "u_b": "t_d"}

    loose = _metrics(predicted, bridge)["gt_double_view"]
    assert loose["predicted_double_view"] == 1
    assert loose["predicted_exact"] == 0
    assert loose["association_recall"] == 0.5
    assert loose["association_recall_exact"] == 0.0

    strict = _metrics(predicted, bridge, jaccard_floor=0.5)["gt_double_view"]
    assert strict["predicted_double_view"] == 0
    assert strict["association_recall"] == 0.0
    assert strict["association_recall_exact"] == 0.0


def test_unmatched_partner_is_credited_only_without_the_floor():
    # u_b matches no truth instance, so the predicted object resolves to {t_a} and
    # its Jaccard with gt1 is exactly 1/2 — the documented floor cannot reject it.
    predicted = [_object("p1", "u_a", "u_b")]
    block = _metrics(predicted, {"u_a": "t_a"})["gt_double_view"]
    assert block["predicted_double_view"] == 1
    assert block["predicted_exact"] == 0
    assert block["association_recall"] == 0.5
    assert block["association_recall_exact"] == 0.0
