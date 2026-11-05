"""Read-only probes for the code-level defects of the DS/VI cascade.

Does not modify production code, configs, or existing results.  The report is
written next to this file as ``code_defect_probe_results.json``.

Run (from the project root):
    <python> -B DS_VI/probes/code_defect_probe.py [--ds-only] [--decision] [--all]

Checks
------
D3  adaptive_fusion can never select PCR6 (distance bound < theta)
D4  RDC_fusion's conflict_fallback -> PCR6 branch is unreachable
D1  decision.py's "top_association" is sum/#supporting, not a max, and can exceed 1
D6  a detection with a missing score becomes strong background evidence
D7  the DS frame of discernment is binary {S, B}
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from random import Random
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from DS_VI.models.ds_fusion.ds_rules import (  # noqa: E402
    FRAME, VACUOUS, adaptive_fusion, betp, dempster_combine,
    discount_bpa, jousselme_distance, make_bpa, make_decision)

ALPHAS = {"match": (0.8, 0.7), "time": (0.8, 0.8)}
THETA = 0.5


# --------------------------------------------------------------------- D3/D7


def probe_ds():
    out = {"D7_frame": list(FRAME), "D3": {}, "D4": {}, "D3_single_accept": {}}
    for family, (a1, a2) in ALPHAS.items():
        worst = 0.0
        rules = set()
        for sa, sb, g in product((0.0, 1.0), repeat=3):
            m1 = make_bpa(sa, reliability=a1, support=g, support_weight=0.5)
            m2 = make_bpa(sb, reliability=a2, support=g, support_weight=0.5)
            worst = max(worst, jousselme_distance(m1, m2))
            rules.add(adaptive_fusion(m1, m2, theta=THETA, with_info=True)[1]["rule"])
        out["D3"][family] = {
            "cube_vertex_max_distance": round(worst, 10),
            "theta": THETA,
            "distance_always_below_theta": worst < THETA,
            "reachable_rules": sorted(rules),
        }
    worst = 0.0
    rules = set()
    for s, g in product((0.0, 1.0), repeat=2):
        m1 = make_bpa(s, reliability=0.5, support=None, support_weight=0.0)
        worst = max(worst, jousselme_distance(m1, VACUOUS))
        rules.add(adaptive_fusion(m1, VACUOUS, theta=THETA, with_info=True)[1]["rule"])
    out["D3"]["single_plus_vacuous"] = {
        "cube_vertex_max_distance": round(worst, 10),
        "distance_always_below_theta": worst < THETA,
        "reachable_rules": sorted(rules),
    }
    # dense random confirmation
    rng = Random(7)
    seen, peak = set(), 0.0
    for _ in range(100000):
        sa, sb, g = rng.random(), rng.random(), rng.random()
        m1 = make_bpa(sa, reliability=0.8, support=g, support_weight=0.5)
        m2 = make_bpa(sb, reliability=0.7, support=g, support_weight=0.5)
        peak = max(peak, jousselme_distance(m1, m2))
        seen.add(adaptive_fusion(m1, m2, with_info=True)[1]["rule"])
    out["D3"]["match_random_1e5"] = {"max_distance": round(peak, 6),
                                     "reachable_rules": sorted(seen)}

    # --- D4: is dempster_combine's singular branch reachable inside RDC?
    def conflict_after_discount(m1, m2):
        d = jousselme_distance(m1, m2)
        a = max(0.0, min(1.0, 1.0 - d))
        return dempster_combine(discount_bpa(m1, a), discount_bpa(m2, a))[1]

    def rand_bpa(rng):
        x, y = rng.random(), rng.random()
        if x + y > 1.0:
            x, y = 1.0 - x, 1.0 - y
        return {"S": x, "B": y, "SB": 1.0 - x - y}

    rng = Random(11)
    peak_r, none_count = 0.0, 0
    for _ in range(200000):
        m1, m2 = rand_bpa(rng), rand_bpa(rng)
        d = jousselme_distance(m1, m2)
        a = max(0.0, min(1.0, 1.0 - d))
        comb, conf = dempster_combine(discount_bpa(m1, a), discount_bpa(m2, a))
        if comb is None:
            none_count += 1
        peak_r = max(peak_r, conf)
    grid = [i / 40 for i in range(41)]
    peak_g = 0.0
    for s1 in grid:
        for s2 in grid:
            for sb1 in grid:
                for sb2 in grid:
                    if s1 + sb1 > 1.0 or s2 + sb2 > 1.0:
                        continue
                    m1 = {"S": s1, "B": 1.0 - s1 - sb1, "SB": sb1}
                    m2 = {"S": s2, "B": 1.0 - s2 - sb2, "SB": sb2}
                    peak_g = max(peak_g, conflict_after_discount(m1, m2))
    out["D4"] = {
        "random_2e5_none_count": none_count,
        "random_2e5_max_conflict": round(peak_r, 12),
        "grid_41pow4_max_conflict": round(peak_g, 12),
        "singular_threshold": 1.0 - 1e-12,
        "branch_unreachable": peak_g < 1.0 - 1e-12 and none_count == 0,
        "note": "conflict inside RDC = agreement^2 * undiscounted_cross_conflict <= 0.5",
    }

    # --- accept_betp=0.3 on default singles
    lo, hi = 1.0, 0.0
    for i in range(1001):
        mass, _ = adaptive_fusion(make_bpa(i / 1000, reliability=0.5), VACUOUS,
                                  theta=THETA, with_info=True)
        b = betp(mass)["S"]
        lo, hi = min(lo, b), max(hi, b)
    out["D3_single_accept"] = {
        "betp_min": round(lo, 8), "betp_max": round(hi, 8),
        "rejected_by_accept_betp_0.3": sum(
            1 for i in range(1001)
            if betp(adaptive_fusion(make_bpa(i / 1000, reliability=0.5), VACUOUS,
                                    theta=THETA))["S"] < 0.3),
    }
    return out


# ----------------------------------------------------------------------- D1


def probe_decision(families=("match", "time"), limit=6):
    from DS_VI.models.cascade.vi import BEST_CONFIG, run_vi_pair
    from DS_VI.models.ds_fusion.config import load_config
    from DS_VI.models.ds_fusion.pipeline import (
        DualDetectionProvider, load_dataset, run_pair)
    from DS_VI.models.shared_identity import inference as inf
    import DS_VI.models.cascade.vi as vm

    cfg = load_config({"decision": {"accept_betp": 0.3, "matched_fallback": True}})
    rows, path_rows = [], []
    for family in families:
        provider = DualDetectionProvider(cfg, family)
        for pair in load_dataset(cfg, family)[:limit]:
            det = {v: provider.detect(pair["base_id"], v, cfg["targets"]) for v in ("a", "b")}
            ds = run_pair(pair, det, cfg)
            matched = [o for o in ds["objects"] if o.get("source") == "matched"]
            for mode, arg in (("vi_alone", None),
                              ("matched", matched)):
                if mode == "matched" and not matched:
                    continue
                captured = {}
                orig = inf.run_joint_inference

                def cap(*a, **k):
                    state = orig(*a, **k)
                    captured["state"] = state
                    return state

                with patch.object(vm, "run_joint_inference", cap):
                    run_vi_pair(pair, det, cfg, family, matched=arg, config_path=BEST_CONFIG,
                                decision_overrides={"provenance": {"grouping_mode": "observation"}},
                                attribute_names=("category",))
                state = captured.get("state")
                if state is None:
                    continue
                objs = {}
                for obs_id, row in state.association.items():
                    for obj_id, p in row.items():
                        if obj_id == "__NULL__":
                            continue
                        objs.setdefault(obj_id, []).append(float(p))
                for obj_id, probs in objs.items():
                    support_mass = sum(p for p in probs if p > 0.0)
                    n_support = sum(1 for p in probs if p >= 0.5)
                    if n_support == 0:
                        continue          # step 1 returns early; step 3 never runs
                    decide_val = support_mass / max(n_support, 1)
                    true_max = max(probs)
                    rec = {"family": family, "pair": pair["base_id"], "mode": mode,
                           "n_supporting": n_support,
                           "decide_value": round(decide_val, 6),
                           "true_max_association": round(true_max, 6),
                           "support_mass": round(support_mass, 6),
                           "exceeds_one": decide_val > 1.0}
                    rows.append(rec)
                    if abs(decide_val - true_max) > 1e-9:
                        path_rows.append(rec)
    return {"objects_reaching_step3": len(rows),
            "objects_where_value_differs_from_true_max": len(path_rows),
            "objects_with_value_above_1": sum(1 for r in rows if r["exceeds_one"]),
            "max_decide_value": max((r["decide_value"] for r in rows), default=None),
            "max_true_max_association": max((r["true_max_association"] for r in rows), default=None),
            "worst_examples": sorted(rows, key=lambda r: -r["decide_value"])[:5]}


# ----------------------------------------------------------------------- D6


def probe_missing_score():
    """A detection without a usable score is turned into strong background mass."""
    from DS_VI.models.ds_fusion.evidence import build_unmatched_evidence
    from DS_VI.models.ds_fusion.config import load_config

    cfg = load_config()
    detection = {"bbox": [10, 10, 60, 60], "label": "ship", "index": 0}
    detection["score"] = 0.0     # what pipeline.detect writes when the field is absent
    m1, m2 = build_unmatched_evidence("view_a", detection, None, None,
                                      {"alpha_single": cfg["evidence"]["alpha_single"]})
    decision = make_decision(m1, accept_betp=0.3, abstain_band=0.0)
    untouched = {k: detection[k] for k in ("bbox", "label")}
    untouched["score"] = detection.get("score", 0.0)
    return {
        "note": "pipeline.detect does float(detection.get('score', 0.0)); a missing "
                "score becomes 0.0, which is NOT filtered because score_threshold=0.0",
        "detection_used": untouched,
        "bpa_single": {k: round(v, 6) for k, v in m1.items()},
        "betp_ship": round(decision["betp_ship"], 6),
        "decision_at_accept_betp_0.3": decision["decision"],
        "source_lines": ["DS_VI/models/ds_fusion/pipeline.py:131-136",
                         "DS_VI/models/ds_fusion/ds_rules.py:54"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds-only", action="store_true")
    ap.add_argument("--decision", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pairs", type=int, default=6)
    args = ap.parse_args()
    run_ds = args.ds_only or args.all or not (args.decision)
    run_dec = args.decision or args.all or not (args.ds_only)

    report = {"scope": "read-only code-defect probes; D1/D3/D4/D6/D7"}
    if run_ds:
        report["ds_layer"] = probe_ds()
        report["missing_score"] = probe_missing_score()
    if run_dec:
        report["decision"] = probe_decision(limit=args.pairs)
    target = Path(__file__).with_name("code_defect_probe_results.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
