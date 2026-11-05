"""BPA construction for matched and unmatched detections."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .ds_rules import VACUOUS, make_bpa


def build_matched_evidence(
    rgb_box: Mapping[str, Any],
    sar_box: Mapping[str, Any],
    pair_info: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Two BPAs for one associated RGB/SAR pair.

    Both sources keep their own detector reliability; the geometric agreement
    of the association is mixed into the detector score with
    ``evidence.support_weight``.
    """
    cfg = dict(config or {})
    support = None
    if pair_info is not None:
        support = float(pair_info.get("match_score", 0.0))
    m1 = make_bpa(
        rgb_box.get("score"),
        reliability=float(cfg.get("alpha_rgb", 0.8)),
        support=support,
        support_weight=float(cfg.get("support_weight", 0.5)),
    )
    m2 = make_bpa(
        sar_box.get("score"),
        reliability=float(cfg.get("alpha_sar", 0.7)),
        support=support,
        support_weight=float(cfg.get("support_weight", 0.5)),
    )
    return m1, m2


def build_unmatched_evidence(
    source: str,
    box: Mapping[str, Any],
    rgb_img: Optional[str] = None,
    sar_img: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """BPA for a single-modality detection plus a counter-evidence BPA.

    The counter-evidence defaults to the vacuous BPA, so the conflict-free
    fusion keeps the single-source evidence while the adaptive rule still sees
    a large evidence distance and routes to PCR6.
    """
    cfg = dict(config or {})
    m1 = make_bpa(
        box.get("score"),
        reliability=float(cfg.get("alpha_single", 0.5)),
        support=None,
        support_weight=0.0,
    )
    counter_mode = str(cfg.get("unmatched_counter", "vacuous")).strip().lower()
    if counter_mode in {"vacuous", "vacuous_bpa", "none"}:
        m2 = dict(VACUOUS)
    else:
        prior = float(cfg.get("unmatched_counter_prior", 0.5))
        m2 = make_bpa(prior, reliability=float(cfg.get("unmatched_counter_reliability", 0.3)))
    return m1, m2
