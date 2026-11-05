"""Shared-identity VI runs on dual-view data (match / time), multi-class.

Two registration modes:

* ``vi_alone``: every cross-view pair candidate per class is registered and the
  joint A/Q inference decides which candidates are confirmed.
* ``vi_matched``: only the Stage-1 D-S matched pairs are registered (one object
  per pair); used by the cascade confirmation stage.

Observations and claims are built with the project's canonical builders and the
provenance graph is filled with :func:`register_problem_provenance`, so the
weights and inference stay identical to the store-ingestion path.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from DS_VI.models.shared_identity.attribute_consistency import (
    accept as consistency_accept,
    consistency,
    pair_weights,
)
from DS_VI.models.shared_identity.config import project_path as resolve_config

from DS_VI.models.shared_identity.config import load_config as load_shared_config
from DS_VI.models.shared_identity.decision import decide_object_truth
from DS_VI.models.shared_identity.evidence_state import build_evidence_state, state_probability
from DS_VI.models.shared_identity.evidence_store import EvidenceStore
from DS_VI.models.shared_identity.inference import run_joint_inference
from DS_VI.models.shared_identity.observation_builder import (
    build_claim,
    build_observation,
)
from DS_VI.models.shared_identity.physical_features import compute_pair_features
from DS_VI.models.shared_identity.provenance import register_problem_provenance
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    Modality,
    ObjectHypothesis,
    SpatialSupport,
    stable_id,
)


#: The one shipped shared-identity configuration (``runtime_best.yaml``). It
#: freezes every revision that measured as a win: uncapped evidence, outlier
#: bound, time-invariant attributes, a per-family independent-candidate width
#: (match pruned / time mutual), the 2026-11 annotation release (``color``
#: replaces ``state``), and the association-precision levers (prior floor 0.2,
#: displacement gate 2.0).  Historical versions live in ``archive/pre-v6/`` and are
#: not runnable from here.
BEST_CONFIG = "DS_VI/configs/shared_identity/runtime_best.yaml"


def _weighted_bbox(store: EvidenceStore, object_id: str) -> Optional[List[float]]:
    state = store.inference_state
    if state is None:
        return None
    total = 0.0
    sums = [0.0, 0.0, 0.0, 0.0]
    for observation_id, observation in store.observations.items():
        bbox = observation.spatial_support.bbox
        if bbox is None:
            continue
        weight = state_probability(state, observation_id, object_id)
        if weight <= 0.0:
            continue
        total += weight
        for index in range(4):
            sums[index] += weight * float(bbox[index])
    if total <= 0.0:
        return None
    return [value / total for value in sums]


def _modality(name: str) -> Modality:
    return Modality.SAR if str(name).lower() == "sar" else Modality.OPTICAL


def _pruned_cross_view_hypotheses(
    observations_a: Sequence[Any],
    observations_b: Sequence[Any],
    capacity: int,
    class_name: str,
    relation_group: str,
    max_links: int = 4,
    center_scale_factor: float = 0.75,
    prior_floor: float = 0.0,
    origin: str = "DS_VI.models.vi_pruned",
    bypass_gates: bool = False,
    margin: float = 0.0,
    margin_mode: str = "fixed",
    margin_quantile: float = 0.25,
    assignment: str = "many_to_one",
) -> List[ObjectHypothesis]:
    """Bounded cross-view candidates: each view-A observation links to its few
    nearest view-B observations (dense time frames can exceed 700 detections and
    the full O(n^2) candidate set is both slow and mostly gated away later).

    Ambiguity (ratio) test, applied when ``margin > 0`` or ``margin_mode`` is not
    ``fixed``: if the best and the second-best candidate of this observation score
    within ``margin`` of each other, the observation is skipped entirely. Two
    similarly good candidates mean the geometry cannot tell which counterpart is
    the right one, and a confirmed wrong pair costs more than a missed one (it
    consumes two boxes and inflates the over-merge count).

    ``margin_mode="quantile"`` derives the threshold from the pair's own margin
    distribution (reject the least-discriminative ``margin_quantile`` fraction)
    instead of a hand-set ratio, so the strictness follows scene density. It falls
    back to the fixed ``margin`` when this class group has fewer than four
    observations with at least two candidates.

    ``assignment="one_to_one"`` additionally forces every observation to be used by
    at most one candidate: the surviving pairs are sorted by ``diagnostic_score``
    and taken greedily, skipping any pair whose members are already consumed. The
    default ``"many_to_one"`` keeps the historical behaviour, where one view-B box
    can be the partner of several view-A boxes (the match family's measured
    over-merge source: one SAR box reused by up to 55 RGB leftovers).
    """
    from DS_VI.models.shared_identity.physical_features import bbox_center
    from DS_VI.models.shared_identity.scope import observations_are_related

    def _distance_rows(left: Any) -> List[Any]:
        rows = []
        for right in observations_b:
            if right.spatial_support.bbox is None:
                continue
            if not observations_are_related(left, right):
                continue
            left_center = bbox_center(left.spatial_support.bbox)
            right_center = bbox_center(right.spatial_support.bbox)
            if left_center is None or right_center is None:
                continue
            distance = (left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2
            rows.append((distance, right))
        rows.sort(key=lambda item: item[0])
        return rows

    def _margin_of(left: Any, scored: Sequence[Any]) -> Optional[float]:
        # Ratio test on the geometry score, evaluated on the nearest few by centre
        # distance so the cost stays bounded on dense frames.
        probe = scored[: max(3, int(max_links) * 2)]
        ranked = []
        for _distance, right in probe:
            candidate_features = compute_pair_features(
                left.spatial_support.bbox,
                right.spatial_support.bbox,
                observation_confidence=left.detection_confidence,
                object_confidence=right.detection_confidence,
                center_scale_factor=float(center_scale_factor),
            )
            ranked.append((candidate_features.diagnostic_score, right))
        ranked.sort(key=lambda item: -item[0])
        best_score = float(ranked[0][0])
        second_score = float(ranked[1][0])
        if best_score <= 0.0:
            return None
        return (best_score - second_score) / best_score

    rows_by_left: Dict[str, List[Any]] = {}
    margins: List[float] = []
    for left in observations_a:
        if left.spatial_support.bbox is None:
            continue
        scored = _distance_rows(left)
        rows_by_left[left.observation_id] = scored
        if (margin > 0.0 or str(margin_mode).lower() == "quantile") and len(scored) >= 2:
            value = _margin_of(left, scored)
            if value is not None:
                margins.append(value)

    quantile_threshold: Optional[float] = None
    if str(margin_mode).lower() == "quantile" and len(margins) >= 4:
        ordered = sorted(margins)
        target = min(max(float(margin_quantile), 0.0), 1.0)
        position = int(round(target * (len(ordered) - 1)))
        quantile_threshold = ordered[max(0, min(position, len(ordered) - 1))]

    hypotheses = []
    pending: List[Any] = []
    one_to_one = str(assignment).lower() == "one_to_one"
    for left in observations_a:
        scored = rows_by_left.get(left.observation_id)
        if not scored:
            continue
        if quantile_threshold is not None and len(scored) >= 2:
            value = _margin_of(left, scored)
            if value is not None and value < quantile_threshold:
                continue
        elif margin > 0.0 and len(scored) >= 2:
            value = _margin_of(left, scored)
            if value is not None and value < float(margin):
                continue
        for _distance, right in scored[: max(1, int(max_links))]:
            features = compute_pair_features(
                left.spatial_support.bbox,
                right.spatial_support.bbox,
                observation_confidence=left.detection_confidence,
                object_confidence=right.detection_confidence,
                center_scale_factor=float(center_scale_factor),
            )
            hypothesis = ObjectHypothesis(
                object_id=stable_id("DS_VI.models.vi", {"class": class_name, "a": left.observation_id, "b": right.observation_id}),
                spatial_support=SpatialSupport(bbox=list(left.spatial_support.bbox)),
                capacity=capacity,
                attribute_hints={"category": class_name},
                context={
                    "candidate_type": "cross_image_pair",
                    "relation_group_id": relation_group,
                    "member_observation_ids": [left.observation_id, right.observation_id],
                    "member_image_group_ids": [
                        left.context.get("image_group_id"),
                        right.context.get("image_group_id"),
                    ],
                    "pair_features": features.to_dict(),
                    "origin": origin,
                    **(({"prior_floor": float(prior_floor)}) if prior_floor > 0.0 else {}),
                    **(({"bypass_gates": True}) if bypass_gates else {}),
                },
            )
            if one_to_one:
                pending.append((
                    float(features.diagnostic_score),
                    str(left.observation_id),
                    str(right.observation_id),
                    hypothesis,
                ))
            else:
                hypotheses.append(hypothesis)

    if pending:
        # Deterministic greedy matching: best diagnostic score first, ties broken
        # by ids so two runs on the same pair produce the same candidate set.
        pending.sort(key=lambda item: (-item[0], item[1], item[2]))
        consumed: set = set()
        for _score, left_id, right_id, hypothesis in pending:
            if left_id in consumed or right_id in consumed:
                continue
            consumed.add(left_id)
            consumed.add(right_id)
            hypotheses.append(hypothesis)
    return hypotheses


def _mutual_cross_view_hypotheses(
    observations_a: Sequence[Any],
    observations_b: Sequence[Any],
    capacity: int,
    class_name: str,
    relation_group: str,
    center_scale_factor: float = 0.75,
    prior_floor: float = 0.5,
    origin: str = "ds_cascade_extra",
) -> List[ObjectHypothesis]:
    """One-to-one candidates among leftovers: mutual nearest neighbours only.

    The pruned builder lets one observation sit in several candidates, which is
    right for recall but leaves the association update undecided (each row splits
    its mass over several hypotheses). For the independent-data channel we want
    the opposite: one candidate per leftover observation, so VI can actually
    confirm or reject it. A pair is kept only when each side is the other's
    nearest same-class counterpart.
    """
    from DS_VI.models.shared_identity.physical_features import bbox_center, compute_pair_features
    from DS_VI.models.shared_identity.scope import observations_are_related

    def center(observation):
        box = observation.spatial_support.bbox
        return bbox_center(box) if box else None

    def distance(left, right):
        center_left, center_right = center(left), center(right)
        if center_left is None or center_right is None:
            return None
        return (center_left[0] - center_right[0]) ** 2 + (center_left[1] - center_right[1]) ** 2

    def nearest(source, targets):
        best, best_distance = None, None
        for target in targets:
            value = distance(source, target)
            if value is None or not observations_are_related(source, target):
                continue
            if best_distance is None or value < best_distance:
                best, best_distance = target, value
        return best

    hypotheses: List[ObjectHypothesis] = []
    for left in observations_a:
        right = nearest(left, observations_b)
        if right is None or nearest(right, observations_a) is not left:
            continue
        features = compute_pair_features(
            left.spatial_support.bbox,
            right.spatial_support.bbox,
            observation_confidence=left.detection_confidence,
            object_confidence=right.detection_confidence,
            center_scale_factor=float(center_scale_factor),
        )
        hypotheses.append(ObjectHypothesis(
            object_id=stable_id("DS_VI.models.vi_extra", {"class": class_name, "a": left.observation_id, "b": right.observation_id}),
            spatial_support=SpatialSupport(bbox=list(left.spatial_support.bbox)),
            capacity=capacity,
            attribute_hints={"category": class_name},
            context={
                "candidate_type": "cross_image_pair",
                "relation_group_id": relation_group,
                "member_observation_ids": [left.observation_id, right.observation_id],
                "member_image_group_ids": [
                    left.context.get("image_group_id"),
                    right.context.get("image_group_id"),
                ],
                "pair_features": features.to_dict(),
                "origin": origin,
                "prior_floor": float(prior_floor),
                "bypass_gates": True,
            },
        ))
    return hypotheses


def build_observations(
    pair: Mapping[str, Any],
    detections: Mapping[str, List[Dict[str, Any]]],
    config: Mapping[str, Any],
    family: str,
    extra_context: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the observations and claims of one pair.

    ``extra_context`` maps a builder observation id to context covariates that
    must reach the observation (visibility / evidence_quality / detection_count /
    registration_*).  The scope keys built here (relation_group_id,
    image_group_id, related_image_group_ids, channel) always win: ctx scope
    naming uses RGB/SAR while VI speaks in a/b.
    """
    spec = dict(config["families"][family])
    images = pair["images"]
    relation_group = f"{family}:{images['a']}:{images['b']}"
    observations = []
    claims = []
    observations_by_view: Dict[str, List[Any]] = {"a": [], "b": []}
    by_view_index: Dict[tuple, str] = {}
    class_of: Dict[str, str] = {}
    by_class: Dict[str, List[str]] = defaultdict(list)
    for view_key in ("a", "b"):
        view_spec = dict(spec["views"][view_key])
        other_key = "b" if view_key == "a" else "a"
        image = images[view_key]
        image_group = f"{family}:{view_key}:{image}"
        related = [f"{family}:{other_key}:{images[other_key]}"]
        for index, detection in enumerate(detections.get(view_key, [])):
            observation = build_observation(
                source_tool="DualDetection",
                modality=_modality(view_spec.get("modality", "optical")),
                bbox=detection["bbox"],
                time_key=str(view_spec.get("time_key", "t0")),
                acquisition_id=image,
                detection_confidence=float(detection.get("score") or 0.0),
                frame_id=image,
                input_roots=[image],
                context={
                    "relation_group_id": relation_group,
                    "image_group_id": image_group,
                    "related_image_group_ids": related,
                    "channel": view_key,
                },
                index=index,
            )
            covariates = dict((extra_context or {}).get(observation.observation_id) or {})
            if covariates:
                # Builder scope keys stay authoritative over ctx covariates.
                merged = dict(covariates)
                merged.update(observation.context)
                observation.context = merged
            observations.append(observation)
            observations_by_view[view_key].append(observation)
            source_index = detection.get("index")
            by_view_index[(view_key, source_index if source_index is not None else index)] = observation.observation_id
            class_of[observation.observation_id] = str(detection.get("label"))
            by_class[str(detection.get("label"))].append(observation.observation_id)
            claims.append(build_claim(
                observation,
                "category",
                detection.get("label"),
                source_channel=_modality(view_spec.get("modality", "optical")).value,
                raw_confidence=float(detection.get("score") or 0.0),
            ))
    return {
        "observations": observations,
        "claims": claims,
        "observations_by_view": observations_by_view,
        "by_view_index": by_view_index,
        "class_of": class_of,
        "by_class": dict(by_class),
        "relation_group": relation_group,
    }


def _member_bbox(store: EvidenceStore, state, object_id: str) -> Optional[List[float]]:
    weighted = _weighted_bbox(store, object_id) if state is not None else None
    if weighted:
        return weighted
    hypothesis = store.objects.get(object_id)
    if hypothesis is not None and hypothesis.spatial_support.bbox:
        return list(hypothesis.spatial_support.bbox)
    return None


def _object_record(
    store: EvidenceStore,
    evidence_state,
    object_id: str,
    class_hint: str,
    index: Optional[int],
    config: Mapping[str, Any],
    attribute_names: Sequence[str] = (),
) -> Dict[str, Any]:
    decision = decide_object_truth(evidence_state, object_id, attribute="category", config=config)
    summary = evidence_state.objects.get(object_id)
    p_class = 0.0
    attribute_beliefs: Dict[str, Dict[str, Any]] = {}
    if summary is not None:
        for _key, belief in summary.truth_beliefs.items():
            attribute = str(belief.get("attribute"))
            if attribute in set(attribute_names):
                distribution = {
                    str(key): round(float(value), 6)
                    for key, value in (belief.get("distribution") or {}).items()
                }
                attribute_beliefs[f"{attribute}@{belief.get('time_key')}"] = {
                    "attribute": attribute,
                    "time_key": belief.get("time_key"),
                    "value": belief.get("value"),
                    "probability": round(float(belief.get("probability", 0.0)), 6),
                    "distribution": distribution,
                }
            if attribute != "category":
                continue
            p_class = max(p_class, float((belief.get("distribution") or {}).get(class_hint, 0.0)))
    members = list((store.objects[object_id].context or {}).get("member_observation_ids") or [])
    weights = [state_probability(store.inference_state, member, object_id) for member in members]
    association = sum(weights) / len(weights) if weights else 0.0
    bbox = _member_bbox(store, store.inference_state, object_id)
    record = {
        "index": index,
        "object_id": object_id,
        "class": class_hint,
        "status": decision.status,
        "value": decision.value,
        "confidence": round(float(decision.confidence or 0.0), 6),
        "p_class": round(float(p_class), 6),
        "association": round(float(association), 6),
        "members": members,
        "bbox": [round(float(v), 3) for v in bbox] if bbox else None,
        "attribute_beliefs": attribute_beliefs,
        "reasons": list(decision.reasons),
    }
    member_boxes = {}
    for member in members:
        observation = store.observations.get(member)
        if observation is not None and observation.spatial_support.bbox:
            member_boxes[member] = list(observation.spatial_support.bbox)
    record["member_bboxes"] = member_boxes
    record["member_views"] = {
        member: str((store.observations[member].context or {}).get("channel"))
        for member in members
        if member in store.observations
    }
    return record


def shipped_cascade_settings(shared: Mapping[str, Any], family: str) -> Dict[str, Any]:
    """Resolve the shipped independent-channel settings for one family.

    Everything the cascade needs beyond the plain D-S handoff lives in the
    ``cascade`` section of the shared-identity YAML, with per-family mappings
    where the two task families need different operating points
    (``extra_margin``, ``extra_consistency``, ``max_links``, ...). Callers may
    override any of it explicitly; ``None`` means "use the shipped value".
    """
    cascade_cfg = dict(shared.get("cascade") or {})

    def per_family(key: str, default: Any) -> Any:
        value = cascade_cfg.get(key)
        if isinstance(value, Mapping):
            return value.get(family, default)
        return default if value is None else value

    return {
        "extra_links": int(cascade_cfg.get("extra_links", 0) or 0),
        # A mapping lets each family pick its own candidate width (the time
        # family benefits from mutual-nearest construction, the match family
        # needs the wider pruned set). A scalar keeps the old behaviour.
        "extra_mode": str(per_family("extra_mode", "pruned")),
        "extra_prior_floor": float(cascade_cfg.get("extra_prior_floor", 0.5)),
        "extra_margin": float(per_family("extra_margin", 0.0)),
        "extra_margin_mode": str(cascade_cfg.get("extra_margin_mode", "fixed") or "fixed"),
        "extra_margin_quantile": float(
            per_family("extra_margin_quantile", 0.25)),
        "extra_consistency": tuple(per_family("extra_consistency", ()) or ()),
        # One-to-one vs many-to-one assignment among the independent candidates.
        # ``one_to_one`` is opt-in: it removes the reuse of a member box by several
        # candidates, which is the measured over-merge source in the match family.
        "extra_assignment": str(per_family("extra_assignment", "many_to_one") or "many_to_one"),
        "consistency_mode": str(cascade_cfg.get("consistency_mode", "weighted")),
        "consistency_tau": float(cascade_cfg.get("consistency_tau", 0.0)),
        "consistency_min_weight": float(cascade_cfg.get("consistency_min_weight", 0.5)),
        "extra_policy": str(cascade_cfg.get("extra_policy", "confirmed")),
        "policy": str(cascade_cfg.get("policy", "status")),
        "blend_weight": float(cascade_cfg.get("blend_weight", 0.5)),
        "max_links": int(per_family("max_links", 4)),
        "center_scale_factor": float(per_family("center_scale_factor", 0.75)),
        # D-S acceptance operating point, fitted per family on all samples.
        # ``None`` keeps the D-S config's own value (v4 behaviour).
        "accept_betp": per_family("accept_betp", None),
        "single_accept_betp": cascade_cfg.get("single_accept_betp"),
        # Displacement-field gate on independent candidates (0 = off).
        "extra_displacement_gate": per_family("extra_displacement_gate", 0.0),
    }


def resolve_cascade_settings(
    vi_config: str, family: str, overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a shared-identity YAML and return the shipped settings for ``family``."""
    shared = load_shared_config(resolve_config(vi_config), overrides=overrides)
    return shipped_cascade_settings(shared, family)


def run_vi_pair(
    pair: Mapping[str, Any],
    detections: Mapping[str, List[Dict[str, Any]]],
    config: Mapping[str, Any],
    family: str,
    matched: Optional[Sequence[Mapping[str, Any]]] = None,
    config_path: str = BEST_CONFIG,
    decision_overrides: Optional[Mapping[str, Any]] = None,
    extra_claims: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    attribute_names: Sequence[str] = (),
    extra_context: Optional[Mapping[str, Mapping[str, Any]]] = None,
    extra_links: Optional[int] = None,
    extra_max_observations: int = 400,
    extra_prior_floor: Optional[float] = None,
    extra_mode: Optional[str] = None,
    extra_margin: Optional[float] = None,
    extra_margin_mode: Optional[str] = None,
    extra_margin_quantile: Optional[float] = None,
    extra_consistency: Optional[Sequence[str]] = None,
    extra_assignment: Optional[str] = None,
    consistency_mode: Optional[str] = None,
    consistency_tau: Optional[float] = None,
    consistency_min_weight: Optional[float] = None,
    attribute_snapshot: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run VI on one pair.

    ``matched=None`` registers all per-class cross-view candidates (VI alone).
    ``matched=[...]`` registers exactly one object per Stage-1 matched item and
    returns records aligned to that list.

    ``extra_links > 0`` additionally registers candidates built from the
    detections the D-S stage left unused: each leftover observation is linked to
    its ``extra_links`` nearest same-class counterparts in the other view. This
    is the "independent data" channel — the unmatched detections get a chance to
    become a confirmed pair without ever re-litigating a D-S decision — and it is
    deliberately bounded (nearest links only, top ``extra_max_observations`` per
    view by detection confidence). Their records come back under ``extras`` and
    are not aligned to ``matched``.
    """
    shared = load_shared_config(resolve_config(config_path), overrides=decision_overrides)
    built = build_observations(pair, detections, config, family, extra_context=extra_context)
    observations = built["observations"]
    claims = built["claims"]
    observations_by_view = built["observations_by_view"]
    capacity = dict(shared.get("association") or {}).get("pair_capacity")
    capacity = None if capacity is None else max(1, int(capacity))
    cascade_cfg = dict(shared.get("cascade") or {})
    # Shipped settings for this family; explicit arguments win when provided.
    shipped = shipped_cascade_settings(shared, family)
    if extra_links is None:
        extra_links = shipped["extra_links"]
    if extra_mode is None:
        extra_mode = shipped["extra_mode"]
    if extra_prior_floor is None:
        extra_prior_floor = shipped["extra_prior_floor"]
    if extra_margin is None:
        extra_margin = shipped["extra_margin"]
    if extra_margin_mode is None:
        extra_margin_mode = shipped["extra_margin_mode"]
    if extra_margin_quantile is None:
        extra_margin_quantile = shipped["extra_margin_quantile"]
    if extra_consistency is None:
        extra_consistency = shipped["extra_consistency"]
    if extra_assignment is None:
        extra_assignment = shipped["extra_assignment"]
    if consistency_mode is None:
        consistency_mode = shipped["consistency_mode"]
    if consistency_tau is None:
        consistency_tau = shipped["consistency_tau"]
    if consistency_min_weight is None:
        consistency_min_weight = shipped["consistency_min_weight"]
    # max_links / center_scale_factor accept either a per-family mapping or a
    # single scalar (the tuner overrides them with scalars).
    max_links_value = cascade_cfg.get("max_links")
    if isinstance(max_links_value, Mapping):
        family_max_links = int(max_links_value.get(family, 4))
    elif max_links_value is None:
        family_max_links = 4
    else:
        family_max_links = int(max_links_value)
    center_value = cascade_cfg.get("center_scale_factor")
    if isinstance(center_value, Mapping):
        family_center = float(center_value.get(family, 0.75))
    elif center_value is None:
        family_center = 0.75
    else:
        family_center = float(center_value)
    objects: List[ObjectHypothesis] = []
    records: List[Dict[str, Any]] = []

    if matched is None:
        for class_name, members in built["by_class"].items():
            member_ids = set(members)
            obs_a = [obs for obs in observations_by_view["a"] if obs.observation_id in member_ids]
            obs_b = [obs for obs in observations_by_view["b"] if obs.observation_id in member_ids]
            objects.extend(_pruned_cross_view_hypotheses(
                obs_a, obs_b, capacity, class_name, built["relation_group"],
                max_links=family_max_links, center_scale_factor=family_center,
            ))
    else:
        by_view_index = built["by_view_index"]
        for index, item in enumerate(matched):
            view_a_id = by_view_index.get(("a", item.get("view_a_index")))
            view_b_id = by_view_index.get(("b", item.get("view_b_index")))
            if view_a_id is None or view_b_id is None:
                records.append({"index": index, "class": item.get("label"), "status": "Uncertain",
                                "value": None, "p_class": 0.0, "association": 0.0, "bbox": None,
                                "error": "missing_member"})
                continue
            view_a_obs = next(obs for obs in observations_by_view["a"] if obs.observation_id == view_a_id)
            view_b_obs = next(obs for obs in observations_by_view["b"] if obs.observation_id == view_b_id)
            features = compute_pair_features(
                view_a_obs.spatial_support.bbox,
                view_b_obs.spatial_support.bbox,
                observation_confidence=view_a_obs.detection_confidence,
                object_confidence=view_b_obs.detection_confidence,
            )
            hypothesis = ObjectHypothesis(
                object_id=stable_id("dual_pair", {"family": family, "index": index, "a": view_a_id, "b": view_b_id}),
                spatial_support=SpatialSupport(bbox=list(view_a_obs.spatial_support.bbox)),
                capacity=None if capacity is None else max(1, capacity),
                attribute_hints={"category": str(item.get("label"))},
                context={
                    "candidate_type": "cross_image_pair",
                    "relation_group_id": built["relation_group"],
                    "member_observation_ids": [view_a_id, view_b_id],
                    "member_image_group_ids": [
                        view_a_obs.context.get("image_group_id"),
                        view_b_obs.context.get("image_group_id"),
                    ],
                    "pair_features": features.to_dict(),
                    "origin": "ds_cascade",
                },
            )
            objects.append(hypothesis)
            records.append({
                "index": index,
                "class": str(item.get("label")),
                "pending_object": hypothesis.object_id,
            })

    # Independent-data channel: leftovers get bounded cross-view candidates.
    extra_first = len(objects)
    if matched is not None and int(extra_links) > 0:
        used: set = set()
        for hypothesis in objects:
            used.update(str(value) for value in (hypothesis.context or {}).get("member_observation_ids") or [])
        cap = max(1, int(extra_max_observations))
        for class_name, member_ids in built["by_class"].items():
            leftover = {str(value) for value in member_ids if str(value) not in used}
            obs_a = [obs for obs in observations_by_view["a"] if obs.observation_id in leftover]
            obs_b = [obs for obs in observations_by_view["b"] if obs.observation_id in leftover]
            if not obs_a or not obs_b:
                continue
            # Bound the cost on dense frames: keep the most confident leftovers.
            obs_a = sorted(obs_a, key=lambda obs: -(obs.detection_confidence or 0.0))[:cap]
            obs_b = sorted(obs_b, key=lambda obs: -(obs.detection_confidence or 0.0))[:cap]
            if str(extra_mode).lower() == "mutual":
                objects.extend(_mutual_cross_view_hypotheses(
                    obs_a, obs_b, capacity, class_name, built["relation_group"],
                    center_scale_factor=family_center,
                    prior_floor=float(extra_prior_floor), origin="ds_cascade_extra",
                ))
            else:
                objects.extend(_pruned_cross_view_hypotheses(
                    obs_a, obs_b, capacity, class_name, built["relation_group"],
                    max_links=int(extra_links), center_scale_factor=family_center,
                    prior_floor=float(extra_prior_floor), origin="ds_cascade_extra",
                    bypass_gates=True, margin=float(extra_margin),
                    margin_mode=str(extra_margin_mode),
                    margin_quantile=float(extra_margin_quantile),
                    assignment=str(extra_assignment),
                ))

    # Attribute/size consistency gate on the independent candidates. The weights
    # come from measured per-(modality, attribute) informativeness, so a channel
    # at chance level (SAR subtype, orientation) abstains instead of vetoing.
    extra_rejected_inconsistent = 0
    if extra_links and extra_consistency and str(consistency_mode).lower() == "weighted":
        # Prefer the source attribute snapshot: reading claims here would force
        # every attribute to be injected as a claim, which splits the provenance
        # budget and weakens the category evidence behind the association.
        attributes_by_observation: Dict[str, Dict[str, Any]] = {
            str(key): dict(value) for key, value in (attribute_snapshot or {}).items()
        }
        if not attributes_by_observation:
            for claim in claims:
                attributes_by_observation.setdefault(claim.observation_id, {})[str(claim.attribute)] = claim.value
        weights = pair_weights(family, extra_consistency)
        kept: List[ObjectHypothesis] = []
        for hypothesis in objects[extra_first:]:
            members = [str(value) for value in (hypothesis.context or {}).get("member_observation_ids") or []]
            if len(members) < 2:
                kept.append(hypothesis)
                continue
            score = consistency(
                attributes_by_observation.get(members[0], {}),
                attributes_by_observation.get(members[1], {}),
                weights,
            )
            if consistency_accept(score, float(consistency_tau), float(consistency_min_weight)):
                kept.append(hypothesis)
            else:
                extra_rejected_inconsistent += 1
        objects = objects[:extra_first] + kept

    if not observations:
        return {"pair_id": str(pair["base_id"]), "family": family, "objects": records, "singles": [],
                "extras": [], "diagnostics": {"observations": 0}}

    store = EvidenceStore(shared)
    for observation in observations:
        store.observations[observation.observation_id] = observation
    for claim in claims:
        store.claims[claim.claim_id] = claim
    for observation_id, claim_payloads in (extra_claims or {}).items():
        observation = store.observations.get(observation_id)
        if observation is None:
            continue
        channel = str(
            (observation.context or {}).get("channel")
            or observation.modality.value
        )
        for payload in claim_payloads:
            attribute = str(payload.get("attribute") or "").strip()
            value = payload.get("value")
            if not attribute or value in (None, ""):
                continue
            claim = build_claim(
                observation,
                attribute,
                value,
                source_channel=payload.get("source_channel") or channel,
                raw_confidence=payload.get("raw_confidence"),
                source_tool=payload.get("source_tool"),
            )
            store.claims[claim.claim_id] = claim
    for hypothesis in objects:
        hypothesis.validate()
        store.objects[hypothesis.object_id] = hypothesis
    store.autoseed_objects = False
    store._recompute_coverage()

    problem = store.build_problem(seed_objects=False)
    register_problem_provenance(problem, store.provenance)
    state = run_joint_inference(problem, store.config, provenance_graph=store.provenance)
    store.inference_state = state
    store._inference_revision = store._revision
    evidence_state = build_evidence_state(store, state)

    extras: List[Dict[str, Any]] = []
    if matched is None:
        for hypothesis in objects:
            class_name = str((hypothesis.attribute_hints or {}).get("category"))
            records.append(_object_record(
                store, evidence_state, hypothesis.object_id, class_name, None, shared, attribute_names,
            ))
    else:
        resolved = []
        for record in records:
            if record.get("error"):
                resolved.append(record)
                continue
            object_id = record.pop("pending_object")
            class_name = str(record.get("class"))
            resolved.append(_object_record(
                store, evidence_state, object_id, class_name, record.get("index"), shared, attribute_names,
            ))
        records = resolved

        # Extra candidates keep their member indices so the cascade can take over
        # the observations they cover instead of emitting them twice.
        index_of = {observation_id: key for key, observation_id in built["by_view_index"].items()}
        edge_counts: Dict[str, int] = {}
        for edge in problem.candidate_edges:
            edge_counts[str(edge.object_id)] = edge_counts.get(str(edge.object_id), 0) + 1
        for hypothesis in objects[extra_first:]:
            class_name = str((hypothesis.attribute_hints or {}).get("category"))
            record = _object_record(
                store, evidence_state, hypothesis.object_id, class_name, None, shared, attribute_names,
            )
            members = [str(value) for value in record.get("members") or []]
            record["view_a_index"] = next(
                (index_of[member][1] for member in members if index_of.get(member, (None,))[0] == "a"), None
            )
            record["view_b_index"] = next(
                (index_of[member][1] for member in members if index_of.get(member, (None,))[0] == "b"), None
            )
            record["kind"] = "extra"
            record["candidate_edges"] = edge_counts.get(hypothesis.object_id, 0)
            extras.append(record)

    # Output completeness: any observation whose best real-object association
    # stays below the residual threshold is reported as a single-view detection.
    # This lets the VI-alone arm express left / appeared changes and recover
    # candidates the cascade handoff left unmatched.
    singles: List[Dict[str, Any]] = []
    if matched is None and store.inference_state is not None:
        residual_threshold = float(cascade_cfg.get("residual_threshold", 0.5))
        for observation in observations:
            bbox = observation.spatial_support.bbox
            if not bbox:
                continue
            row = store.inference_state.association_row(observation.observation_id)
            best_association = 0.0
            for object_id, probability in row.items():
                if object_id == NULL_OBJECT_ID:
                    continue
                best_association = max(best_association, float(probability))
            if best_association >= residual_threshold:
                continue
            singles.append({
                "observation_id": observation.observation_id,
                "class": built["class_of"].get(observation.observation_id),
                "bbox": [round(float(v), 3) for v in bbox],
                "view": str((observation.context or {}).get("channel") or ""),
                "score": round(float(observation.detection_confidence or 0.0), 6),
                "association": round(best_association, 6),
                "source": "vi_single",
            })

    diagnostics = {
        "observations": len(observations),
        "objects": len(objects),
        "extra_candidates": len(extras),
        "extra_rejected_inconsistent": int(extra_rejected_inconsistent),
        "extra_confirmed": sum(
            1 for record in extras
            if str(record.get("status")) in {"Confirmed", "Weak Support"}
            and str(record.get("value")) == str(record.get("class"))
        ),
        "iterations": evidence_state.inference_diagnostics.get("iterations"),
        "converged": evidence_state.inference_diagnostics.get("converged"),
        "cascade_settings": {
            "extra_links": int(extra_links),
            "extra_mode": str(extra_mode),
            "extra_prior_floor": float(extra_prior_floor),
            "extra_margin": float(extra_margin),
            "extra_margin_mode": str(extra_margin_mode),
            "extra_margin_quantile": float(extra_margin_quantile),
            "extra_consistency": list(extra_consistency or ()),
            "extra_assignment": str(extra_assignment),
            "consistency_mode": str(consistency_mode),
            "consistency_tau": float(consistency_tau),
        },
        "mean_null_mass": (
            sum(evidence_state.null_mass.values()) / len(evidence_state.null_mass)
            if evidence_state.null_mass
            else 0.0
        ),
    }
    return {"pair_id": str(pair["base_id"]), "family": family, "objects": records,
            "singles": singles, "extras": extras, "diagnostics": diagnostics}


def vi_boxes(
    records: Mapping[str, Any],
    policy: str = "status",
    threshold: float = 0.0,
    include_singles: bool = True,
) -> List[Dict[str, Any]]:
    predictions = []
    for record in records.get("objects", []):
        if record.get("error"):
            continue
        bbox = record.get("bbox")
        if not bbox:
            continue
        if policy == "status":
            keep = record.get("status") in {"Confirmed", "Weak Support"} and str(record.get("value")) == str(record.get("class"))
            score = float(record.get("confidence") or 0.0)
        elif policy == "posterior":
            keep = float(record.get("p_class") or 0.0) >= float(threshold)
            score = float(record.get("p_class") or 0.0)
        elif policy == "rank":
            rank = float(record.get("p_class") or 0.0) * float(record.get("association") or 0.0)
            keep = rank >= float(threshold)
            score = rank
        else:
            raise ValueError(f"Unsupported VI policy: {policy!r}")
        if not keep:
            continue
        predictions.append({
            "bbox": [float(v) for v in bbox[:4]],
            "label": str(record.get("class")),
            "score": score,
            "source": "vi",
            "vi_status": record.get("status"),
            "p_class": record.get("p_class"),
        })
    if include_singles:
        for single in records.get("singles") or []:
            bbox = single.get("bbox")
            if not bbox:
                continue
            predictions.append({
                "bbox": [float(v) for v in bbox[:4]],
                "label": str(single.get("class")),
                "score": float(single.get("score") or 0.0),
                "source": str(single.get("source") or "vi_single"),
                "vi_status": "single",
                "association": single.get("association"),
            })
    return predictions


def vi_time_objects(
    records: Mapping[str, Any],
    policy: str = "status",
    threshold: float = 0.0,
    include_singles: bool = True,
) -> List[Dict[str, Any]]:
    from DS_VI.models.ds_fusion.pipeline import time_change_label

    objects = []
    for record in records.get("objects", []):
        if record.get("error"):
            continue
        if policy == "status":
            keep = record.get("status") in {"Confirmed", "Weak Support"} and str(record.get("value")) == str(record.get("class"))
            score = float(record.get("confidence") or 0.0)
        elif policy == "posterior":
            keep = float(record.get("p_class") or 0.0) >= float(threshold)
            score = float(record.get("p_class") or 0.0)
        elif policy == "rank":
            rank = float(record.get("p_class") or 0.0) * float(record.get("association") or 0.0)
            keep = rank >= float(threshold)
            score = rank
        else:
            raise ValueError(f"Unsupported VI policy: {policy!r}")
        if not keep:
            continue
        member_boxes = record.get("member_bboxes") or {}
        member_views = record.get("member_views") or {}
        bbox_a = bbox_b = None
        for member, bbox in member_boxes.items():
            if member_views.get(member) == "a":
                bbox_a = bbox
            elif member_views.get(member) == "b":
                bbox_b = bbox
        objects.append({
            "bbox": [float(v) for v in (record["bbox"] or bbox_a or bbox_b)],
            "bbox_t0": [float(v) for v in bbox_a] if bbox_a else None,
            "bbox_t1": [float(v) for v in bbox_b] if bbox_b else None,
            "label": str(record.get("class")),
            "score": score,
            "source": "matched" if bbox_a and bbox_b else "view_a_only" if bbox_a else "view_b_only",
            "change": time_change_label(bbox_a, bbox_b),
            "vi_status": record.get("status"),
            "p_class": record.get("p_class"),
            "association": record.get("association"),
        })
    if include_singles:
        for single in records.get("singles") or []:
            bbox = single.get("bbox")
            if not bbox:
                continue
            view = str(single.get("view") or "")
            bbox_a = [float(v) for v in bbox[:4]] if view == "a" else None
            bbox_b = [float(v) for v in bbox[:4]] if view == "b" else None
            objects.append({
                "bbox": [float(v) for v in bbox[:4]],
                "bbox_t0": bbox_a,
                "bbox_t1": bbox_b,
                "label": str(single.get("class")),
                "score": float(single.get("score") or 0.0),
                "source": "view_a_only" if bbox_a else "view_b_only",
                "change": "left" if bbox_a else "appeared",
                "vi_status": "single",
                "p_class": None,
                "association": single.get("association"),
            })
    return objects
