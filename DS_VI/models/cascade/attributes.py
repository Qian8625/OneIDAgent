"""Cascade v2 adapter: D-S matched candidates -> VI identity with annotation attributes.

Attributes come from an explicit ctx dataset rebuilt from the workbook and
current detections, with the selected noise rate. Detector categories are not
corrupted. This is mechanism replay using annotation-derived attributes.

Only matched object candidates are registered, but all observations enter VI.
The original adapter reads attribute claims; ctx covariates are not forwarded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from DS_VI.models.cascade.status import object_status
from DS_VI.models.cascade.vi import build_observations, run_vi_pair


VIEW_TAG = {"match": {"a": "RGB", "b": "SAR"}, "time": {"a": "time1", "b": "time2"}}
#: Attribute axes this legacy ctx-replay adapter requests.  ``color`` replaced
#: ``state`` on the 2026-11 sheets; ``state`` is still listed so an older ctx
#: dataset keeps working (missing values simply contribute nothing).
ATTRIBUTES = {
    "match": ("subtype", "color", "size_bin"),
    "time": ("subtype", "color", "state", "size_bin"),
}

#: Observation-context covariates the contextual reliability model consumes.
#: Only these may travel from the ctx dataset into an observation's context;
#: everything else in the ctx stays out of the model.
CONTEXT_COVARIATE_KEYS = (
    "visibility",
    "evidence_quality",
    "detection_count",
    "registration_offset_norm",
    "registration_counterpart_present",
)


def observation_id_for(base_id: str, family: str, view: str, index: Any) -> str:
    tag = VIEW_TAG[family][view]
    return f"{base_id}-{tag}-{int(index):03d}"


class CtxIndex:
    """Index the frozen ctx dataset by sample / observation."""

    def __init__(self, path: Optional[Path] = None, payload: Optional[Mapping[str, Any]] = None):
        if payload is None:
            if path is None:
                raise ValueError("An explicit ctx path or payload is required")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.samples = {sample["sample_id"]: sample for sample in payload["samples"]}
        self._claims: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for sample_id, sample in self.samples.items():
            by_observation: Dict[str, List[Dict[str, Any]]] = {}
            for claim in sample.get("claims") or []:
                by_observation.setdefault(str(claim["observation_id"]), []).append(dict(claim))
            self._claims[sample_id] = by_observation
        self._contexts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for sample_id, sample in self.samples.items():
            by_observation_context: Dict[str, Dict[str, Any]] = {}
            for observation in sample.get("observations") or []:
                observation_id = str(observation.get("observation_id") or "")
                if not observation_id:
                    continue
                by_observation_context[observation_id] = dict(observation.get("context") or {})
            self._contexts[sample_id] = by_observation_context

    def observation_context(
        self,
        sample_id: str,
        family: str,
        view: str,
        index: Any,
    ) -> Dict[str, Any]:
        """Whitelisted context covariates of one ctx observation.

        Returns only the keys ``ReliabilityModel`` actually reads, so the ctx's
        annotation-derived extras never leak into the inference.  Missing keys
        are simply absent: the feature extractor turns that into ``*_missing``.
        """
        observation_id = observation_id_for(sample_id, family, view, index)
        context = (self._contexts.get(sample_id) or {}).get(observation_id) or {}
        return {
            key: context[key]
            for key in CONTEXT_COVARIATE_KEYS
            if context.get(key) is not None
        }

    def has_sample(self, sample_id: str) -> bool:
        return sample_id in self.samples

    def sample_ids(self) -> List[str]:
        return sorted(self.samples)

    def gt_assignment(self, sample_id: str) -> Dict[str, str]:
        return dict((self.samples.get(sample_id) or {}).get("gt_assignment") or {})

    def gt_truth(self, sample_id: str) -> Dict[str, Any]:
        return dict((self.samples.get(sample_id) or {}).get("gt_truth") or {})

    def annotation_claims(
        self,
        sample_id: str,
        family: str,
        view: str,
        index: Any,
        attributes: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        wanted = set(attributes or ATTRIBUTES[family])
        observation_id = observation_id_for(sample_id, family, view, index)
        claims = self._claims.get(sample_id, {}).get(observation_id, [])
        return [
            {
                "attribute": str(claim.get("attribute")),
                "value": claim.get("value"),
                "raw_confidence": claim.get("raw_confidence"),
            }
            for claim in claims
            if str(claim.get("attribute")) in wanted
        ]


def run_cascade_attr_pair(
    pair: Mapping[str, Any],
    detections: Mapping[str, List[Dict[str, Any]]],
    matched: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    family: str,
    ctx: CtxIndex,
    attributes: Optional[Sequence[str]] = None,
    decision_overrides: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the cascade confirmation with annotation attribute claims attached.

    Returns the VI result plus a ``claim_coverage`` block (how many matched
    member observations found ctx attribute claims).
    """
    attrs = tuple(attributes or ATTRIBUTES[family])
    built = build_observations(pair, detections, config, family)
    by_view_index = built["by_view_index"]
    extra_claims: Dict[str, List[Dict[str, Any]]] = {}
    extra_context: Dict[str, Dict[str, Any]] = {}
    coverage = {
        "members": 0,
        "members_with_claims": 0,
        "members_with_context": 0,
        "objects_with_any_claim": 0,
        "objects_with_any_context": 0,
    }
    for item in matched:
        object_has_claims = False
        object_has_context = False
        for view, index_key in (("a", "view_a_index"), ("b", "view_b_index")):
            index = item.get(index_key)
            observation_id = by_view_index.get((view, index))
            if observation_id is None:
                continue
            coverage["members"] += 1
            claims = ctx.annotation_claims(pair["base_id"], family, view, index, attrs)
            if claims:
                coverage["members_with_claims"] += 1
                object_has_claims = True
                extra_claims.setdefault(observation_id, []).extend(claims)
            covariates = ctx.observation_context(pair["base_id"], family, view, index)
            if covariates:
                coverage["members_with_context"] += 1
                object_has_context = True
                extra_context[observation_id] = covariates
        if object_has_claims:
            coverage["objects_with_any_claim"] += 1
        if object_has_context:
            coverage["objects_with_any_context"] += 1

    # Theta is a per-claim reliability, not a matched-only correction: every
    # observation of the pair gets its covariates, matched or not.
    for (view, index), observation_id in by_view_index.items():
        if observation_id in extra_context:
            continue
        covariates = ctx.observation_context(pair["base_id"], family, view, index)
        if covariates:
            extra_context[observation_id] = covariates

    kwargs: Dict[str, Any] = {
        "matched": matched,
        "attribute_names": attrs,
        "extra_claims": extra_claims,
        "extra_context": extra_context,
    }
    if decision_overrides is not None:
        kwargs["decision_overrides"] = decision_overrides
    if config_path is not None:
        kwargs["config_path"] = config_path
    result = run_vi_pair(pair, detections, config, family, **kwargs)
    result["claim_coverage"] = coverage
    return result


STATUS_ACCEPT = {"Confirmed", "Weak Support"}


def _vi_keeps(vi: Mapping[str, Any], item: Mapping[str, Any], policy: str, threshold: float):
    expected = str(item.get("label"))
    if policy == "status":
        keep = str(vi.get("status")) in STATUS_ACCEPT and str(vi.get("value")) == expected
        score = float(vi.get("confidence") or 0.0)
    elif policy == "posterior":
        keep = float(vi.get("p_class") or 0.0) >= float(threshold)
        score = float(vi.get("p_class") or 0.0)
    elif policy == "rank":
        keep = float(vi.get("p_class") or 0.0) * float(vi.get("association") or 0.0) >= float(threshold)
        score = float(vi.get("p_class") or 0.0) * float(vi.get("association") or 0.0)
    elif policy == "blend":
        # Keep every D-S candidate (no recall loss) and re-rank it by mixing the
        # D-S fusion score with the VI object posterior. AP@50 is rank-based, so
        # a better ordering raises F1 without deleting true positives.
        base = float(item.get("betp") or item.get("score") or 0.0)
        weight = float(threshold)
        if weight <= 0.0 or weight >= 1.0:
            weight = 0.5
        keep = True
        score = (1.0 - weight) * base + weight * float(vi.get("p_class") or 0.0)
    elif policy == "all":
        keep = True
        score = float(vi.get("p_class") or 0.0) * max(float(vi.get("association") or 0.0), 1e-6)
    else:
        raise ValueError(f"Unsupported cascade policy: {policy!r}")
    return keep, score

def _single_keys(item: Mapping[str, Any]) -> set:
    """``(view, index)`` keys a single-view item occupies."""
    keys = set()
    for view, column in (("a", "view_a_index"), ("b", "view_b_index")):
        value = item.get(column)
        if value is not None:
            keys.add((view, int(value)))
    return keys


def _time_change_of(record: Mapping[str, Any]) -> Optional[str]:
    """Change label of a two-member record, from its member boxes."""
    from DS_VI.models.ds_fusion.pipeline import time_change_label

    boxes = dict(record.get("member_bboxes") or {})
    views = dict(record.get("member_views") or {})
    bbox_a = next((bbox for member, bbox in boxes.items() if views.get(member) == "a"), None)
    bbox_b = next((bbox for member, bbox in boxes.items() if views.get(member) == "b"), None)
    if bbox_a is None or bbox_b is None:
        return None
    return time_change_label(bbox_a, bbox_b)


def _extra_object(record: Mapping[str, Any], extra_policy: str) -> Optional[Dict[str, Any]]:
    """Turn one VI candidate record into a prediction object, or ``None``."""
    if record.get("error"):
        return None
    if extra_policy == "all":
        keep, score = True, float(record.get("confidence") or 0.0)
    elif extra_policy == "posterior":
        keep, score = bool(float(record.get("p_class") or 0.0) >= 0.5), float(record.get("p_class") or 0.0)
    else:
        # ``confirmed``: an extra candidate is only reported when VI confirms it.
        keep, _ = _vi_keeps(record, {"label": record.get("class")}, "status", 0.0)
        score = float(record.get("confidence") or 0.0)
    if not keep:
        return None
    return {
        "label": str(record.get("class")),
        "bbox": list(record.get("bbox") or []),
        "score": score,
        "betp": score,
        "change": _time_change_of(record),
        "source": "vi_extra",
        "view_a_index": record.get("view_a_index"),
        "view_b_index": record.get("view_b_index"),
        "vi_status": record.get("status"),
        "vi_p_class": record.get("p_class"),
        "vi_association": record.get("association"),
        "attribute_beliefs": dict(record.get("attribute_beliefs") or {}),
    }


def _resolve_family(
    family: Optional[str],
    matched: Sequence[Mapping[str, Any]],
    single: Sequence[Mapping[str, Any]],
) -> str:
    """Family of the assembled objects: explicit, else read off the D-S records."""
    if family:
        return str(family)
    for item in list(matched) + list(single):
        value = item.get("family")
        if value:
            return str(value)
    return "match"


def _with_status(item: Dict[str, Any], family: str) -> Dict[str, Any]:
    """Attach the reported status label to one output object."""
    has_a = item.get("view_a_index") is not None
    has_b = item.get("view_b_index") is not None
    change = item.get("change")
    if change is None and family == "time" and has_a and has_b:
        change = "stayed"
    item["status"] = object_status(family, has_a, has_b, change)
    return item


def cascade_objects(
    matched: Sequence[Mapping[str, Any]],
    single: Sequence[Mapping[str, Any]],
    vi_records: Sequence[Mapping[str, Any]],
    policy: str = "status",
    threshold: float = 0.0,
    extra_records: Optional[Sequence[Mapping[str, Any]]] = None,
    extra_policy: str = "confirmed",
    family: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Assemble the prediction objects of one pair.

    ``matched``/``vi_records`` are the Stage-1 pairs and their VI verdicts.
    ``extra_records`` are VI candidates built from detections the D-S stage left
    unused; they are reported only when VI confirms them (``extra_policy``),
    and a confirmed one takes over its member observations, so the same box is
    never emitted twice (once as a single, once as a confirmed pair).

    Every emitted object carries a ``status`` from
    :mod:`DS_VI.models.cascade.status`: ``match/rgb-only/sar-only/unknown`` for
    the match family, ``static/moved/time1-only/time2-only`` for the time family.
    """
    extras: List[Dict[str, Any]] = []
    taken: set = set()
    for record in extra_records or []:
        entry = _extra_object(record, extra_policy)
        if entry is None or not entry["bbox"]:
            continue
        extras.append(entry)
        if entry["view_a_index"] is not None:
            taken.add(("a", int(entry["view_a_index"])))
        if entry["view_b_index"] is not None:
            taken.add(("b", int(entry["view_b_index"])))

    final: List[Dict[str, Any]] = [
        dict(item) for item in single
        if not (_single_keys(item) & taken)
    ]
    for index, item in enumerate(matched):
        vi = vi_records[index] if index < len(vi_records) else None
        if vi is None or vi.get("error"):
            continue
        keep, score = _vi_keeps(vi, item, policy, threshold)
        if not keep:
            continue
        kept = dict(item)
        kept["score"] = score
        kept["vi_status"] = vi.get("status")
        kept["vi_p_class"] = vi.get("p_class")
        kept["vi_association"] = vi.get("association")
        kept["attribute_beliefs"] = dict(vi.get("attribute_beliefs") or {})
        final.append(kept)
    final.extend(extras)
    resolved = _resolve_family(family, matched, single)
    return [_with_status(dict(item), resolved) for item in final]
