"""Data layer for the 5-fold cross-fitting experiment (exp_3).

Everything here is read-only: the module loads the frozen tool/truth property
workbooks, builds the claim-level table the contextual reliability model is
fitted on, and assigns pair-grouped folds.

Why the claim table is rebuilt instead of shipped: the artifact records
``data/shared_identity_real_eval_ctx_clean.json`` (sha256 pinned), and that file
is not part of the repository.  Rebuilding it from the workbooks is the only
reproducible route inside this checkout; ``build_claim_rows`` documents every
mapping decision so the reconstruction can be audited.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Dataset root.  Overridable through the documented env var.
DATA_ROOT = Path(os.environ.get("ONEID_DATA_ROOT") or (REPO_ROOT / "data"))

#: family -> workbook pair / view tags used by the identities sheet.
FAMILIES: Dict[str, Dict[str, Any]] = {
    "match": {
        "tool": "tool_task4_property.xlsx",
        "truth": "truth_task4_property.xlsx",
        "views": {"a": "rgb", "b": "sar"},
        "image_dir": "match_images",
    },
    "time": {
        "tool": "tool_task5_property.xlsx",
        "truth": "truth_task5_property.xlsx",
        "views": {"a": "time1", "b": "time2"},
        "image_dir": "time_images",
    },
}

#: Attribute axes carried by the property workbooks.  ``category`` reaches the
#: engine through the detector label, the rest through the attribute table.
DEFAULT_ATTRIBUTES: Tuple[str, ...] = ("category", "subtype", "color", "size_bin", "orientation")

#: IoU used to join a tool detection to its annotated truth object.  This is the
#: frozen instance-level join of ``property_metrics.match_image``.
INSTANCE_IOU = 0.5

#: Views whose modality key is ``sar``; everything else is optical.
_SAR_VIEWS = {"sar"}


def family_views(family: str) -> Mapping[str, str]:
    return FAMILIES[family]["views"]


def tool_workbook(family: str, data_root: Path = DATA_ROOT) -> Path:
    return Path(data_root) / FAMILIES[family]["tool"]


def truth_workbook(family: str, data_root: Path = DATA_ROOT) -> Path:
    return Path(data_root) / FAMILIES[family]["truth"]


def modality_of(family: str, view: str) -> str:
    """``rgb``/``sar``/``time1``/``time2`` -> ``optical`` / ``sar``.

    This mirrors ``shared_identity.contextual_reliability.claim_modality``; the
    fit and the engine must agree or the coefficients land on the wrong channel.
    """
    return "sar" if str(view).strip().lower() in _SAR_VIEWS else "optical"


def load_book(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read one workbook into ``{sheet: [row dicts]}``."""
    from DS_VI.experiments.metrics.property_metrics import read_book

    return read_book(Path(path))


def _norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


@dataclass
class ClaimRow:
    """One attribute claim of one detection against its annotated truth object."""

    pair_id: str
    view: str
    modality: str
    attribute: str
    value: Optional[str]
    truth: Optional[str]
    correct: int
    visibility: str
    detection_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "view": self.view,
            "modality": self.modality,
            "attribute": self.attribute,
            "value": self.value,
            "truth": self.truth,
            "correct": int(self.correct),
            "visibility": self.visibility,
            "detection_count": int(self.detection_count),
        }

    def context(self) -> Dict[str, Any]:
        """The observation context handed to ``extract_features``."""
        return {"visibility": self.visibility, "detection_count": self.detection_count}


def build_claim_rows(
    family: str,
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    data_root: Path = DATA_ROOT,
    iou_threshold: float = INSTANCE_IOU,
) -> List[ClaimRow]:
    """Join tool detections to annotated objects and emit claim-level rows.

    A row exists only when *both* sides assert the attribute: a blank cell means
    "not annotated / not reported", not "wrong".  ``correct`` is therefore the
    agreement of two asserted values, which is exactly what the reliability
    model estimates -- the probability that this source is right.
    """
    from DS_VI.experiments.metrics.property_metrics import match_image

    tool = load_book(tool_workbook(family, data_root))
    truth = load_book(truth_workbook(family, data_root))

    tool_images: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in tool.get("instances") or []:
        tool_images[(str(row.get("pair_id")), str(row.get("view")))].append(row)
    truth_images: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in truth.get("instances") or []:
        truth_images[(str(row.get("pair_id")), str(row.get("view")))].append(row)

    rows: List[ClaimRow] = []
    for key in sorted(tool_images):
        pair_id, view = key
        detections = tool_images[key]
        annotations = truth_images.get(key) or []
        if not annotations:
            continue
        record = match_image(detections, annotations, float(iou_threshold))
        detection_count = len(detections)
        for match in record.get("matches") or []:
            source = match["tool"]
            gold = match["truth"]
            visibility = _norm(source.get("visibility")) or "missing"
            for attribute in attributes:
                claimed = _norm(source.get(attribute))
                annotated = _norm(gold.get(attribute))
                if claimed is None or annotated is None:
                    continue
                rows.append(ClaimRow(
                    pair_id=pair_id,
                    view=view,
                    modality=modality_of(family, view),
                    attribute=str(attribute),
                    value=claimed,
                    truth=annotated,
                    correct=int(claimed == annotated),
                    visibility=visibility,
                    detection_count=detection_count,
                ))
    return rows


def summarize_rows(rows: Sequence[ClaimRow]) -> Dict[str, Any]:
    """Support / accuracy per (modality, attribute) cell -- the fit's audit trail."""
    cells: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        bucket = cells[row.modality].setdefault(
            row.attribute, {"support": 0, "correct": 0})
        bucket["support"] += 1
        bucket["correct"] += row.correct
    out: Dict[str, Any] = {"rows": len(rows), "pairs": len({r.pair_id for r in rows}),
                           "cells": {}}
    for modality, per_attribute in sorted(cells.items()):
        out["cells"][modality] = {}
        for attribute, bucket in sorted(per_attribute.items()):
            support = bucket["support"]
            out["cells"][modality][attribute] = {
                "support": support,
                "accuracy": round(bucket["correct"] / support, 6) if support else None,
            }
    return out


def load_all_claims(
    families: Sequence[str],
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    data_root: Path = DATA_ROOT,
) -> List[ClaimRow]:
    rows: List[ClaimRow] = []
    for family in families:
        rows.extend(build_claim_rows(family, attributes, data_root))
    return rows


def available_pairs(family: str, data_root: Path = DATA_ROOT) -> List[str]:
    """Pairs that have both workbooks' rows -- the evaluation universe."""
    tool = load_book(tool_workbook(family, data_root))
    truth = load_book(truth_workbook(family, data_root))
    tool_pairs = {str(row.get("pair_id")) for row in tool.get("instances") or []}
    truth_pairs = {str(row.get("pair_id")) for row in truth.get("instances") or []}
    return sorted(tool_pairs & truth_pairs)


def pair_family(pair_id: str) -> str:
    """``task4_*`` -> match, ``task5_*`` -> time."""
    return "match" if str(pair_id).startswith("task4") else "time"


def count_pairs(rows: Sequence[ClaimRow]) -> Counter:
    return Counter(row.pair_id for row in rows)


__all__ = [
    "ClaimRow",
    "DATA_ROOT",
    "DEFAULT_ATTRIBUTES",
    "FAMILIES",
    "INSTANCE_IOU",
    "REPO_ROOT",
    "available_pairs",
    "build_claim_rows",
    "count_pairs",
    "family_views",
    "load_all_claims",
    "load_book",
    "modality_of",
    "pair_family",
    "summarize_rows",
    "tool_workbook",
    "truth_workbook",
]
