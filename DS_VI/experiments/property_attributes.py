"""Tool attribute claims for the shared-identity VI.

The detection tools report subtype / color / size_bin / orientation in the
property workbooks, but the trunk only fed ``category`` into VI, so every
attribute except category was invisible to the inference.  This module turns
workbook rows into the ``extra_claims`` payloads
:func:`DS_VI.models.cascade.vi.run_vi_pair` already accepts.

Keys are ``(observation_id, attribute)``: the workbook's ``(pair, view, index)``
maps to an observation through :func:`build_observations`, whose ids are
deterministic, so the same mapping works for the trunk entry point and for the
comparison arms.

Only asserted (non-empty) values become claims; a blank cell contributes
nothing.  ``visibility`` is deliberately *not* injected here: it is acquisition
metadata for the contextual reliability model, not a target attribute, and
wiring it in would activate a theta artifact fitted on this same dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from DS_VI.models.cascade.attributes import VIEW_TAG

#: Attributes injected by default.  ``category`` already arrives through the
#: detector label, so it is not repeated here.  ``color`` replaced the old
#: ``state`` column in the 2026-11 annotation release; ``state`` is still read
#: when a workbook carries it.
DEFAULT_ATTRIBUTES: Tuple[str, ...] = ("subtype", "color", "size_bin", "orientation")

#: Observation-context covariates the contextual reliability model consumes.
#: The property workbooks carry ``visibility`` (clear/partial/poor), which is
#: exactly the feature set the fitted theta artifact expects.
DEFAULT_CONTEXT: Tuple[str, ...] = ("visibility",)

AttributeTable = Dict[str, Dict[str, Dict[int, Dict[str, str]]]]
"""``pair_id -> view('a'|'b') -> detection index -> {attribute: value}``."""


def _view_key(family: str, view: Any) -> Optional[str]:
    """Workbook view (``rgb``/``sar``/``time1``/``time2``) -> ``a``/``b``."""
    text = str(view or "").strip().lower()
    for key, tag in (VIEW_TAG.get(family) or {}).items():
        if str(tag).lower() == text:
            return key
    return None


def load_attribute_table(
    workbook_path: Path,
    family: str,
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
) -> AttributeTable:
    """Read a tool workbook once into a lookup table.

    Row order inside a ``(pair, view)`` block is the detection order of the tool
    JSON, so the position is the detection index the pipeline uses.
    """
    from DS_VI.experiments.metrics.property_metrics import read_book

    book = read_book(Path(workbook_path))
    counters: Dict[Tuple[str, str], int] = {}
    table: AttributeTable = {}
    for row in book.get("instances") or []:
        pair_id = str(row.get("pair_id"))
        raw_view = str(row.get("view"))
        key = (pair_id, raw_view)
        index = counters.get(key, 0)
        counters[key] = index + 1
        view = _view_key(family, raw_view)
        if view is None:
            continue
        values: Dict[str, str] = {}
        for attribute in attributes:
            value = row.get(attribute)
            if value is None:
                continue
            text = str(value).strip().lower()
            if text:
                values[attribute] = text
        if values:
            table.setdefault(pair_id, {}).setdefault(view, {})[index] = values
    return table


def claims_from_table(
    table: AttributeTable,
    pair_id: str,
    by_view_index: Mapping[Tuple[str, int], str],
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    source_tool: str = "property_workbook",
) -> Dict[str, List[Dict[str, Any]]]:
    """``observation_id -> claim payloads`` for one pair."""
    payloads: Dict[str, List[Dict[str, Any]]] = {}
    for view, by_index in (table.get(str(pair_id)) or {}).items():
        for index, values in by_index.items():
            observation_id = by_view_index.get((view, index))
            if observation_id is None:
                continue
            for attribute in attributes:
                value = values.get(attribute)
                if value in (None, ""):
                    continue
                payloads.setdefault(observation_id, []).append({
                    "attribute": str(attribute),
                    "value": value,
                    "source_channel": view,
                    "source_tool": source_tool,
                })
    return payloads


def load_context_table(
    workbook_path: Path,
    family: str,
    context_keys: Sequence[str] = DEFAULT_CONTEXT,
) -> AttributeTable:
    """Read the workbook context covariates (``visibility``) per detection."""
    from DS_VI.experiments.metrics.property_metrics import read_book

    book = read_book(Path(workbook_path))
    counters: Dict[Tuple[str, str], int] = {}
    table: AttributeTable = {}
    for row in book.get("instances") or []:
        pair_id = str(row.get("pair_id"))
        raw_view = str(row.get("view"))
        key = (pair_id, raw_view)
        index = counters.get(key, 0)
        counters[key] = index + 1
        view = _view_key(family, raw_view)
        if view is None:
            continue
        values: Dict[str, str] = {}
        for name in context_keys:
            value = row.get(name)
            if value is None:
                continue
            text = str(value).strip().lower()
            if text:
                values[name] = text
        if values:
            table.setdefault(pair_id, {}).setdefault(view, {})[index] = values
    return table


def context_from_table(
    table: AttributeTable,
    pair_id: str,
    by_view_index: Mapping[Tuple[str, int], str],
) -> Dict[str, Dict[str, Any]]:
    """``observation_id -> context covariates`` for one pair."""
    context: Dict[str, Dict[str, Any]] = {}
    for view, by_index in (table.get(str(pair_id)) or {}).items():
        for index, values in by_index.items():
            observation_id = by_view_index.get((view, index))
            if observation_id is not None and values:
                context[observation_id] = dict(values)
    return context


def context_from_items(
    items: Iterable[Mapping[str, Any]],
    by_view_index: Mapping[Tuple[str, int], str],
    context_keys: Sequence[str] = DEFAULT_CONTEXT,
) -> Dict[str, Dict[str, Any]]:
    """Same as :func:`context_from_table`, for normalized comparison rows."""
    context: Dict[str, Dict[str, Any]] = {}
    for item in items:
        observation_id = by_view_index.get((str(item.get("view")), int(item.get("index", -1))))
        if observation_id is None:
            continue
        row = item.get("row") or {}
        values = {}
        for name in context_keys:
            value = row.get(name)
            if value is None:
                continue
            text = str(value).strip().lower()
            if text:
                values[name] = text
        if values:
            context[observation_id] = values
    return context


def claims_from_items(
    items: Iterable[Mapping[str, Any]],
    by_view_index: Mapping[Tuple[str, int], str],
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    source_tool: str = "property_workbook",
) -> Dict[str, List[Dict[str, Any]]]:
    """Same as :func:`claims_from_table`, for callers holding normalized rows.

    ``items`` are mappings with ``view``, ``index``, ``attributes`` and
    ``score`` (the comparison layer's normalized detections).
    """
    payloads: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        observation_id = by_view_index.get((str(item.get("view")), int(item.get("index", -1))))
        if observation_id is None:
            continue
        values = item.get("attributes") or {}
        for attribute in attributes:
            value = values.get(attribute)
            if value in (None, ""):
                continue
            payloads.setdefault(observation_id, []).append({
                "attribute": str(attribute),
                "value": value,
                "source_channel": str(item.get("view")),
                "source_tool": source_tool,
                "raw_confidence": float(item.get("score") or 0.0),
            })
    return payloads


def attributes_from_table(
    table: AttributeTable,
    pair_id: str,
    by_view_index: Mapping[Tuple[str, int], str],
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
) -> Dict[str, Dict[str, str]]:
    """``observation_id -> {attribute: value}`` for one pair.

    This is the *source* view of the tool attributes: it feeds the
    consistency gate without turning every attribute into a claim. Injecting
    them as claims would split the provenance budget across five axes and
    weaken the category evidence that drives association.
    """
    snapshot: Dict[str, Dict[str, str]] = {}
    for view, by_index in (table.get(str(pair_id)) or {}).items():
        for index, values in by_index.items():
            observation_id = by_view_index.get((view, index))
            if observation_id is None:
                continue
            kept = {name: values[name] for name in attributes if values.get(name)}
            if kept:
                snapshot[observation_id] = kept
    return snapshot


def attributes_from_items(
    items: Iterable[Mapping[str, Any]],
    by_view_index: Mapping[Tuple[str, int], str],
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
) -> Dict[str, Dict[str, str]]:
    """Same as :func:`attributes_from_table`, for normalized comparison rows."""
    snapshot: Dict[str, Dict[str, str]] = {}
    for item in items:
        observation_id = by_view_index.get((str(item.get("view")), int(item.get("index", -1))))
        if observation_id is None:
            continue
        values = item.get("attributes") or {}
        kept = {name: values[name] for name in attributes if values.get(name)}
        if kept:
            snapshot[observation_id] = kept
    return snapshot


def observation_ids(
    pair: Mapping[str, Any],
    detections: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    family: str,
) -> Dict[Tuple[str, int], str]:
    """``(view, detection index) -> observation id`` for one pair."""
    from DS_VI.models.cascade.vi import build_observations

    return dict(build_observations(pair, detections, config, family)["by_view_index"])


def attribute_names(attributes: Sequence[str] = DEFAULT_ATTRIBUTES) -> Tuple[str, ...]:
    """``attribute_names`` for ``run_vi_pair``: the injected axes plus category."""
    names = ["category"]
    for attribute in attributes:
        if attribute not in names:
            names.append(str(attribute))
    return tuple(names)


__all__ = [
    "DEFAULT_ATTRIBUTES",
    "AttributeTable",
    "attribute_names",
    "attributes_from_items",
    "attributes_from_table",
    "claims_from_items",
    "claims_from_table",
    "context_from_items",
    "context_from_table",
    "load_context_table",
    "load_attribute_table",
    "observation_ids",
]
