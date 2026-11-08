"""Pair-grouped k-fold assignment.

Grouping unit is the **pair** (both views and their annotations stay on the same
side of the split).  Splitting at instance level would leak the scene, its
illumination and its annotation conventions into the training fold, and the
cross-fitted number would stop being a held-out estimate.

``group_key`` accepts a future site/cluster mapping: pass ``{pair_id: group}``
and whole groups move together.  Until the RSTIBench scene metadata is available
the default is the identity mapping, i.e. one group per pair -- documented in
``README.md`` as the remaining gap for the "holdout-sites" wording.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _shuffle(items: List[str], seed: int) -> List[str]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def make_folds(
    pairs: Sequence[str],
    k: int = 5,
    seed: int = 20261107,
    *,
    family: Optional[Mapping[str, str]] = None,
    group_key: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, object]]:
    """Return ``[{"fold": i, "train": [...], "test": [...]}, ...]``.

    ``family`` maps ``pair_id -> 'match'|'time'``.  Folds are balanced per family
    first (round-robin over the shuffled ids) so a fold is never all-match or
    all-time; the fold index is then re-sorted for stable output.
    """
    pair_ids = [str(pair) for pair in pairs]
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(pair_ids) < k:
        raise ValueError(f"not enough pairs ({len(pair_ids)}) for {k} folds")

    if family is None:
        family = {pair: ("match" if pair.startswith("task4") else "time") for pair in pair_ids}
    if group_key is None:
        group_key = {pair: pair for pair in pair_ids}

    groups: Dict[str, List[str]] = defaultdict(list)
    for pair in pair_ids:
        groups[str(group_key.get(pair, pair))].append(pair)

    # Balance by group size: largest group first, each into the lightest fold.
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    buckets: List[List[str]] = [[] for _ in range(k)]
    for index, (_group, members) in enumerate(ordered_groups):
        target = index % k
        buckets[target].extend(members)

    folds: List[Dict[str, object]] = []
    for index, test in enumerate(buckets):
        test_set = sorted(set(test))
        train = sorted(set(pair_ids) - set(test_set))
        folds.append({
            "fold": index,
            "train": train,
            "test": test_set,
            "train_by_family": _by_family(train, family),
            "test_by_family": _by_family(test_set, family),
        })
    return folds


def _by_family(pairs: Sequence[str], family: Mapping[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for pair in pairs:
        counts[str(family.get(pair, "unknown"))] += 1
    return dict(sorted(counts.items()))


def describe(folds: Sequence[Mapping[str, object]]) -> str:
    lines = [f"{len(folds)} folds"]
    for fold in folds:
        lines.append(
            f"  fold {fold['fold']}: test={len(fold['test'])} {fold['test_by_family']} "
            f"train={len(fold['train'])} {fold['train_by_family']}"
        )
    return "\n".join(lines)


__all__ = ["describe", "make_folds"]
