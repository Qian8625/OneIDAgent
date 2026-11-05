"""Provenance graph, group weight budgets, and exact-replay deduplication.

Two failure modes this module prevents:

1. **Correlated evidence double-counting.**  One image -> one detector -> one crop
   -> three attribute claims is *one* piece of evidence, not three.  Claims in a
   derivation group share a weight budget of at most 1.
2. **Replay amplification.**  Re-running the same tool with the same normalized
   inputs must not strengthen the posterior.  Identical fingerprints contribute
   once.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.schema import (
    Claim,
    InferenceProblem,
    PrimitiveObservation,
    ProvenanceNode,
    SchemaValidationError,
    fingerprint,
)


def normalize_tool_inputs(payload: Any) -> Any:
    """Canonicalize a tool payload for fingerprinting.

    Volatile keys (ids, timestamps, output paths, round counters) are dropped so
    that a genuine replay of the same request fingerprints identically.
    """
    volatile = {
        "request_id",
        "round_id",
        "timestamp",
        "elapsed",
        "worker_id",
        "artifacts",
        "result_id",
        "visualization",
        "output_path",
        "trace_id",
        "uuid",
    }
    if isinstance(payload, Mapping):
        return {
            str(key): normalize_tool_inputs(value)
            for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
            if str(key) not in volatile
        }
    if isinstance(payload, (list, tuple)):
        return [normalize_tool_inputs(item) for item in payload]
    if isinstance(payload, float):
        return round(payload, 6)
    return payload


def compute_evidence_fingerprint(
    tool_name: str,
    normalized_inputs: Any,
    parameters: Optional[Mapping[str, Any]] = None,
    input_roots: Optional[Sequence[str]] = None,
) -> str:
    """Stable fingerprint over (tool, normalized inputs, parameters, roots)."""
    return fingerprint(
        {
            "tool": str(tool_name),
            "inputs": normalize_tool_inputs(normalized_inputs),
            "parameters": normalize_tool_inputs(parameters or {}),
            "roots": sorted(str(r) for r in (input_roots or [])),
        }
    )


class ProvenanceGraph:
    """Derivation DAG over observations and claims, with replay bookkeeping."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        cfg = dict(config or load_config())
        section = dict(cfg.get("provenance") or {})
        self.group_budget = float(section.get("group_budget", 1.0))
        self.dedup_exact_replay = bool(section.get("dedup_exact_replay", True))
        self.grouping_mode = str(section.get("grouping_mode", "connected_component"))
        self.remove_deterministic_restatements = bool(
            section.get("remove_deterministic_restatements", True)
        )
        self.nodes: Dict[str, ProvenanceNode] = {}
        self._fingerprint_owner: Dict[str, str] = {}

    def register(
        self,
        node_id: str,
        kind: str,
        tool_name: str,
        fingerprint_value: str,
        parents: Optional[Sequence[str]] = None,
        input_roots: Optional[Sequence[str]] = None,
        group_id: Optional[str] = None,
        round_id: Optional[int] = None,
        duplicate_of: Optional[str] = None,
    ) -> ProvenanceNode:
        """Register a node, marking it a duplicate if its fingerprint recurs.

        Returns the stored node.  When the fingerprint has been seen before and
        dedup is enabled, ``duplicate_of`` points at the original owner and the
        node will receive zero weight.

        Args:
            duplicate_of: Forces the node to be treated as a replay of the given
                node.  Callers use this when they detect a replay at a level the
                fingerprint cannot see, such as a nondeterministic tool that
                returns jittered output for an identical call.
        """
        if duplicate_of is None and fingerprint_value and self.dedup_exact_replay:
            owner = self._fingerprint_owner.get(fingerprint_value)
            if owner is not None and owner != node_id:
                duplicate_of = owner
            else:
                self._fingerprint_owner.setdefault(fingerprint_value, node_id)

        node = ProvenanceNode(
            node_id=node_id,
            kind=kind,
            tool_name=tool_name,
            fingerprint=fingerprint_value,
            parents=list(parents or []),
            input_roots=list(input_roots or []),
            group_id=group_id,
            round_id=round_id,
            duplicate_of=duplicate_of,
        )
        node.validate()
        self.nodes[node_id] = node
        return node

    def is_duplicate(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        return bool(node and node.duplicate_of)

    def roots_of(self, node_id: str, _seen: Optional[Set[str]] = None) -> List[str]:
        """Transitively resolve the input roots feeding ``node_id``."""
        seen = _seen if _seen is not None else set()
        if node_id in seen:
            return []
        seen.add(node_id)
        node = self.nodes.get(node_id)
        if node is None:
            return []
        roots = list(node.input_roots)
        for parent in node.parents:
            roots.extend(self.roots_of(parent, seen))
        # Preserve order while removing duplicates.
        deduped: List[str] = []
        for root in roots:
            if root not in deduped:
                deduped.append(root)
        return deduped

    def summarize(self) -> Dict[str, Any]:
        duplicates = [n.node_id for n in self.nodes.values() if n.duplicate_of]
        return {
            "node_count": len(self.nodes),
            "duplicate_count": len(duplicates),
            "duplicate_nodes": duplicates,
            "distinct_fingerprints": len(self._fingerprint_owner),
            "group_budget": self.group_budget,
            "grouping_mode": self.grouping_mode,
            "remove_deterministic_restatements": self.remove_deterministic_restatements,
        }


def register_problem_provenance(
    problem: InferenceProblem,
    graph: ProvenanceGraph,
    replays: Optional[Mapping[str, str]] = None,
    auto_detect_replays: bool = False,
) -> Dict[str, Any]:
    """Register observation and claim nodes for one inference problem.

    This is the evaluator-side mirror of ``EvidenceStore.ingest_tool_response``:
    nodes are registered so ``compute_group_weights`` can see duplicates, and
    ``replays`` forces a node to be treated as a replay of another when the
    caller declared it (a nondeterministic tool returning jittered output for an
    identical call).

    ``auto_detect_replays`` is off by default.  A detection-content fingerprint
    (tool + acquisition + bbox) cannot distinguish a genuine re-issued call from
    two identical detections inside one call, and EvidenceStore fingerprints the
    call arguments instead.  Callers that really have call-level fingerprints
    may opt in; everyone else must declare replays explicitly rather than have
    legitimate duplicate detections silently suppressed.

    Without this registration the graph stays empty, ``is_duplicate`` is always
    false, and the provenance variant is indistinguishable from the baseline.
    """
    forced = dict(replays or {})
    claims_by_observation: Dict[str, List[Claim]] = {}
    for claim in problem.claims:
        claims_by_observation.setdefault(claim.observation_id, []).append(claim)

    registered = 0
    duplicates = 0
    for observation in problem.observations:
        observation_fingerprint = compute_evidence_fingerprint(
            observation.source_tool,
            {
                "acquisition_id": observation.acquisition_id,
                "bbox": list(observation.spatial_support.bbox or []),
                "time": observation.time_scope.key,
            },
            input_roots=list(observation.input_roots),
        )
        node_fingerprint = (
            observation_fingerprint
            if auto_detect_replays
            else f"{observation_fingerprint}:{observation.observation_id}"
        )
        graph.register(
            node_id=observation.observation_id,
            kind="observation",
            tool_name=observation.source_tool,
            fingerprint_value=node_fingerprint,
            parents=list(observation.derivation_parents),
            input_roots=list(observation.input_roots),
            duplicate_of=forced.get(observation.observation_id),
        )
        registered += 1
        owner = graph.nodes[observation.observation_id].duplicate_of
        if owner:
            duplicates += 1
        for claim in claims_by_observation.get(observation.observation_id, []):
            claim_fingerprint = compute_evidence_fingerprint(
                observation.source_tool,
                {
                    "observation": claim.observation_id,
                    "attribute": claim.attribute,
                    "value": claim.value,
                    "time": claim.time_scope.key,
                },
                input_roots=list(observation.input_roots),
            )
            node_fingerprint = (
                claim_fingerprint
                if auto_detect_replays
                else f"{claim_fingerprint}:{claim.claim_id}"
            )
            graph.register(
                node_id=claim.claim_id,
                kind="claim",
                tool_name=observation.source_tool,
                fingerprint_value=node_fingerprint,
                parents=[observation.observation_id],
                input_roots=list(observation.input_roots),
                duplicate_of=forced.get(claim.claim_id) or owner,
            )
            registered += 1

    summary = graph.summarize()
    return {
        "registered_nodes": registered,
        "duplicate_nodes": duplicates,
        "graph": summary,
    }


def build_observation_components(
    observations: Mapping[str, PrimitiveObservation],
) -> Dict[str, str]:
    """Connected components of the observation dependence graph.

    Two observations are connected when they share any recorded raw-input root
    or when one is an explicit derivation parent of the other.  The component
    id is the lexicographically smallest observation id, so the grouping is
    deterministic regardless of input ordering.  Membership depends only on
    recorded ancestry and roots, never on which object currently wins.
    """
    parent: Dict[str, str] = {observation_id: observation_id for observation_id in observations}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # Keep the smaller id as the root so the component id is canonical.
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    roots_index: Dict[str, str] = {}
    for observation_id in sorted(observations):
        observation = observations[observation_id]
        for root in observation.input_roots or []:
            if root in roots_index:
                union(roots_index[root], observation_id)
            else:
                roots_index[root] = observation_id
    for observation_id in sorted(observations):
        observation = observations[observation_id]
        for ancestor in observation.derivation_parents or []:
            if ancestor in parent:
                union(ancestor, observation_id)

    return {observation_id: f"grp_cc_{find(observation_id)}" for observation_id in observations}


def deterministic_restatement_ids(claims: Sequence[Claim]) -> Set[str]:
    """Claims that restate another claim deterministically.

    Detection is explicit, never inferred from coincidentally equal values: a
    claim marked ``deterministic_restatement_of`` (or belonging to an
    observation marked ``derivation_kind=deterministic_transform``) is excluded
    as an additional likelihood term.  Inferring restatements from equal values
    would suppress legitimate duplicate detections.
    """
    claim_ids = {claim.claim_id for claim in claims}
    restatements: Set[str] = set()
    for claim in claims:
        target = claim.context.get("deterministic_restatement_of")
        if target and str(target) in claim_ids:
            restatements.add(claim.claim_id)
    return restatements


def restatements_from_observations(
    claims: Sequence[Claim],
    observations: Mapping[str, PrimitiveObservation],
) -> Set[str]:
    """Extend explicit claim-level restatements with observation-level marks."""
    restatements = deterministic_restatement_ids(claims)
    deterministic_observations = {
        observation_id
        for observation_id, observation in observations.items()
        if str(observation.context.get("derivation_kind") or "").lower()
        == "deterministic_transform"
    }
    if deterministic_observations:
        for claim in claims:
            if claim.observation_id in deterministic_observations:
                restatements.add(claim.claim_id)
    return restatements


def assign_group_ids(
    claims: Sequence[Claim],
    observations: Mapping[str, PrimitiveObservation],
    grouping_mode: str = "connected_component",
) -> Dict[str, str]:
    """Assign a derivation group to every claim that lacks one.

    ``grouping_mode``:
      * ``connected_component`` (default, paper A.19): claims join the component
        of their observation under shared raw-input roots and derivation edges.
      * ``root``: same graph, but roots are not transitively merged through
        derivation parents (kept for a paired ablation).
      * ``observation``: legacy per-observation budget.
    Explicit ``provenance_group_id`` values set by an adapter are respected and
    win over the computed grouping.
    """
    mode = str(grouping_mode or "connected_component").lower()
    if mode not in ("connected_component", "root", "observation"):
        raise SchemaValidationError(
            f"unknown provenance grouping mode: {grouping_mode!r}"
        )
    if mode == "connected_component":
        observation_groups = build_observation_components(observations)
    elif mode == "root":
        observation_groups = {}
        roots_index: Dict[str, str] = {}
        for observation_id in sorted(observations):
            observation = observations[observation_id]
            component = None
            for root in observation.input_roots or []:
                if root in roots_index:
                    component = roots_index[root]
                    break
            if component is None:
                component = f"grp_cc_{observation_id}"
            for root in observation.input_roots or []:
                roots_index.setdefault(root, component)
            observation_groups[observation_id] = component
    else:
        observation_groups = {
            observation_id: f"grp_{observation_id}" for observation_id in observations
        }

    assigned: Dict[str, str] = {}
    for claim in claims:
        if claim.provenance_group_id:
            assigned[claim.claim_id] = claim.provenance_group_id
            continue
        group_id = observation_groups.get(
            claim.observation_id, f"grp_{claim.observation_id}"
        )
        claim.provenance_group_id = group_id
        assigned[claim.claim_id] = group_id
    return assigned


def compute_group_weights(
    claims: Sequence[Claim],
    graph: Optional[ProvenanceGraph] = None,
    group_budget: float = 1.0,
    budget_enabled: bool = True,
    restatements: Optional[Set[str]] = None,
    informativeness: Optional[Mapping[str, float]] = None,
    weighting: str = "uniform",
) -> Dict[str, float]:
    """Compute ``varpi_i`` for each claim under its group budget.

    Within each group the budget is split evenly (``weighting="uniform"``,
    ``varpi_i = budget / |g|``) or in proportion to ``informativeness``
    (``weighting="informativeness"``), with exact replays and deterministic
    restatements given weight ``0`` and excluded from the group size so they
    neither add nor dilute evidence.

    ``informativeness`` is the chance-normalized discriminative power of a claim,
    ``max(0, (r - 1/C) / (1 - 1/C))``. A channel at chance level therefore
    receives zero share, which is what stops a chance-level attribute from
    diluting the category evidence that drives association. When every claim in a
    group is uninformative the split falls back to uniform, so evidence is never
    silently erased. Without a budget the uniform rule applies.

    Guarantees ``sum(varpi_i) <= group_budget`` for every group.
    """
    budget = float(group_budget if graph is None else graph.group_budget)
    removed = set(restatements or ())
    use_informativeness = (
        str(weighting).lower() == "informativeness"
        and bool(informativeness)
        and budget_enabled
    )

    groups: Dict[str, List[Claim]] = {}
    for claim in claims:
        group_id = claim.provenance_group_id or f"grp_{claim.observation_id}"
        groups.setdefault(group_id, []).append(claim)

    weights: Dict[str, float] = {}
    for group_claims in groups.values():
        effective = [
            claim
            for claim in group_claims
            if claim.claim_id not in removed
            and (graph is None or not graph.is_duplicate(claim.claim_id))
        ]
        for claim in group_claims:
            if claim not in effective:
                weights[claim.claim_id] = 0.0
        if not effective:
            continue
        if not budget_enabled:
            for claim in effective:
                weights[claim.claim_id] = 1.0
            continue
        if use_informativeness:
            values = [
                max(float((informativeness or {}).get(claim.claim_id, 0.0)), 0.0)
                for claim in effective
            ]
            total = sum(values)
            if total > 0.0:
                for claim, value in zip(effective, values, strict=True):
                    weights[claim.claim_id] = budget * (value / total)
                continue
        share = budget / len(effective)
        for claim in effective:
            weights[claim.claim_id] = share
    return weights


def verify_group_budget(
    claims: Sequence[Claim],
    weights: Mapping[str, float],
    group_budget: float = 1.0,
) -> Dict[str, Any]:
    """Confirm no derivation group exceeds its weight budget."""
    sums: Dict[str, float] = {}
    for claim in claims:
        group_id = claim.provenance_group_id or f"grp_{claim.observation_id}"
        sums[group_id] = sums.get(group_id, 0.0) + float(weights.get(claim.claim_id, 0.0))
    violations = {
        group_id: total
        for group_id, total in sums.items()
        if total > float(group_budget) + 1e-6
    }
    return {"ok": not violations, "group_sums": sums, "violations": violations}


__all__ = [
    "ProvenanceGraph",
    "normalize_tool_inputs",
    "compute_evidence_fingerprint",
    "register_problem_provenance",
    "build_observation_components",
    "deterministic_restatement_ids",
    "restatements_from_observations",
    "assign_group_ids",
    "compute_group_weights",
    "verify_group_budget",
]
