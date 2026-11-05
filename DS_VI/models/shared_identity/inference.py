"""Joint shared-identity association (A) and object truth (Q) inference.

The structural core of the revision.  Two coupled variables:

* ``A[m, k]`` — probability that **observation** ``m`` is object ``k``.  One row
  per observation, *not* per claim.
* ``Q[k, a, t, v]`` — probability that object ``k``'s attribute ``a`` at time
  ``t`` has value ``v``.

Update for A, accumulating **all** claims of an observation into that single row:

.. code-block:: text

    S[m,k] = sum_{i in claims(m)} varpi_i * E_Q[log P(y_i | Q[k,a_i,t_i], r_i)]
    log A_tilde[m,k] = log p[m,k] + S[m,k] / eta
    A_tilde[m,:] = softmax(log A_tilde[m,:])

then a KL capacity projection, then the Q update:

.. code-block:: text

    Q[k,a,t,v] ∝ prior[k,a,t,v] * exp(sum_i varpi_i * A[m(i),k] * log P(y_i | v, r_i))

No decision labels are produced here.  That is :mod:`decision`'s job.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from DS_VI.models.shared_identity.candidate_graph import candidate_mask as build_candidate_mask
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.likelihood import (
    DEFAULT_EPSILON,
    expected_log_likelihood,
    normalize_log_weights,
    safe_log,
    truth_log_posterior,
)
from DS_VI.models.shared_identity.physical_prior import compute_physical_prior, prior_matrix
from DS_VI.models.shared_identity.projection import (
    build_capacities,
    project_association_block_kl,
    project_association_kl,
)
from DS_VI.models.shared_identity.provenance import (
    ProvenanceGraph,
    assign_group_ids,
    compute_group_weights,
    restatements_from_observations,
)
from DS_VI.models.shared_identity.reliability import ReliabilityModel
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    Claim,
    InferenceProblem,
    InferenceState,
    PrimitiveObservation,
    SchemaValidationError,
)

TruthKey = Tuple[str, str, str]  # (object_id, attribute, time_key)

#: Truth time bucket used by attributes that cannot change between views.
INVARIANT_TIME = "*"


def truth_time(claim: Claim, invariant: Optional[Set[str]] = None) -> str:
    """Time bucket a claim speaks about.

    ``category``/``subtype``/``size_bin``/``orientation`` describe the target
    itself, so a cross-view pair asserting the same object twice would
    otherwise split its evidence over two truth cells (``t0`` and ``t1``),
    halving the posterior and starving the association update. Configured
    time-invariant attributes pool into one bucket instead; ``state`` and
    ``motion`` stay per-time because they genuinely change.
    """
    if invariant and str(claim.attribute) in invariant:
        return INVARIANT_TIME
    return claim.time_scope.key


def _value_domain(
    attribute: str,
    claims: Sequence[Claim],
    configured: Mapping[str, Sequence[Any]],
) -> List[Any]:
    """Value domain for an attribute: configured values plus observed values.

    Observed-but-unconfigured values are appended so a tool can never assert a
    value that the truth variable has no room for.
    """
    domain: List[Any] = [v for v in (configured.get(attribute) or [])]
    seen = {str(v).strip().lower() for v in domain}
    for claim in claims:
        if claim.attribute != attribute:
            continue
        key = str(claim.value).strip().lower()
        if key not in seen:
            seen.add(key)
            domain.append(claim.value)
    return domain or ["unknown"]


def build_value_domains(
    problem: InferenceProblem,
    config: Mapping[str, Any],
) -> Dict[str, List[Any]]:
    """Resolve the value domain of every attribute present in the problem."""
    configured: Dict[str, Sequence[Any]] = dict(config.get("value_domains") or {})
    configured.update({k: list(v) for k, v in (problem.value_domains or {}).items()})
    attributes = {claim.attribute for claim in problem.claims}
    return {
        attribute: _value_domain(attribute, problem.claims, configured)
        for attribute in sorted(attributes)
    }


def _truth_keys(
    problem: InferenceProblem,
    object_ids: Sequence[str],
    invariant: Optional[Set[str]] = None,
) -> List[TruthKey]:
    """Every ``(object, attribute, time)`` triple the claims can speak about."""
    axes = sorted({(claim.attribute, truth_time(claim, invariant)) for claim in problem.claims})
    return [
        (object_id, attribute, time_key)
        for object_id in object_ids
        for attribute, time_key in axes
    ]


def initialize_truth(
    problem: InferenceProblem,
    object_ids: Sequence[str],
    domains: Mapping[str, List[Any]],
    smoothing: float = 1.0,
    invariant: Optional[Set[str]] = None,
) -> Dict[TruthKey, Dict[Any, float]]:
    """Initialize Q from the configured priors, defaulting to uniform."""
    truth: Dict[TruthKey, Dict[Any, float]] = {}
    for key in _truth_keys(problem, object_ids, invariant):
        _, attribute, _ = key
        domain = domains.get(attribute) or ["unknown"]
        prior = problem.truth_priors.get(key)
        if prior:
            total = sum(float(p) for p in prior.values())
            if total > 0.0:
                truth[key] = {v: float(prior.get(v, 0.0)) / total for v in domain}
                continue
        share = 1.0 / len(domain)
        truth[key] = {value: share for value in domain}
    return truth


def truth_priors_for(
    problem: InferenceProblem,
    domains: Mapping[str, List[Any]],
    smoothing: float,
) -> Dict[TruthKey, Dict[Any, float]]:
    """Smoothed priors used inside the Q update."""
    priors: Dict[TruthKey, Dict[Any, float]] = {}
    for key, prior in problem.truth_priors.items():
        _, attribute, _ = key
        domain = domains.get(attribute) or list(prior.keys())
        alpha = max(float(smoothing), 0.0)
        raw = {v: float(prior.get(v, 0.0)) + alpha for v in domain}
        total = sum(raw.values())
        if total > 0.0:
            priors[key] = {v: p / total for v, p in raw.items()}
    return priors


#: Observation-context keys the contextual reliability model reads.  Kept here
#: so the coverage diagnostic counts exactly what the model consumes.
RELIABILITY_CONTEXT_KEYS = (
    "visibility",
    "evidence_quality",
    "detection_count",
    "registration_offset_norm",
    "registration_counterpart_present",
)


def _reliability_context_coverage(problem: InferenceProblem) -> Dict[str, Any]:
    """Share of claims whose observation carries usable context covariates.

    A theta-enabled run with coverage 0 means every theta cell falls back to the
    "all covariates missing" value: the switch is on but changes nothing.
    """
    by_observation = {observation.observation_id: observation for observation in problem.observations}
    total = 0
    with_context = 0
    per_attribute: Dict[str, int] = {}
    for claim in problem.claims:
        total += 1
        observation = by_observation.get(claim.observation_id)
        context = dict((observation.context if observation is not None else {}) or {})
        if any(context.get(key) is not None for key in RELIABILITY_CONTEXT_KEYS):
            with_context += 1
            per_attribute[str(claim.attribute)] = per_attribute.get(str(claim.attribute), 0) + 1
    return {
        "claims": total,
        "with_context": with_context,
        "ratio": (with_context / total) if total else 0.0,
        "attributes_with_context": dict(sorted(per_attribute.items())),
    }


def update_association(
    problem: InferenceProblem,
    observation_ids: Sequence[str],
    object_ids: Sequence[str],
    prior: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
    truth: Mapping[TruthKey, Mapping[Any, float]],
    reliabilities: Mapping[str, float],
    weights: Mapping[str, float],
    domains: Mapping[str, List[Any]],
    eta: float,
    epsilon: float,
    outlier_mixture: float = 0.0,
    evidence_cap: Optional[float] = None,
    feedback_damping: float = 1.0,
    feedback_attributes: Optional[Sequence[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    invariant: Optional[Set[str]] = None,
) -> List[List[float]]:
    """One A update: accumulate every claim of an observation into its single row.

    The shared-identity invariant lives in this function.  ``claims_by_observation``
    is iterated per observation, and the resulting ``S[m, k]`` is applied to the
    whole row.  There is no per-claim association anywhere.

    Robustness controls, all neutral by default:

    * ``outlier_mixture`` bounds each claim's likelihood ratio.
    * ``evidence_cap`` clips the per-claim contribution to ``[-cap, +cap]`` so a
      single claim cannot dominate the row; ``diagnostics`` receives the number
      of clipped contributions when supplied.
    * ``feedback_damping`` scales the whole truth-derived evidence term.  At 0.0
      the row is the physical prior, which is the no-feedback baseline.
    """
    claims_by_observation = problem.claims_by_observation()
    temperature = max(float(eta), epsilon)
    cap = None if evidence_cap is None else abs(float(evidence_cap))
    damping = max(float(feedback_damping), 0.0)
    clipped = 0
    updated: List[List[float]] = []

    for m, observation_id in enumerate(observation_ids):
        observation_claims = claims_by_observation.get(observation_id) or []
        log_row: List[float] = []
        for k, object_id in enumerate(object_ids):
            if not mask[m][k]:
                log_row.append(float("-inf"))
                continue

            evidence = 0.0
            if object_id != NULL_OBJECT_ID:
                # Shared identity: one accumulation loop for all claims of m.
                for claim in observation_claims:
                    if feedback_attributes is not None and claim.attribute not in feedback_attributes:
                        continue
                    weight = float(weights.get(claim.claim_id, 0.0))
                    if weight <= 0.0:
                        continue
                    truth_key = (object_id, claim.attribute, truth_time(claim, invariant))
                    distribution = truth.get(truth_key)
                    if not distribution:
                        continue
                    domain_size = len(domains.get(claim.attribute) or distribution)
                    contribution = weight * expected_log_likelihood(
                        claim.value,
                        distribution,
                        float(reliabilities.get(claim.claim_id, 0.5)),
                        domain_size,
                        epsilon,
                        outlier_mixture,
                    )
                    if cap is not None and abs(contribution) > cap:
                        contribution = cap if contribution > 0.0 else -cap
                        clipped += 1
                    evidence += contribution
            else:
                # NULL carries no attribute evidence: an unassigned observation
                # explains its claims at the domain-uniform rate. The cap applies
                # here too; capping only real objects would bias rows toward NULL.
                for claim in observation_claims:
                    if feedback_attributes is not None and claim.attribute not in feedback_attributes:
                        continue
                    weight = float(weights.get(claim.claim_id, 0.0))
                    if weight <= 0.0:
                        continue
                    domain_size = max(len(domains.get(claim.attribute) or []), 1)
                    contribution = weight * safe_log(1.0 / domain_size, epsilon)
                    if cap is not None and abs(contribution) > cap:
                        contribution = cap if contribution > 0.0 else -cap
                        clipped += 1
                    evidence += contribution

            log_row.append(
                safe_log(prior[m][k], epsilon) + damping * evidence / temperature
            )

        probabilities = normalize_log_weights(log_row)
        updated.append(
            [p if mask[m][k] else 0.0 for k, p in enumerate(probabilities)]
        )
    if diagnostics is not None:
        diagnostics["clipped_claim_contributions"] = clipped
    return updated


def update_truth(
    problem: InferenceProblem,
    observation_ids: Sequence[str],
    object_ids: Sequence[str],
    association: Sequence[Sequence[float]],
    reliabilities: Mapping[str, float],
    weights: Mapping[str, float],
    domains: Mapping[str, List[Any]],
    priors: Mapping[TruthKey, Mapping[Any, float]],
    epsilon: float,
    outlier_mixture: float = 0.0,
    invariant: Optional[Set[str]] = None,
) -> Dict[TruthKey, Dict[Any, float]]:
    """One Q update: each ``(object, attribute, time)`` normalized independently."""
    row_of = {obs_id: m for m, obs_id in enumerate(observation_ids)}
    claims_by_axis: Dict[Tuple[str, str], List[Claim]] = {}
    for claim in problem.claims:
        claims_by_axis.setdefault((claim.attribute, truth_time(claim, invariant)), []).append(claim)

    truth: Dict[TruthKey, Dict[Any, float]] = {}
    for k, object_id in enumerate(object_ids):
        if object_id == NULL_OBJECT_ID:
            continue
        for (attribute, time_key), axis_claims in claims_by_axis.items():
            domain = domains.get(attribute) or ["unknown"]
            weighted: List[Tuple[Any, float, float]] = []
            for claim in axis_claims:
                m = row_of.get(claim.observation_id)
                if m is None:
                    continue
                weight = float(weights.get(claim.claim_id, 0.0)) * float(association[m][k])
                if weight <= 0.0:
                    continue
                weighted.append(
                    (claim.value, weight, float(reliabilities.get(claim.claim_id, 0.5)))
                )
            key: TruthKey = (object_id, attribute, time_key)
            truth[key] = truth_log_posterior(
                domain, weighted, priors.get(key), epsilon, outlier_mixture
            )
    return truth


def _objective(
    problem: InferenceProblem,
    observation_ids: Sequence[str],
    object_ids: Sequence[str],
    association: Sequence[Sequence[float]],
    prior: Sequence[Sequence[float]],
    truth: Mapping[TruthKey, Mapping[Any, float]],
    reliabilities: Mapping[str, float],
    weights: Mapping[str, float],
    domains: Mapping[str, List[Any]],
    epsilon: float,
    outlier_mixture: float = 0.0,
    invariant: Optional[Set[str]] = None,
) -> float:
    """Expected complete-data log-likelihood; monitored for convergence.

    The objective tracks the likelihood the updates actually optimize, so the
    outlier mixture enters here as well.  ``evidence_cap`` and
    ``feedback_damping`` deliberately do not: they modify the A update rather
    than the model, so the reported objective stays comparable across them.
    """
    row_of = {obs_id: m for m, obs_id in enumerate(observation_ids)}
    total = 0.0
    for m in range(len(observation_ids)):
        for k in range(len(object_ids)):
            a = float(association[m][k])
            if a > 0.0:
                total += a * safe_log(prior[m][k], epsilon)
    for claim in problem.claims:
        m = row_of.get(claim.observation_id)
        if m is None:
            continue
        weight = float(weights.get(claim.claim_id, 0.0))
        if weight <= 0.0:
            continue
        reliability = float(reliabilities.get(claim.claim_id, 0.5))
        domain_size = max(len(domains.get(claim.attribute) or []), 1)
        for k, object_id in enumerate(object_ids):
            a = float(association[m][k])
            if a <= 0.0:
                continue
            if object_id == NULL_OBJECT_ID:
                total += a * weight * safe_log(1.0 / domain_size, epsilon)
                continue
            distribution = truth.get((object_id, claim.attribute, truth_time(claim, invariant)))
            if not distribution:
                continue
            total += a * weight * expected_log_likelihood(
                claim.value, distribution, reliability, domain_size, epsilon,
                outlier_mixture,
            )
    return total


def _run_joint_inference_single(
    problem: InferenceProblem,
    config: Optional[Mapping[str, Any]] = None,
    provenance_graph: Optional[ProvenanceGraph] = None,
    validate: bool = True,
) -> InferenceState:
    """Run the alternating A / projection / Q loop for one relation group.

    Args:
        problem: Fully populated :class:`InferenceProblem`.  When
            ``candidate_edges`` is empty the graph is not inferred; callers
            should build it with :func:`candidate_graph.build_candidate_graph`.
        config: Optional shared-identity config.
        provenance_graph: Optional graph supplying replay dedup and group budget.
        validate: Whether to run full schema validation first.

    Returns:
        An :class:`InferenceState` with A, Q, the reliability table, objective
        and max-delta histories, iteration count, convergence flag, and
        diagnostics.  No decision labels are included.
    """
    cfg = dict(config or load_config())
    inference_cfg = dict(cfg.get("inference") or {})
    eta = float(inference_cfg.get("eta", 1.0))
    max_iter = int(inference_cfg.get("max_iter", 50))
    tol = float(inference_cfg.get("tol", 1e-5))
    epsilon = float(inference_cfg.get("epsilon", DEFAULT_EPSILON))
    smoothing = float(inference_cfg.get("truth_prior_smoothing", 1.0))
    # None means "no per-object column cap" (paper Eq.(29) caps per block).
    raw_default_capacity = inference_cfg.get("default_capacity", 2)
    default_capacity = (
        None if raw_default_capacity is None else int(raw_default_capacity)
    )
    capacity_projection = bool(inference_cfg.get("capacity_projection", True))
    capacity_mode = str(inference_cfg.get("capacity_mode", "legacy_global"))
    auto_block_scoped_capacity = bool(
        inference_cfg.get("auto_block_scoped_capacity", True)
    )
    block_capacity = float(inference_cfg.get("block_capacity", 1.0))
    projection_cfg = dict(inference_cfg.get("projection") or {})
    raw_cap = inference_cfg.get("evidence_cap")
    evidence_cap = None if raw_cap is None else abs(float(raw_cap))
    feedback_damping = float(inference_cfg.get("feedback_damping", 1.0))
    # ``constant`` reproduces the fixed relaxation factor. ``ramp`` starts at zero
    # and reaches ``feedback_damping`` after ``damping_ramp_iterations`` rounds:
    # while Q is still (near) uniform, the evidence push is not informative — the
    # first round can move A *away* from the true object — so it enters gradually
    # as the attribute posterior sharpens.
    damping_schedule = str(inference_cfg.get("damping_schedule", "constant") or "constant").lower()
    damping_ramp_iterations = max(1, int(inference_cfg.get("damping_ramp_iterations", 3)))
    invariant = {str(value) for value in (inference_cfg.get("time_invariant_attributes") or [])}
    feedback_attributes_raw = inference_cfg.get("feedback_attributes")
    feedback_attributes = (
        [str(value) for value in feedback_attributes_raw]
        if feedback_attributes_raw is not None
        else None
    )
    association_cfg = dict(cfg.get("association") or {})
    scoped_association = (
        str(association_cfg.get("mode", "legacy_global"))
        == "cross_image_independent"
        and bool(association_cfg.get("forbid_same_image", True))
    )
    scope_has_image_metadata = any(
        bool(
            observation.context.get("image_group_id")
            or observation.acquisition_id
        )
        for observation in problem.observations
    )
    if (
        auto_block_scoped_capacity
        and scoped_association
        and scope_has_image_metadata
        and capacity_mode == "legacy_global"
    ):
        # A global object capacity still permits two detections from the same
        # image to consume the pair identity.  Use image blocks so one object
        # receives at most one row per image while retaining RGB/SAR capacity.
        capacity_mode = "block_object_column"
    if feedback_damping < 0.0:
        raise SchemaValidationError(
            f"inference.feedback_damping must be >= 0, got {feedback_damping}"
        )

    observations: Dict[str, PrimitiveObservation] = {
        observation.observation_id: observation for observation in problem.observations
    }

    # Reliability and value domains are needed *before* the provenance budget:
    # with ``provenance.weighting: informativeness`` a claim's share is
    # proportional to how much it can distinguish values (``r - 1/C``), so a
    # chance-level channel (e.g. orientation) stops diluting the informative ones.
    reliability_model = ReliabilityModel(cfg)
    reliabilities = reliability_model.build_table(problem.claims, observations)
    outlier_mixture = reliability_model.outlier_mixture
    context_coverage = _reliability_context_coverage(problem)
    domains = build_value_domains(problem, cfg)

    # Provenance grouping and weights must exist before validation checks the
    # group budget invariant.
    provenance_cfg = dict(cfg.get("provenance") or {})
    grouping_mode = str(provenance_cfg.get("grouping_mode", "connected_component"))
    remove_restatements = bool(
        provenance_cfg.get("remove_deterministic_restatements", True)
    )
    assign_group_ids(problem.claims, observations, grouping_mode=grouping_mode)
    restatements = (
        restatements_from_observations(problem.claims, observations)
        if remove_restatements
        else set()
    )
    weighting = str(provenance_cfg.get("weighting", "uniform") or "uniform").lower()
    informativeness: Dict[str, float] = {}
    if weighting == "informativeness":
        for claim in problem.claims:
            reliability = float(reliabilities.get(claim.claim_id, 0.5))
            domain_size = max(len(domains.get(claim.attribute) or []), 2)
            chance = 1.0 / float(domain_size)
            informativeness[claim.claim_id] = max(
                0.0, (reliability - chance) / max(1.0 - chance, 1e-9)
            )
    weights = compute_group_weights(
        problem.claims,
        provenance_graph,
        float(provenance_cfg.get("group_budget", 1.0)),
        budget_enabled=bool(provenance_cfg.get("budget_enabled", True)),
        restatements=restatements,
        informativeness=informativeness or None,
        weighting=weighting,
    )
    problem.provenance_weights = dict(weights)
    problem.provenance_budget_enabled = bool(provenance_cfg.get("budget_enabled", True))
    group_sizes: Dict[str, int] = {}
    weight_totals: Dict[str, float] = {}
    for claim in problem.claims:
        group_id = claim.provenance_group_id or f"grp_{claim.observation_id}"
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
        weight_totals[group_id] = weight_totals.get(group_id, 0.0) + float(
            weights.get(claim.claim_id, 0.0)
        )

    if validate:
        problem.validate()

    observation_ids = [observation.observation_id for observation in problem.observations]
    object_ids = problem.object_ids(include_null=True)

    priors = truth_priors_for(problem, domains, smoothing)

    if not observation_ids:
        return InferenceState(
            reliability=reliabilities,
            converged=True,
            diagnostics={
                "reason": "no_observations",
                "reliability_model": reliability_model.describe(),
                "reliability_context_coverage": context_coverage,
            },
        )

    physical_prior = compute_physical_prior(problem.candidate_edges, cfg, problem.objects)
    prior = prior_matrix(physical_prior, observation_ids, object_ids)
    mask = build_candidate_mask(problem.candidate_edges, observation_ids, object_ids)

    # Every row must retain at least NULL so it can be normalized.
    null_index = object_ids.index(NULL_OBJECT_ID)
    for m in range(len(observation_ids)):
        if not any(mask[m]):
            mask[m][null_index] = True
        if sum(prior[m]) <= 0.0:
            prior[m][null_index] = 1.0

    capacities = (
        build_capacities(object_ids, NULL_OBJECT_ID, problem.capacities, default_capacity)
        if capacity_projection
        else [None] * len(object_ids)
    )
    if capacity_projection and capacity_mode == "block_object_column":
        # The block projection uses ``None`` to mark the unconstrained NULL
        # column, so an uncapped real column must be ``inf`` rather than None:
        # otherwise the block capacity would never bind.
        capacities = [
            None
            if object_id == NULL_OBJECT_ID
            else (math.inf if capacities[index] is None else capacities[index])
            for index, object_id in enumerate(object_ids)
        ]

    association = [list(row) for row in prior]
    truth = initialize_truth(problem, object_ids, domains, smoothing, invariant)

    objective_history: List[float] = []
    max_delta_history: List[float] = []
    projection_diagnostics: Dict[str, Any] = {}
    association_diagnostics: Dict[str, Any] = {}
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        previous = [list(row) for row in association]
        if damping_schedule == "ramp":
            damping_effective = feedback_damping * min(
                1.0, float(iteration) / float(damping_ramp_iterations)
            )
        else:
            damping_effective = feedback_damping

        a_tilde = update_association(
            problem, observation_ids, object_ids, prior, mask, truth,
            reliabilities, weights, domains, eta, epsilon,
            outlier_mixture=outlier_mixture,
            evidence_cap=evidence_cap,
            feedback_damping=damping_effective,
            feedback_attributes=feedback_attributes,
            diagnostics=association_diagnostics,
            invariant=invariant,
        )
        if capacity_projection and capacity_mode == "block_object_column":
            block_ids = []
            for observation in problem.observations:
                if scoped_association and scope_has_image_metadata:
                    block_ids.append(
                        str(
                            observation.context.get("image_group_id")
                            or observation.acquisition_id
                            # No image metadata at all: keep the observation its own
                            # block so one row per block stays satisfiable.
                            or observation.observation_id
                        )
                    )
                else:
                    channel = str(
                        observation.context.get("channel")
                        or observation.modality.value
                        or "unknown"
                    )
                    acquisition = (
                        observation.acquisition_id or observation.observation_id
                    )
                    block_ids.append(f"{acquisition}|{channel}")
            association, projection_diagnostics = project_association_block_kl(
                a_tilde,
                capacities,
                mask,
                block_ids,
                block_capacity=block_capacity,
                max_iter=int(projection_cfg.get("max_iter", 100)),
                tol=float(projection_cfg.get("tol", 1e-6)),
            )
        else:
            association, projection_diagnostics = project_association_kl(
                a_tilde,
                capacities,
                mask,
                max_iter=int(projection_cfg.get("max_iter", 100)),
                tol=float(projection_cfg.get("tol", 1e-6)),
            )
        truth = update_truth(
            problem, observation_ids, object_ids, association,
            reliabilities, weights, domains, priors, epsilon,
            outlier_mixture=outlier_mixture, invariant=invariant,
        )

        max_delta = 0.0
        for m in range(len(observation_ids)):
            for k in range(len(object_ids)):
                max_delta = max(max_delta, abs(association[m][k] - previous[m][k]))
        max_delta_history.append(max_delta)
        objective_history.append(
            _objective(
                problem, observation_ids, object_ids, association, prior,
                truth, reliabilities, weights, domains, epsilon,
                outlier_mixture, invariant=invariant,
            )
        )
        if max_delta <= tol:
            converged = True
            break

    association_out = {
        observation_id: {
            object_ids[k]: association[m][k]
            for k in range(len(object_ids))
            if association[m][k] > 0.0
        }
        for m, observation_id in enumerate(observation_ids)
    }

    diagnostics = {
        "eta": eta,
        "tol": tol,
        "max_iter": max_iter,
        "epsilon": epsilon,
        "object_ids": list(object_ids),
        "observation_ids": list(observation_ids),
        "value_domains": {k: list(v) for k, v in domains.items()},
        "physical_prior": {k: dict(v) for k, v in physical_prior.items()},
        "projection": dict(projection_diagnostics),
        "projection_feasible": bool(projection_diagnostics.get("converged", False)),
        "capacity_projection": capacity_projection,
        "capacity_mode": capacity_mode,
        "outlier_mixture": outlier_mixture,
        "reliability_context_coverage": context_coverage,
        "reliability_shrinkage": reliability_model.shrinkage,
        "evidence_cap": evidence_cap,
        "feedback_damping": feedback_damping,
        "damping_schedule": damping_schedule,
        "damping_ramp_iterations": damping_ramp_iterations,
        "final_damping": damping_effective,
        "prior_scale": float((cfg.get("physical") or {}).get("prior_scale", 6.0)),
        "provenance_weighting": str(provenance_cfg.get("weighting", "uniform") or "uniform"),
        "clipped_claim_contributions": int(
            association_diagnostics.get("clipped_claim_contributions", 0)
        ),
        "provenance_weights": dict(weights),
        "provenance_budget_enabled": bool(provenance_cfg.get("budget_enabled", True)),
        "provenance_grouping_mode": grouping_mode,
        "time_invariant_attributes": sorted(invariant),
        "provenance_group_sizes": dict(sorted(group_sizes.items())),
        "provenance_group_weight_totals": {
            key: round(value, 6) for key, value in sorted(weight_totals.items())
        },
        "provenance_restatements_removed": sorted(restatements),
        "reliability_model": reliability_model.describe(),
        "final_objective": objective_history[-1] if objective_history else None,
        "final_max_delta": max_delta_history[-1] if max_delta_history else None,
        # Mean admissible objects per observation. Ground-truth candidate_recall
        # needs labels and is computed by the evaluator via candidate_recall().
        "mean_candidates": (
            sum(sum(1 for value in row if value) for row in mask) / len(mask)
            if mask
            else 0.0
        ),
    }

    state = InferenceState(
        association=association_out,
        truth=truth,
        reliability=reliabilities,
        objective_history=objective_history,
        max_delta_history=max_delta_history,
        iterations=iterations,
        converged=converged,
        diagnostics=diagnostics,
    )
    # Structural invariant: shared identity means every claim of an observation
    # reads the same association row, so this must stay 0.0. Recorded on every
    # run so a regression surfaces immediately instead of silently.
    state.diagnostics["identity_contradiction_rate"] = identity_contradiction_rate(
        state, problem
    )
    return state


def _relation_groups(problem: InferenceProblem) -> Dict[str, List[PrimitiveObservation]]:
    """Partition observations into independently inferred relation groups."""
    from DS_VI.models.shared_identity.scope import relation_group_id

    groups: Dict[str, List[PrimitiveObservation]] = {}
    for observation in problem.observations:
        groups.setdefault(str(relation_group_id(observation)), []).append(observation)
    return groups


def _merge_independent_states(
    states: Sequence[Tuple[str, InferenceState]],
) -> InferenceState:
    """Merge disjoint group posteriors while retaining group diagnostics."""
    association: Dict[str, Dict[str, float]] = {}
    truth: Dict[TruthKey, Dict[Any, float]] = {}
    reliability: Dict[str, float] = {}
    objective_history: List[float] = []
    max_delta_history: List[float] = []
    converged = True
    iterations = 0
    group_diagnostics: Dict[str, Any] = {}

    for group_id, state in states:
        overlap = set(association).intersection(state.association)
        if overlap:
            raise SchemaValidationError(
                f"independent association groups overlap observations: {sorted(overlap)}"
            )
        association.update(state.association)
        truth.update(state.truth)
        reliability.update(state.reliability)
        objective_history.extend(state.objective_history)
        max_delta_history.extend(state.max_delta_history)
        converged = converged and state.converged
        iterations = max(iterations, state.iterations)
        group_diagnostics[group_id] = dict(state.diagnostics)

    return InferenceState(
        association=association,
        truth=truth,
        reliability=reliability,
        objective_history=objective_history,
        max_delta_history=max_delta_history,
        iterations=iterations,
        converged=converged,
        diagnostics={
            "mode": "cross_image_independent",
            "group_count": len(states),
            "groups": group_diagnostics,
            "identity_contradiction_rate": 0.0,
        },
    )


def run_joint_inference(
    problem: InferenceProblem,
    config: Optional[Mapping[str, Any]] = None,
    provenance_graph: Optional[ProvenanceGraph] = None,
    validate: bool = True,
) -> InferenceState:
    """Run shared-identity inference independently for each relation group."""
    cfg = dict(config or load_config())
    association_cfg = dict(cfg.get("association") or {})
    mode = str(association_cfg.get("mode", "legacy_global"))

    if mode != "cross_image_independent":
        return _run_joint_inference_single(
            problem,
            cfg,
            provenance_graph=provenance_graph,
            validate=validate,
        )

    groups = _relation_groups(problem)
    if len(groups) <= 1:
        state = _run_joint_inference_single(
            problem,
            cfg,
            provenance_graph=provenance_graph,
            validate=validate,
        )
        state.diagnostics.setdefault("mode", "cross_image_independent")
        state.diagnostics.setdefault("group_count", len(groups))
        return state

    states: List[Tuple[str, InferenceState]] = []
    for group_id, observations in groups.items():
        observation_ids = {item.observation_id for item in observations}
        object_ids = {
            edge.object_id
            for edge in problem.candidate_edges
            if edge.observation_id in observation_ids and not edge.is_null
        }
        sub_problem = InferenceProblem(
            observations=list(observations),
            claims=[
                claim
                for claim in problem.claims
                if claim.observation_id in observation_ids
            ],
            objects=[
                obj for obj in problem.objects if obj.object_id in object_ids
            ],
            candidate_edges=[
                edge
                for edge in problem.candidate_edges
                if edge.observation_id in observation_ids
            ],
            relations=[
                relation
                for relation in problem.relations
                if relation.source_observation_id in observation_ids
                and relation.target_observation_id in observation_ids
            ],
            capacities={
                object_id: value
                for object_id, value in problem.capacities.items()
                if object_id in object_ids
            },
            value_domains=dict(problem.value_domains),
            truth_priors={
                key: value
                for key, value in problem.truth_priors.items()
                if key[0] in object_ids
            },
        )
        states.append(
            (
                group_id,
                _run_joint_inference_single(
                    sub_problem,
                    cfg,
                    provenance_graph=provenance_graph,
                    validate=validate,
                ),
            )
        )

    return _merge_independent_states(states)


def identity_contradiction_rate(state: InferenceState, problem: InferenceProblem) -> float:
    """Fraction of multi-claim observations whose claims disagree about identity.

    For the shared-identity model this must be ``0.0``: all claims of an
    observation read one row.  A non-zero value indicates a structural
    regression, not a threshold to tune.

    To be load-bearing this must look for evidence that identity leaked to the
    claim level, since simply re-reading the observation's row can only ever
    agree with itself.  Two leaks are detectable:

    1. ``state.association`` carries a row keyed by ``claim_id`` that differs
       from its observation's row, i.e. a per-claim association was fitted.
    2. A claim carries one of :attr:`Claim.FORBIDDEN_FIELDS` in its context,
       i.e. an object id was attached directly to the claim.
    """
    claims_by_observation = problem.claims_by_observation()
    contradictions = 0
    considered = 0

    for observation_id, claims in claims_by_observation.items():
        if len(claims) < 2:
            continue
        considered += 1
        observation_row = tuple(sorted(state.association_row(observation_id).items()))

        rows = {observation_row}
        for claim in claims:
            # A claim-keyed row must not exist; if it does, it is the leak.
            if claim.claim_id in state.association:
                rows.add(tuple(sorted(state.association_row(claim.claim_id).items())))
            for forbidden in Claim.FORBIDDEN_FIELDS:
                if forbidden in (claim.context or {}):
                    rows.add((("__leaked__", forbidden),))
        if len(rows) > 1:
            contradictions += 1

    return (contradictions / considered) if considered else 0.0


__all__ = [
    "TruthKey",
    "build_value_domains",
    "initialize_truth",
    "truth_priors_for",
    "update_association",
    "update_truth",
    "run_joint_inference",
    "identity_contradiction_rate",
]
