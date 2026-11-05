"""Log-domain symmetric noise likelihood for attribute claims.

For a claim reporting value ``y`` about a truth variable with domain size
``C``, under reliability ``r``:

.. code-block:: text

    P(y | v = y, r) = r
    P(y | v != y, r) = (1 - r) / (C - 1)

Everything is computed in log space with ``logsumexp`` so that long evidence
chains never underflow.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_EPSILON = 1e-12


def safe_log(value: float, epsilon: float = DEFAULT_EPSILON) -> float:
    """``log(value)`` with a floor at ``epsilon`` to avoid ``-inf``."""
    return math.log(max(float(value), epsilon))


def logsumexp(values: Iterable[float]) -> float:
    """Numerically stable ``log(sum(exp(values)))``.

    Returns ``-inf`` for an empty or all-``-inf`` input.
    """
    items = [v for v in values if not math.isinf(v) or v > 0]
    finite = [v for v in items if not math.isinf(v)]
    if not finite:
        return float("-inf")
    peak = max(finite)
    total = sum(math.exp(v - peak) for v in finite)
    if total <= 0.0:
        return float("-inf")
    return peak + math.log(total)


def normalize_log_weights(log_weights: Sequence[float]) -> List[float]:
    """Softmax a vector of log weights into a probability simplex."""
    if not log_weights:
        return []
    total = logsumexp(log_weights)
    if math.isinf(total):
        uniform = 1.0 / len(log_weights)
        return [uniform] * len(log_weights)
    probs = [math.exp(w - total) for w in log_weights]
    mass = sum(probs)
    if mass <= 0.0:
        uniform = 1.0 / len(log_weights)
        return [uniform] * len(log_weights)
    return [p / mass for p in probs]


def claim_log_likelihood(
    observed_value: Any,
    candidate_value: Any,
    reliability: float,
    domain_size: int,
    epsilon: float = DEFAULT_EPSILON,
    outlier_mixture: float = 0.0,
) -> float:
    """``log P(observed_value | candidate_value, reliability)``.

    A domain of size 1 carries no information, so the likelihood is flat
    (``log 1 = 0``) rather than degenerate.

    ``outlier_mixture`` (``e``) mixes in a uniform outlier component:

    .. code-block:: text

        P'(y | v) = (1 - e) * P(y | v) + e / C

    ``e = 0`` is the pure symmetric noise model.  A positive ``e`` bounds the
    log-likelihood ratio between agreeing and disagreeing values, so one claim
    can no longer drive a truth variable to certainty no matter how reliable its
    cell claims to be.

    Reliability is used as given, with only a numerical floor at ``epsilon``.
    A channel measured *below* chance (``r < 1 / C``) is anti-correlated: its
    likelihood ratio ``log((C-1) r / (1-r))`` is negative, so agreement with a
    candidate lowers that candidate's odds. The paper relies on this sign
    (Eq. 7 and Appendix C.7): a miscalibrated channel inverts the attribute's
    effect instead of being silently flattened to "uninformative".
    """
    size = max(int(domain_size), 1)
    if size == 1:
        return 0.0
    r = min(max(float(reliability), epsilon), 1.0 - epsilon)
    if _values_equal(observed_value, candidate_value):
        probability = r
    else:
        probability = (1.0 - r) / (size - 1)
    e = float(outlier_mixture)
    if e > 0.0:
        probability = (1.0 - e) * probability + e / size
    return safe_log(probability, epsilon)


def _values_equal(a: Any, b: Any) -> bool:
    """Compare claim values with light normalization (case/space insensitive)."""
    if a is b:
        return True
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip().lower() == str(b).strip().lower()
    return a == b


def expected_log_likelihood(
    observed_value: Any,
    truth_distribution: Mapping[Any, float],
    reliability: float,
    domain_size: Optional[int] = None,
    epsilon: float = DEFAULT_EPSILON,
    outlier_mixture: float = 0.0,
) -> float:
    """``E_Q[log P(observed_value | v, r)]`` under the current truth belief.

    This is the per-claim quantity accumulated into ``S[m, k]`` when updating the
    association matrix.
    """
    if not truth_distribution:
        return 0.0
    size = int(domain_size) if domain_size else len(truth_distribution)
    total_mass = sum(float(p) for p in truth_distribution.values())
    if total_mass <= 0.0:
        return 0.0
    accumulated = 0.0
    for candidate_value, probability in truth_distribution.items():
        weight = float(probability) / total_mass
        if weight <= 0.0:
            continue
        accumulated += weight * claim_log_likelihood(
            observed_value, candidate_value, reliability, size, epsilon, outlier_mixture
        )
    return accumulated


def truth_log_posterior(
    candidate_values: Sequence[Any],
    weighted_claims: Sequence[tuple],
    prior: Optional[Mapping[Any, float]] = None,
    epsilon: float = DEFAULT_EPSILON,
    outlier_mixture: float = 0.0,
) -> Dict[Any, float]:
    """Posterior over one truth variable given association-weighted claims.

    Args:
        candidate_values: The value domain of this truth variable.
        weighted_claims: Sequence of ``(observed_value, weight, reliability)``
            where ``weight = varpi_i * A[m(i), k]``.
        prior: Optional prior over values; uniform when omitted.
        epsilon: Numerical floor.
        outlier_mixture: Uniform outlier mass mixed into every claim likelihood.

    Returns:
        A normalized ``{value: probability}`` mapping.
    """
    if not candidate_values:
        return {}
    size = len(candidate_values)
    log_scores: List[float] = []
    for candidate in candidate_values:
        prior_mass = float((prior or {}).get(candidate, 1.0 / size))
        score = safe_log(prior_mass, epsilon)
        for observed_value, weight, reliability in weighted_claims:
            w = float(weight)
            if w <= 0.0:
                continue
            score += w * claim_log_likelihood(
                observed_value, candidate, reliability, size, epsilon, outlier_mixture
            )
        log_scores.append(score)
    probabilities = normalize_log_weights(log_scores)
    return {value: prob for value, prob in zip(candidate_values, probabilities, strict=False)}


def entropy(distribution: Mapping[Any, float], epsilon: float = DEFAULT_EPSILON) -> float:
    """Shannon entropy in nats of a (possibly unnormalized) distribution."""
    values = [float(p) for p in distribution.values() if float(p) > 0.0]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return -sum((p / total) * math.log(max(p / total, epsilon)) for p in values)


def normalized_entropy(
    distribution: Mapping[Any, float],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Entropy scaled to ``[0, 1]`` by ``log(support size)``."""
    support = len([p for p in distribution.values() if float(p) > 0.0])
    if support <= 1:
        return 0.0
    return entropy(distribution, epsilon) / math.log(support)


__all__ = [
    "DEFAULT_EPSILON",
    "safe_log",
    "logsumexp",
    "normalize_log_weights",
    "claim_log_likelihood",
    "expected_log_likelihood",
    "truth_log_posterior",
    "entropy",
    "normalized_entropy",
]
