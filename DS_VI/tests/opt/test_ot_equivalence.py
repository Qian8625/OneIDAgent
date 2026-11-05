"""The torch OT port must reproduce the numpy reference it was derived from.

This is the equivalence gate from the plan: the torch implementation is only
allowed to be used by an experiment when it matches the readable numpy reference
on the same problem to float64 precision.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from DS_VI.models.opt import numpy_ot
from DS_VI.models.opt.torch_ot import log_sinkhorn as torch_log_sinkhorn
from DS_VI.models.opt.torch_ot import with_dustbin_square


CASES = (
    (np.array([[0.1, 0.9], [0.8, 0.2]]), None),
    (np.array([[0.0, 1.0, 0.5], [0.4, 0.2, 0.9]]), np.array([[True, True, False], [True, True, True]])),
    (np.array([[0.3, 0.4], [0.5, 0.2], [0.9, 0.1]]), None),
)


@pytest.mark.parametrize("cost, mask", CASES)
def test_torch_matches_the_numpy_reference(cost, mask):
    reference = numpy_ot.log_sinkhorn(cost, reg=0.1, mask=mask)
    ported = torch_log_sinkhorn(cost, reg=0.1, mask=mask, device="cpu")
    assert np.allclose(ported["plan"], reference["plan"], atol=1e-9)


@pytest.mark.parametrize("cost, mask", CASES)
def test_torch_matches_the_numpy_reference_on_the_dustbin_matrix(cost, mask):
    augmented = with_dustbin_square(cost, dustbin_cost=0.5)
    augmented_mask = np.ones_like(augmented, dtype=bool)
    if mask is not None:
        augmented_mask[: cost.shape[0], : cost.shape[1]] = mask
    reference = numpy_ot.log_sinkhorn(augmented, reg=0.1, mask=augmented_mask)
    ported = torch_log_sinkhorn(augmented, reg=0.1, mask=augmented_mask, device="cpu")
    assert np.allclose(ported["plan"], reference["plan"], atol=1e-9)


def test_torch_cpu_is_deterministic():
    cost = np.array([[0.2, 0.7], [0.6, 0.1], [0.9, 0.9]])
    first = torch_log_sinkhorn(cost, reg=0.1, device="cpu")
    second = torch_log_sinkhorn(cost, reg=0.1, device="cpu")
    assert np.array_equal(first["plan"], second["plan"])


def test_cuda_matches_cpu_within_float32_tolerance():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cost = np.array([[0.2, 0.7], [0.6, 0.1], [0.9, 0.9]])
    cpu = torch_log_sinkhorn(cost, reg=0.1, device="cpu")
    gpu = torch_log_sinkhorn(cost, reg=0.1, device="cuda")
    assert np.allclose(gpu["plan"], cpu["plan"], atol=1e-3)
