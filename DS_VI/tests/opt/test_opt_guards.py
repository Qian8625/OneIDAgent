"""Guards of the optional torch layer: clear failures, no silent fallbacks."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from DS_VI.models.opt import (
    TorchUnavailableError,
    pick_device,
    pick_dtype,
    require_torch,
    torch_available,
)
from DS_VI.models.opt.torch_ot import log_sinkhorn


def test_require_torch_returns_the_module():
    assert require_torch() is torch


def test_torch_available_is_true_in_a_torch_environment():
    assert torch_available()


def test_cpu_dtype_is_float64_for_determinism():
    assert pick_dtype(pick_device("cpu")) is torch.float64


def test_cuda_request_fails_loudly_when_cuda_is_missing():
    if torch.cuda.is_available():
        pytest.skip("CUDA available; the failure path cannot be exercised")
    with pytest.raises(TorchUnavailableError):
        pick_device("cuda")


def test_auto_device_picks_what_is_available():
    device = pick_device("auto")
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_empty_cost_matrix_is_handled_without_a_solve():
    result = log_sinkhorn(np.zeros((0, 3)), reg=0.1)
    assert result["plan"].shape == (0, 3)
    assert result["total_mass"] == 0.0
