"""Optional torch layer: nothing here is imported by the shipped numpy trunk.

The cascade, the D-S fusion and the shared-identity inference stay torch-free so
that ``requirements.txt`` alone reproduces every published number. This package
only holds the pieces that genuinely need torch:

* :mod:`DS_VI.models.opt.torch_ot` — log-domain Sinkhorn / unbalanced OT used by
  the OT-association experiments (``association_mode="ot"``).
* :mod:`DS_VI.models.opt.sweep` — batched OT hyper-parameter scans on GPU.

Everything is guarded: a missing torch (or a torch without CUDA) raises a clear
error instead of silently falling back to a different algorithm, because a silent
fallback would make an OT arm indistinguishable from a Hungarian one.
"""
from __future__ import annotations

from typing import Any, Dict


class TorchUnavailableError(RuntimeError):
    """Raised when a torch-only experiment path is used without torch installed."""


_TORCH_STATE: Dict[str, Any] = {}


def torch_available() -> bool:
    """True only when ``import torch`` actually succeeds (cached).

    ``importlib.util.find_spec`` is not enough: a partially installed or
    ABI-broken torch has a spec but fails to import, and picking it would turn
    every OT pair into an error.
    """
    if "available" not in _TORCH_STATE:
        try:
            import importlib

            importlib.import_module("torch")
            _TORCH_STATE["available"] = True
        except Exception:
            _TORCH_STATE["available"] = False
    return bool(_TORCH_STATE["available"])


def require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise TorchUnavailableError(
            "this experiment needs torch; install requirements-torch.txt "
            f"(python -m pip install -r requirements-torch.txt): {exc}"
        ) from exc
    return torch


def pick_device(requested: str = "cpu") -> Any:
    """Resolve a device string; ``auto`` prefers CUDA when it is usable offline."""
    torch = require_torch()
    wanted = str(requested or "cpu").strip().lower()
    if wanted == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        raise TorchUnavailableError("cuda requested but torch.cuda.is_available() is False")
    return torch.device(wanted)


def pick_dtype(device: Any, exact: bool = True) -> Any:
    """float64 on CPU for determinism, float32 on GPU for throughput."""
    torch = require_torch()
    if exact and getattr(device, "type", "cpu") == "cpu":
        return torch.float64
    return torch.float32


__all__ = [
    "TorchUnavailableError",
    "pick_device",
    "pick_dtype",
    "require_torch",
    "torch_available",
]
