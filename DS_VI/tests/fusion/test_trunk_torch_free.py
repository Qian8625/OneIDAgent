"""Merge gate: the shipped inference path must stay torch-free.

Torch is an optional experiment dependency. If any module on the inference path
started importing it, then (a) a torch-free installation would break, and (b) the
published numbers would depend on which optional packages happen to be installed.
The check runs in a subprocess so the test process's own torch import cannot mask
a regression.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PROBE = """
import sys
import DS_VI.models.cascade.vi  # noqa: F401
import DS_VI.models.cascade.attributes  # noqa: F401
import DS_VI.models.ds_fusion.pipeline  # noqa: F401
import DS_VI.models.shared_identity.inference  # noqa: F401
import DS_VI.experiments.metrics.property_metrics  # noqa: F401
print("torch" in sys.modules)
"""


def test_inference_path_does_not_import_torch():
    result = subprocess.run(
        [sys.executable, "-B", "-c", PROBE],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("False"), result.stdout
