"""Test-suite bootstrap.

The optional torch experiments load a second OpenMP/MKL runtime next to Anaconda's
numpy/scipy. On Windows that combination aborts the interpreter ("Fatal Python
error: Aborted") the first time numpy hits a threaded kernel after torch was
imported. ``KMP_DUPLICATE_LIB_OK`` is the documented escape hatch for the duplicate
runtime; it only affects this test process, never a library import path.
"""
from __future__ import annotations

import os

# Only Windows needs the escape hatch: the duplicate OpenMP/MKL runtime that
# aborts the interpreter there does not exist on Linux, and setting it
# unconditionally would mask real threading problems there.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
