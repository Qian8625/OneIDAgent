#!/usr/bin/env python3
"""Run the OneID contract tests without pytest.

pytest is not installed in the runtime environments (see docs/dev_env.md), so the
contract tests are plain assertion functions. This runner imports every
``test_*.py`` under ``tests/contract`` (plus the legacy dependency guard) and
calls each ``test_*`` function, reporting a PASS/FAIL table.

Usage:
    python3 scripts/ci/run_contract_tests.py [--verbose]
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGET_DIRS = [REPO_ROOT / "tests" / "contract"]
EXTRA_FILES = [REPO_ROOT / "tests" / "skill_eval" / "test_cross_modal_dependency_guard.py"]


def _load(path: Path):
    name = "oneid_contract_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    verbose = "--verbose" in sys.argv
    files = []
    for directory in TARGET_DIRS:
        files.extend(sorted(directory.glob("test_*.py")))
    files.extend(path for path in EXTRA_FILES if path.is_file())

    passed = failed = 0
    skipped = []
    failures = []
    for path in files:
        try:
            module = _load(path)
        except ModuleNotFoundError as exc:
            # 例：agent_runner 会经 ToolManager 牵出 torch，而运行环境没有 torch。
            skipped.append((path.name, str(exc)))
            print(f"SKIP  {path.name} <import: {exc}>")
            continue
        except Exception:
            failed += 1
            failures.append((path.name, "<import>", traceback.format_exc(limit=3)))
            print(f"FAIL  {path.name} <import>")
            continue
        tests = [name for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))]
        for name in tests:
            try:
                getattr(module, name)()
            except Exception:
                failed += 1
                failures.append((path.name, name, traceback.format_exc(limit=3)))
                print(f"FAIL  {path.name}::{name}")
            else:
                passed += 1
                if verbose:
                    print(f"PASS  {path.name}::{name}")

    total = passed + failed
    print(f"\nSUMMARY: {passed}/{total} passed, {len(skipped)} skipped"
          + "".join(f"\n  SKIP {name}: {reason}" for name, reason in skipped))
    for file_name, test_name, detail in failures:
        print(f"\n--- {file_name}::{test_name} ---\n{detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
