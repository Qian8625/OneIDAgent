"""Run every OneID end-to-end suite in order.

    python scripts/ci/run_e2e.py              # all suites
    python scripts/ci/run_e2e.py --only static,worker

Each suite is an independent process, so a failure in one does not mask the
others. Exit code is non-zero if any suite failed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = Path(__file__).resolve().parent
SUITES = {
    "static": "e2e_static.py",
    "worker": "e2e_worker.py",
    "skill": "e2e_skill_chain.py",
    "ui": "e2e_ui.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma separated subset of: " + ", ".join(SUITES))
    args = parser.parse_args()

    wanted = [name.strip() for name in args.only.split(",") if name.strip()] or list(SUITES)
    unknown = [name for name in wanted if name not in SUITES]
    if unknown:
        print("unknown suite(s):", ", ".join(unknown))
        return 2

    failures = []
    for name in wanted:
        script = CI / SUITES[name]
        print(f"\n──────── {name} ────────")
        result = subprocess.run([sys.executable, str(script)], cwd=str(CI))
        if result.returncode != 0:
            failures.append(name)

    print("\n════════ SUMMARY ════════")
    for name in wanted:
        print(f"  {name:8} {'FAIL' if name in failures else 'ok'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
