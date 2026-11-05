import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_server.tool_workers.skills.validators import (
    validate_skill_manifests,
    format_validation_report,
)


def main():
    ok, errors = validate_skill_manifests()
    print(format_validation_report(ok, errors))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
