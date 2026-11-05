"""Three-hop end-to-end check: skill -> controller -> tool.

Starts a controller, the ``CrossModalInterpret`` worker and the
``CrossModalMatchSkill`` worker, then calls the *skill* over HTTP. The skill
resolves the interpret worker address through the controller, so this exercises
the real deployment topology (and the D-S -> evidence -> cascade chain inside).

    python scripts/ci/e2e_skill_chain.py
"""

from __future__ import annotations

import os
import subprocess
import sys

from e2e_common import (
    PORT_CONTROLLER,
    PORT_INTERPRET,
    PORT_MATCH,
    ROOT,
    Reporter,
    WORKERS,
    base_env,
    post,
    sample,
    stop_all,
    wait_port,
)

SKILL = ROOT / "tool_server" / "tool_workers" / "skills" / "plugins" / "cross_modal_match" / "worker.py"


def _spawn(args, log_name: str) -> subprocess.Popen:
    (ROOT / ".e2e_tmp").mkdir(exist_ok=True)
    log = open(ROOT / ".e2e_tmp" / f"{log_name}.log", "w", encoding="utf-8")
    from e2e_common import WORKER_PYTHON, _PROCESSES

    proc = subprocess.Popen(args, cwd=str(ROOT), env=base_env(), stdout=log, stderr=subprocess.STDOUT)
    _PROCESSES.append(proc)
    return proc


def main() -> int:
    report = Reporter("skill -> controller -> tool e2e")
    from e2e_common import WORKER_PYTHON

    controller_addr = f"http://127.0.0.1:{PORT_CONTROLLER}"

    controller = _spawn(
        [WORKER_PYTHON, str(WORKERS / "controller.py"), "--host", "127.0.0.1", "--port", str(PORT_CONTROLLER)],
        "controller")
    interpret = skill = None
    try:
        if not report.check("controller starts", wait_port(PORT_CONTROLLER, controller, 120)):
            return report.finish()

        interpret = _spawn(
            [WORKER_PYTHON, str(WORKERS / "CrossModalInterpret_worker.py"),
             "--host", "127.0.0.1", "--port", str(PORT_INTERPRET),
             "--controller-address", controller_addr], "interpret_reg")
        report.check("interpret worker registers", wait_port(PORT_INTERPRET, interpret, 120))

        skill = _spawn(
            [WORKER_PYTHON, str(SKILL), "--host", "127.0.0.1", "--port", str(PORT_MATCH),
             "--controller-address", controller_addr], "match_skill_reg")
        report.check("match skill registers", wait_port(PORT_MATCH, skill, 120))

        registered = post(PORT_CONTROLLER, "/list_models", {})
        models = registered.get("models") or registered.get("model_names") or registered
        text = str(models)
        report.check("controller sees interpret worker", "CrossModalInterpret" in text, text[:120])
        report.check("controller sees match skill", "CrossModalMatchSkill" in text, text[:120])

        response = post(PORT_MATCH, "/worker_generate", {
            "rgb_image": str(sample("match_rgb.jpg")),
            "sar_image": str(sample("match_sar.jpg")),
            "target": "ship",
            "rgb_detections": [
                {"bbox": [420, 300, 520, 360], "label": "ship", "score": 0.92},
                {"bbox": [700, 520, 800, 590], "label": "ship", "score": 0.81},
            ],
            "sar_candidates": [
                {"bbox": [425, 302, 522, 362], "label": "ship", "score": 0.88},
            ],
            "visualize": False,
        }, timeout=600)

        report.check("skill returns ok", int(response.get("error_code", 1)) == 0,
                     str(response.get("text"))[:150])
        structured = response.get("structured") or {}
        report.check("skill used the shared-identity backend",
                     str(structured.get("mode")) == "shared_identity",
                     f"mode={structured.get('mode')} degraded={structured.get('degraded')}")
        pairs = response.get("matched_pairs") or []
        report.check("skill returned pairs with VI score", bool(pairs),
                     f"pairs={len(pairs)}")
        if pairs:
            first = pairs[0]
            report.check("pair carries VI verdict", first.get("vi_status") is not None,
                         f"vi_status={first.get('vi_status')} match_score={first.get('match_score')}")
            report.check("pair keeps geometry score",
                         first.get("geometry_match_score") is not None,
                         f"geometry_match_score={first.get('geometry_match_score')}")
            check = structured.get("pairing") or {}
            report.check("pairing metadata present", check.get("paired_by") == "CrossModalInterpret",
                         f"paired_by={check.get('paired_by')}")
    finally:
        stop_all()

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
