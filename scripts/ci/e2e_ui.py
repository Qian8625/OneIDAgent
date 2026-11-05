"""UI end-to-end check: start ``app/app.py`` and inspect what it serves.

The Gradio app is started from this checkout and queried over HTTP; the served
configuration must carry the OneID header and none of the old institutional
branding. Requires a Python with ``gradio`` installed (``ONEID_E2E_UI_PYTHON``).

    python scripts/ci/e2e_ui.py
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys

from e2e_common import PORT_UI, ROOT, Reporter, UI_PYTHON, base_env, get, stop_all

TITLE = "OneID : Remote Sensing Identity Interpretation"
SUBTITLE = "Dual-source (RGB/SAR) shared-identity reasoning for remote sensing"
STALE = (r"\bNPU\b", r"nwhu", r"nwpu", r"Li Huihui")


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    report = Reporter("UI e2e")

    probe = subprocess.run([UI_PYTHON, "-c", "import gradio"], capture_output=True)
    if probe.returncode != 0:
        report.skip("UI launch", f"gradio not available in {UI_PYTHON}")
        return 0

    (ROOT / ".e2e_tmp").mkdir(exist_ok=True)
    log = open(ROOT / ".e2e_tmp" / "ui.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [UI_PYTHON, "app/app.py"], cwd=str(ROOT), env=base_env(),
        stdout=log, stderr=subprocess.STDOUT)
    from e2e_common import _PROCESSES

    _PROCESSES.append(proc)
    try:
        from e2e_common import wait_port

        if not report.check("UI starts", wait_port(PORT_UI, proc, 240),
                            f"port {PORT_UI}"):
            return report.finish()

        for path in ("/config", "/"):
            body = get(PORT_UI, path)
            report.check(f"GET {path} serves OneID header", TITLE in body and SUBTITLE in body,
                         f"{len(body)} bytes")
            hits = [pattern for pattern in STALE if re.search(pattern, body, 0 if "NPU" in pattern else re.I)]
            report.check(f"GET {path} has no institutional branding", not hits, ", ".join(hits))
    finally:
        stop_all()

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
