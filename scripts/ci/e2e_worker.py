"""Worker-level end-to-end checks over real HTTP.

Starts two workers from this checkout and talks to them the same way the tool
controller does:

* ``AddText`` renders a Chinese caption with the bundled font;
* ``CrossModalInterpret`` runs the DS+VI shared-identity chain.

    python scripts/ci/e2e_worker.py
"""

from __future__ import annotations

import sys

from e2e_common import (
    PORT_ADDTEXT,
    PORT_INTERPRET,
    ROOT,
    Reporter,
    WORKERS,
    base_env,
    post,
    sample,
    spawn,
    stop_all,
    wait_port,
)

OUTPUT_DIRS = [ROOT / "tools_output", ROOT / "tool_server" / "tool_workers" / "tools_output"]


def snapshot() -> dict:
    """Map output file -> mtime (ns), so an overwritten file still counts as new."""
    found = {}
    for directory in OUTPUT_DIRS:
        if directory.exists():
            for path in directory.glob("*"):
                if path.is_file():
                    found[path] = path.stat().st_mtime_ns
    return found


def main() -> int:
    report = Reporter("worker HTTP e2e")
    (ROOT / ".e2e_tmp").mkdir(exist_ok=True)

    # ---- AddText: bundled font -> Chinese caption ----
    before = snapshot()
    add_text = spawn(WORKERS / "AddText_worker.py", PORT_ADDTEXT, "addtext")
    try:
        if report.check("AddText worker starts", wait_port(PORT_ADDTEXT, add_text, 120)):
            response = post(PORT_ADDTEXT, "/worker_generate", {
                "image": str(sample("match_rgb.jpg")),
                "text": "光学 SAR 目标 OneID",
                "position": "lt",
                "font_size": 40,
                "color": "red",
            })
            report.check("AddText renders Chinese text", int(response.get("error_code", 1)) == 0,
                         str(response.get("text"))[:120])
            after = snapshot()
            produced = sorted(
                path for path, mtime in after.items()
                if before.get(path) != mtime
            )
            report.check("AddText wrote an output image", bool(produced),
                         str([p.name for p in produced[:3]]))
            if produced:
                from PIL import Image

                with Image.open(produced[0]) as rendered, Image.open(sample("match_rgb.jpg")) as src:
                    report.check("annotated image keeps source geometry", rendered.size == src.size,
                                 f"{rendered.size}")
    finally:
        stop_all()

    # ---- CrossModalInterpret: shared-identity chain ----
    interpret = spawn(WORKERS / "CrossModalInterpret_worker.py", PORT_INTERPRET, "interpret")
    try:
        if report.check("CrossModalInterpret starts", wait_port(PORT_INTERPRET, interpret, 120)):
            instruction = post(PORT_INTERPRET, "/tool_instruction", {})
            name = (instruction.get("tool_instruction") or {}).get("function", {}).get("name")
            report.check("tool instruction served", name == "CrossModalInterpret", str(name))

            response = post(PORT_INTERPRET, "/worker_generate", {
                "target": "ship",
                "task": "crosscheck_exists",
                "mode": "shared_identity",
                "rgb_detections": [
                    {"bbox": [100, 100, 200, 200], "label": "ship", "score": 0.9},
                    {"bbox": [300, 300, 400, 400], "label": "ship", "score": 0.8},
                ],
                "sar_detections": [
                    {"bbox": [102, 101, 201, 199], "label": "ship", "score": 0.85},
                ],
            })
            report.check("CrossModalInterpret returns ok", int(response.get("error_code", 1)) == 0,
                         str(response.get("error_code")))
            report.check("mode is shared_identity", response.get("mode") == "shared_identity",
                         f"mode={response.get('mode')} degraded={response.get('degraded')}")
            summary = response.get("summary") or {}
            report.check("pairing produced a match", int(summary.get("matched_count", 0)) >= 1,
                         f"matched={summary.get('matched_count')} dedup={response.get('dedup_count')}")
            report.check("evidence layer reported", "evidence_summary" in response,
                         f"iterations={(response.get('inference') or {}).get('iterations')}")
    finally:
        stop_all()

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
