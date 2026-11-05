"""Shared helpers for the OneID end-to-end checks.

Every check is a standalone script that prints PASS/FAIL lines and exits
non-zero on failure, so they can run directly or through ``run_e2e.py``.

Ports and the interpreter used to launch workers are configurable:

    ONEID_E2E_INTERPRET_PORT   default 20035
    ONEID_E2E_ADDTEXT_PORT     default 20036
    ONEID_E2E_CONTROLLER_PORT  default 20001
    ONEID_E2E_MATCH_PORT       default 20132
    ONEID_E2E_UI_PORT          default 4444
    ONEID_E2E_PYTHON           python used for worker processes (default sys.executable)
    ONEID_E2E_UI_PYTHON        python used for the UI (default: ONEID_E2E_PYTHON)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
WORKERS = ROOT / "tool_server" / "tool_workers" / "online_workers"
SAMPLES = ROOT / "assets" / "samples"

WORKER_PYTHON = os.environ.get("ONEID_E2E_PYTHON") or sys.executable
UI_PYTHON = os.environ.get("ONEID_E2E_UI_PYTHON") or WORKER_PYTHON

PORT_INTERPRET = int(os.environ.get("ONEID_E2E_INTERPRET_PORT", "20035"))
PORT_ADDTEXT = int(os.environ.get("ONEID_E2E_ADDTEXT_PORT", "20036"))
PORT_CONTROLLER = int(os.environ.get("ONEID_E2E_CONTROLLER_PORT", "20001"))
PORT_MATCH = int(os.environ.get("ONEID_E2E_MATCH_PORT", "20132"))
PORT_UI = int(os.environ.get("ONEID_E2E_UI_PORT", "4444"))

_PROCESSES: List[subprocess.Popen] = []


class Reporter:
    """Collect PASS/FAIL results and exit accordingly."""

    def __init__(self, title: str):
        self.title = title
        self.results: List[Tuple[bool, str]] = []
        print(f"\n=== {title} ===")

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((ok, name))
        print(("PASS  " if ok else "FAIL  ") + name + (f" | {detail}" if detail else ""))
        return ok

    def skip(self, name: str, reason: str) -> None:
        print(f"SKIP  {name} | {reason}")

    def finish(self) -> int:
        failed = [name for ok, name in self.results if not ok]
        total = len(self.results)
        print(f"\n{self.title}: {total - len(failed)}/{total} passed")
        return 1 if failed else 0


def base_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("ONEID_DATA_ROOT", str(ROOT / "data"))
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    if extra:
        env.update(extra)
    return env


def wait_port(port: int, proc: subprocess.Popen, seconds: int = 120) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def spawn(script: Path, port: int, log_name: str, controller: Optional[int] = None) -> subprocess.Popen:
    """Start a worker (or the controller) in the background."""
    args = [WORKER_PYTHON, str(script), "--port", str(port)]
    if script.name != "controller.py":
        args.append("--no-register") if controller is None else args.extend(
            ["--controller-address", f"http://127.0.0.1:{controller}"])
    log = open(ROOT / ".e2e_tmp" / f"{log_name}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=str(ROOT), env=base_env(), stdout=log, stderr=subprocess.STDOUT)
    _PROCESSES.append(proc)
    return proc


def post(port: int, path: str, payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(port: int, path: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return response.read().decode("utf-8", "ignore")


def stop_all() -> None:
    for proc in _PROCESSES:
        if proc.poll() is None:
            proc.terminate()
    for proc in _PROCESSES:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    _PROCESSES.clear()


def sample(name: str) -> Path:
    return SAMPLES / name
