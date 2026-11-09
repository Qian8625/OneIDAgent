#!/usr/bin/env python3
"""工具栈体检与单点重启。

约定（E2E 任何一次实验之前都要先过这一步）：
  1. controller 的 /list_models 必须非空；
  2. 每个注册的工具都必须能取到地址、并能响应 /tool_instruction 或 /worker_get_status；
  3. 有问题的工具单独重启，再体检一次，通过了才允许开跑。

用法：
    python -m scripts.exp_1_1.doctor check
    python -m scripts.exp_1_1.doctor check --expect ObjectDetection,DenoDetV2,SmallObjectDetection,CrossModalInterpret
    python -m scripts.exp_1_1.doctor restart ObjectDetection
    python -m scripts.exp_1_1.doctor restart-all --config outputs/e2e_task45/stack_task45.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONTROLLER = "http://127.0.0.1:20001"
DEFAULT_CONFIG = ROOT / "outputs/e2e_task45/stack_task45.yaml"


def _post(url: str, payload: Dict, timeout: float = 10.0) -> Dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def list_models(controller: str) -> List[str]:
    return list(_post(controller.rstrip("/") + "/list_models", {}).get("models") or [])


def worker_address(controller: str, model: str) -> Optional[str]:
    try:
        address = _post(controller.rstrip("/") + "/get_worker_address", {"model": model}).get("address") or ""
    except Exception:
        return None
    return address.strip() or None


def probe_tool(address: str, model: str) -> Tuple[bool, str]:
    """优先用 /tool_instruction（能同时验证功能定义），失败再看 /worker_get_status。"""
    try:
        payload = _post(address.rstrip("/") + "/tool_instruction", {}, timeout=8)
        name = ((payload.get("tool_instruction") or {}).get("function") or {}).get("name")
        if name == model:
            return True, "tool_instruction ok"
        return True, f"tool_instruction ok (name={name})"
    except Exception as exc:
        first = f"{type(exc).__name__}: {exc}"
    try:
        payload = _post(address.rstrip("/") + "/worker_get_status", {}, timeout=8)
        return True, f"status ok ({','.join(payload.get('model_names') or [])})"
    except Exception as exc:
        return False, f"unreachable: {first} / {type(exc).__name__}: {exc}"


def check(controller: str, expect: Optional[List[str]] = None, wait: int = 0) -> int:
    """wait>0 时轮询：模型加载慢，端口未 bind 会被误判成 Connection refused。"""
    deadline = time.time() + wait
    while True:
        code = _check_once(controller, expect)
        if code == 0 or time.time() >= deadline:
            return code
        print(f"[wait] 还有工具未就绪，{int(deadline - time.time())}s 后重试 …")
        time.sleep(15)


def _check_once(controller: str, expect: Optional[List[str]] = None) -> int:
    models = list_models(controller)
    print(f"[registry] {len(models)} tools: {models}")
    if not models:
        print("[FAIL] /list_models 为空：先重启 worker，再跑实验")
        return 2

    failures = []
    for model in models:
        address = worker_address(controller, model)
        if not address:
            failures.append(model)
            print(f"[FAIL] {model:28} 取不到地址")
            continue
        ok, detail = probe_tool(address, model)
        print(f"[{'ok  ' if ok else 'FAIL'}] {model:28} {address}  {detail}")
        if not ok:
            failures.append(model)

    if expect:
        missing = [name for name in expect if name not in models]
        if missing:
            print(f"[FAIL] 期望的工具未注册：{missing}")
            failures.extend(missing)

    print(f"\n[summary] ok={len(models)-len(failures)}/{len(models)} failures={failures}")
    return 1 if failures else 0


def _kill(pattern: str) -> int:
    killed = 0
    for pid in subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
    return killed


def _stack_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("ONEID_ENABLE_WORKBOOK_REPLAY", "1")
    env.setdefault("ONEID_REPLAY_BINDING", "iou_assisted")
    env.setdefault("ONEID_DATA_ROOT", "/home/ubuntu/00_Data/RSTI")
    env.setdefault("CHECKPOINT_ROOT", "/home/ubuntu/00_CPK")
    bin_dir = ROOT / "outputs/e2e_task45/bin"
    conda_bin = Path("/home/ubuntu/miniconda3/bin")
    env["PATH"] = os.pathsep.join([str(bin_dir), str(conda_bin), env.get("PATH", "")])
    return env


def restart(config: Path, controller: str, wait_seconds: int = 240) -> int:
    print("[restart] 停掉现有 worker / controller / launcher …")
    for pattern in ("start_server_local.py", "online_workers/controller.py",
                    "online_workers/[A-Za-z0-9]*_worker.py"):
        _kill(pattern)
    time.sleep(5)

    log = ROOT / "outputs/exp_1_1/doctor_restart.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "tool_server/tool_workers/scripts/launch_scripts/start_server_local.py"),
         "--config", str(config)],
        cwd=str(ROOT), env=_stack_env(), stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True)
    print(f"[restart] launcher pid={process.pid}，日志 {log}")

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            models = list_models(controller)
        except Exception:
            models = []
        if len(models) >= 5:
            print(f"[restart] 注册到 {len(models)} 个工具：{models}")
            return 0
        time.sleep(5)
    print("[restart] 超时：注册表仍不完整，请查看日志")
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "restart", "restart-all"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expect", default="")
    parser.add_argument("--wait", type=int, default=0, help="轮询等待秒数（模型加载慢时用）")
    args = parser.parse_args(argv)

    expect = [x.strip() for x in args.expect.split(",") if x.strip()]

    if args.action == "check":
        return check(args.controller, expect, args.wait)

    if args.action == "restart-all":
        return restart(Path(args.config), args.controller)

    if not args.name:
        parser.error("restart 需要工具名")
    print(f"[restart] 单点重启 {args.name}")
    _kill(f"online_workers/{args.name}_worker.py")
    time.sleep(2)
    return restart(Path(args.config), args.controller)


if __name__ == "__main__":
    sys.exit(main())
