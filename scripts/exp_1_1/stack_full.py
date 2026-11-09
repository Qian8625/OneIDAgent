#!/usr/bin/env python3
"""起齐全部工具：逐个独立拉起，不因单个失败阻塞其他 worker。

start_server_local 会对每个 worker 无限等待地址，任何一个起不来就卡住后面的；
这里改成"全部拉起 → 统一轮询注册表 → 报告谁没起来"，然后再用 doctor 体检。

    python -m scripts.exp_1_1.stack_full --config tool_server/.../all_service_example_local.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exp_1_1 import doctor  # noqa: E402

DEFAULT_CONFIG = ROOT / "tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml"
LOG_DIR = ROOT / "outputs/exp_1_1/stackfull"


def stack_env(cuda: str | None) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("ONEID_ENABLE_WORKBOOK_REPLAY", "1")
    env.setdefault("ONEID_REPLAY_BINDING", "iou_assisted")
    env.setdefault("ONEID_DATA_ROOT", "/home/ubuntu/00_Data/RSTI")
    env.setdefault("CHECKPOINT_ROOT", "/home/ubuntu/00_CPK")
    env["PYTHONPATH"] = str(ROOT)
    env["PATH"] = os.pathsep.join([str(ROOT / "outputs/e2e_task45/bin"),
                                   "/home/ubuntu/miniconda3/bin", env.get("PATH", "")])
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    return env


def command_for(spec: Dict) -> List[str]:
    cmd_cfg = dict(spec.get("cmd") or {})
    script = str(cmd_cfg.pop("script-addr"))
    args = [script]
    for key, value in cmd_cfg.items():
        args += [f"--{key}", str(value)]
    conda_env = spec.get("conda_env")
    if conda_env:
        return ["/home/ubuntu/miniconda3/bin/conda", "run", "--no-capture-output", "-n", conda_env,
                "env", f"PYTHONPATH={ROOT}", "python", *args]
    return ["python", *args]


def spawn(spec: Dict, job_name: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"{job_name}.log", "w", encoding="utf-8")
    process = subprocess.Popen(command_for(spec), cwd=str(ROOT), env=stack_env(spec.get("cuda_visible_devices")),
                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    print(f"[start] {job_name:28} pid={process.pid}")
    return process


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--wait", type=int, default=420)
    args = parser.parse_args(argv)

    import yaml

    from tool_server.utils.paths import resolve_config_paths

    config = resolve_config_paths(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))

    print("[stop] 清理既有 worker / controller / launcher")
    for pattern in ("start_server_local.py", "online_workers/controller.py",
                    "online_workers/[A-Za-z0-9]*_worker.py"):
        doctor._kill(pattern)
    time.sleep(5)

    controller = dict(config["controller_config"])
    spawn(controller, "controller")
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            if doctor.list_models("http://127.0.0.1:20001"):
                break
        except Exception:
            pass
        time.sleep(2)

    started: List[Tuple[str, subprocess.Popen]] = []
    for entry in config.get("tool_worker_config") or []:
        name = next(iter(entry))
        started.append((name, spawn(entry[name], name)))

    deadline = time.time() + args.wait
    models: List[str] = []
    while time.time() < deadline:
        try:
            models = doctor.list_models("http://127.0.0.1:20001")
        except Exception:
            models = []
        if len(models) >= len(started) * 0.8:
            break
        time.sleep(5)

    print(f"\n[registry] {len(models)}/{len(started)}: {models}")
    missing = [name for name, _ in started if name not in models]
    dead = [name for name, process in started if process.poll() is not None]
    if missing:
        print(f"[missing] {missing}")
        print(f"[dead]    {dead}")
        print(f"[logs]    {LOG_DIR}")
    return 0 if len(models) >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
