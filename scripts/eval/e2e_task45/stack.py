"""环境预检 + 最小 worker 栈的 yaml 生成。

task4/task5 只需要 6 个 worker；而 ``start_server_local.py`` 只能整份 yaml 全起
（20 个，含 3 个 GPU 重模型）。四臂 × 717 样本反复起停整栈会互相抢显存，所以这里
从既有配置里裁一份最小栈，并检查几个"配错了会让某个臂系统性变差"的环境项。

注意：``ONEID_ENABLE_WORKBOOK_REPLAY`` 与 ``ONEID_DATA_ROOT`` 是 **worker 进程级**
开关，必须在起 worker 之前导出；请求参数无法覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

#: 只保留 task4/task5 会用到的在线 worker。
TASK45_WORKERS: Dict[str, List[str]] = {
    "task4": ["ObjectDetection", "DenoDetV2", "CrossModalInterpret"],
    "task5": ["SmallObjectDetection", "CrossModalInterpret"],
    "shared": ["RegionAttributeDescription", "Calculator"],
}


def required_workers(tasks: List[str]) -> List[str]:
    names: List[str] = []
    for task in tasks:
        for name in TASK45_WORKERS.get(task, []):
            if name not in names:
                names.append(name)
    for name in TASK45_WORKERS["shared"]:
        if name not in names:
            names.append(name)
    return names


def preflight(data_root: Path, tasks: List[str], oneid_arms: bool) -> List[Dict[str, str]]:
    """返回 [{level, item, detail}]，level ∈ ok|warn|error。"""
    checks: List[Dict[str, str]] = []

    def add(level: str, item: str, detail: str) -> None:
        checks.append({"level": level, "item": item, "detail": detail})

    dataset = data_root / "test.jsonl"
    add("ok" if dataset.is_file() else "error", "test.jsonl", str(dataset))

    for name in ("tool_task4_property.xlsx", "tool_task5_property.xlsx"):
        path = data_root / name
        add("ok" if path.is_file() else "warn", name, str(path))

    for task in tasks:
        folder = data_root / ("match_images" if task == "task4" else "time_images")
        add("ok" if folder.is_dir() else "error", f"{task} images", str(folder))

    if oneid_arms:
        flag = os.environ.get("ONEID_ENABLE_WORKBOOK_REPLAY", "")
        add("ok" if flag == "1" else "warn",
            "ONEID_ENABLE_WORKBOOK_REPLAY",
            f"{flag or '<unset>'}（OneID 臂的 workbook replay 需要它，且必须是 worker 启动时的环境）")
        root = os.environ.get("ONEID_DATA_ROOT", "")
        add("ok" if root else "warn", "ONEID_DATA_ROOT", root or "<unset>（默认 <repo>/data，通常不是数据集所在）")

    for env_name in ("ONEID_ABLATION_BASE_MODEL", "ONEID_ABLATION_SFT_MODEL"):
        value = os.environ.get(env_name, "")
        add("ok" if value and Path(value).exists() else "warn", env_name, value or "<unset>")
    return checks


def render_stack_yaml(config_path: Path, out_path: Path, tasks: List[str],
                      exclude: tuple = ()) -> Path:
    """从既有全量配置裁出最小栈；保留 controller 与所需 worker 的原始端口。

    ``exclude`` 用来临时摘掉还没备好权重的 worker（例如依赖 BaseModel 的
    RegionAttributeDescription），而不是让它在注册表里"在但永远失败"。
    """
    import yaml  # 延迟导入：只在需要落 yaml 时才依赖

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    wanted = set(required_workers(tasks)) - set(exclude)

    kept = []
    for entry in raw.get("tool_worker_config") or []:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        name, spec = next(iter(entry.items()))
        if name in wanted:
            kept.append({name: spec})

    dropped = [
        next(iter(entry)) for entry in (raw.get("tool_worker_config") or [])
        if isinstance(entry, dict) and len(entry) == 1 and next(iter(entry)) not in wanted
    ]
    raw["tool_worker_config"] = kept
    raw["model_worker_config"] = []
    raw["_ablations_note"] = (
        f"task45 最小栈：保留 {sorted(wanted)}；移除 {len(dropped)} 个无关 worker。"
        "启动前请确保 ONEID_ENABLE_WORKBOOK_REPLAY / ONEID_DATA_ROOT 已导出。"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_path


def stack_command(config_path: Path) -> str:
    return (
        "python tool_server/tool_workers/scripts/launch_scripts/start_server_local.py "
        f"--config {config_path}"
    )
