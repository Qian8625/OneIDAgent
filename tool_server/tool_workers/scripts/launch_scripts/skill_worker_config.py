from typing import Dict

from tool_server.tool_workers.skills.manifest import iter_skill_worker_configs
from tool_server.tool_workers.skills.registry import SKILL_MANIFESTS


def append_manifest_skill_workers(config: Dict) -> Dict:
    """
    Append skill workers declared by skills/plugins/*/manifest.yaml
    into config["tool_worker_config"].

    Existing YAML entries win, so this is backward-compatible.
    """
    config = dict(config or {})
    tool_worker_config = list(config.get("tool_worker_config") or [])

    existing_worker_names = set()
    for item in tool_worker_config:
        if not isinstance(item, dict) or not item:
            continue
        worker_name = next(iter(item.keys()))
        existing_worker_names.add(worker_name)

    for skill_cfg in iter_skill_worker_configs(SKILL_MANIFESTS):
        worker_name = skill_cfg["worker_name"]

        if not worker_name:
            continue

        if worker_name in existing_worker_names:
            continue

        cmd = {
            "script-addr": skill_cfg["script_addr"],
            "host": skill_cfg.get("host", "0.0.0.0"),
        }

        if skill_cfg.get("port") is not None:
            cmd["port"] = skill_cfg["port"]

        entry = {
            worker_name: {
                "worker_name": worker_name,
                "job_name": skill_cfg.get("job_name") or f"{worker_name.lower()}_skill",
                "calculate_type": skill_cfg.get("calculate_type") or "control",
                "cmd": cmd,
            }
        }

        if skill_cfg.get("conda_env"):
            entry[worker_name]["conda_env"] = skill_cfg["conda_env"]

        if skill_cfg.get("cuda_visible_devices"):
            entry[worker_name]["cuda_visible_devices"] = skill_cfg["cuda_visible_devices"]

        tool_worker_config.append(entry)
        existing_worker_names.add(worker_name)

    config["tool_worker_config"] = tool_worker_config
    return config
