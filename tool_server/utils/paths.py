import os
import re
from pathlib import Path
from typing import Any


_BASE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_LOADED = False

_DEFAULT_ENV = {
    "CHECKPOINT_ROOT": "../../00_CPK",
    "LOCAL_MODEL_ROOT": "models",
    "TOOL_WORKERS_DIR": "tool_server/tool_workers",
    "SERVER_LOG_DIR": "${TOOL_WORKERS_DIR}/logs/server_log",
    "TOOLS_OUTPUT_DIR": "${TOOL_WORKERS_DIR}/tools_output",
    "CONTROLLER_ADDR_FILE": "${TOOL_WORKERS_DIR}/online_workers/controller_addr/controller_addr.json",
    "GRADIO_TEMP_DIR": "app/gradio_tmp",
    "PRETRAINED_PATH": "${CHECKPOINT_ROOT}/OpenEarthAgent",
    "BASE_MODEL_PATH": "${CHECKPOINT_ROOT}/Qwen2.5-VL-3B-Instruct",
    "LOGO_PATH": "",
    "QGIS_PREFIX_PATH": "../../miniconda3/envs/tool1",
    "SAR_TO_RGB_MODEL_PATH": "${LOCAL_MODEL_ROOT}/Seg-CycleGAN/175_net_G_A.pth",
    "TEXT_TO_BBOX_MODEL_PATH": "${LOCAL_MODEL_ROOT}/cpk/groundingdino_swint_ogc.pth",
    "TEXT_TO_BBOX_CONFIG_PATH": "${LOCAL_MODEL_ROOT}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "SAM_MODEL_PATH": "${LOCAL_MODEL_ROOT}/sam-2/checkpoints/sam2.1_hiera_large.pt",
    "SAM_MODEL_CONFIG_PATH": "${LOCAL_MODEL_ROOT}/sam-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    "CHANGE_DETECTION_MODEL_BASE": "${CHECKPOINT_ROOT}/TEOChat",
    "OBJECT_DETECTION_MODEL_PATH": "${LOCAL_MODEL_ROOT}/cpk/lae_dino_swint_lae1m-28ca3a15.pth",
    "OBJECT_DETECTION_CONFIG_PATH": "${LOCAL_MODEL_ROOT}/LAE-DINO/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py",
    "GEE_KEY_FILE": "gee_key/my-project-1.json",
    "TIME_SERIES_EXCEL_PATH": "database/database.xlsx",
    "TIME_SERIES_IMAGE_BASE_DIR": "data/train_part1",
    "TIME_SERIES_METADATA_BASE_DIR": "database",
    "EMRDM_ROOT": "${LOCAL_MODEL_ROOT}/EMRDM",
    "CLOUD_REMOVAL_CONFIG_PATH": "${LOCAL_MODEL_ROOT}/cpk/cuhk-20260410T131636Z-3-001/cuhk/configs/2024-11-09T08-59-25-project.yaml",
    "CLOUD_REMOVAL_MODEL_PATH": "${LOCAL_MODEL_ROOT}/cpk/cuhk-20260410T131636Z-3-001/cuhk/checkpoints/last.ckpt",
    "SMALL_OBJECT_DETECTION_MODEL_PATH": "${LOCAL_MODEL_ROOT}/cpk/yolo11s-obb.pt",
    "DENO_DET_V2_MODEL_PATH": "${LOCAL_MODEL_ROOT}/cpk/SARDet-100K/best_coco_bbox_mAP_epoch_12.pth",
    "DENO_DET_V2_CONFIG_PATH": "${LOCAL_MODEL_ROOT}/cpk/SARDet-100K/gfl_fft_attention_r50_fpn_1x_SARDet_100k.py",
    "DENO_DET_V2_CODE_DIR": "${LOCAL_MODEL_ROOT}/DenoDetV2",
}

_PATH_CONFIG_KEYS = {
    "base_dir",
    "log_folder",
    "tools_output_dir",
    "controller_addr_location",
    "script-addr",
    "save_path",
    "model-path",
    "model_path",
    "model-config",
    "model_config",
    "model_base",
    "sam_model_path",
    "sam_model_config",
    "key_file",
    "emrdm-root",
    "emrdm_root",
    "config-path",
    "config_path",
    "excel-path",
    "excel_path",
    "image-base-dir",
    "image_base_dir",
    "metadata-base-dir",
    "metadata_base_dir",
    "denodet-code-dir",
}

_REMOTE_MODEL_KEYS = {"model-path", "model_path", "model_base"}
_LOCAL_PATH_PREFIXES = {
    ".",
    "..",
    "~",
    "assets",
    "data",
    "database",
    "gee_key",
    "models",
    "requirements",
    "scripts",
    "tool_server",
}


def _read_env_file(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_project_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_file = Path(os.environ.get("ENV_FILE", _BASE_PROJECT_ROOT / ".env")).expanduser()
    _read_env_file(env_file)
    os.environ.setdefault("PROJECT_ROOT", str(_BASE_PROJECT_ROOT))
    for key, value in _DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    _ENV_LOADED = True


def expand_env_value(value: str) -> str:
    load_project_env()
    previous = None
    expanded = str(value)
    for _ in range(10):
        if expanded == previous:
            break
        previous = expanded
        expanded = os.path.expandvars(expanded)
    return expanded


def get_project_root() -> Path:
    load_project_env()
    return Path(expand_env_value(os.environ["PROJECT_ROOT"])).expanduser().resolve()


PROJECT_ROOT = get_project_root()


def project_path(value) -> str:
    expanded = expand_env_value(os.fspath(value))
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((PROJECT_ROOT / path).resolve())


def env_path(name: str, default) -> str:
    return project_path(os.environ.get(name, os.fspath(default)))


def existing_or_project_path(value: str) -> str:
    expanded = expand_env_value(value)
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return project_path(expanded)


def _is_url(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value))


def _looks_like_remote_model_id(value: str, key) -> bool:
    if key not in _REMOTE_MODEL_KEYS:
        return False
    if value.startswith(("/", ".", "~", "$")):
        return False
    first_part = value.split("/", 1)[0]
    if first_part in _LOCAL_PATH_PREFIXES:
        return False
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value))


def _should_resolve_path(value: str, key) -> bool:
    if not value or value == "ENV" or _is_url(value):
        return False
    if _looks_like_remote_model_id(value, key):
        return False
    return True


def resolve_config_paths(value: Any, key=None) -> Any:
    if isinstance(value, dict):
        return {k: resolve_config_paths(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_config_paths(item, key) for item in value]
    if not isinstance(value, str):
        return value

    expanded = expand_env_value(value)
    if key in _PATH_CONFIG_KEYS and _should_resolve_path(expanded, key):
        return project_path(expanded)
    return expanded
