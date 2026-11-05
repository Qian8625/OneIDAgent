import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def _is_languagebind_image_tower(path):
    if path is None:
        return False

    path = str(path)
    name = os.path.basename(path.rstrip("/"))

    return (
        path == "LanguageBind/LanguageBind_Image"
        or path.endswith("LanguageBind/LanguageBind_Image")
        or name == "LanguageBind_Image"
    )


def _is_languagebind_video_tower(path):
    if path is None:
        return False

    path = str(path)
    name = os.path.basename(path.rstrip("/"))

    return (
        path == "LanguageBind/LanguageBind_Video_merge"
        or path.endswith("LanguageBind/LanguageBind_Video_merge")
        or name == "LanguageBind_Video_merge"
    )


def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(
        image_tower_cfg,
        "mm_image_tower",
        getattr(image_tower_cfg, "image_tower", None)
    )

    cache_dir = getattr(image_tower_cfg, "cache_dir", "./cache_dir")

    if image_tower is None:
        raise ValueError("image_tower is None")

    # 注意：LanguageBind 判断必须放在 os.path.exists 前面
    # 否则 /home/ubuntu/00_CPK/LanguageBind_Image 会被误分发到 CLIPVisionTower
    if _is_languagebind_image_tower(image_tower):
        print("[DEBUG] build_image_tower -> LanguageBindImageTower:", image_tower)
        return LanguageBindImageTower(
            image_tower,
            args=image_tower_cfg,
            cache_dir=cache_dir,
            **kwargs
        )

    if (
        os.path.exists(image_tower)
        or image_tower.startswith("openai")
        or image_tower.startswith("laion")
    ):
        print("[DEBUG] build_image_tower -> CLIPVisionTower:", image_tower)
        return CLIPVisionTower(
            image_tower,
            args=image_tower_cfg,
            **kwargs
        )

    raise ValueError(f"Unknown image tower: {image_tower}")


def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(
        video_tower_cfg,
        "mm_video_tower",
        getattr(video_tower_cfg, "video_tower", None)
    )

    cache_dir = getattr(video_tower_cfg, "cache_dir", "./cache_dir")

    if video_tower is None:
        raise ValueError("video_tower is None")

    if _is_languagebind_video_tower(video_tower):
        print("[DEBUG] build_video_tower -> LanguageBindVideoTower:", video_tower)
        return LanguageBindVideoTower(
            video_tower,
            args=video_tower_cfg,
            cache_dir=cache_dir,
            **kwargs
        )

    raise ValueError(f"Unknown video tower: {video_tower}")

# ============================================================================================================