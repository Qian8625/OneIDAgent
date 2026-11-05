import json
import os
import sys
import tempfile
import time
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.utils.paths import env_path, load_project_env
from tool_server.tool_workers.skills.plugins.cross_modal_match.worker import CrossModalMatchSkillWorker
from tool_server.tool_workers.skills.plugins.cross_modal_task.worker import CrossModalTaskSkillWorker
from tool_server.tool_workers.skills.plugins.object_detect_select.worker import ObjectDetectSelectSkillWorker
from tool_server.tool_workers.skills.plugins.object_measure.worker import ObjectMeasureSkillWorker
from tool_server.tool_workers.skills.plugins.object_visual_output.worker import ObjectVisualOutputSkillWorker
from tool_server.tool_workers.skills.plugins.sar_evidence_extract.worker import SAREvidenceExtractSkillWorker

load_project_env()

"""
Test script for all registered skill workers.

Design goals:
1. Keep the calling style close to tools_single_test.py / tools_test.py.
2. Run every skill once with a representative example payload.
3. Reuse repo-local fixtures when available, and gracefully skip tests when
   required sample data is missing in the current environment.
4. Print structured results for quick manual verification.
"""

LEGACY_REPO_ROOTS = [
    os.environ.get("PROJECT_ROOT"),
    os.environ.get("OPENEARTHAGENT_ROOT"),
    os.environ.get("WEAVELIGHT_ROOT"),
    os.environ.get("LEGACY_FIXTURE_ROOT"),
]


def abs_path(rel_path: str) -> str:
    """Convert a path relative to this script into an absolute path."""
    return os.path.abspath(os.path.join(SCRIPT_DIR, rel_path))


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_existing_path(*candidates: Optional[str]) -> Optional[str]:
    """
    Resolve the first existing path from a list of absolute/relative candidates.

    Search order for relative paths:
      - relative to this script
      - relative to repository root
      - relative to common legacy roots
    """
    probe_paths: List[str] = []

    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expandvars(os.path.expanduser(str(candidate).strip()))
        if not expanded:
            continue

        probe_paths.append(expanded)
        if not os.path.isabs(expanded):
            probe_paths.append(os.path.join(SCRIPT_DIR, expanded))
            probe_paths.append(os.path.join(REPO_ROOT, expanded))
            for root in LEGACY_REPO_ROOTS:
                if root:
                    probe_paths.append(os.path.join(root, expanded))

    for path in _dedupe_keep_order(probe_paths):
        if os.path.exists(path):
            return os.path.abspath(path)

    return None


def build_scratch_path(scratch_dir: str, filename: str) -> str:
    return os.path.abspath(os.path.join(scratch_dir, filename))


def build_direct_skill_workers() -> Dict[str, Any]:
    """Build local skill workers for smoke tests when no controller is running."""
    controller_addr = "http://localhost:20001"
    workers = {
        "ObjectDetectSelectSkill": ObjectDetectSelectSkillWorker(
            controller_addr=controller_addr,
            no_register=True,
            port=20127,
        ),
        "ObjectMeasureSkill": ObjectMeasureSkillWorker(
            controller_addr=controller_addr,
            no_register=True,
            port=20128,
        ),
        "ObjectVisualOutputSkill": ObjectVisualOutputSkillWorker(
            controller_addr=controller_addr,
            no_register=True,
            port=20129,
        ),
        "SAREvidenceExtractSkill": SAREvidenceExtractSkillWorker(
            controller_addr=controller_addr,
            no_register=True,
            port=20130,
        ),
        "CrossModalTaskSkill": CrossModalTaskSkillWorker(
            controller_addr=controller_addr,
            no_register=True,
            port=20131,
        ),
        "CrossModalMatchSkill": CrossModalMatchSkillWorker(
            controller_addr=controller_addr,
            no_register=True,
            port=20132,
        ),
    }

    def fixture_detect_targets(self, image, target, detector, confidence_threshold, iou_threshold, width, height):
        bbox = [max(0, width // 4), max(0, height // 4), max(1, width // 4 + 40), max(1, height // 4 + 60)]
        bbox[2] = min(width, bbox[2])
        bbox[3] = min(height, bbox[3])
        return [
            {"bbox": bbox, "score": max(float(confidence_threshold), 0.9), "label": str(target or "object")}
        ], [{"tool": "direct_fixture", "status": "success", "parsed_detections": 1}]

    def fixture_run_detector(self, image, detector, confidence_threshold, skill_trace, internal_tool_calls):
        skill_trace.append({"stage": "detect_sar_candidates", "status": "success", "tool": "direct_fixture"})
        return [
            {"bbox": [10, 10, 60, 60], "score": max(float(confidence_threshold), 0.9), "label": "ship", "class_id": 0}
        ]

    workers["ObjectDetectSelectSkill"]._detect_targets = MethodType(fixture_detect_targets, workers["ObjectDetectSelectSkill"])
    workers["SAREvidenceExtractSkill"]._run_detector = MethodType(fixture_run_detector, workers["SAREvidenceExtractSkill"])
    return workers


def ensure_rgb_proxy_for_sar(
    tm: ToolManager,
    sar_image: Optional[str],
    preferred_rgb: Optional[str],
    scratch_dir: str,
) -> Optional[str]:
    """
    Try to prepare an RGB-like proxy image for cross-modal skill tests.

    Priority:
      1. existing preferred_rgb fixture
      2. generate via SARToRGB
    """
    if preferred_rgb and os.path.isfile(preferred_rgb):
        return preferred_rgb
    if not sar_image or not os.path.isfile(sar_image):
        return None

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    sar_to_rgb_out = build_scratch_path(scratch_dir, "sar_rgb_proxy.png")
    if "SARToRGB" in tm.available_tools:
        candidates.append((
            "SARToRGB",
            {"image": sar_image, "output_path": sar_to_rgb_out},
        ))

    for tool_name, params in candidates:
        try:
            resp = tm.call_tool(tool_name, params)
            image_path = resp.get("image") or resp.get("output_path")
            if resp.get("error_code") == 0 and image_path and os.path.isfile(image_path):
                print(f"Prepared proxy RGB for SAR using {tool_name}: {image_path}")
                return image_path
            print(f"⚠️  Failed to prepare proxy RGB with {tool_name}: {resp.get('text', '')}")
        except Exception as exc:
            print(f"⚠️  Exception while preparing proxy RGB with {tool_name}: {exc}")

    return None


def discover_fixtures(tm: ToolManager, scratch_dir: str) -> Dict[str, Optional[str]]:
    """Discover sample assets used by the skill test suite."""
    rgb_image = resolve_existing_path(
        "tools_test_images/1.jpg",
        "scripts/tools_test/tools_test_images/1.jpg",
    )
    change_pre = resolve_existing_path(
        "tools_test_images/xBD_cls_1.png",
        "scripts/tools_test/tools_test_images/xBD_cls_1.png",
    )
    change_post = resolve_existing_path(
        "tools_test_images/xBD_cls_2.png",
        "scripts/tools_test/tools_test_images/xBD_cls_2.png",
    )
    geotiff = resolve_existing_path(
        "tools_test_images/S_07.tif",
        "scripts/tools_test/tools_test_images/S_07.tif",
    )

    sar_raw = resolve_existing_path(
        "scripts/tools_test/tools_test_images/sar_test_1.png",
    )
    sar_rgb = resolve_existing_path(
        env_path("SAR_TEST_RGB_IMAGE", "scripts/tools_test/tools_test_images/sar_to_rgb_result.png"),
        "scripts/tools_test/tools_test_images/sar_to_rgb_result.png",
    )
    sar_rgb = ensure_rgb_proxy_for_sar(
        tm=tm,
        sar_image=sar_raw,
        preferred_rgb=sar_rgb,
        scratch_dir=scratch_dir,
    )
    assets_rgb = resolve_existing_path(
        "assets/Singapore_clip_6_9_RGB.tif",
        "assets/Singapore_clip_6_9_RGB_preview.png",
    )
    assets_sar = resolve_existing_path(
        "assets/Singapore_clip_6_9_SAR.tif",
        "assets/Singapore_clip_6_9_SAR_preview.png",
    )

    return {
        "rgb_image": rgb_image,
        "change_pre": change_pre,
        "change_post": change_post,
        "geotiff": geotiff,
        "sar_raw": sar_raw,
        "sar_rgb": sar_rgb,
        "assets_rgb": assets_rgb,
        "assets_sar": assets_sar,
    }


def required_paths_exist(paths: Iterable[Optional[str]]) -> Tuple[bool, List[str]]:
    missing = [p for p in paths if not p or not os.path.exists(p)]
    return len(missing) == 0, missing


def print_fixture_summary(fixtures: Dict[str, Optional[str]]) -> None:
    print("\n=== Fixture Discovery Summary ===")
    for key, value in fixtures.items():
        status = value if value else "<missing>"
        print(f"- {key}: {status}")
    print()


def build_skill_examples(fixtures: Dict[str, Optional[str]], scratch_dir: str):
    return [
        {
            "skill_name": "ObjectDetectSelectSkill",
            "required_paths": [fixtures["rgb_image"]],
            "params": {
                "image": fixtures["rgb_image"],
                "target": "red car",
                "selection_mode": "all",
                "detector": "text_to_bbox",
                "visualize": True,
                "max_draw": 3,
            },
        },
        {
            "skill_name": "ObjectMeasureSkill",
            "required_paths": [fixtures["rgb_image"]],
            "params": {
                "image": fixtures["rgb_image"],
                "target": "red car",
                "measure_type": "area",
                "bbox": [289, 645, 330, 724],
                "gsd_m_per_pixel": 0.3,
                "visualize": True,
            },
        },
        {
            "skill_name": "ObjectVisualOutputSkill",
            "required_paths": [fixtures["rgb_image"]],
            "params": {
                "image": fixtures["rgb_image"],
                "visual_type": "box_and_label",
                "bbox": [289, 645, 330, 724],
                "labels": ["red car"],
                "output_path": build_scratch_path(scratch_dir, "object_visual_red_car.png"),
            },
        },
        {
            "skill_name": "SAREvidenceExtractSkill",
            "required_paths": [fixtures["sar_raw"]],
            "params": {
                "sar_image": fixtures["sar_raw"],
                "target": "ship",
                "preprocess_mode": "auto",
                "confidence_threshold": 0.25,
                "visualize": True,
                "max_candidates": 20,
            },
        },
        {
            "skill_name": "CrossModalMatchSkill",
            "required_paths": [fixtures["assets_rgb"], fixtures["assets_sar"]],
            "params": {
                "rgb_image": fixtures["assets_rgb"],
                "sar_image": fixtures["assets_sar"],
                "target": "ship",
                "rgb_detections": [{"bbox": [10, 10, 60, 60], "score": 0.9, "label": "ship"}],
                "sar_candidates": [{"bbox": [12, 12, 62, 62], "score": 0.8, "label": "ship"}],
                "match_strategy": "hybrid",
                "score_threshold": 0.3,
                "visualize": True,
                "max_draw": 5,
            },
        },
        {
            "skill_name": "CrossModalTaskSkill",
            "required_paths": [fixtures["assets_rgb"], fixtures["assets_sar"]],
            "params": {
                "rgb_image": fixtures["assets_rgb"],
                "sar_image": fixtures["assets_sar"],
                "target": "ship",
                "task_type": "optical_sar_verify",
                "question": "Verify with SAR evidence whether the optical ship detections are credible.",
                "matched_pairs": [
                    {
                        "rgb_bbox": [10, 10, 60, 60],
                        "sar_bbox": [12, 12, 62, 62],
                        "match_score": 0.82,
                        "iou": 0.7,
                        "match_type": "strong_match",
                        "rgb_label": "ship",
                        "sar_label": "ship",
                    }
                ],
                "visualize": True,
                "max_draw": 5,
            },
        },
    ]


def main():
    print("\n=== Initializing Tool Manager ===")
    tm = ToolManager(controller_url_location=None)
    scratch_dir = tempfile.mkdtemp(prefix="skill_test_")
    print(f"Using scratch directory: {scratch_dir}")
    available_tool_names = set(getattr(tm, "internal_available_tools", []) or []) | set(getattr(tm, "available_tools", []) or [])
    direct_workers = build_direct_skill_workers()
    if not any(name in available_tool_names for name in direct_workers):
        print("No online skill workers are registered; using direct local skill workers for smoke tests.")

    fixtures = discover_fixtures(tm, scratch_dir)
    print_fixture_summary(fixtures)

    example_calls = build_skill_examples(fixtures, scratch_dir)

    print("\n=== Running tests for all skill workers ===\n")

    skills_with_errors: List[str] = []
    skills_successful: List[str] = []
    skipped_skills: List[str] = []

    skipped = 0
    success = 0
    error = 0

    for item in example_calls:
        skill_name = item["skill_name"]
        params = item["params"]
        required_paths = item.get("required_paths", [])

        use_direct_worker = skill_name not in available_tool_names and skill_name in direct_workers
        if skill_name not in available_tool_names and not use_direct_worker:
            print(f"⚠️  {skill_name} is not registered or unavailable — skipping.\n")
            skipped += 1
            skipped_skills.append(skill_name)
            continue

        ok, missing = required_paths_exist(required_paths)
        if not ok:
            print(f"⚠️  {skill_name} is skipped because required fixtures are missing:")
            for path in missing:
                print(f"   - {path}")
            print()
            skipped += 1
            skipped_skills.append(skill_name)
            continue

        mode = "direct local worker" if use_direct_worker else "registered ToolManager worker"
        print(f"🔹 Testing {skill_name} via {mode} with parameters:")
        print(json.dumps(params, indent=2, ensure_ascii=False))

        try:
            if use_direct_worker:
                result = direct_workers[skill_name].generate(params)
            else:
                result = tm.call_tool(skill_name, params)

            if result.get("error_code") == 0:
                print(f"✅ {skill_name} executed successfully.")
                print(f"Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                success += 1
                skills_successful.append(skill_name)
            else:
                print(f"❌ {skill_name} returned an error.")
                print(f"Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                error += 1
                skills_with_errors.append(skill_name)

        except Exception as exc:
            print(f"❌ Exception while testing {skill_name}: {exc}\n")
            error += 1
            skills_with_errors.append(skill_name)

        print("-" * 100 + "\n")
        time.sleep(0.5)

    print("\n🎯 All example skills executed in sequence.\n")
    print("📊 Summary:")
    print(f"   ✅ Successful : {success}")
    print(f"   ❌ Failed     : {error}")
    print(f"   ⚠️  Skipped    : {skipped}\n")

    if skills_successful:
        print(f"✅ Successful skills:\n   {sorted(set(skills_successful))}\n")

    if skills_with_errors:
        print(f"❌ Failed skills:\n   {sorted(set(skills_with_errors))}\n")

    if skipped_skills:
        print(f"⚠️  Skipped skills:\n   {sorted(set(skipped_skills))}\n")

    if error == 0 and skipped == 0:
        print("🎉 All skills ran successfully without errors.")


if __name__ == "__main__":
    main()
