from tool_server.tf_eval.agent_runtime.agent_runner import (
    build_oracle_skill_args,
    inject_runtime_arguments_with_overrides,
)


def test_deno_det_v2_prefers_sar_image():
    registry = {"primary_image": "/tmp/rgb.png", "sar_image": "/tmp/sar.png"}
    args, overrides = inject_runtime_arguments_with_overrides("DenoDetV2", {"image": "primary_image"}, registry, None)
    assert args["image"] == "/tmp/sar.png"
    assert overrides


def test_oracle_skill_args_for_cross_modal_task():
    registry = {"primary_image": "/tmp/rgb.png", "optical_image": "/tmp/rgb.png", "sar_image": "/tmp/sar.png"}
    args = build_oracle_skill_args("CrossModalTaskSkill", "Please cross-validate whether aircraft targets exist in the RGB and SAR images.", registry, {"task_family": "cross_modal_task"})
    assert args["skill_name"] == "CrossModalTaskSkill"
    assert args["task_type"] == "optical_sar_verify"
    assert args["rgb_image"] == "/tmp/rgb.png"
    assert args["sar_image"] == "/tmp/sar.png"
