from tool_server.tool_workers.skills.router import SkillRouter


def test_router_prefers_task_level_cross_modal():
    router = SkillRouter(min_score=0.0)
    result = router.route("Please cross-validate whether aircraft targets exist in the RGB and SAR images, and give the final conclusion.", {"primary_image": "/tmp/rgb.png", "sar_image": "/tmp/sar.png"})
    assert result
    assert result[0]["skill_name"] == "CrossModalTaskSkill"


def test_router_prefers_match_for_pairs():
    router = SkillRouter(min_score=0.0)
    result = router.route("Match RGB objects with SAR candidates and return matched pairs.", {"primary_image": "/tmp/rgb.png", "sar_image": "/tmp/sar.png"})
    assert result
    assert result[0]["skill_name"] == "CrossModalMatchSkill"
