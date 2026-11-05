from tool_server.tf_eval.agent_runtime.run_config import RunMode


def test_oracle_forced_mode_exists():
    assert RunMode.normalize("oracle_forced_skill") == RunMode.ORACLE_FORCED_SKILL
