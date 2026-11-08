"""静态接线断言：共享实现只能有一份，六个 app 变体都不能漏接线。

app 变体不能真的 import（模块级会拉 gradio/torch 并构建 UI），所以用源码检查代替：
每个 app 变体都必须同时具备"登记检测"和"注入检测"两个调用点，且共享实现只有一份。
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_FILES = ["app/app.py"]
EVIDENCE_MODULE = REPO_ROOT / "tool_server" / "tool_workers" / "cross_modal_evidence.py"


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def test_every_app_registers_and_injects_detections():
    for name in APP_FILES:
        source = _read(name)
        assert "from tool_server.tool_workers.cross_modal_evidence import" in source, name
        assert 'register_detections(registry, api_name, tool_resp.get("structured"), round_id, api_args)' in source, name
        assert "prepare_interpret_args(injected_args, registry)" in source, name
        # 陈旧绑定不得再出现
        assert "georef_image" not in source, name


def test_each_app_has_exactly_one_call_site():
    for name in APP_FILES:
        source = _read(name)
        # 导入语句没有括号，因此计数 1 就是"恰好一处调用"
        assert source.count("register_detections(") == 1, name
        assert source.count("prepare_interpret_args(") == 1, name


def test_evidence_pipeline_has_a_single_implementation():
    definitions = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {"DS_VI", "models", "tests", "__pycache__", ".git"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "def prepare_interpret_args(" in text:
            definitions.append(path.relative_to(REPO_ROOT).as_posix())
    assert definitions == ["tool_server/tool_workers/cross_modal_evidence.py"], definitions


def test_agent_runner_delegates_to_shared_module():
    source = _read("tool_server/tf_eval/agent_runtime/agent_runner.py")
    assert "from tool_server.tool_workers.cross_modal_evidence import" in source
    assert "def prepare_cross_modal_interpret_args(" not in source
    assert "register_detections(registry, api_name, structured_result, round_id, api_args)" in source


def test_replay_eval_binding_does_not_inject_live_georef():
    import ast
    import typing
    source = _read("tool_server/tf_eval/agent_runtime/agent_runner.py")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "inject_runtime_arguments"]
    ns = {"Dict": typing.Dict, "Any": typing.Any, "Optional": typing.Optional,
          "TOOL_ARG_BINDINGS": {"CrossModalInterpret": {"georef_image": "optical_image"}},
          "TOOL_DEFAULT_ARGUMENTS": {}, "GPKG_REQUIRED_TOOLS": set(),
          "resolve_argument_references": lambda args, registry: args}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "runner_binding", "exec"), ns)
    result = ns["inject_runtime_arguments"]("CrossModalInterpret", {"source_mode": "workbook_replay"}, {"optical_image": "/data/rgb.png"}, None)
    assert result == {"source_mode": "workbook_replay"}
