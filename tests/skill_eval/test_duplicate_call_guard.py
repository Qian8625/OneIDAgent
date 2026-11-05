from tool_server.tf_eval.agent_runtime.agent_runner import call_signature


def test_call_signature_is_stable():
    first = call_signature("OCR", {"image": "/tmp/a.png", "input_registry": {"primary_image": "/tmp/a.png"}})
    second = call_signature("OCR", {"input_registry": {"primary_image": "/tmp/a.png"}, "image": "/tmp/a.png"})
    assert first == second
