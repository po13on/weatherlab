from policy import allow_tool, deny_result, normalize_role, openai_tools_for_role, redact_observation
from tools import OPENAI_TOOLS, dispatch


def test_guest_cannot_read_logs_or_runbooks():
    assert allow_tool("analyst", "extract_log_error") is True
    assert allow_tool("guest", "extract_log_error") is False
    denied = dispatch("extract_log_error", {"run_id": "run-fail-oom"}, role="guest")
    assert denied["error"] == "tool_denied"
    names = [
        item["function"]["name"]
        for item in openai_tools_for_role("guest", OPENAI_TOOLS)
    ]
    assert "extract_log_error" not in names
    assert "compare_metrics" in names


def test_guest_redacts_if_payload_leaks():
    raw = {
        "ok": True,
        "payload": {"lines": ["secret", "ERROR oom"], "first_error": "ERROR oom"},
    }
    redacted = redact_observation("guest", "extract_log_error", raw)
    assert redacted["payload"]["lines"][0].startswith("[redacted]")


def test_unknown_role_is_guest_not_analyst():
    assert normalize_role("admin") == "guest"
    assert normalize_role(None) == "analyst"
    assert allow_tool("admin", "extract_log_error") is False


def test_deny_result_shape():
    payload = deny_result("guest", "retrieve_runbook")
    assert payload["error"] == "tool_denied"
    assert payload["role"] == "guest"
