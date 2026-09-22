from types import SimpleNamespace

from agent import run_agent


def _response(calls):
    tool_calls = []
    for index, (name, arguments) in enumerate(calls):
        tool_calls.append(
            SimpleNamespace(
                id=f"call-{index}",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        )
    message = SimpleNamespace(content="", tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_hitl_reject_stops_at_insufficient():
    def completer(messages, tools):
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"open_question","text":"need review"}]}',
                )
            ]
        )

    result = run_agent(
        "review me",
        {"run_id": "run-fail-nolog"},
        completer,
        max_steps=1,
        hitl_gate=lambda claims: {"action": "reject"},
    )
    assert result["verdict"] == "insufficient"
    assert any(item.get("why") == "hitl_rejected" for item in result["missing"])


def test_hitl_timeout_is_insufficient():
    def completer(messages, tools):
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"open_question","text":"need review"}]}',
                )
            ]
        )

    result = run_agent(
        "review me",
        {"exp_ids": ["exp-afno"]},
        completer,
        max_steps=2,
        hitl_gate=lambda claims: {"action": "timeout"},
    )
    assert result["verdict"] == "insufficient"
    assert any(item.get("why") == "hitl_timeout" for item in result["missing"])
