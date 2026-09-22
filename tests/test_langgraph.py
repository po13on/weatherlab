import pytest

from runners.langgraph_baseline import HAS_LANGGRAPH


@pytest.mark.skipif(not HAS_LANGGRAPH, reason="langgraph extra needs Python 3.10+")
def test_langgraph_runner_matches_handwritten_contract():
    from types import SimpleNamespace

    from runners.langgraph_baseline import run_langgraph_agent

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

    turns = [
        _response(
            [
                (
                    "compare_metrics",
                    '{"exp_ids":["exp-afno","exp-afno-wmse"],"metric":"ts_1.0"}',
                )
            ]
        )
    ]

    def completer(messages, tools):
        if turns:
            return turns.pop(0)
        import json

        evidence_id = None
        for message in messages:
            if message.get("role") == "tool":
                evidence_id = json.loads(message["content"]).get("evidence_id")
        payload = (
            '{"claims":[{"kind":"fact","text":"TS@1.0 rose","evidence_ids":["'
            + str(evidence_id)
            + '"]}]}'
        )
        return _response([("finalize_report", payload)])

    result = run_langgraph_agent(
        "why",
        {"exp_ids": ["exp-afno", "exp-afno-wmse"]},
        completer,
    )
    assert result["runner"] == "langgraph"
    assert result["verdict"] == "supported"
    assert result["claims"][0]["evidence_ids"]
