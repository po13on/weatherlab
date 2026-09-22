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


def test_mocked_loop_finalizes_with_evidence():
    turns = [
        _response(
            [
                (
                    "compare_metrics",
                    '{"exp_ids":["exp-afno","exp-afno-wmse"],"metric":"ts_1.0"}',
                )
            ]
        ),
        None,
    ]

    def completer(messages, tools):
        result = turns[0]
        if turns[0] is not None:
            turns[0] = None
            return result
        evidence_id = None
        for message in messages:
            if message.get("role") == "tool":
                import json

                evidence_id = json.loads(message["content"]).get("evidence_id")
        payload = (
            '{"claims":[{"kind":"fact","text":"TS@1.0 rose","evidence_ids":["'
            + str(evidence_id)
            + '"]}]}'
        )
        return _response([("finalize_report", payload)])

    result = run_agent("why", {"exp_ids": ["exp-afno", "exp-afno-wmse"]}, completer)
    assert result["verdict"] == "supported"
    assert result["claims"][0]["evidence_ids"]
    assert result["run_metrics"]["llm_calls"] >= 1
    assert result["run_metrics"]["wall_ms"] >= 0
    assert result["result_status"] == "success"


def test_mocked_loop_rejects_root_cause_without_evidence():
    def completer(messages, tools):
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"fact","text":"根因是雷达缺测","evidence_ids":[]}]}',
                )
            ]
        )

    result = run_agent("root cause?", {"run_id": "run-fail-nolog"}, completer, max_steps=1)
    assert result["verdict"] == "insufficient"


def test_run_without_log_cannot_be_supported_by_runbook():
    def completer(messages, tools):
        if completer.step == 0:
            completer.step = 1
            return _response([("extract_log_error", '{"run_id":"run-fail-nolog"}')])
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"open_question","text":"unknown"}]}',
                )
            ]
        )

    completer.step = 0
    result = run_agent(
        "root cause?", {"run_id": "run-fail-nolog"}, completer, max_steps=3
    )
    assert result["verdict"] == "insufficient"
    assert any(item.get("why") == "run_evidence_unavailable" for item in result["missing"])


def test_timeout_injection_degrades():
    def completer(messages, tools):
        if completer.step == 0:
            completer.step = 1
            return _response([("extract_log_error", '{"run_id":"run-fail-oom"}')])
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"open_question","text":"tool timed out"}]}',
                )
            ]
        )

    completer.step = 0
    result = run_agent(
        "log first error?",
        {"run_id": "run-fail-oom", "inject_timeout_tool": "extract_log_error"},
        completer,
        max_steps=3,
    )
    assert result["verdict"] == "insufficient"
    assert result["result_status"] == "degrade"
    assert any(
        (step.get("observation") or {}).get("error") == "timeout"
        for step in result["tool_trace"]
    )


def test_guest_denied_log_degrades():
    def completer(messages, tools):
        names = [item["function"]["name"] for item in tools]
        assert "extract_log_error" not in names
        if completer.step == 0:
            completer.step = 1
            return _response([("extract_log_error", '{"run_id":"run-fail-oom"}')])
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"open_question","text":"no log access"}]}',
                )
            ]
        )

    completer.step = 0
    result = run_agent(
        "root cause?",
        {"run_id": "run-fail-oom"},
        completer,
        max_steps=3,
        role="guest",
    )
    assert result["verdict"] == "insufficient"
    assert result["result_status"] == "degrade"
    assert any(
        (step.get("observation") or {}).get("error") == "tool_denied"
        for step in result["tool_trace"]
    )


def test_guest_cannot_support_oom_from_completeness_only():
    turns = []

    def completer(messages, tools):
        if completer.step == 0:
            completer.step = 1
            return _response(
                [("check_data_completeness", '{"run_id":"run-fail-oom"}')]
            )
        if not turns:
            import json

            evidence_id = None
            for message in messages:
                if message.get("role") == "tool":
                    evidence_id = json.loads(message["content"]).get("evidence_id")
            turns.append(evidence_id)
            payload = (
                '{"claims":[{"kind":"inference","text":"根因是 CUDA OOM","evidence_ids":["'
                + str(evidence_id)
                + '"]}]}'
            )
            return _response([("finalize_report", payload)])
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"open_question","text":"no log access"}]}',
                )
            ]
        )

    completer.step = 0
    result = run_agent(
        "root cause?",
        {"run_id": "run-fail-oom"},
        completer,
        max_steps=4,
        role="guest",
    )
    assert result["verdict"] == "insufficient"
    assert result["result_status"] == "degrade"
    assert any(item.get("why") == "role_not_permitted" for item in result["missing"])


def test_timeout_not_washed_by_completeness():
    def completer(messages, tools):
        if completer.step == 0:
            completer.step = 1
            return _response([("extract_log_error", '{"run_id":"run-fail-oom"}')])
        if completer.step == 1:
            completer.step = 2
            return _response(
                [("check_data_completeness", '{"run_id":"run-fail-oom"}')]
            )
        return _response(
            [
                (
                    "finalize_report",
                    '{"claims":[{"kind":"open_question","text":"log timed out"}]}',
                )
            ]
        )

    completer.step = 0
    result = run_agent(
        "log first error?",
        {"run_id": "run-fail-oom", "inject_timeout_tool": "extract_log_error"},
        completer,
        max_steps=4,
    )
    assert result["verdict"] == "insufficient"
    assert result["result_status"] == "degrade"
    assert any(item.get("why") == "run_evidence_unavailable" for item in result["missing"])


def test_repeated_tool_call_stops():
    def completer(messages, tools):
        return _response(
            [("list_runs", '{"exp_ids":["exp-afno"]}')]
        )

    result = run_agent("list", {"exp_ids": ["exp-afno"]}, completer, max_steps=3)
    assert result["verdict"] == "insufficient"
    assert result["missing"][0]["why"] == "repeated_tool_call"
