from eval_cases import EVAL_CASES
from eval_suite import extract_json_object, run_eval, score_result


def test_twenty_machine_readable_cases():
    assert [item["id"] for item in EVAL_CASES] == [f"Q{i}" for i in range(1, 21)]
    cats = {item.get("category") for item in EVAL_CASES}
    assert {"normal", "noise", "failure"} <= cats
    for case in EVAL_CASES:
        assert case["question"]
        assert case["scope"]
        assert case["expected_verdict"] in {"supported", "insufficient", "contradictory"}
        assert "must_tools" in case


def test_score_q7_insufficient_and_q10_forbidden():
    q7 = next(item for item in EVAL_CASES if item["id"] == "Q7")
    q10 = next(item for item in EVAL_CASES if item["id"] == "Q10")
    good = score_result(
        q7,
        {
            "verdict": "insufficient",
            "tool_trace": [
                {"name": "extract_log_error", "observation": {"error": "log_unavailable"}},
                {"name": "check_data_completeness", "observation": {"error": "manifest_unavailable"}},
            ],
            "claims": [{"kind": "open_question", "text": "no local evidence", "evidence_ids": []}],
            "evidence_ids": ["ev-1", "ev-2"],
        },
    )
    assert good["verdict_ok"] is True
    assert good["must_tools_ok"] is True

    bad = score_result(
        q10,
        {
            "verdict": "supported",
            "tool_trace": [{"name": "extract_log_error", "observation": {"ok": True}}],
            "claims": [
                {
                    "kind": "inference",
                    "text": "根因是硬件故障",
                    "evidence_ids": ["ev-oom"],
                }
            ],
            "evidence_ids": ["ev-oom"],
        },
    )
    assert "硬件故障" in bad["forbidden_hit"]


def test_run_eval_with_mock_agent_does_not_call_llm():
    def runner(case):
        return {
            "verdict": case["expected_verdict"],
            "tool_trace": [{"name": name, "observation": {"ok": True}} for name in case["must_tools"]],
            "claims": [{"kind": "fact", "text": "fixture fact", "evidence_ids": ["ev-1"]}],
            "evidence_ids": ["ev-1"],
            "steps_used": len(case["must_tools"]),
        }

    report = run_eval(runner, kind="system")
    assert report["summary"]["n"] == 20
    assert report["summary"]["verdict_accuracy"] == 1.0
    assert report["summary"]["tool_selection_accuracy"] == 1.0
    assert report["kind"] == "system"
    assert set(report["summary"]["by_category"]) >= {"normal", "noise", "failure"}
    assert "p50_wall_ms" in report["summary"]
    assert "p95_wall_ms" in report["summary"]


def test_extract_json_from_fenced_text():
    parsed = extract_json_object('```json\n{"verdict":"insufficient","claims":[]}\n```')
    assert parsed["verdict"] == "insufficient"
