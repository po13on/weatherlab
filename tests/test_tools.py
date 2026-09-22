from tools import (
    check_data_completeness,
    compare_metrics,
    dispatch,
    extract_log_error,
    get_config_diff,
    list_runs,
    retrieve_runbook,
)


def test_list_runs_unknown_exp():
    assert list_runs(["no-such"])["error"] == "unknown_exp"


def test_config_diff_only_loss():
    result = get_config_diff("exp-afno", "exp-afno-wmse")
    assert result["ok"] is True
    fields = {item["field"] for item in result["payload"]["diffs"]}
    assert fields == {"loss"}


def test_compare_metrics_wmse_vs_afno():
    result = compare_metrics(["exp-afno", "exp-afno-wmse"], "ts_1.0")
    assert result["payload"]["stats"]["exp-afno"]["mean"] == 0.41
    assert result["payload"]["stats"]["exp-afno-wmse"]["mean"] == 0.47
    assert result["payload"]["pairwise"][0]["rel_pct"] == 14.6
    assert result["payload"]["same_direction_across_seeds"] is True


def test_compare_metrics_corr_not_same_direction():
    result = compare_metrics(["exp-afno", "exp-afno-corr"], "ts_1.0")
    assert result["payload"]["same_direction_across_seeds"] is False


def test_compare_metrics_rejects_train_split():
    result = compare_metrics(["exp-afno", "exp-afno-wmse"], "ts_1.0", split="train")
    assert result["error"] == "incomparable_metric"


def test_missing_manifest_and_log():
    assert check_data_completeness("run-fail-nolog")["error"] == "manifest_unavailable"
    assert extract_log_error("run-fail-nolog")["error"] == "log_unavailable"


def test_oom_log_is_untrusted_and_keeps_cuda_error():
    result = extract_log_error("run-fail-oom")
    assert result["payload"]["first_error"].startswith("ERROR torch.cuda")
    assert result["payload"]["lines"][0].startswith("[untrusted_log]")
    assert any("hardware failure" in line.lower() for line in result["payload"]["lines"])


def test_dispatch_adds_evidence_id():
    result = dispatch("list_runs", {"exp_ids": ["exp-afno"]})
    assert result["ok"] is True
    assert result["latency_ms"] >= 0
    timed = dispatch(
        "extract_log_error",
        {"run_id": "run-fail-oom"},
        inject_timeout=True,
    )
    assert timed["error"] == "timeout"
    assert timed["retries"] == 1
    assert "evidence_id" not in timed
    assert result["evidence_id"].startswith("ev-")


def test_empty_runbook_query_returns_nothing():
    empty = retrieve_runbook()
    assert empty["payload"]["count"] == 0
    oom = retrieve_runbook("CUDA OOM correction head halve batch_size")
    ids = [hit["doc_id"] for hit in oom["payload"]["hits"]]
    assert ids
    assert ids[0] == "rb-oom"
    tagged = retrieve_runbook(tag="oom")
    assert {hit["doc_id"] for hit in tagged["payload"]["hits"]} >= {"rb-oom"}
