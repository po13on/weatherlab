from critic import critique


def test_rejects_claim_without_pointer():
    judged = critique(
        [{"kind": "fact", "text": "根因肯定是雷达缺测", "evidence_ids": []}],
        ["ev-1"],
    )
    assert judged["ok"] is False
    assert judged["verdict"] == "insufficient"


def test_rejects_unknown_evidence_id():
    judged = critique(
        [{"kind": "fact", "text": "x", "evidence_ids": ["ev-missing"]}],
        ["ev-1"],
    )
    assert judged["ok"] is False


def test_accepts_supported_fact():
    judged = critique(
        [
            {"kind": "fact", "text": "TS@1.0 0.41 -> 0.47", "evidence_ids": ["ev-1"]},
            {"kind": "open_question", "text": "没有权重表", "evidence_ids": []},
        ],
        ["ev-1"],
    )
    assert judged == {"ok": True, "verdict": "supported", "supported_count": 1}


def test_rejects_hardware_story_when_log_is_cuda_oom():
    judged = critique(
        [{"kind": "inference", "text": "根因是硬件故障", "evidence_ids": ["ev-log"]}],
        ["ev-log"],
        {
            "ev-log": {
                "name": "extract_log_error",
                "payload": {
                    "first_error": "ERROR torch.cuda: CUDA out of memory. Tried to allocate 2.41 GiB"
                },
            }
        },
    )
    assert judged["ok"] is False
    assert judged["missing"][0]["why"] == "claim_contradicts_log_error"


def test_rejects_runbook_claim_on_the_wrong_page():
    judged = critique(
        [
            {
                "kind": "fact",
                "text": "halve batch_size",
                "page": 1,
                "span": "halve batch_size",
                "evidence_ids": ["ev-rb"],
            }
        ],
        ["ev-rb"],
        {
            "ev-rb": {
                "name": "retrieve_runbook",
                "payload": {
                    "hits": [
                        {
                            "doc_id": "rb-oom",
                            "page": 2,
                            "quote": "First action: halve batch_size.",
                        }
                    ]
                },
            }
        },
    )
    assert judged["ok"] is False
    assert judged["missing"][0]["why"] == "wrong_page"


def test_rejects_same_page_number_from_another_runbook():
    judged = critique(
        [
            {
                "kind": "fact",
                "text": "halve batch_size",
                "doc_id": "rb-oom",
                "page": 2,
                "evidence_ids": ["ev-rb"],
            }
        ],
        ["ev-rb"],
        {
            "ev-rb": {
                "name": "retrieve_runbook",
                "payload": {
                    "hits": [
                        {
                            "doc_id": "rb-missing-radar",
                            "page": 2,
                            "quote": "Do not impute composites across hours.",
                        }
                    ]
                },
            }
        },
    )
    assert judged["ok"] is False
    assert judged["missing"][0]["why"] == "wrong_page"


def test_rejects_page_claim_without_runbook_evidence():
    judged = critique(
        [{"kind": "fact", "text": "halve batch_size", "page": 2, "evidence_ids": ["ev-metric"]}],
        ["ev-metric"],
        {"ev-metric": {"name": "compare_metrics", "payload": {"delta": 1}}},
    )
    assert judged["ok"] is False
    assert judged["missing"][0]["why"] == "runbook_page_without_retrieval"
