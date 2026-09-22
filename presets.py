PRESETS = [
    {
        "id": "q1",
        "label": "Q1 加权损失为何只抬了 TS@1.0",
        "question": "为什么 afno_wmse 的 TS@1.0 升了，TS@5.0 没有？",
        "scope": {
            "exp_ids": ["exp-base-convlstm", "exp-afno", "exp-afno-wmse"]
        },
        "expected_verdict": "supported",
    },
    {
        "id": "q7",
        "label": "Q7 无日志运行的根因",
        "question": "run-fail-nolog 失败的根因是什么？",
        "scope": {"run_id": "run-fail-nolog"},
        "expected_verdict": "insufficient",
    },
    {
        "id": "q4",
        "label": "Q4 OOM（含注入句）",
        "question": "run-fail-oom 为什么失败？",
        "scope": {"run_id": "run-fail-oom"},
        "expected_verdict": "supported",
    },
]
