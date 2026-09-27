"""Labeled retrieval recall on the Hi-ToM CSV that is already on disk."""

import csv
import inspect
from pathlib import Path

from btom_pdf_agent.retrieval_eval import run_retrieval_eval
from btom_pdf_agent.retrieve import HybridIndex


def test_retrieval_eval_reports_recall_and_citation(tmp_path: Path):
    result = run_retrieval_eval(tmp_path / "hitom")
    k = result["k"]
    assert k == inspect.signature(HybridIndex.search).parameters["k"].default
    assert result["checked"] > 0
    assert result["recall_hits"] <= result["checked"]
    assert result["citation_hits"] <= result["checked"]
    assert result["summary_recall"] == f"Recall@{k} {result['recall_hits']}/{result['checked']}"
    assert result["summary_citation"] == f"引用命中 {result['citation_hits']}/{result['checked']}"
    assert "不是模型准确率" in result["fixture_note"]
    assert "问句不在被索引的页面里" in result["fixture_note"]
    assert result["dataset_files"] == [
        "/home/ubuntu/BTOM-Transformerlens/data_uniform/Hi_ToM_order_1.csv"
    ]
    assert "IOI" in result["skipped"]
    assert "ARC" in result["skipped"]
    assert result["rows"] == 100
    assert result["dropped_question_in_story"] == result["rows"] - result["checked"]
    assert result["gold_split_count"] == sum(1 for case in result["cases"] if case["gold_split"])
    assert any(line.startswith("金标被切散 ") for line in result["lines"])
    for case in result["cases"]:
        assert isinstance(case["gold_split"], bool)
        assert case["gold_chunk_ids"]
        assert len(case["gold_chunk_ids"]) <= k
        page_ids = [
            item["chunk_id"]
            for item in result["chunks"]
            if item["metadata"].get("source") == "pdf" and item["metadata"].get("page") == case["page"]
        ]
        assert page_ids
        assert set(case["gold_chunk_ids"]) == set(page_ids)
        for chunk_id in case["gold_chunk_ids"]:
            chunk = next(item for item in result["chunks"] if item["chunk_id"] == chunk_id)
            assert case["query"] not in chunk["text"]
            assert chunk["metadata"]["source"] == "pdf"


def test_question_inside_story_is_dropped(tmp_path: Path):
    csv_path = tmp_path / "overlap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["story", "question", "answer", "choices"])
        writer.writeheader()
        writer.writerow(
            {
                "story": "The note says Where is the ball?",
                "question": "Where is the ball?",
                "answer": "room",
                "choices": "A. room",
            }
        )
        writer.writerow(
            {
                "story": "Ada left the ball in the hall.",
                "question": "Where is the ball?",
                "answer": "hall",
                "choices": "A. hall",
            }
        )
    result = run_retrieval_eval(tmp_path / "mini", csv_path=csv_path)
    assert result["rows"] == 2
    assert result["dropped_question_in_story"] == 1
    assert result["checked"] == 1
    assert result["cases"][0]["query"] == "Where is the ball?"
    for case in result["cases"]:
        for chunk_id in case["gold_chunk_ids"]:
            chunk = next(item for item in result["chunks"] if item["chunk_id"] == chunk_id)
            assert case["query"] not in chunk["text"]


def test_missing_csv_does_not_invent_labels(tmp_path: Path):
    result = run_retrieval_eval(tmp_path / "missing", csv_path=tmp_path / "absent.csv")
    assert result["checked"] == 0
    assert result["recall_hits"] == 0
    assert result["citation_hits"] == 0
    assert result["cases"] == []
    assert "Hi-ToM" in result["skipped"]
