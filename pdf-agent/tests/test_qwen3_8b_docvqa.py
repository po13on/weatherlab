import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

DATA = (
    Path(__file__).resolve().parents[1]
    / "docvqa"
    / "nielsr_docvqa_1200_examples_train.jsonl"
)
USAGE = Path(__file__).resolve().parents[1] / "USAGE.md"
ID_SHA256 = "51e4b4141037d18219291be5b4a8972bc1df070d3f04dafbf220654b3f5268c0"


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen3_8b_docvqa.py"
    spec = importlib.util.spec_from_file_location("qwen3_8b_docvqa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anls_takes_max_gold_then_averages_questions():
    module = _module()
    assert module.levenshtein("kitten", "sitting") == 3
    assert module.normalized_edit_similarity("Italy", "italy") == 1.0
    assert module.normalized_edit_similarity("foobar", "foo") == 0.0
    partial = module.normalized_edit_similarity("fo", "foo")
    assert partial == pytest.approx(1.0 - 1.0 / 3.0)
    assert module.anls_one("5", ["5.", "5"]) == 1.0
    assert module.anls_one("nope", []) == 0.0
    mean = module.anls_mean([module.anls_one("italy", ["Italy"]), module.anls_one("x", ["italy"])])
    assert mean == pytest.approx(0.5)


def test_prompt_is_page_text_and_not_an_image():
    module = _module()
    user = module.build_user_text("City: Hong Kong", "Where?", truncated=False)
    assert "City: Hong Kong" in user
    assert "Where?" in user
    assert "不是图像" in user
    system = module.build_system_text()
    assert "不能看页面图像" in system
    assert "不是 DocVQA 2026" in system
    assert module.answer_text("<think>hidden</think>\n1/8/93") == "1/8/93"


def test_bundled_split_is_the_same_1000_questions_without_images():
    module = _module()
    assert DATA.stat().st_size < 50 * 1024 * 1024
    rows = module.load_questions(DATA)
    assert len(rows) == 1000
    assert rows[0]["id"] == "train_0"
    assert rows[0]["question"] == "what is the date mentioned in this letter?"
    assert rows[0]["answers"] == ["1/8/93"]
    assert rows[0]["page_text"].startswith("Confidential")
    assert rows[-1]["id"] == "train_1239"
    digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    assert digest == ID_SHA256
    raw = DATA.read_text(encoding="utf-8")
    assert '"image"' not in raw
    assert ".parquet" not in raw


def test_vision_weights_are_refused_before_loading_a_model(tmp_path: Path):
    module = _module()
    vision = tmp_path / "vl"
    vision.mkdir()
    (vision / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "vision_config": {"depth": 24},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="视觉权重已拒绝"):
        module.reject_vision_weights(vision)

    sneaky = tmp_path / "sneaky"
    sneaky.mkdir()
    (sneaky / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "vision_config": {"hidden_size": 1280}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="视觉权重已拒绝"):
        module.reject_vision_weights(sneaky)

    text = tmp_path / "text"
    text.mkdir()
    (text / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    config = module.reject_vision_weights(text)
    assert config["model_type"] == "qwen3"


def test_parquet_and_image_rows_are_refused(tmp_path: Path):
    module = _module()
    parquet = tmp_path / "val.parquet"
    parquet.write_bytes(b"not-read")
    with pytest.raises(SystemExit, match="parquet"):
        module.load_questions(parquet)

    pictured = tmp_path / "rows.jsonl"
    row = {
        "id": "x",
        "question": "q",
        "answers": ["a"],
        "page_text": "page",
        "image": "page.png",
    }
    pictured.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="页面图片"):
        module.load_questions(pictured)


def test_usage_points_at_the_1000_question_anls_path():
    text = USAGE.read_text(encoding="utf-8")
    assert "pdf-agent/docvqa/nielsr_docvqa_1200_examples_train.jsonl" in text
    assert "scripts/qwen3_8b_docvqa.py" in text
    assert "--weights /path/to/Qwen3-8B" in text
    assert "归一化编辑相似度" in text
    assert "554/1000" in text
    assert "不建议用 8B 测" in text
    assert "不能看图" in text
