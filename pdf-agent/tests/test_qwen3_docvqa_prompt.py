import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen3_8b_docvqa2026.py"
    spec = importlib.util.spec_from_file_location("qwen3_8b_docvqa2026", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_uses_ocr_text_and_final_answer_marker():
    module = _module()
    pages, truncated = module.trim_pages(["alpha page", "beta page"], max_chars=5)
    assert truncated is True
    assert pages == ["alpha"]
    user = module.build_user_text(["City: Hong Kong"], "Where?", truncated=False)
    assert "City: Hong Kong" in user
    assert "FINAL ANSWER:" in user
    assert "OCR" in user
    system = module.build_system_text("format rules")
    assert "不能看页面图像" in system
    assert "FINAL ANSWER:" in system
    assert "format rules" in system
