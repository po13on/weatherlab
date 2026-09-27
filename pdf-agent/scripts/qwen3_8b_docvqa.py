"""用本地 Qwen3-8B 文本权重回答 DocVQA 1000 问，并计算 ANLS。

数据是仓库里的 nielsr/docvqa_1200_examples train，和检索 Recall@4 554/1000
用的是同一批问题。模型只读 page_text，不看图。本脚本不下载权重。

ANLS（DocVQA 常用定义，阈值 τ = 0.5）：
对预测 p 和某一个金标 g，先转成小写并去掉首尾空白。
NL(p, g) = Lev(p, g) / max(|p|, |g|)；两边都是空串时 NL = 0。
归一化编辑相似度 s(p, g) = 1 - NL(p, g)，当 NL(p, g) < 0.5；否则 s(p, g) = 0。
一题得分 = 该题全部金标的 s 的最大值；没有金标时为 0。
ANLS = 全部问题得分的算术平均。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "docvqa"
    / "nielsr_docvqa_1200_examples_train.jsonl"
)
QUESTION_COUNT = 1000
ANLS_TAU = 0.5
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, ca in enumerate(left, start=1):
        current = [i]
        for j, cb in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_similarity(prediction: str, gold: str, tau: float = ANLS_TAU) -> float:
    """One gold answer. Below tau the DocVQA score is zero."""

    pred = prediction.lower().strip()
    gt = gold.lower().strip()
    if not pred and not gt:
        nl = 0.0
    else:
        nl = levenshtein(pred, gt) / max(len(pred), len(gt))
    if nl < tau:
        return 1.0 - nl
    return 0.0


def anls_one(prediction: str, golds: list[str], tau: float = ANLS_TAU) -> float:
    """Max normalized edit similarity over the gold answers of one question."""

    if not golds:
        return 0.0
    return max(normalized_edit_similarity(prediction, gold, tau) for gold in golds)


def anls_mean(scores: list[float]) -> float:
    """Average of per-question scores. This average is ANLS."""

    if not scores:
        raise ValueError("没有问题，无法平均")
    return sum(scores) / len(scores)


def answer_text(raw: str) -> str:
    return _THINK.sub("", raw).strip()


def build_system_text() -> str:
    return (
        "你正在回答 DocVQA。当前模型是 Qwen3-8B 文本权重，不能看页面图像。\n"
        "输入只有这一页的文本，没有图像张量。不要声称看见了原图。\n"
        "这不是 DocVQA 2026。\n"
        "只输出答案本身，不要解释，不要加引号。"
    )


def build_user_text(page_text: str, question: str, truncated: bool) -> str:
    note = "页面文本在送入模型前被截断。\n\n" if truncated else ""
    body = page_text if page_text else "(empty page text)"
    return (
        f"{note}"
        "以下是页面文本，不是图像。\n\n"
        f"{body}\n\n"
        f"Question: {question}"
    )


def trim_page(page_text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(page_text) <= max_chars:
        return page_text, False
    return page_text[:max_chars], True


def load_questions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        raise SystemExit(f"不要读 parquet：{path}")
    if not path.is_file():
        raise SystemExit(f"找不到数据文件：{path}")
    items: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_no} 不是对象")
        if "image" in row or "page_image" in row or "image_bytes" in row:
            raise SystemExit(f"{path}:{line_no} 含有页面图片，已拒绝")
        question = str(row.get("question") or "").strip()
        answers = row.get("answers")
        page_text = row.get("page_text")
        if not question or not isinstance(answers, list) or not answers:
            raise SystemExit(f"{path}:{line_no} 缺少问题或金标")
        if not isinstance(page_text, str) or not page_text.strip():
            raise SystemExit(f"{path}:{line_no} 缺少页面文本")
        if not all(isinstance(answer, str) for answer in answers):
            raise SystemExit(f"{path}:{line_no} 金标必须是字符串")
        items.append(
            {
                "id": str(row.get("id") or line_no),
                "question": question,
                "answers": answers,
                "page_text": page_text,
            }
        )
    if len(items) != QUESTION_COUNT:
        raise SystemExit(f"{path} 应有 {QUESTION_COUNT} 问，实际 {len(items)} 问")
    return items


def reject_vision_weights(weights: Path) -> dict[str, Any]:
    """Refuse a vision checkpoint before any model code is imported."""

    if not weights.is_dir():
        raise SystemExit(f"找不到权重目录：{weights}")
    config_path = weights / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"找不到文本权重配置：{config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"无法读取权重配置：{config_path}（{exc}）") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"权重配置不是对象：{config_path}")
    model_type = str(config.get("model_type") or "")
    architectures = [str(item) for item in (config.get("architectures") or [])]
    vision_keys = [key for key in config if "vision" in key.lower()]
    blob = " ".join([model_type, *architectures]).lower()
    if vision_keys or "vision_config" in config or "vl" in blob or "vision" in blob:
        raise SystemExit(
            f"视觉权重已拒绝：{weights}（model_type={model_type!r}）。"
            "Qwen3-8B 文本模型不能看图。"
        )
    if model_type != "qwen3":
        raise SystemExit(f"权重要是 Qwen3 文本模型，当前 model_type={model_type!r}")
    return config


def load_text_model(weights: Path, trust_remote_code: bool):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("没有可用的显卡，已停止。不要在这台机器上加载 Qwen3-8B。")
    config = AutoConfig.from_pretrained(
        weights,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model_type = str(getattr(config, "model_type", ""))
    if model_type != "qwen3" or getattr(config, "vision_config", None) is not None:
        raise SystemExit(f"视觉或非文本权重已拒绝，当前 model_type={model_type!r}")
    tokenizer = AutoTokenizer.from_pretrained(
        weights,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        weights,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, page_text: str, question: str, truncated: bool) -> str:
    messages = [
        {"role": "system", "content": build_system_text()},
        {"role": "user", "content": build_user_text(page_text, question, truncated)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用本地 Qwen3-8B 文本权重回答仓库里的 DocVQA 1000 问，并计算 ANLS。"
    )
    parser.add_argument("--weights", required=True, type=Path, help="本地 Qwen3-8B 文本权重目录")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="仓库里的 1000 问 jsonl。默认就是这份 train 文本。",
    )
    parser.add_argument("--out", required=True, type=Path, help="写出 predictions.jsonl 和 summary.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-page-chars",
        type=int,
        default=24000,
        help="每问送入的页面文本字符上限。0 表示不截断。这份数据最长约 8000 字符。",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 问。0 表示全部 1000 问。")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reject_vision_weights(args.weights)
    questions = load_questions(args.data)
    if args.limit > 0:
        questions = questions[: args.limit]
    model, tokenizer = load_text_model(args.weights, args.trust_remote_code)

    args.out.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out / "predictions.jsonl"
    scores: list[float] = []
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item in questions:
            page_text, truncated = trim_page(item["page_text"], args.max_page_chars)
            prompt = render_prompt(tokenizer, page_text, item["question"], truncated)
            raw = generate_answer(model, tokenizer, prompt, args.max_new_tokens)
            prediction = answer_text(raw)
            score = anls_one(prediction, item["answers"])
            scores.append(score)
            handle.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "answers": item["answers"],
                        "prediction": prediction,
                        "anls": score,
                        "truncated_page": truncated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

    total = len(questions)
    mean = anls_mean(scores)
    summary = {
        "model": "Qwen3-8B-text",
        "weights": str(args.weights),
        "dataset": "nielsr/docvqa_1200_examples",
        "split": "train",
        "task": "DocVQA",
        "metric": "ANLS",
        "tau": ANLS_TAU,
        "anls": mean,
        "questions": total,
        "sees_images": False,
        "input": "page_text",
        "not_docvqa_2026": True,
        "retrieval_recall_at_4_is_not_anls": "554/1000",
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ANLS {mean:.4f}（{total} 问）")
    print("计分：对每个金标算归一化编辑相似度，取最大，再对问题平均。")
    print("阈值：归一化编辑距离 NL < 0.5 时相似度是 1-NL，否则该金标记 0。")
    print("这不是 DocVQA 2026。")
    print("554/1000 是此前哈希嵌入检索的 Recall@4，不是这次 ANLS。")
    print("输入：页面文本。Qwen3-8B 文本权重不能看图。")


if __name__ == "__main__":
    main()
