"""DocVQA 2026 validation with a local Qwen3-8B text checkpoint.

Qwen3-8B does not see images. Each question is answered from OCR page text.
The raw generation is scored by the official
``evaluate_docvqa_prediction`` in VLR-CVC/DocVQA2026 ``eval_utils.py``.
That function rejects any string that does not contain ``FINAL ANSWER:``.

This script never downloads weights. Pass a directory that is already on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable


def build_system_text(official_prompt: str) -> str:
    return (
        "你正在回答 DocVQA 2026。当前模型是 Qwen3-8B 文本权重，不能看页面图像。\n"
        "输入只有 OCR 之后的页面文本，没有图像张量。不要声称看见了原图。\n"
        "下面是竞赛基线提示里的格式规则。其中的 images 在本脚本里对应这些 OCR 文本。\n\n"
        f"{official_prompt}\n\n"
        "输出必须包含字面量 FINAL ANSWER: 。计分只取最后一次出现之后的文本。"
    )


def build_user_text(pages: list[str], question: str, truncated: bool) -> str:
    blocks = [f"[page {index}]\n{text}" for index, text in enumerate(pages, start=1)]
    note = "OCR 文本在送入模型前被截断。\n\n" if truncated else ""
    body = "\n\n".join(blocks) if blocks else "(no OCR text)"
    return (
        f"{note}"
        "以下是 OCR 后的页面文本，不是图像。\n\n"
        f"{body}\n\n"
        f"Question: {question}\n\n"
        "回答末尾必须有一行包含字面量 FINAL ANSWER: ，后面紧跟答案。"
    )


def trim_pages(pages: list[str], max_chars: int) -> tuple[list[str], bool]:
    if max_chars <= 0:
        return pages, False
    kept: list[str] = []
    used = 0
    truncated = False
    for text in pages:
        if used >= max_chars:
            truncated = True
            break
        room = max_chars - used
        if len(text) > room:
            kept.append(text[:room])
            truncated = True
            break
        kept.append(text)
        used += len(text)
    if len(kept) < len(pages):
        truncated = True
    return kept, truncated


def load_official_scorer(path: Path) -> tuple[Callable[..., Any], str]:
    spec = importlib.util.spec_from_file_location("docvqa2026_eval_utils", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载官方评测脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score = getattr(module, "evaluate_docvqa_prediction", None)
    if not callable(score):
        raise SystemExit(f"{path} 里没有 evaluate_docvqa_prediction")
    prompt = ""
    get_prompt = getattr(module, "get_evaluation_prompt", None)
    if callable(get_prompt):
        prompt = str(get_prompt())
    _check_scorer(score)
    return score, prompt


def _check_scorer(score: Callable[..., Any]) -> None:
    missing, _ = score("Indiana", "Indiana")
    if missing is not False:
        raise SystemExit("官方计分自检失败：没有 FINAL ANSWER: 时应判错")
    ok, _ = score("FINAL ANSWER: Indiana", "Indiana")
    if ok is not True:
        raise SystemExit("官方计分自检失败：FINAL ANSWER: Indiana 对 Indiana 时应判对")


def load_questions(parquet_path: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        parquet_path,
        columns=["doc_id", "doc_category", "questions", "answers"],
    )
    items: list[dict[str, str]] = []
    for row in table.to_pylist():
        questions = row["questions"]
        answers = row["answers"]
        gold = dict(zip(answers["question_id"], answers["answer"]))
        for question_id, question in zip(questions["question_id"], questions["question"]):
            if question_id not in gold:
                raise SystemExit(f"{row['doc_id']} 的问题 {question_id} 没有答案")
            items.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "doc_category": str(row["doc_category"]),
                    "question_id": str(question_id),
                    "question": str(question),
                    "answer": str(gold[question_id]),
                }
            )
    if not items:
        raise SystemExit(f"{parquet_path} 里没有问题")
    return items


def load_ocr_pages(ocr_root: Path, doc_id: str) -> list[str]:
    folder = ocr_root / doc_id
    files = sorted(folder.glob("p*.txt"))
    if not files:
        raise SystemExit(f"没有 OCR 页文本：{folder}（期望 p0000.txt 这种按页文件）")
    return [path.read_text(encoding="utf-8", errors="replace") for path in files]


def load_text_model(weights: Path, trust_remote_code: bool):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(
        weights,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model_type = str(getattr(config, "model_type", ""))
    if model_type != "qwen3" or getattr(config, "vision_config", None) is not None:
        raise SystemExit(
            f"权重要是不能看图的 Qwen3 文本模型，当前 model_type={model_type!r}"
        )
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


def render_prompt(tokenizer, pages: list[str], question: str, official_prompt: str, truncated: bool) -> str:
    messages = [
        {"role": "system", "content": build_system_text(official_prompt)},
        {"role": "user", "content": build_user_text(pages, question, truncated)},
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
        description="用本地 Qwen3-8B 文本权重回答 DocVQA 2026，并用官方函数计分。"
    )
    parser.add_argument("--weights", required=True, type=Path, help="本地 Qwen3-8B 文本权重目录")
    parser.add_argument(
        "--eval-utils",
        required=True,
        type=Path,
        help="官方仓库 VLR-CVC/DocVQA2026 的 eval_utils.py",
    )
    parser.add_argument("--parquet", required=True, type=Path, help="DocVQA 2026 的 val.parquet")
    parser.add_argument(
        "--ocr-root",
        required=True,
        type=Path,
        help="OCR 页文本根目录，每份文档一个子目录，页文件名为 p0000.txt",
    )
    parser.add_argument("--out", required=True, type=Path, help="写出 predictions.jsonl 和 summary.json")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-ocr-chars",
        type=int,
        default=24000,
        help="每问送入的 OCR 字符上限。0 表示不截断。",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 问。0 表示全部。")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.weights.is_dir():
        raise SystemExit(f"找不到权重目录：{args.weights}")
    if not args.eval_utils.is_file():
        raise SystemExit(f"找不到官方 eval_utils.py：{args.eval_utils}")
    if not args.parquet.is_file():
        raise SystemExit(f"找不到 parquet：{args.parquet}")
    if not args.ocr_root.is_dir():
        raise SystemExit(f"找不到 OCR 目录：{args.ocr_root}")

    score, official_prompt = load_official_scorer(args.eval_utils)
    questions = load_questions(args.parquet)
    if args.limit > 0:
        questions = questions[: args.limit]
    model, tokenizer = load_text_model(args.weights, args.trust_remote_code)

    args.out.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out / "predictions.jsonl"
    correct = 0
    by_category: dict[str, list[int]] = {}
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item in questions:
            pages = load_ocr_pages(args.ocr_root, item["doc_id"])
            pages, truncated = trim_pages(pages, args.max_ocr_chars)
            prompt = render_prompt(tokenizer, pages, item["question"], official_prompt, truncated)
            raw = generate_answer(model, tokenizer, prompt, args.max_new_tokens)
            ok, extracted = score(raw, item["answer"])
            correct += int(bool(ok))
            bucket = by_category.setdefault(item["doc_category"], [0, 0])
            bucket[0] += int(bool(ok))
            bucket[1] += 1
            handle.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "doc_id": item["doc_id"],
                        "doc_category": item["doc_category"],
                        "correct": bool(ok),
                        "extracted": extracted,
                        "has_final_answer": "FINAL ANSWER:" in raw,
                        "truncated_ocr": truncated,
                        "prediction": raw,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

    total = len(questions)
    summary = {
        "model": "Qwen3-8B-text",
        "weights": str(args.weights),
        "task": "DocVQA 2026",
        "metric": "Accuracy",
        "scorer": "evaluate_docvqa_prediction",
        "correct": correct,
        "total": total,
        "accuracy": f"{correct}/{total}",
        "by_category": {
            name: {"correct": pair[0], "total": pair[1]} for name, pair in sorted(by_category.items())
        },
        "sees_images": False,
        "input": "ocr_page_text",
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Accuracy {correct}/{total}")
    for name, pair in sorted(by_category.items()):
        print(f"{name} {pair[0]}/{pair[1]}")
    print("计分函数：官方 evaluate_docvqa_prediction")
    print("输入：OCR 页面文本。Qwen3-8B 文本权重不能看图。")
    print("这不是哈希嵌入检索的 Recall。")


if __name__ == "__main__":
    main()
