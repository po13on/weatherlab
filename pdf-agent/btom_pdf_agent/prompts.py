"""Load the versioned system prompt."""

from __future__ import annotations

from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "v1.md"


def load_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if not text.startswith("version: v1"):
        raise RuntimeError(f"unexpected prompt header in {PROMPT_PATH}")
    return text


def prompt_version() -> str:
    return "v1"
