from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=16)
def load_prompt(name: str) -> dict[str, Any]:
    path = PROMPTS_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"prompt {name} is not a mapping")
    return payload


def prompt_version(name: str = "system") -> str:
    return str(load_prompt(name).get("version") or "unknown")


def prompt_template(name: str = "system") -> str:
    return str(load_prompt(name).get("template") or "").strip()
