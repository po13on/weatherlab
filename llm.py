from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

MODEL_ALIASES = {
    "deepseek-v4.1-flash": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
    "qwen-plus": "qwen-plus",
    "qwen2.5": "qwen-plus",
    "qwen2.5-72b-instruct": "qwen-plus",
    "llama-3-70b": "llama-3-70b",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def settings() -> dict[str, str]:
    raw_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    return {
        "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        "model": MODEL_ALIASES.get(raw_model, raw_model),
        "requested_model": raw_model,
    }


def _create_kwargs(tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    cfg = settings()
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "temperature": 0,
        "timeout": 60,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if "deepseek" in cfg["model"] or "deepseek" in cfg["base_url"]:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return kwargs


async def complete_async(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    cfg = settings()
    if not cfg["api_key"]:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    kwargs = _create_kwargs(tools)
    kwargs["messages"] = messages
    return await client.chat.completions.create(**kwargs)


def complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(complete_async(messages, tools))
    cfg = settings()
    if not cfg["api_key"]:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    kwargs = _create_kwargs(tools)
    kwargs["messages"] = messages
    return client.chat.completions.create(**kwargs)
