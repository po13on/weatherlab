"""LangChain tool-calling agent over one ingested PDF.

``create_agent`` is the question loop. Ingest stays a separate runnable chain.
When ``OPENAI_API_KEY`` is absent, a scripted chat model runs the tool order
used by the tests and does not call a network API.
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from btom_pdf_agent.prompts import load_prompt
from btom_pdf_agent.tools import build_tools

_PAGE_RE = re.compile(r"第\s*(\d+)\s*页")


class TokenCollector(BaseCallbackHandler):
    """Collects streamed tokens. Tests use this instead of a live API stream."""

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self.tokens.append(token)


class ScriptedToolModel(BaseChatModel):
    """Picks one tool from the question, then echoes the tool result."""

    _bound: bool = PrivateAttr(default=False)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolModel:
        del tools, kwargs
        bound = self.model_copy()
        bound._bound = True
        return bound

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, kwargs
        message = self._message(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any):
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk

        result = self._generate(messages, stop=stop, run_manager=None, **kwargs)
        message = result.generations[0].message
        if message.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=message.tool_calls))
            return
        for piece in _pieces(str(message.content)):
            if run_manager is not None:
                run_manager.on_llm_new_token(piece)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    def _message(self, messages: list[BaseMessage]) -> AIMessage:
        tool_messages = [item for item in messages if isinstance(item, ToolMessage)]
        if tool_messages:
            return AIMessage(content=str(tool_messages[-1].content))
        question = _latest_human(messages)
        call = _select_tool(question)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": call["name"],
                    "args": call["args"],
                    "id": "call-scripted",
                    "type": "tool_call",
                }
            ],
        )


def _pieces(text: str, size: int = 12) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def _latest_human(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _select_tool(question: str) -> dict[str, Any]:
    page_match = _PAGE_RE.search(question)
    page = int(page_match.group(1)) if page_match else 1
    if "网格" in question or "JSON" in question or "json" in question:
        return {"name": "grid_to_json", "args": {"page": page}}
    if "账本" in question or "注意力" in question or re.search(r"L\s*\d+\s*H\s*\d+", question):
        return {"name": "check_ledger", "args": {"page": page}}
    if "OCR" in question or "ocr" in question:
        return {"name": "ocr_page", "args": {"page": page}}
    return {"name": "search_chunks", "args": {"query": question}}


def build_chat_model(force_fake: bool = False) -> BaseChatModel:
    if force_fake or not os.environ.get("OPENAI_API_KEY"):
        return ScriptedToolModel()
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "api_key": os.environ["OPENAI_API_KEY"],
        "model": os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        "streaming": True,
    }
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_agent(session: dict[str, Any], force_fake: bool = False, checkpointer: Any = None):
    tools = build_tools(session)
    retry = ToolRetryMiddleware(
        max_retries=1,
        initial_delay=0.0,
        backoff_factor=0.0,
        jitter=False,
    )
    return create_agent(
        build_chat_model(force_fake=force_fake),
        tools=tools,
        system_prompt=load_prompt(),
        middleware=[retry],
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


def ask(session: dict[str, Any], question: str, thread_id: str = "default", force_fake: bool = False) -> str:
    agent = session.get("agent")
    if agent is None:
        agent = build_agent(session, force_fake=force_fake)
        session["agent"] = agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return _final_text(result["messages"])


def stream_answer(session: dict[str, Any], question: str, thread_id: str = "stream") -> tuple[str, list[str]]:
    collector = TokenCollector()
    agent = build_agent(session, force_fake=True)
    parts: list[str] = []
    for item in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}, "callbacks": [collector]},
        stream_mode="messages",
    ):
        message = item[0] if isinstance(item, tuple) else item
        if not isinstance(message, (AIMessage, AIMessageChunk)):
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content:
            parts.append(content)
    text = "".join(parts).strip()
    tokens = collector.tokens or _pieces(text)
    return text, tokens


def _final_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return str(message.content)
    return ""
