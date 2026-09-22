from __future__ import annotations

import json
import time
from typing import Any, Callable, TypedDict

from critic import critique
from policy import openai_tools_for_role
from prompts.loader import prompt_template
from tools import OPENAI_TOOLS, dispatch

HAS_LANGGRAPH = False
try:
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except Exception:
    StateGraph = None  # type: ignore
    END = "end"
    START = "start"


class GraphState(TypedDict, total=False):
    messages: list
    role: str
    scope: dict
    remaining: int
    seen: list
    facts: list
    tool_trace: list
    evidence_ids: list
    evidence_records: dict
    claims: list
    verdict: str
    missing: list
    finished: bool
    pending: list
    llm_calls: int
    token_in: int
    token_out: int
    question: str


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _assistant_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
    }
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return payload


def build_langgraph(
    completer: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any],
    role: str,
    max_steps: int,
):
    if not HAS_LANGGRAPH:
        raise ImportError("langgraph is not installed. Use Python 3.10+ and pip install -r requirements.txt")

    offered = openai_tools_for_role(role, OPENAI_TOOLS)

    def model_node(state: GraphState) -> GraphState:
        if state.get("finished") or (state.get("remaining") or 0) <= 0:
            state["finished"] = True
            state["verdict"] = state.get("verdict") or "insufficient"
            state["missing"] = state.get("missing") or [{"why": "budget_exceeded"}]
            return state
        response = completer(state["messages"], offered)
        tin, tout = _usage(response)
        state["llm_calls"] = (state.get("llm_calls") or 0) + 1
        state["token_in"] = (state.get("token_in") or 0) + tin
        state["token_out"] = (state.get("token_out") or 0) + tout
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        state["messages"] = list(state["messages"]) + [_assistant_message(message, tool_calls)]
        if not tool_calls:
            state["finished"] = True
            state["verdict"] = "insufficient"
            state["missing"] = [{"why": "model_returned_no_tool_call"}]
            return state
        pending = []
        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            pending.append({"id": call.id, "name": call.function.name, "args": args})
        state["pending"] = pending
        return state

    def tools_node(state: GraphState) -> GraphState:
        from agent import _apply_run_evidence_gate

        pending_finalize = None
        seen = set(state.get("seen") or [])
        scope = state.get("scope") or {}
        for call in state.get("pending") or []:
            name = call["name"]
            args = dict(call["args"])
            if name == "compare_metrics":
                for key in ("split", "region", "lead_hours"):
                    if key in scope and key not in args:
                        args[key] = scope[key]
            key = name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)
            if name != "finalize_report" and key in seen:
                state["finished"] = True
                state["verdict"] = "insufficient"
                state["missing"] = [{"why": "repeated_tool_call", "tool": name}]
                return state
            if name != "finalize_report":
                seen.add(key)
            result = dispatch(
                name,
                args,
                role=state["role"],
                inject_timeout=scope.get("inject_timeout_tool") == name,
            )
            evidence_id = result.get("evidence_id")
            if name != "finalize_report" and evidence_id:
                state.setdefault("evidence_ids", []).append(evidence_id)
                state.setdefault("evidence_records", {})[evidence_id] = {
                    "name": name,
                    "payload": result.get("payload") or result,
                }
            if result.get("facts"):
                state.setdefault("facts", []).extend(result["facts"])
            observation = {k: v for k, v in result.items() if k != "facts"}
            state.setdefault("tool_trace", []).append(
                {
                    "name": name,
                    "args": args,
                    "observation": observation,
                    "evidence_id": evidence_id,
                    "latency_ms": result.get("latency_ms"),
                    "retries": result.get("retries"),
                }
            )
            state["messages"] = list(state["messages"]) + [
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            ]
            if name == "finalize_report":
                pending_finalize = args
        state["seen"] = list(seen)
        state["pending"] = []
        state["remaining"] = (state.get("remaining") or 1) - 1
        if pending_finalize is not None:
            claims = pending_finalize.get("claims") or []
            judged = critique(claims, state.get("evidence_ids") or [], state.get("evidence_records") or {})
            state["claims"] = claims if isinstance(claims, list) else []
            if judged.get("ok"):
                state["verdict"] = judged["verdict"]
                state["finished"] = True
            else:
                state["missing"] = judged.get("missing") or []
                if (state.get("remaining") or 0) <= 0:
                    state["verdict"] = "insufficient"
                    state["finished"] = True
                else:
                    state["messages"] = list(state["messages"]) + [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "critic": judged,
                                    "instruction": "Revise claims or gather missing evidence.",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    ]
        if state.get("finished"):
            packed = {
                "scope": state.get("scope") or {},
                "role": state.get("role"),
                "verdict": state.get("verdict"),
                "claims": state.get("claims") or [],
                "facts": state.get("facts") or [],
                "evidence_ids": state.get("evidence_ids") or [],
                "evidence_records": state.get("evidence_records") or {},
                "tool_trace": state.get("tool_trace") or [],
                "missing": state.get("missing") or [],
                "question": state.get("question") or "",
                "llm_calls": state.get("llm_calls") or 0,
                "token_in": state.get("token_in") or 0,
                "token_out": state.get("token_out") or 0,
                "prompt_version": state.get("prompt_version"),
            }
            _apply_run_evidence_gate(packed)
            state.update(packed)
        return state

    def route(state: GraphState) -> str:
        if state.get("finished") or (state.get("remaining") or 0) <= 0:
            return END
        if state.get("pending"):
            return "tools"
        return "model"

    graph = StateGraph(dict)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route, {"model": "model", "tools": "tools", END: END})
    return graph.compile()


def run_langgraph_agent(
    question: str,
    scope: dict[str, Any],
    completer: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any],
    max_steps: int = 8,
    role: str = "analyst",
) -> dict[str, Any]:
    from agent import _finish
    from policy import PROMPT_VERSION, normalize_role

    role = normalize_role(role)
    graph = build_langgraph(completer, role, max_steps)
    start: GraphState = {
        "question": question,
        "scope": scope,
        "role": role,
        "remaining": max_steps,
        "seen": [],
        "facts": [],
        "tool_trace": [],
        "evidence_ids": [],
        "evidence_records": {},
        "claims": [],
        "missing": [],
        "finished": False,
        "pending": [],
        "llm_calls": 0,
        "token_in": 0,
        "token_out": 0,
        "prompt_version": PROMPT_VERSION,
        "t0": time.time(),
        "messages": [
            {"role": "system", "content": prompt_template("system")},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "scope": {k: v for k, v in scope.items() if not str(k).startswith("inject_")},
                        "constraints": {"max_steps": max_steps, "runner": "langgraph", "role": role},
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    final = graph.invoke(start)
    packed = {
        "question": question,
        "scope": scope,
        "role": role,
        "verdict": final.get("verdict") or "insufficient",
        "claims": final.get("claims") or [],
        "facts": final.get("facts") or [],
        "evidence_ids": final.get("evidence_ids") or [],
        "evidence_records": final.get("evidence_records") or {},
        "tool_trace": final.get("tool_trace") or [],
        "missing": final.get("missing") or [{"why": "budget_exceeded"}],
        "llm_calls": final.get("llm_calls") or 0,
        "token_in": final.get("token_in") or 0,
        "token_out": final.get("token_out") or 0,
        "prompt_version": PROMPT_VERSION,
    }
    result = _finish(packed)
    result["runner"] = "langgraph"
    return result
