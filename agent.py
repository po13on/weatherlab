from __future__ import annotations

import json
import time
from typing import Any, Callable

from critic import critique
from policy import PROMPT_VERSION, normalize_role, openai_tools_for_role
from prompts.loader import prompt_template
from tools import OPENAI_TOOLS, dispatch

MAX_STEPS = 8
SYSTEM_PROMPT = prompt_template("system")


def _args_hash(name: str, args: dict[str, Any]) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def run_agent(
    question: str,
    scope: dict[str, Any],
    completer: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any],
    max_steps: int = MAX_STEPS,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    hitl_gate: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    role: str = "analyst",
) -> dict[str, Any]:
    role = normalize_role(role)
    t0 = time.time()
    def emit(event_type: str, payload: dict[str, Any]) -> None:
        if on_event:
            on_event(event_type, payload)

    emit("phase", {"phase": "BUILD_CONTEXT", "role": role, "prompt_version": PROMPT_VERSION})
    state: dict[str, Any] = {
        "question": question,
        "scope": scope,
        "role": role,
        "phase": "BUILD_CONTEXT",
        "facts": [],
        "tool_trace": [],
        "evidence_ids": [],
        "evidence_records": {},
        "claims": [],
        "verdict": None,
        "missing": [],
        "budget": {"max_steps": max_steps, "remaining_steps": max_steps},
        "llm_calls": 0,
        "token_in": 0,
        "token_out": 0,
        "prompt_version": PROMPT_VERSION,
        "t0": t0,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "scope": {k: v for k, v in scope.items() if not str(k).startswith("inject_")},
                    "constraints": {
                        "max_steps": max_steps,
                        "stop": "finalize_report or budget_exceeded or repeated_tool_call",
                        "role": role,
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
    seen: set[str] = set()
    offered_tools = openai_tools_for_role(role, OPENAI_TOOLS)

    while state["budget"]["remaining_steps"] > 0:
        state["phase"] = "PLAN"
        emit("phase", {"phase": "PLAN", "remaining_steps": state["budget"]["remaining_steps"]})
        response = completer(messages, offered_tools)
        state["llm_calls"] += 1
        tin, tout = _usage(response)
        state["token_in"] += tin
        state["token_out"] += tout
        choice = response.choices[0]
        message = choice.message
        tool_calls = message.tool_calls or []
        messages.append(_assistant_message(message, tool_calls))

        if not tool_calls:
            state["verdict"] = "insufficient"
            state["missing"] = [{"why": "model_returned_no_tool_call"}]
            break

        pending_finalize: dict[str, Any] | None = None
        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                result: dict[str, Any] = {"error": "invalid_json_args"}
            else:
                if name == "compare_metrics":
                    args = dict(args)
                    for key in ("split", "region", "lead_hours"):
                        if key in scope and key not in args:
                            args[key] = scope[key]
                state["phase"] = "EXECUTE_STEP"
                emit("phase", {"phase": "EXECUTE_STEP", "tool": name})
                result = dispatch(
                    name,
                    args,
                    role=role,
                    inject_timeout=scope.get("inject_timeout_tool") == name,
                )

            emit("tool.start", {"name": name, "args": args})
            key = _args_hash(name, args)
            if name != "finalize_report":
                if key in seen:
                    state["verdict"] = "insufficient"
                    state["missing"] = [{"why": "repeated_tool_call", "tool": name}]
                    finished = _finish(state)
                    emit("final", finished)
                    return finished
                seen.add(key)

            evidence_id = result.get("evidence_id")
            if name != "finalize_report" and evidence_id:
                state["evidence_ids"].append(evidence_id)
                state["evidence_records"][evidence_id] = {
                    "name": name,
                    "payload": result.get("payload") or result,
                }
            if result.get("facts"):
                state["facts"].extend(result["facts"])

            observation = {k: v for k, v in result.items() if k != "facts"}
            state["tool_trace"].append(
                {
                    "name": name,
                    "args": args,
                    "observation": observation,
                    "evidence_id": evidence_id,
                    "latency_ms": result.get("latency_ms"),
                    "retries": result.get("retries"),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )
            emit(
                "tool.end",
                {
                    "name": name,
                    "args": args,
                    "evidence_id": evidence_id,
                    "ok": "error" not in result,
                    "error": result.get("error"),
                    "latency_ms": result.get("latency_ms"),
                    "retries": result.get("retries"),
                },
            )
            if name == "finalize_report":
                pending_finalize = args

        if pending_finalize is not None:
            claims = pending_finalize.get("claims") or []
            if hitl_gate:
                decision = hitl_gate(claims if isinstance(claims, list) else [])
                action = (decision or {}).get("action")
                if action == "reject":
                    state["missing"] = [{"why": "hitl_rejected"}]
                    emit("hitl", {"action": "reject"})
                    if state["budget"]["remaining_steps"] <= 1:
                        state["verdict"] = "insufficient"
                        finished = _finish(state)
                        emit("final", finished)
                        return finished
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "hitl": "rejected",
                                    "instruction": "Human rejected the report. Revise or finalize as insufficient.",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    state["budget"]["remaining_steps"] -= 1
                    continue
                if action == "edit_claims" and decision.get("claims"):
                    claims = decision["claims"]
                    pending_finalize["claims"] = claims
                elif action == "timeout":
                    state["verdict"] = "insufficient"
                    state["missing"] = [{"why": "hitl_timeout"}]
                    finished = _finish(state)
                    emit("final", finished)
                    return finished
            state["phase"] = "VALIDATE"
            emit("phase", {"phase": "VALIDATE"})
            judged = critique(
                claims, state["evidence_ids"], state["evidence_records"]
            )
            state["claims"] = claims if isinstance(claims, list) else []
            emit("critic", judged)
            if judged["ok"]:
                state["verdict"] = judged["verdict"]
                if state["verdict"] == "supported":
                    state["missing"] = []
                finished = _finish(state)
                emit("final", finished)
                return finished
            state["missing"] = judged.get("missing", [])
            if state["budget"]["remaining_steps"] <= 1:
                state["verdict"] = "insufficient"
                finished = _finish(state)
                emit("final", finished)
                return finished
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "critic": judged,
                            "instruction": "Revise claims or gather missing evidence. Do not invent a root cause.",
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        state["budget"]["remaining_steps"] -= 1

    if state["verdict"] is None:
        state["verdict"] = "insufficient"
        state["missing"] = state["missing"] or [{"why": "budget_exceeded"}]
    finished = _finish(state)
    emit("final", finished)
    return finished


def _apply_run_evidence_gate(state: dict[str, Any]) -> None:
    run_id = (state.get("scope") or {}).get("run_id")
    if not run_id:
        return
    role = normalize_role(state.get("role"))
    log_ok = False
    log_gap = False
    completeness_ok = False
    completeness_gap = False
    for step in state.get("tool_trace") or []:
        if (step.get("args") or {}).get("run_id") != run_id:
            continue
        name = step.get("name")
        error = (step.get("observation") or {}).get("error")
        ok = bool((step.get("observation") or {}).get("ok"))
        if name == "extract_log_error":
            if error in {"log_unavailable", "timeout", "tool_denied"}:
                log_gap = True
            elif ok:
                log_ok = True
        elif name == "check_data_completeness":
            if error in {"manifest_unavailable", "timeout", "tool_denied"}:
                completeness_gap = True
            elif ok:
                completeness_ok = True
    blocked = False
    why = None
    if role == "guest" and not log_ok:
        blocked = True
        why = "role_not_permitted"
    elif log_gap and not log_ok:
        blocked = True
        why = "run_evidence_unavailable"
    elif (log_gap or completeness_gap) and not log_ok and not completeness_ok:
        blocked = True
        why = "run_evidence_unavailable"
    if blocked:
        state["verdict"] = "insufficient"
        missing = state.setdefault("missing", [])
        if not any(item.get("why") == why for item in missing):
            missing.append({"why": why, "run_id": run_id})


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


def _result_status(state: dict[str, Any]) -> str:
    errors = [
        (step.get("observation") or {}).get("error")
        for step in state.get("tool_trace") or []
    ]
    why = (state.get("missing") or [{}])[0].get("why") if state.get("missing") else None
    degrade_errors = {
        "timeout",
        "tool_denied",
        "log_unavailable",
        "manifest_unavailable",
        "incomparable_metric",
        "unknown_exp",
        "unknown_run",
    }
    if why in {"budget_exceeded", "model_returned_no_tool_call"}:
        return "fail"
    if why in {"role_not_permitted", "run_evidence_unavailable"}:
        return "degrade"
    if any(err in degrade_errors for err in errors):
        return "degrade"
    if state.get("verdict") in {"supported", "insufficient", "contradictory"}:
        return "success"
    return "fail"


def _finish(state: dict[str, Any]) -> dict[str, Any]:
    _apply_run_evidence_gate(state)
    state["phase"] = "OUTPUT"
    tool_metrics = [
        {
            "name": step.get("name"),
            "latency_ms": step.get("latency_ms"),
            "error": (step.get("observation") or {}).get("error"),
            "retries": step.get("retries") or 0,
        }
        for step in state.get("tool_trace") or []
        if step.get("name") != "finalize_report"
    ]
    cited = []
    for claim in state.get("claims") or []:
        if isinstance(claim, dict) and claim.get("kind") in {"fact", "inference"}:
            cited.extend(claim.get("evidence_ids") or [])
    return {
        "verdict": state["verdict"],
        "question": state["question"],
        "scope": state["scope"],
        "claims": state["claims"],
        "facts": state["facts"],
        "evidence_ids": state["evidence_ids"],
        "tool_trace": state["tool_trace"],
        "missing": state["missing"],
        "steps_used": len(state["tool_trace"]),
        "result_status": _result_status(state),
        "run_metrics": {
            "prompt_version": state.get("prompt_version"),
            "role": state.get("role"),
            "llm_calls": state.get("llm_calls") or 0,
            "token_in": state.get("token_in") or 0,
            "token_out": state.get("token_out") or 0,
            "tool_calls": tool_metrics,
            "citations": cited,
            "result_status": _result_status(state),
            "stop_condition": (state.get("missing") or [{}])[0].get("why") if state.get("verdict") != "supported" else "finalize_ok",
            "wall_ms": int((time.time() - (state.get("t0") or time.time())) * 1000),
        },
        "synthetic_fixture": True,
        "not_production": True,
    }
