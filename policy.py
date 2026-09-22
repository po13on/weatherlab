from __future__ import annotations

from typing import Any

from prompts.loader import prompt_version

PROMPT_VERSION = prompt_version("system")

ROLES = {
    "analyst": {
        "tools": {
            "list_runs",
            "get_config_diff",
            "compare_metrics",
            "check_data_completeness",
            "extract_log_error",
            "retrieve_runbook",
            "ocr_page",
            "finalize_report",
        }
    },
    "guest": {
        "tools": {
            "list_runs",
            "get_config_diff",
            "compare_metrics",
            "check_data_completeness",
            "finalize_report",
        }
    },
}


def normalize_role(role: str | None) -> str:
    if role is None or not str(role).strip():
        return "analyst"
    name = str(role).strip().lower()
    return name if name in ROLES else "guest"


def allow_tool(role: str, name: str) -> bool:
    return name in ROLES[normalize_role(role)]["tools"]


def openai_tools_for_role(role: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = ROLES[normalize_role(role)]["tools"]
    filtered = []
    for item in catalog:
        name = ((item.get("function") or {}).get("name"))
        if name in allowed:
            filtered.append(item)
    return filtered


def redact_observation(role: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    if normalize_role(role) != "guest":
        return result
    payload = result.get("payload")
    if name == "extract_log_error" and isinstance(payload, dict):
        payload = dict(payload)
        payload["lines"] = ["[redacted] guest role cannot read raw logs"]
        result = dict(result)
        result["payload"] = payload
        result["redacted"] = True
    return result


def deny_result(role: str, name: str) -> dict[str, Any]:
    return {
        "error": "tool_denied",
        "name": name,
        "role": normalize_role(role),
        "why": "role_not_permitted",
    }
