from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def resolve_fixture_path() -> Path:
    """Prefer the copy shipped with this repo so a clone runs on another machine."""
    override = os.getenv("WEATHERLAB_FIXTURES", "").strip()
    bundled = Path(__file__).resolve().parent / "fixtures" / "experiments.json"
    legacy = (
        Path(__file__).resolve().parents[1]
        / "求职作战包"
        / "weatherlab_fixtures"
        / "experiments.json"
    )
    if override:
        chosen = Path(override) / "experiments.json"
        if chosen.exists():
            return chosen
    if bundled.exists():
        return bundled
    return legacy


FIXTURE_PATH = resolve_fixture_path()

METRIC_CONTRACT = {
    "split": "val",
    "region": "north_plain",
    "lead_hours": 2,
}


@lru_cache(maxsize=1)
def load_fixture() -> dict[str, Any]:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"fixture not found: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def experiments() -> dict[str, dict[str, Any]]:
    return {item["exp_id"]: item for item in load_fixture()["experiments"]}


def runs() -> dict[str, dict[str, Any]]:
    return {item["run_id"]: item for item in load_fixture()["runs"]}


def manifests() -> dict[str, dict[str, Any]]:
    return {item["manifest_id"]: item for item in load_fixture()["manifests"]}


def runbooks() -> list[dict[str, Any]]:
    return list(load_fixture()["runbooks"])
