from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "weatherlab.sqlite3"

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_version INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS traces_job_seq ON traces(job_id, seq);
CREATE TABLE IF NOT EXISTS eval_runs (
    eval_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    score_json TEXT NOT NULL,
    FOREIGN KEY(eval_id) REFERENCES eval_runs(eval_id)
);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_version INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id SERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS traces_job_seq ON traces(job_id, seq);
CREATE TABLE IF NOT EXISTS eval_runs (
    eval_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_items (
    id SERIAL PRIMARY KEY,
    eval_id TEXT NOT NULL REFERENCES eval_runs(eval_id),
    case_id TEXT NOT NULL,
    score_json TEXT NOT NULL
);
"""


class CompatConnection:
    def __init__(self, raw: Any, dialect: str) -> None:
        self.raw = raw
        self.dialect = dialect

    def execute(self, sql: str, params: tuple = ()):
        if self.dialect == "postgres":
            sql = sql.replace("?", "%s")
            cur = self.raw.cursor()
            cur.execute(sql, params)
            return cur
        return self.raw.execute(sql, params)

    def executescript(self, sql: str) -> None:
        if self.dialect == "postgres":
            cur = self.raw.cursor()
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            return
        self.raw.executescript(sql)

    def commit(self) -> None:
        self.raw.commit()


def storage_kind() -> str:
    return os.getenv("STORAGE", "sqlite").strip().lower()


def connect(path: Path | None = None) -> CompatConnection:
    if path is None and storage_kind() == "postgres":
        import psycopg
        from psycopg.rows import dict_row

        raw = psycopg.connect(os.getenv("DATABASE_URL", ""), row_factory=dict_row)
        return CompatConnection(raw, "postgres")
    db_path = path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(db_path, check_same_thread=False)
    raw.row_factory = sqlite3.Row
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("PRAGMA foreign_keys=ON")
    return CompatConnection(raw, "sqlite")


def init_db(conn: CompatConnection) -> None:
    conn.executescript(POSTGRES_SCHEMA if conn.dialect == "postgres" else SQLITE_SCHEMA)
    conn.commit()
