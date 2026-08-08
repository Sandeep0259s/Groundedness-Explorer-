"""Persists conversation history to a local SQLite file instead of an
in-memory dict, so an open multi-turn conversation survives a server
restart instead of silently losing its history.

Unlike the old in-memory dict — which was implicitly capped by process
uptime and reset to zero on every restart — this file grows without bound
unless something prunes it: every "New chat" click, or every stateless API
caller minting its own conversation_id, inserts a row that's otherwise
permanent. `prune_old_conversations()` deletes anything untouched for
longer than a cutoff; main.py calls it once per server startup.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import settings

DB_PATH = Path(settings.vectorstore_dir).resolve().parent / "conversations.sqlite3"
DEFAULT_MAX_AGE_DAYS = 30


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # timeout: wait for a competing writer instead of immediately raising
    # "database is locked" — sync FastAPI routes run in a threadpool, so two
    # requests touching the same conversation_id can genuinely overlap.
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "conversation_id TEXT PRIMARY KEY, history TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        with conn:  # commits on clean exit, rolls back on exception
            yield conn
    finally:
        conn.close()


def load_history(conversation_id: str) -> list[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT history FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
    return json.loads(row[0]) if row else []


def save_history(conversation_id: str, history: list[dict]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, history, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(conversation_id) DO UPDATE SET history = excluded.history, updated_at = excluded.updated_at",
            (conversation_id, json.dumps(history), datetime.now(timezone.utc).isoformat()),
        )


def clear_history(conversation_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))


def prune_old_conversations(max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> int:
    """Deletes any conversation not updated within `max_age_days` — a plain
    dict never needed this since a restart wiped it for free; a durable
    store does. Returns the number of rows removed."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,))
        return cursor.rowcount
