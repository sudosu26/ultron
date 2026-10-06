"""SQLite queries for saved conversations and messages."""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from ultron.utils.paths import DATA_DIR

_DATABASE_FILE = DATA_DIR / "ultron.sqlite3"


@contextmanager
def _connection() -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(_DATABASE_FILE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _initialize_database() -> None:
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_conversation_id_id
                ON messages(conversation_id, id);
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )


def _as_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def create_conversation(title: str = "New conversation") -> str:
    conversation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, now, now),
        )
    return conversation_id


def list_conversations() -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return _as_dict(row)


def update_conversation_title(conversation_id: str, title: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, now, conversation_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Conversation not found: {conversation_id}")


def add_message(conversation_id: str, role: str, content: str) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported conversation role: {role}")

    now = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, now),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Conversation not found: {conversation_id}")
        return int(cursor.lastrowid)


def list_messages(conversation_id: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def count_messages(conversation_id: str) -> int:
    with _connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return int(row[0])


def get_setting(key: str, default: str | None = None) -> str | None:
    with _connection() as connection:
        row = connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()
    return str(row["value"]) if row is not None else default


def set_setting(key: str, value: str) -> None:
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


_initialize_database()
