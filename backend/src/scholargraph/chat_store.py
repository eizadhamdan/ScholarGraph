import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = Path(
    os.getenv("CHAT_HISTORY_DB", BACKEND_ROOT / "data" / "chat_history.sqlite3")
).expanduser()
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = BACKEND_ROOT / DATABASE_PATH


class ConversationNotFoundError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
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
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS messages_conversation_id_id
                ON messages(conversation_id, id);
            """
        )


def create_conversation() -> dict:
    now = _now()
    conversation = {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "created_at": now,
        "updated_at": now,
    }
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (:id, :title, :created_at, :updated_at)
            """,
            conversation,
        )
    return conversation


def list_conversations() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with _connect() as connection:
        conversation = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None

        messages = connection.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()

    return {**dict(conversation), "messages": [dict(row) for row in messages]}


def add_user_message(
    conversation_id: str, content: str
) -> tuple[dict, dict, list[dict]] | None:
    now = _now()
    with _connect() as connection:
        conversation = connection.execute(
            "SELECT id, title, created_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None

        previous_messages = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT 12
            """,
            (conversation_id,),
        ).fetchall()
        history = [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(previous_messages)
        ]
        title = conversation["title"]
        if not previous_messages:
            title = content[:64]
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, now, conversation_id),
        )
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, 'user', ?, ?)
            """,
            (conversation_id, content, now),
        )
        message = {
            "id": cursor.lastrowid,
            "role": "user",
            "content": content,
            "created_at": now,
        }
        summary = {
            "id": conversation_id,
            "title": title,
            "created_at": conversation["created_at"],
            "updated_at": now,
        }
    return summary, message, history


def add_assistant_message(conversation_id: str, content: str) -> dict:
    now = _now()
    with _connect() as connection:
        if (
            connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            is None
        ):
            raise ConversationNotFoundError(conversation_id)
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, 'assistant', ?, ?)
            """,
            (conversation_id, content, now),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        return {
            "id": cursor.lastrowid,
            "role": "assistant",
            "content": content,
            "created_at": now,
        }


def delete_conversation(conversation_id: str) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
    return cursor.rowcount > 0
