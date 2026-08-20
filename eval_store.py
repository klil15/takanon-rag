"""
מסד הנתונים לסט הזהב (gold dataset) של שאלות הערכה — טבלת eval_queries ב-Neon.

כל פריט: מזהה יציב (id), שאלה, תשובה, רשימת סעיפים רלוונטיים, דגל
"מאומת" והערת בודק. המפתח היציב (id) הוא זה שמאפשר להריץ שוב בלי ליצור
כפילויות — עריכת השאלה עצמה לא יוצרת פריט חדש.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import psycopg
from psycopg.rows import dict_row

from config import Settings, load_settings

TABLE = "eval_queries"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id                TEXT        PRIMARY KEY,
    query             TEXT        NOT NULL,
    answer            TEXT,
    relevant_sections TEXT[]      NOT NULL DEFAULT '{{}}',
    verified          BOOLEAN     NOT NULL DEFAULT false,
    reviewer_note     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

RETURNING_COLS = "id, query, answer, relevant_sections, verified, reviewer_note, updated_at"

_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


@contextmanager
def connect(settings: Settings | None = None) -> Iterator[psycopg.Connection]:
    cfg = settings or get_settings()
    conn = psycopg.connect(cfg.database_url, row_factory=dict_row, connect_timeout=15)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(settings: Settings | None = None) -> None:
    """Creates the table if it does not exist yet."""
    with connect(settings) as conn:
        conn.execute(SCHEMA_SQL)


def list_items(settings: Settings | None = None) -> list[dict[str, Any]]:
    with connect(settings) as conn:
        return conn.execute(f"SELECT {RETURNING_COLS} FROM {TABLE} ORDER BY created_at;").fetchall()


def upsert_item(
    item_id: str,
    query: str,
    answer: str | None,
    relevant_sections: list[str],
    verified: bool,
    reviewer_note: str | None,
    settings: Settings | None = None,
) -> None:
    sql = f"""
        INSERT INTO {TABLE} (id, query, answer, relevant_sections, verified, reviewer_note)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
        SET query             = EXCLUDED.query,
            answer             = EXCLUDED.answer,
            relevant_sections  = EXCLUDED.relevant_sections,
            verified           = EXCLUDED.verified,
            reviewer_note      = EXCLUDED.reviewer_note,
            updated_at         = now();
    """
    with connect(settings) as conn:
        conn.execute(sql, (item_id, query, answer, relevant_sections, verified, reviewer_note))


def upsert_items(items: Iterable[dict[str, Any]], settings: Settings | None = None) -> int:
    """Upserts many items at once. Returns how many were written."""
    rows = list(items)
    if not rows:
        return 0
    sql = f"""
        INSERT INTO {TABLE} (id, query, answer, relevant_sections, verified, reviewer_note)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
        SET query             = EXCLUDED.query,
            answer             = EXCLUDED.answer,
            relevant_sections  = EXCLUDED.relevant_sections,
            verified           = EXCLUDED.verified,
            reviewer_note      = EXCLUDED.reviewer_note,
            updated_at         = now();
    """
    params = [
        (
            item["id"],
            item["query"],
            item.get("answer") or None,
            item.get("relevant_sections") or [],
            bool(item.get("verified", False)),
            item.get("reviewer_note") or None,
        )
        for item in rows
    ]
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params)
    return len(rows)


def delete_items(item_ids: Iterable[str], settings: Settings | None = None) -> int:
    """Deletes items by id (used when an item was removed in the UI). Returns how many were deleted."""
    ids = list(item_ids)
    if not ids:
        return 0
    sql = f"DELETE FROM {TABLE} WHERE id = ANY(%s);"
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ids,))
            return cur.rowcount
