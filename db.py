"""
Database layer — Neon Postgres in the cloud (not a local file on the machine).

The connection is always made through DATABASE_URL coming from the .env file,
exactly as it is (including the ?sslmode=require suffix).

The table stores the regulation sections:
    section_number  The section number (unique identifier)
    chapter         The chapter the section belongs to
    content         The text of the section
    version         Version number — incremented by 1 on every update
    created_at      When it was created
    updated_at      When it was last updated
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import psycopg
from psycopg.rows import dict_row

from config import Settings, load_settings

TABLE = "takanon_sections"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id             BIGSERIAL PRIMARY KEY,
    section_number TEXT        NOT NULL,
    chunk_index    INTEGER     NOT NULL DEFAULT 0,
    chapter        TEXT,
    content        TEXT        NOT NULL,
    summary        TEXT,
    page           INTEGER,
    version        INTEGER     NOT NULL DEFAULT 1,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Bring a table created by an earlier version up to the current shape. The
-- first version stored one row per section and had UNIQUE (section_number);
-- chunking means several rows per section, so that constraint has to go.
-- Every statement here is idempotent, and a no-op on a freshly created table.
ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0;
ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS summary     TEXT;
ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS page        INTEGER;
ALTER TABLE {TABLE} DROP CONSTRAINT IF EXISTS {TABLE}_section_number_key;

CREATE INDEX IF NOT EXISTS {TABLE}_chapter_idx ON {TABLE} (chapter);

-- A long section is stored as several chunks, so the section number alone is
-- not unique. The natural key is the section together with its chunk index.
CREATE UNIQUE INDEX IF NOT EXISTS {TABLE}_section_chunk_idx
    ON {TABLE} (section_number, chunk_index);
"""

RETURNING_COLS = (
    "section_number, chunk_index, chapter, content, summary, page, "
    "version, created_at, updated_at"
)

_settings: Settings | None = None


def get_settings() -> Settings:
    """Loads the settings once and caches them in memory."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


@contextmanager
def connect(settings: Settings | None = None) -> Iterator[psycopg.Connection]:
    """
    Connects to Neon and yields a connection.
    At the end of the block: automatic commit if everything succeeded,
    rollback if there was an error, and then the connection is closed.
    """
    cfg = settings or get_settings()
    conn = psycopg.connect(cfg.database_url, row_factory=dict_row, connect_timeout=15)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(settings: Settings | None = None) -> None:
    """Creates the table (and the index) if they do not exist yet."""
    with connect(settings) as conn:
        conn.execute(SCHEMA_SQL)


def upsert_section(
    section_number: str,
    chapter: str | None,
    content: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Inserts a new section; if it already exists — updates it and increments the version by 1.
    Returns the row as it was saved.
    """
    sql = f"""
        INSERT INTO {TABLE} (section_number, chapter, content)
        VALUES (%s, %s, %s)
        ON CONFLICT (section_number, chunk_index) DO UPDATE
        SET chapter    = EXCLUDED.chapter,
            content    = EXCLUDED.content,
            version    = {TABLE}.version + 1,
            updated_at = now()
        RETURNING {RETURNING_COLS};
    """
    with connect(settings) as conn:
        row = conn.execute(sql, (section_number, chapter, content)).fetchone()
    return row


def upsert_sections(
    sections: Iterable[tuple[str, str | None, str]],
    settings: Settings | None = None,
) -> int:
    """
    Inserts/updates many sections at once (takes triples: number, chapter, content).
    Returns the number of sections saved.
    """
    rows = list(sections)
    if not rows:
        return 0

    sql = f"""
        INSERT INTO {TABLE} (section_number, chapter, content)
        VALUES (%s, %s, %s)
        ON CONFLICT (section_number, chunk_index) DO UPDATE
        SET chapter    = EXCLUDED.chapter,
            content    = EXCLUDED.content,
            version    = {TABLE}.version + 1,
            updated_at = now();
    """
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
    return len(rows)


def get_section(
    section_number: str, settings: Settings | None = None
) -> dict[str, Any] | None:
    """
    Returns a single section by its number, or None if it does not exist.

    A long section is stored as several chunks; this returns the first one.
    Use get_section_chunks() to get all of them in order.
    """
    sql = (
        f"SELECT {RETURNING_COLS} FROM {TABLE} WHERE section_number = %s "
        "ORDER BY chunk_index LIMIT 1;"
    )
    with connect(settings) as conn:
        return conn.execute(sql, (section_number,)).fetchone()


def get_section_chunks(
    section_number: str, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Returns every chunk of one section, in reading order."""
    sql = (
        f"SELECT {RETURNING_COLS} FROM {TABLE} WHERE section_number = %s "
        "ORDER BY chunk_index;"
    )
    with connect(settings) as conn:
        return conn.execute(sql, (section_number,)).fetchall()


def get_sections(
    section_numbers: Sequence[str], settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Returns several sections given a list of section numbers."""
    if not section_numbers:
        return []
    sql = f"""
        SELECT {RETURNING_COLS} FROM {TABLE}
        WHERE section_number = ANY(%s)
        ORDER BY section_number;
    """
    with connect(settings) as conn:
        return conn.execute(sql, (list(section_numbers),)).fetchall()


def list_sections(
    chapter: str | None = None,
    limit: int = 100,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Returns a list of sections, optionally filtered by chapter."""
    if chapter is None:
        sql = f"SELECT {RETURNING_COLS} FROM {TABLE} ORDER BY section_number LIMIT %s;"
        params: tuple[Any, ...] = (limit,)
    else:
        sql = (
            f"SELECT {RETURNING_COLS} FROM {TABLE} WHERE chapter = %s "
            "ORDER BY section_number LIMIT %s;"
        )
        params = (chapter, limit)
    with connect(settings) as conn:
        return conn.execute(sql, params).fetchall()


def count_sections(settings: Settings | None = None) -> int:
    """How many sections are currently stored in the database."""
    with connect(settings) as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {TABLE};").fetchone()
    return int(row["n"])


def delete_section(section_number: str, settings: Settings | None = None) -> bool:
    """Deletes a section by its number. Returns True if something was actually deleted."""
    sql = f"DELETE FROM {TABLE} WHERE section_number = %s;"
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (section_number,))
            return cur.rowcount > 0


def server_version(settings: Settings | None = None) -> str:
    """Returns the server's Postgres version — a quick check that the connection is alive."""
    with connect(settings) as conn:
        row = conn.execute("SELECT version() AS v;").fetchone()
    return row["v"]


def get_all_chunks(settings: Settings | None = None) -> list[dict[str, Any]]:
    """
    Returns every chunk with its row id, for the chunk-review UI.

    The id (not section_number + chunk_index) is what an edit is applied
    against, so that editing the section number or chunk index itself is
    still safe — see update_chunk_by_id().
    """
    sql = f"SELECT id, {RETURNING_COLS} FROM {TABLE} ORDER BY section_number, chunk_index;"
    with connect(settings) as conn:
        return conn.execute(sql).fetchall()


def update_chunk_by_id(
    row_id: int,
    section_number: str,
    chunk_index: int,
    chapter: str | None,
    content: str,
    summary: str | None,
    page: int | None,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    """
    Updates one existing row by its id (never inserts). Returns the saved
    row, or None if no row has that id.
    """
    sql = f"""
        UPDATE {TABLE}
        SET section_number = %s,
            chunk_index    = %s,
            chapter        = %s,
            content        = %s,
            summary        = %s,
            page           = %s,
            version        = version + 1,
            updated_at     = now()
        WHERE id = %s
        RETURNING id, {RETURNING_COLS};
    """
    with connect(settings) as conn:
        return conn.execute(
            sql, (section_number, chunk_index, chapter, content, summary, page, row_id)
        ).fetchone()


def upsert_chunk(
    section_number: str,
    chunk_index: int,
    chapter: str | None,
    content: str,
    summary: str | None,
    page: int | None,
    settings: Settings | None = None,
) -> bool:
    """
    Inserts a new chunk, or updates the existing one with the same
    (section_number, chunk_index). Returns True if a new row was inserted,
    False if an existing row was updated.

    Used for loading uploaded chunks (PDF or JSON) — never touches rows
    outside the given (section_number, chunk_index) key.
    """
    sql = f"""
        INSERT INTO {TABLE} (section_number, chunk_index, chapter, content, summary, page)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (section_number, chunk_index) DO UPDATE
        SET chapter    = EXCLUDED.chapter,
            content    = EXCLUDED.content,
            summary    = EXCLUDED.summary,
            page       = EXCLUDED.page,
            version    = {TABLE}.version + 1,
            updated_at = now()
        RETURNING (xmax = 0) AS inserted;
    """
    with connect(settings) as conn:
        row = conn.execute(
            sql, (section_number, chunk_index, chapter, content, summary, page)
        ).fetchone()
    return bool(row["inserted"])
