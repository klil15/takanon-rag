"""
קריאה וכתיבה של קובץ סט הזהב (data/eval_queries.json).

לכל פריט יש מזהה יציב (id) שנקבע פעם אחת ונשמר בקובץ. המזהה הזה, ולא
תוכן השאלה, הוא מה שמאפשר לשמור שוב ושוב בלי ליצור כפילויות — עריכת
נוסח השאלה לא הופכת אותה לפריט חדש.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "eval_queries.json"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def new_item() -> dict[str, Any]:
    return {
        "id": _new_id(),
        "query": "",
        "answer": "",
        "relevant_sections": [],
        "verified": False,
        "reviewer_note": "",
    }


def load_items(path: Path = DEFAULT_PATH) -> list[dict[str, Any]]:
    """
    Loads the gold set. Legacy items (query + relevant_sections only, no id)
    get the missing fields filled in with defaults and a fresh id assigned —
    persisted the next time the caller saves.
    """
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for entry in raw:
        items.append(
            {
                "id": str(entry.get("id") or _new_id()),
                "query": str(entry.get("query") or ""),
                "answer": str(entry.get("answer") or ""),
                "relevant_sections": [
                    str(s) for s in (entry.get("relevant_sections") or [])
                ],
                "verified": bool(entry.get("verified", False)),
                "reviewer_note": str(entry.get("reviewer_note") or ""),
            }
        )
    return items


def save_items(items: list[dict[str, Any]], path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
