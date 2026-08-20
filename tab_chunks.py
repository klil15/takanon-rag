"""
תת-לשונית "סקירת צ'אנקים" בתוך "צ'אנקים וסט זהב".

מציגה את כל הצ'אנקים מטבלת takanon_sections בטבלה ניתנת לעריכה, עם סינון
לפי סעיף וחיפוש חופשי. שמירה מעדכנת שורות קיימות לפי המזהה הפנימי שלהן
(id) — לעולם לא מוסיפה שורה חדשה. הוספת שורות חדשות אפשרית רק דרך העלאת
קובץ (PDF או JSON), ורק אחרי שדוח האימות מוצג ומאושר בנפרד.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st

import db
import load_takanon as lt
from config import Settings

EDITABLE_COLUMNS = ["section_number", "chapter", "summary", "content", "page", "chunk_index"]
COLUMN_LABELS = {
    "section_number": "מספר סעיף",
    "chapter": "כותרת פרק",
    "summary": "תקציר",
    "content": "טקסט",
    "page": "עמוד מקור",
    "chunk_index": "אינדקס צ'אנק",
}


@dataclass
class ValidationReport:
    valid: list[dict[str, Any]] = field(default_factory=list)
    invalid: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    will_insert: int = 0
    will_update: int = 0


def _load_chunks_df(settings: Settings) -> pd.DataFrame:
    rows = db.get_all_chunks(settings)
    if not rows:
        return pd.DataFrame(
            columns=["id", "section_number", "chunk_index", "chapter", "content", "summary", "page"]
        )
    return pd.DataFrame(rows)


def _read_pdf_pages_from_stream(file: Any) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(file)
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — עמוד בעייתי אחד לא יעצור את הפענוח
            text = ""
        pages.append((number, text))
    return pages


def _parse_pdf_upload(file: Any) -> tuple[list[dict[str, Any]], list[str]]:
    pages = _read_pdf_pages_from_stream(file)
    section_style = lt.detect_heading_style(
        pages, lt.SECTION_STYLES, ("keyword_after", "keyword_before")
    )
    chapter_style = lt.detect_heading_style(
        pages, lt.CHAPTER_STYLES, ("keyword_after", "keyword_before")
    )
    sections = lt.split_into_sections(pages, section_style, chapter_style)
    if not sections:
        return [], ["לא זוהה אף סעיף בקובץ — ייתכן שהמבנה שונה מהמוכר."]

    chunks = lt.build_chunks(sections, lt.MAX_CHARS, lt.OVERLAP_CHARS)
    rows = [
        {
            "section_number": c.section_number,
            "chapter": c.chapter_title,
            "content": c.text,
            "summary": None,  # לא נקרא ל-LLM מכאן — הסיכום מולא ידנית בטבלה
            "page": c.page,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    return rows, []


def _parse_json_upload(file: Any) -> tuple[list[dict[str, Any]], list[str]]:
    import json

    try:
        raw = json.loads(file.read().decode("utf-8"))
    except Exception as err:  # noqa: BLE001
        return [], [f"הקובץ אינו JSON תקין: {err}"]

    if not isinstance(raw, list):
        return [], ["הקובץ חייב להכיל רשימת אובייקטים (JSON array)."]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    per_section_counter: dict[str, int] = {}

    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            errors.append(f"פריט {i}: אינו אובייקט JSON")
            continue
        section_number = str(entry.get("section_number") or "").strip()
        content = str(entry.get("text") or entry.get("content") or "").strip()
        chapter = str(entry.get("chapter_title") or entry.get("chapter") or "").strip() or None
        summary = str(entry.get("summary") or "").strip() or None
        page = entry.get("page")
        chunk_index = entry.get("chunk_index")
        if chunk_index is None:
            chunk_index = per_section_counter.get(section_number, 0)
        per_section_counter[section_number] = int(chunk_index) + 1
        rows.append(
            {
                "section_number": section_number,
                "chapter": chapter,
                "content": content,
                "summary": summary,
                "page": page,
                "chunk_index": int(chunk_index),
            }
        )
    return rows, errors


def _validate_candidates(
    candidates: list[dict[str, Any]], existing_keys: set[tuple[str, int]]
) -> ValidationReport:
    report = ValidationReport()
    seen: set[tuple[str, int]] = set()
    for row in candidates:
        if not row["content"].strip():
            report.invalid.append((row, "הטקסט ריק"))
            continue
        if not row["section_number"].strip():
            report.invalid.append((row, "אין מספר סעיף"))
            continue
        key = (row["section_number"], row["chunk_index"])
        if key in seen:
            report.invalid.append(
                (row, f"כפילות בתוך הקובץ: סעיף {key[0]} צ'אנק {key[1]}")
            )
            continue
        seen.add(key)
        report.valid.append(row)

    report.will_update = sum(
        1 for r in report.valid if (r["section_number"], r["chunk_index"]) in existing_keys
    )
    report.will_insert = len(report.valid) - report.will_update
    return report


def render(settings: Settings) -> None:
    if "chunks_df" not in st.session_state:
        try:
            st.session_state.chunks_df = _load_chunks_df(settings)
        except Exception as err:  # noqa: BLE001
            st.error(f"טעינת הצ'אנקים ממסד הנתונים נכשלה: {err}")
            return

    df: pd.DataFrame = st.session_state.chunks_df

    st.subheader("סינון וחיפוש")
    col1, col2 = st.columns(2)
    section_filter = col1.text_input("סינון לפי מספר סעיף (התחלת המספר)")
    search_text = col2.text_input("חיפוש חופשי בטקסט או בתקציר")

    view = df
    if section_filter:
        view = view[view["section_number"].astype(str).str.startswith(section_filter)]
    if search_text:
        mask = (
            view["content"].fillna("").str.contains(search_text, case=False, na=False)
            | view["summary"].fillna("").str.contains(search_text, case=False, na=False)
        )
        view = view[mask]

    st.caption(f"מוצגות {len(view)} מתוך {len(df)} שורות.")

    edited = st.data_editor(
        view,
        column_order=EDITABLE_COLUMNS,
        column_config={
            key: st.column_config.Column(label=label) for key, label in COLUMN_LABELS.items()
        },
        num_rows="fixed",
        width="stretch",
        key="chunks_editor",
    )

    if st.button("💾 שמירת שינויים", key="save_chunks"):
        changed = 0
        errors = 0
        for _, row in edited.iterrows():
            original_rows = df.loc[df["id"] == row["id"]]
            if original_rows.empty:
                continue
            original = original_rows.iloc[0]
            if any(row[c] != original[c] for c in EDITABLE_COLUMNS):
                try:
                    db.update_chunk_by_id(
                        int(row["id"]),
                        str(row["section_number"]),
                        int(row["chunk_index"]),
                        row["chapter"] or None,
                        str(row["content"]),
                        row["summary"] or None,
                        int(row["page"]) if pd.notna(row["page"]) else None,
                        settings,
                    )
                    changed += 1
                except Exception as err:  # noqa: BLE001
                    errors += 1
                    st.error(f"עדכון שורה {row['id']} נכשל: {err}")
        st.success(f"עודכנו {changed} שורות.")
        if errors:
            st.warning(f"{errors} עדכונים נכשלו — ראו את השגיאות למעלה.")
        st.session_state.chunks_df = _load_chunks_df(settings)
        st.rerun()

    st.markdown("---")
    st.subheader("הוספת שורות חדשות מקובץ")
    uploaded = st.file_uploader("העלאת PDF או JSON של צ'אנקים", type=["pdf", "json"])

    if uploaded is not None:
        if uploaded.name.lower().endswith(".pdf"):
            candidates, parse_errors = _parse_pdf_upload(uploaded)
        else:
            candidates, parse_errors = _parse_json_upload(uploaded)

        for err_msg in parse_errors:
            st.error(err_msg)

        if candidates:
            existing_keys = {
                (str(r["section_number"]), int(r["chunk_index"])) for _, r in df.iterrows()
            }
            report = _validate_candidates(candidates, existing_keys)

            st.markdown("#### דוח אימות")
            st.write(f"נמצאו {len(candidates)} צ'אנקים מועמדים בקובץ.")
            st.write(
                f"✅ תקינים: {len(report.valid)} "
                f"(חדשים: {report.will_insert}, עדכון קיימים: {report.will_update})"
            )
            if report.invalid:
                st.write(f"❌ לא תקינים ({len(report.invalid)}) — לא יישמרו:")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "מספר סעיף": r["section_number"],
                                "אינדקס צ'אנק": r["chunk_index"],
                                "סיבה": reason,
                            }
                            for r, reason in report.invalid
                        ]
                    ),
                    width="stretch",
                )

            if st.button(
                "✅ אישור כתיבה למסד הנתונים",
                key="confirm_upload",
                disabled=not report.valid,
            ):
                inserted = updated = errors = 0
                for row in report.valid:
                    try:
                        was_inserted = db.upsert_chunk(
                            row["section_number"],
                            row["chunk_index"],
                            row["chapter"],
                            row["content"],
                            row["summary"],
                            row["page"],
                            settings,
                        )
                    except Exception as err:  # noqa: BLE001
                        errors += 1
                        st.error(
                            f"כתיבת סעיף {row['section_number']}/{row['chunk_index']} נכשלה: {err}"
                        )
                        continue
                    if was_inserted:
                        inserted += 1
                    else:
                        updated += 1
                st.success(f"נכתבו {inserted} שורות חדשות, עודכנו {updated} שורות קיימות.")
                if errors:
                    st.warning(f"{errors} כתיבות נכשלו — ראו את השגיאות למעלה.")
                st.session_state.chunks_df = _load_chunks_df(settings)
                st.rerun()
