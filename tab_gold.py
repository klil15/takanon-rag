"""
תת-לשונית "בניית סט הזהב" בתוך "צ'אנקים וסט זהב".

טוענת את data/eval_queries.json ומציגה פריט אחד בכל פעם לבדיקה ידנית:
עריכת השאלה והתשובה, תיקון הסעיפים הרלוונטיים, סימון מאומת / דורש עבודה,
הערת בודק, מחיקה, והוספת פריטים חדשים. שום פריט לא נחשב מהימן עד שאדם
מאשר אותו — לכן דגל ה"מאומת" והערת הבודק נשמרים לכל פריט, והמסך מציג
כמה פריטים עדיין לא אומתו.

שמירה כותבת גם לקובץ data/eval_queries.json וגם לטבלת eval_queries ב-Neon,
לפי המזהה היציב (id) של כל פריט — כך שהרצה חוזרת מעדכנת ולא משכפלת.
"""

from __future__ import annotations

import streamlit as st

import eval_store
import gold_set
from config import Settings


def _ensure_state() -> None:
    if "gold_items" not in st.session_state:
        items = gold_set.load_items()
        st.session_state.gold_items = items
        st.session_state.gold_original_ids = {item["id"] for item in items}
        st.session_state.gold_index = 0


def render(settings: Settings) -> None:
    _ensure_state()
    items: list[dict] = st.session_state.gold_items

    unverified = sum(1 for it in items if not it["verified"])
    st.info(f"פריטים שטרם אומתו: {unverified} מתוך {len(items)}")

    if not items:
        st.write("אין פריטים בסט הזהב עדיין. הוסיפו פריט חדש למטה.")
    else:
        idx = max(0, min(st.session_state.gold_index, len(items) - 1))
        st.session_state.gold_index = idx
        item = items[idx]

        st.caption(f"פריט {idx + 1} מתוך {len(items)}")
        col_prev, col_next = st.columns(2)
        if col_prev.button("⬅ הקודם", disabled=idx == 0, key="gold_prev"):
            st.session_state.gold_index -= 1
            st.rerun()
        if col_next.button("הבא ➡", disabled=idx == len(items) - 1, key="gold_next"):
            st.session_state.gold_index += 1
            st.rerun()

        item["query"] = st.text_area("שאלה", value=item["query"], key=f"query_{item['id']}")
        item["answer"] = st.text_area("תשובה", value=item["answer"], key=f"answer_{item['id']}")
        sections_text = st.text_input(
            "סעיפים רלוונטיים (מופרדים בפסיקים)",
            value=", ".join(item["relevant_sections"]),
            key=f"sections_{item['id']}",
        )
        item["relevant_sections"] = [s.strip() for s in sections_text.split(",") if s.strip()]
        item["verified"] = st.checkbox("מאומת", value=item["verified"], key=f"verified_{item['id']}")
        item["reviewer_note"] = st.text_input(
            "הערת בודק", value=item["reviewer_note"], key=f"note_{item['id']}"
        )

        if st.button("🗑 מחיקת הפריט הזה", key="delete_item"):
            items.pop(idx)
            st.session_state.gold_index = max(0, idx - 1)
            st.rerun()

    st.markdown("---")
    if st.button("➕ הוספת פריט חדש", key="add_item"):
        items.append(gold_set.new_item())
        st.session_state.gold_index = len(items) - 1
        st.rerun()

    if st.button("💾 שמירת סט הזהב", key="save_gold"):
        gold_set.save_items(items)
        st.success(f"נשמרו {len(items)} פריטים לקובץ data/eval_queries.json.")

        try:
            eval_store.init_db(settings)
            eval_store.upsert_items(items, settings)
            removed_ids = st.session_state.gold_original_ids - {it["id"] for it in items}
            if removed_ids:
                eval_store.delete_items(removed_ids, settings)
            st.session_state.gold_original_ids = {it["id"] for it in items}
            st.success(f"נשמרו {len(items)} פריטים בטבלת eval_queries ב-Neon.")
        except Exception as err:  # noqa: BLE001 — כשל בענן לא מוחק את מה שכבר נשמר לקובץ
            st.error(f"השמירה ל-Neon נכשלה (הקובץ המקומי כן נשמר): {err}")
