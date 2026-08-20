"""
תת-לשונית "הערכת אחזור" — משווה BM25 מול Pinecone על סט הזהב הקיים
(data/eval_queries.json + gold_set.py, אותו סט שנבנה בכרטיסיית
"צ'אנקים וסט זהב"; לא נוצר כאן קובץ נפרד).

שום קריאה כאן לא נוגעת בנתוני האמת: קריאה בלבד מ-Neon ומ-Pinecone, ושמירת
סט הזהב קורית רק כשהמשתמש לוחץ במפורש על כפתור השמירה.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import db
import gold_set
import retrieval_eval as reval
import vectors
from add_to_pinecone import index_dimension
from config import Settings

TOPK_OPTIONS = (5, 10, 20)


# ── עזרי קאשינג ─────────────────────────────────────────────────────────
# הלקוח עצמו הוא משאב שחוזרים ומשתמשים בו (cache_resource); ה-embedding
# של כל שאלה הוא נתון שנגזר ממנה (cache_data), כדי לא לחשב אותו שוב באותה
# הרצה או בהרצה חוזרת עם אותם k / gold set. שני הארגומנטים שאסור לקאש
# לפי ערכם (הלקוח, וההגדרות שנושאות מפתחות) מסומנים בקו-תחתי מוביל, כך
# ש-Streamlit מדלג על ה-hashing שלהם.


@st.cache_resource(show_spinner=False)
def _cached_genai_client(_settings: Settings) -> Any:
    from google import genai

    return genai.Client(api_key=_settings.gemini_api_key)


@st.cache_data(show_spinner=False)
def _cached_query_embedding(_client: Any, model: str, dimension: int, query: str) -> list[float]:
    return reval.embed_query(_client, model, dimension, query)


@st.cache_data(show_spinner=False)
def _cached_corpus(_settings: Settings) -> list[dict[str, Any]]:
    return reval.load_corpus(_settings)


# ── מצב הכרטיסייה ────────────────────────────────────────────────────


def _ensure_state() -> None:
    if "eval_gold_items" not in st.session_state:
        st.session_state.eval_gold_items = gold_set.load_items()
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None  # (bm25_results, dense_results, k)


def _discovery_panel(settings: Settings) -> tuple[int | None, str]:
    """
    מציגה מה התגלה בפועל בפרויקט לפני הרצה: קובץ סט הזהב והשדות שלו,
    טבלת/עמודות Neon, שם אינדקס Pinecone, מודל ה-embedding והמימד שלו.
    מחזירה (מימד האינדקס או None בכשל, הודעת שגיאה אם הייתה).
    """
    st.markdown("#### מה התגלה בפרויקט")

    st.write(
        f"**סט הזהב:** `{gold_set.DEFAULT_PATH.name}` "
        f"(ליד {gold_set.DEFAULT_PATH.parent.name}/) — שדות: "
        "`id`, `query`, `answer`, `relevant_sections`, `verified`, `reviewer_note`. "
        f"נטענו {len(st.session_state.eval_gold_items)} פריטים."
    )
    st.write(
        f"**טבלת Neon:** `{db.TABLE}` — עמודות בשימוש: "
        "`id`, `section_number`, `chunk_index`, `chapter`, `content`, `page`."
    )
    st.write(f"**אינדקס Pinecone:** `{settings.pinecone_index}`")

    try:
        dimension = index_dimension(settings)
    except Exception as err:  # noqa: BLE001
        st.error(f"בדיקת האינדקס נכשלה: {err}")
        return None, str(err)

    st.write(f"**מודל embedding:** `{settings.embedding_model}` — מימד: **{dimension}**")
    return dimension, ""


# ── עריכת סט הזהב (קל, לא כפילות של כרטיסיית סט הזהב) ─────────────────


def _gold_editor() -> None:
    st.markdown("#### סט הזהב לשימוש בהערכה")
    st.caption(
        "זהו אותו קובץ data/eval_queries.json שנבנה בכרטיסיית \"צ'אנקים וסט זהב\". "
        "עריכה כאן משפיעה רק על ההערכה הנוכחית — השינויים לא נשמרים לקובץ "
        "עד לחיצה מפורשת על 'שמירת סט הזהב'."
    )

    uploaded = st.file_uploader(
        "החלפת סט הזהב מקובץ JSON (אופציונלי)", type=["json"], key="eval_gold_upload"
    )
    if uploaded is not None:
        import json

        try:
            raw = pd.read_json(uploaded)
            entries = raw.to_dict(orient="records")
        except Exception:
            uploaded.seek(0)
            entries = json.loads(uploaded.read().decode("utf-8"))
        loaded = []
        for entry in entries:
            loaded.append(
                {
                    "id": str(entry.get("id") or gold_set.new_item()["id"]),
                    "query": str(entry.get("query") or ""),
                    "answer": str(entry.get("answer") or ""),
                    "relevant_sections": [
                        str(s) for s in (entry.get("relevant_sections") or [])
                    ],
                    "verified": bool(entry.get("verified", False)),
                    "reviewer_note": str(entry.get("reviewer_note") or ""),
                }
            )
        st.session_state.eval_gold_items = loaded
        st.success(f"נטענו {len(loaded)} פריטים מהקובץ שהועלה (בזיכרון בלבד, טרם נשמר).")

    df = pd.DataFrame(
        [
            {
                "query": it["query"],
                "relevant_sections": ", ".join(it["relevant_sections"]),
                "verified": it["verified"],
            }
            for it in st.session_state.eval_gold_items
        ]
    )
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", key="eval_gold_editor")

    col1, col2 = st.columns(2)
    if col1.button("↩ החלת עריכות (לזיכרון בלבד)", key="eval_apply_edits"):
        new_items = []
        old_by_query = {it["query"]: it for it in st.session_state.eval_gold_items}
        for _, row in edited.iterrows():
            base = old_by_query.get(row["query"], gold_set.new_item())
            new_items.append(
                {
                    "id": base["id"],
                    "query": str(row["query"]),
                    "answer": base.get("answer", ""),
                    "relevant_sections": [
                        s.strip() for s in str(row["relevant_sections"]).split(",") if s.strip()
                    ],
                    "verified": bool(row["verified"]),
                    "reviewer_note": base.get("reviewer_note", ""),
                }
            )
        st.session_state.eval_gold_items = new_items
        st.rerun()

    if col2.button("💾 שמירת סט הזהב (לקובץ)", key="eval_save_gold"):
        gold_set.save_items(st.session_state.eval_gold_items)
        st.success(f"נשמרו {len(st.session_state.eval_gold_items)} פריטים ל-{gold_set.DEFAULT_PATH}.")


# ── הרצת ההערכה ───────────────────────────────────────────────────────


def _run_evaluation(settings: Settings, dimension: int, k: int, limit: int | None) -> None:
    items = reval.gold_items_from_project(st.session_state.eval_gold_items)
    if limit:
        items = items[:limit]
    if not items:
        st.warning("אין פריטים בסט הזהב להרצה.")
        return

    missing = reval.missing_sections(items, settings)
    if missing:
        st.warning(f"סעיפי זהב שלא קיימים בטבלה כלל (לעולם לא ייענו): {', '.join(missing)}")

    progress = st.progress(0.0, text="מתחיל...")

    def report(fraction: float, label: str) -> None:
        # BM25 תופס את המחצית הראשונה של הפס, Pinecone את השנייה
        progress.progress(min(0.999, fraction), text=label)

    # ── BM25 ────────────────────────────────────────────────────────
    corpus = _cached_corpus(settings)
    bm25 = reval.build_bm25(corpus)
    bm25_results = reval.run_retriever(
        items, reval.BM25_RETRIEVER, reval.BM25_MODEL_NAME,
        lambda q, limit_: reval.bm25_search(bm25, corpus, q, limit_),
        k, lambda f, label: report(f * 0.5, label),
    )

    # ── Pinecone ────────────────────────────────────────────────────
    vectors.use_settings(settings)
    client = _cached_genai_client(settings)

    def dense_search_fn(query: str, limit_: int) -> list[dict[str, Any]]:
        embedding = _cached_query_embedding(client, settings.embedding_model, dimension, query)
        return reval.dense_search(embedding, limit_)

    dense_results = reval.run_retriever(
        items, reval.DENSE_RETRIEVER, settings.embedding_model,
        dense_search_fn, k, lambda f, label: report(0.5 + f * 0.5, label),
    )

    progress.progress(1.0, text="הושלם.")
    st.session_state.eval_results = (bm25_results, dense_results, k)


# ── תצוגת תוצאות ──────────────────────────────────────────────────────


def _results_view(k: int) -> None:
    bm25_results, dense_results, result_k = st.session_state.eval_results
    if result_k != k:
        st.info(f"התוצאות המוצגות הן מהרצה עם k={result_k}. הריצו שוב כדי לעדכן ל-k={k}.")

    bm25_agg = reval.aggregate(bm25_results)
    dense_agg = reval.aggregate(dense_results)

    st.markdown("#### טבלת השוואה")
    compare_df = pd.DataFrame(
        [
            {
                "שיטה": reval.BM25_RETRIEVER,
                "מודל": reval.BM25_MODEL_NAME,
                "k": result_k,
                "שאלות": bm25_agg["queries"],
                "Mean Recall@K": round(bm25_agg["recall"], 4),
                "Mean MRR@K": round(bm25_agg["mrr"], 4),
                "נמצא לפחות סעיף אחד": bm25_agg["found"],
            },
            {
                "שיטה": reval.DENSE_RETRIEVER,
                "מודל": dense_results[0].model if dense_results else "",
                "k": result_k,
                "שאלות": dense_agg["queries"],
                "Mean Recall@K": round(dense_agg["recall"], 4),
                "Mean MRR@K": round(dense_agg["mrr"], 4),
                "נמצא לפחות סעיף אחד": dense_agg["found"],
            },
        ]
    )
    st.dataframe(compare_df, width="stretch")

    st.markdown("#### השוואה גרפית")
    chart_df = pd.DataFrame(
        {
            "Mean Recall@K": [bm25_agg["recall"], dense_agg["recall"]],
            "Mean MRR@K": [bm25_agg["mrr"], dense_agg["mrr"]],
        },
        index=[reval.BM25_RETRIEVER, reval.DENSE_RETRIEVER],
    )
    st.bar_chart(chart_df)

    st.markdown("#### פירוט לפי שאלה")
    for bm25_r, dense_r in zip(bm25_results, dense_results):
        with st.expander(f"❓ {bm25_r.query}"):
            st.write(f"**סעיפים רלוונטיים (זהב):** {', '.join(bm25_r.relevant) or '—'}")

            col_bm25, col_dense = st.columns(2)
            for col, result in ((col_bm25, bm25_r), (col_dense, dense_r)):
                with col:
                    st.markdown(f"**{result.retriever}** · `{result.model}`")
                    st.write(
                        f"Recall@{result_k} = {result.recall:.3f}  ·  "
                        f"RR = {result.reciprocal_rank:.3f}"
                    )
                    if not result.hits:
                        st.caption("אין תוצאות.")
                        continue
                    hits_df = pd.DataFrame(
                        [
                            {
                                "דירוג": h["rank"],
                                "סעיף": h["section_number"],
                                "רלוונטי": "✅" if h["relevant"] else "—",
                                "ציון": h["score"],
                            }
                            for h in result.hits
                        ]
                    )
                    st.dataframe(hits_df, width="stretch", hide_index=True)

    st.markdown("#### ייצוא")
    all_rows = reval.csv_rows(bm25_results) + reval.csv_rows(dense_results)
    csv_bytes = pd.DataFrame(all_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ הורדת תוצאות מפורטות (CSV)",
        data=csv_bytes,
        file_name=f"retrieval_eval_k{result_k}.csv",
        mime="text/csv",
        key="eval_download_csv",
    )


# ── כניסה ─────────────────────────────────────────────────────────────


def render(settings: Settings) -> None:
    _ensure_state()

    dimension, error = _discovery_panel(settings)
    st.markdown("---")
    _gold_editor()
    st.markdown("---")

    if dimension is None:
        st.error("לא ניתן להריץ הערכה בלי אינדקס Pinecone תקין.")
        return

    st.markdown("#### הרצת הערכה")
    k = st.selectbox("top_k", TOPK_OPTIONS, index=1, key="eval_topk")

    col_test, col_full = st.columns(2)
    if col_test.button("🧪 בדיקה מהירה (2 שאלות)", key="eval_run_test"):
        with st.spinner("מריץ בדיקה על 2 שאלות..."):
            _run_evaluation(settings, dimension, k, limit=2)
    if col_full.button("▶ הרצת הערכה מלאה", key="eval_run_full"):
        with st.spinner("מריץ הערכה מלאה..."):
            _run_evaluation(settings, dimension, k, limit=None)

    if st.session_state.eval_results is not None:
        st.markdown("---")
        _results_view(k)
