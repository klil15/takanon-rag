"""
תת-לשונית "שאלה חופשית" — שאלה אחת, חופשית לגמרי, נענית פעמיים באותה
נשימה: פעם אחת עם אחזור סמנטי (Pinecone) ופעם אחת עם BM25, כדי להשוות
ביניהן זו מול זו. קריאה בלבד: לא נכתב כלום ל-Neon, ל-Pinecone או לסט הזהב.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import free_question as fq
from add_to_pinecone import index_dimension
from config import Settings


def _hit_card(h: dict[str, Any]) -> None:
    section = h.get("section_number") or "?"
    chapter = h.get("chapter") or "—"
    page = h.get("page")
    score = h.get("score")
    preview = (h.get("content") or "")[:200]
    st.markdown(f"**סעיף {section}** · {chapter} · עמוד {page if page is not None else '—'} · ציון {score:.4f}")
    st.caption(preview)


def _method_column(result: fq.MethodResult) -> None:
    st.markdown(f"### {result.method}")
    st.caption(f"מודל: `{result.model}`")

    if result.error:
        st.error(result.error)

    if result.hits:
        st.markdown("**3 הקטעים המובילים:**")
        for h in result.hits:
            _hit_card(h)
    else:
        st.info("לא נמצאו קטעים.")

    st.markdown("**התשובה:**")
    st.write(result.answer or "—")

    st.caption(
        f"זמן אחזור: {result.retrieval_ms} ms · זמן מענה: {result.answer_ms} ms · "
        f"**זמן כולל: {result.total_ms} ms**"
    )


@st.cache_resource(show_spinner=False)
def _cached_genai_client(_settings: Settings) -> Any:
    from google import genai

    return genai.Client(api_key=_settings.gemini_api_key)


def render(settings: Settings) -> None:
    st.caption(
        "השוואה חופשית בין שתי שיטות אחזור על אותה שאלה, מאותו קורפוס "
        "(טבלת takanon_sections), עם אותו prompt ואותו מודל מענה. הלשונית "
        "הזו קריאה בלבד — שום דבר לא נכתב ל-Neon, ל-Pinecone או לסט הזהב."
    )

    question = st.text_input("הקלידו שאלה חופשית על התקנון", key="free_question_input")
    run = st.button("🔍 השוואת שתי שיטות אחזור", key="free_question_run")

    if not run or not question.strip():
        return

    try:
        dimension = index_dimension(settings)
    except Exception as err:  # noqa: BLE001
        st.error(f"בדיקת אינדקס Pinecone נכשלה: {err}")
        return

    client = _cached_genai_client(settings)

    with st.spinner("מריץ שני אחזורים ושתי תשובות..."):
        dense_result, sparse_result = fq.run_both(settings, dimension, question.strip(), client)

    col_dense, col_sparse = st.columns(2)
    with col_dense:
        _method_column(dense_result)
    with col_sparse:
        _method_column(sparse_result)

    st.markdown("---")
    st.markdown("#### מה הנתונים מראים")
    st.write(fq.investigate_pattern(st.session_state.get("eval_results")))
