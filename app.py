"""
מערכת התקנון — אפליקציית Streamlit.

כרגע יש כרטיסייה אחת בלבד: "מצב המערכת", שמראה אם החיבורים לענן
(Neon, Pinecone) עובדים, ואם מפתח ה-Gemini קיים בקובץ ה-.env.

עיצוב מכוון: שום בדיקה לא מפילה את האפליקציה. אם שירות לא זמין, או
שמפתח חסר, מוצג X אדום וטקסט השגיאה — במקום קריסה.
"""

from __future__ import annotations

import streamlit as st

import db
import vectors
from config import ENV_PATH, Settings, _load_raw_values

st.set_page_config(page_title="מערכת התקנון", page_icon="📖", layout="centered")

# הופכים את כל האפליקציה ל-RTL (עברית מיושרת לימין)
st.markdown(
    """
    <style>
    html, body, [class*="css"] { direction: rtl; text-align: right; }
    .stTabs [data-baseweb="tab-list"] { direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("מערכת התקנון")


def load_raw_settings() -> Settings:
    """
    טוען את המפתחות ישירות מה-.env, בלי לעצור אם חסר אחד מהם.

    בשונה מ-config.load_settings (שנועד לסקריפטים ועוצר אם משהו חסר),
    כאן המטרה היא להציג בעמוד את הסטטוס של כל שירות בנפרד — כולל השורות
    שבהן דווקא המפתח חסר.
    """
    raw = _load_raw_values(ENV_PATH)
    return Settings(
        gemini_api_key=(raw.get("GEMINI_API_KEY") or "").strip(),
        pinecone_api_key=(raw.get("PINECONE_API_KEY") or "").strip(),
        pinecone_index=(raw.get("PINECONE_INDEX") or "").strip(),
        database_url=(raw.get("DATABASE_URL") or "").strip(),
    )


def status_line(label: str, ok: bool, detail: str = "") -> None:
    icon = "✅" if ok else "❌"
    line = f"{icon} **{label}**"
    if detail:
        line += f" — {detail}"
    st.markdown(line)


def check_neon(settings: Settings) -> None:
    if not settings.database_url:
        status_line("חיבור ל-Neon", False, "מפתח DATABASE_URL חסר בקובץ ה-.env")
        return
    try:
        version_text = db.server_version(settings).split(",")[0]
    except Exception as err:  # noqa: BLE001 — רוצים להציג את השגיאה, לא לקרוס
        status_line("חיבור ל-Neon", False, str(err))
        return
    status_line("חיבור ל-Neon", True, version_text)


def check_pinecone(settings: Settings) -> None:
    if not settings.pinecone_api_key or not settings.pinecone_index:
        status_line("חיבור ל-Pinecone", False, "מפתח PINECONE_API_KEY או PINECONE_INDEX חסר")
        return
    try:
        vectors.use_settings(settings)
        exists = vectors.index_exists()
    except Exception as err:  # noqa: BLE001
        status_line("חיבור ל-Pinecone", False, str(err))
        return
    detail = "האינדקס קיים" if exists else f"האינדקס '{settings.pinecone_index}' עדיין לא נוצר"
    status_line("חיבור ל-Pinecone", True, detail)


def check_gemini(settings: Settings) -> None:
    status_line("מפתח Gemini", bool(settings.gemini_api_key), "" if settings.gemini_api_key else "חסר בקובץ ה-.env")


(tab_status,) = st.tabs(["מצב המערכת"])

with tab_status:
    settings = load_raw_settings()
    check_neon(settings)
    check_pinecone(settings)
    check_gemini(settings)
