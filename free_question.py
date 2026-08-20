"""
ליבת הלוגיקה של לשונית "שאלה חופשית" — עונה על שאלה חופשית פעמיים,
פעם אחת לכל שיטת אחזור, כדי להשוות ביניהן בצורה הוגנת.

שני הצדדים קוראים מאותו קורפוס (retrieval_eval.load_corpus — אותה טבלה
takanon_sections), משתמשים באותו prompt ובאותו מודל מענה. ההבדל היחיד
בין העמודות הוא שיטת האחזור:

    dense (Pinecone)   דמיון קוסינוס בין embedding של השאלה לווקטורים
                       הקיימים באינדקס.
    sparse (BM25)      ניקוד BM25 מילות-מפתח על אותם צ'אנקים.

לא נוצר כאן אינדקס, מודל embedding, קורפוס או צינור אחזור חדשים — הכול
מיובא מ-retrieval_eval.py ו-vectors.py הקיימים.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import retrieval_eval as reval
import vectors
from config import Settings
from load_takanon import DEFAULT_MODEL as ANSWER_MODEL  # אותו מודל Gemini שכבר בשימוש בפרויקט

TOP_K = 3

DENSE_LABEL = "סמנטי / Embeddings (Pinecone)"
SPARSE_LABEL = "BM25"

# אותו prompt לשני הצדדים — ראו ask.py (הגרסה הקודמת של המסך הזה). מבקש
# תשובה מבוססת על ההקשר בלבד, לא ממציא, ומצטט מספרי סעיפים.
ANSWER_PROMPT = """אתה עונה על שאלות בנוגע לתקנון של מכללה אקדמית.

ענה אך ורק על סמך קטעי התקנון שמצורפים למטה. אם הקטעים אינם מכילים את
התשובה, כתוב זאת במפורש ואל תשלים מידע מהידע הכללי שלך.

ציין בסוף התשובה את מספרי הסעיפים שעליהם הסתמכת.
ענה בעברית, בקצרה — שניים עד ארבעה משפטים.

השאלה:
{question}

קטעי התקנון:
{context}

התשובה:"""

# פרוקסי גס ל"מונח מדויק" בשאלה: מספר סעיף בצורה X.Y
SECTION_TOKEN_RE = re.compile(r"\d+\.\d+")


@dataclass
class MethodResult:
    """התוצאה המלאה של צד אחד — אחזור + תשובה + זמנים."""

    method: str
    model: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    error: str = ""
    retrieval_ms: int = 0
    answer_ms: int = 0

    @property
    def total_ms(self) -> int:
        return self.retrieval_ms + self.answer_ms


def build_context(hits: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[סעיף {h.get('section_number') or '?'} · {h.get('chapter') or ''} · עמוד {h.get('page') or '?'}]\n"
        f"{h.get('content') or ''}"
        for h in hits
    )


def generate_answer(client: Any, model: str, question: str, hits: list[dict[str, Any]]) -> str:
    from google.genai import types

    if not hits:
        return "לא נמצאו קטעים רלוונטיים, ולכן אין בסיס לתשובה."

    response = client.models.generate_content(
        model=model,
        contents=ANSWER_PROMPT.format(question=question, context=build_context(hits)),
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return (response.text or "").strip()


def run_side(
    method: str,
    model: str,
    retrieve: Callable[[str], list[dict[str, Any]]],
    client: Any,
    question: str,
) -> MethodResult:
    """
    מריצה צד אחד עד הסוף, עם תזמון נפרד לאחזור ולמענה.

    נפרד בכוונה: זמן האחזור הוא תכונה של שיטת האחזור שמושווית, בעוד זמן
    היצירה הוא אותו מודל בשני הצדדים ולא אמור להטות את ההשוואה.
    """
    result = MethodResult(method=method, model=model)

    started = time.perf_counter()
    try:
        result.hits = retrieve(question)
    except Exception as err:  # noqa: BLE001
        result.error = f"האחזור נכשל: {str(err)[:160]}"
        result.retrieval_ms = int((time.perf_counter() - started) * 1000)
        return result
    result.retrieval_ms = int((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    try:
        result.answer = generate_answer(client, ANSWER_MODEL, question, result.hits)
    except Exception as err:  # noqa: BLE001
        result.error = f"יצירת התשובה נכשלה: {str(err)[:160]}"
    result.answer_ms = int((time.perf_counter() - started) * 1000)
    return result


def run_both(
    settings: Settings, dimension: int, question: str, client: Any
) -> tuple[MethodResult, MethodResult]:
    """
    מריצה את שני הצדדים על אותה שאלה, מאותו קורפוס בדיוק.
    מחזירה (dense, sparse).
    """
    corpus = reval.load_corpus(settings)
    bm25 = reval.build_bm25(corpus)
    vectors.use_settings(settings)

    def dense_retrieve(q: str) -> list[dict[str, Any]]:
        embedding = reval.embed_query(client, settings.embedding_model, dimension, q)
        return reval.dense_search(embedding, TOP_K)

    def sparse_retrieve(q: str) -> list[dict[str, Any]]:
        return reval.bm25_search(bm25, corpus, q, TOP_K)

    dense_result = run_side(DENSE_LABEL, settings.embedding_model, dense_retrieve, client, question)
    sparse_result = run_side(SPARSE_LABEL, reval.BM25_MODEL_NAME, sparse_retrieve, client, question)
    return dense_result, sparse_result


def has_exact_section_token(query: str) -> bool:
    return bool(SECTION_TOKEN_RE.search(query))


def investigate_pattern(eval_results: tuple[list, list, int] | None) -> str:
    """
    בודקת אם התבנית הצפויה (BM25 טוב יותר במונחים מדויקים, Dense טוב יותר
    בפרפרזה) נתמכת בפועל בתוצאות ה"הערכת אחזור" האחרונה שהורצה בסשן הזה.

    לא כותבת מסקנה קבועה מראש: אם אין נתוני הערכה בסשן, או שאין מספיק
    שאלות עם מונח מדויק כדי לבחון את זה, היא אומרת זאת במפורש ומדווחת
    את מה שכן ניתן לראות בנתונים.
    """
    if not eval_results:
        return (
            "טרם הורצה 'הערכת אחזור' בסשן הזה — אין עדיין נתונים להשוואה "
            "שיטתית בין השיטות. הריצו אותה כדי לקבל תובנה מבוססת-נתונים."
        )

    bm25_results, dense_results, k = eval_results
    if not bm25_results:
        return "הרצת ההערכה האחרונה לא הכילה שאלות."

    bm25_agg = reval.aggregate(bm25_results)
    dense_agg = reval.aggregate(dense_results)
    exact_bm25 = [r for r in bm25_results if has_exact_section_token(r.query)]

    if not exact_bm25:
        # לא ניתן לבחון את תת-ההשערה הספציפית על מונחים מדויקים — מדווחים
        # מה שכן עולה מהנתונים: ההשוואה הכללית.
        if dense_agg["recall"] > bm25_agg["recall"] and dense_agg["mrr"] > bm25_agg["mrr"]:
            return (
                f"בהרצת ההערכה האחרונה (k={k}, {bm25_agg['queries']} שאלות), אף שאלה לא הכילה "
                "מספר סעיף מפורש, כך שלא ניתן לבחון את ההשערה הספציפית על מונחים מדויקים. "
                f"מה שכן עולה מהנתונים: Dense עלה בממוצע על BM25 "
                f"(Recall@{k}={dense_agg['recall']:.2f} מול {bm25_agg['recall']:.2f}, "
                f"MRR@{k}={dense_agg['mrr']:.2f} מול {bm25_agg['mrr']:.2f}) — "
                "תואם לציפייה ששאלות מנוסחות בשפה טבעית מתאימות יותר לאחזור סמנטי."
            )
        if bm25_agg["recall"] > dense_agg["recall"]:
            return (
                f"בהרצת ההערכה האחרונה BM25 עלה בממוצע על Dense "
                f"(Recall@{k}={bm25_agg['recall']:.2f} מול {dense_agg['recall']:.2f}) — "
                "בניגוד לציפייה המקורית; ייתכן שנוסח השאלות בסט הזהב חופף מילולית לטקסט התקנון."
            )
        return "התוצאות של שתי השיטות בהרצת ההערכה האחרונה קרובות מדי כדי להצביע על מגמה ברורה."

    # יש שאלות עם מונח מדויק — אפשר לבחון את ההשערה הספציפית
    exact_ids = {r.query for r in exact_bm25}
    exact_dense = [r for r in dense_results if r.query in exact_ids]
    natural_bm25 = [r for r in bm25_results if r.query not in exact_ids]
    natural_dense = [r for r in dense_results if r.query not in exact_ids]

    if not natural_bm25:
        return "כל השאלות בהרצת ההערכה האחרונה הכילו מונח מדויק — אין קבוצת השוואה של ניסוח טבעי."

    exact_bm25_recall = sum(r.recall for r in exact_bm25) / len(exact_bm25)
    exact_dense_recall = sum(r.recall for r in exact_dense) / len(exact_dense)
    natural_bm25_recall = sum(r.recall for r in natural_bm25) / len(natural_bm25)
    natural_dense_recall = sum(r.recall for r in natural_dense) / len(natural_dense)

    pattern_holds = exact_bm25_recall >= exact_dense_recall and natural_dense_recall >= natural_bm25_recall
    if pattern_holds:
        return (
            f"הנתונים תומכים בתבנית הצפויה: בשאלות עם מונח מדויק BM25 השיג Recall={exact_bm25_recall:.2f} "
            f"מול Dense={exact_dense_recall:.2f}, ואילו בשאלות בניסוח טבעי Dense השיג "
            f"Recall={natural_dense_recall:.2f} מול BM25={natural_bm25_recall:.2f}."
        )
    return (
        "הנתונים לא תומכים בתבנית הצפויה באופן חד-משמעי: "
        f"מונח מדויק — BM25={exact_bm25_recall:.2f} מול Dense={exact_dense_recall:.2f}; "
        f"ניסוח טבעי — BM25={natural_bm25_recall:.2f} מול Dense={natural_dense_recall:.2f}."
    )
