"""
ניקוד איכות האחזור מול סט השאלות המאומת (הסט הקיים, לא קובץ חדש).

שני שיטות אחזור מושוות על אותן שאלות:

    dense (Pinecone)   חיפוש וקטורי באינדקס הקיים, עם embedding של השאלה
                       באמצעות אותו מודל Google שכבר בנה את האינדקס.
    sparse (BM25)      BM25 על אותם צ'אנקים, נקראים ישירות מ-Neon
                       (טבלת takanon_sections, אותה טבלה שהפרויקט כבר משתמש בה).

שתי השיטות נשאלות אותן שאלות, מנוקדות באותו אופן, וכל שורת תוצאה נושאת את
שם השיטה והמודל שהפיקו אותה — כדי שכל מספר יהיה ניתן לשחזור.

יחידת הרלוונטיות היא מספר הסעיף, לא הצ'אנק. סט הזהב אומר "7.2 עונה על
השאלה הזו", ושתי השיטות מחזירות צ'אנקים — וסעיף ארוך הוא כמה צ'אנקים.
הפגיעות של כל שיטה מכווצות אפוא למספרי סעיפים ייחודיים, תוך שמירת הדירוג
הטוב ביותר שכל סעיף השיג. עשיית זה באופן זהה בשני הצדדים היא מה שהופך
את ההשוואה להוגנת.

בגלל שהכיווץ מאבד מיקומים, כל שיטה מתבקשת ליותר צ'אנקים מ-k הנמדד
(פי OVERFETCH) ונשמרים k מספרי הסעיפים הייחודיים הראשונים.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import add_to_pinecone as atp
import db
import gold_set
import vectors
from config import Settings

# מבקשים פי כמה צ'אנקים מ-k כדי שיהיה אפשר לשחזר k סעיפים ייחודיים
OVERFETCH = 3

BM25_RETRIEVER = "sparse (BM25)"
DENSE_RETRIEVER = "dense (Pinecone)"
BM25_MODEL_NAME = "rank_bm25 · BM25Okapi"

# task type בזמן שאילתה — שונה מ-task type בזמן אינדוקס (RETRIEVAL_DOCUMENT
# שבו נעשה שימוש ב-add_to_pinecone.py). המודל ממקם שאלות וקטעי טקסט אחרת.
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"

# אותיות עבריות, לועזיות וספרות. כל השאר — פיסוק, מקף, גרשיים כקיצור —
# מפריד בין טוקנים.
TOKEN_RE = re.compile(r"[0-9A-Za-z֐-׿]+")

# חלקיקים בני אות אחת שנדבקים לתחילת מילה בעברית: ו, ב, ה, ל, מ, ש, כ.
# "בקורס" ו"קורס" הן אותה מילה לצורך חיפוש, ול-BM25 אין שורש (stemmer) שיפתור
# את זה בעצמו.
HEBREW_PREFIXES = ("ו", "ב", "ה", "ל", "מ", "ש", "כ")
MIN_STEM_LENGTH = 3


@dataclass
class GoldItem:
    """שאלה אחת והסעיפים שאמורים לענות עליה."""

    query: str
    relevant: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """מה ששיטת אחזור אחת עשתה עבור שאלה אחת."""

    query: str
    retriever: str
    model: str
    k: int
    ranked: list[str]  # מספרי סעיפים ייחודיים, הטוב ביותר קודם
    relevant: list[str]
    hits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def first_relevant_rank(self) -> int:
        for position, section in enumerate(self.ranked[: self.k], start=1):
            if section in self.relevant:
                return position
        return 0

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_relevant_rank
        return 1.0 / rank if rank else 0.0

    @property
    def recall(self) -> float:
        if not self.relevant:
            return 0.0
        found = {s for s in self.ranked[: self.k] if s in self.relevant}
        return len(found) / len(self.relevant)


# ── סט הזהב — נטען מהמודול הקיים gold_set.py, לא מקובץ חדש ────────────


def gold_items_from_project(items: list[dict[str, Any]] | None = None) -> list[GoldItem]:
    """
    ממיר את פריטי gold_set.py (data/eval_queries.json) לצורה שהערכת האחזור
    צריכה. items=None טוען את הקובץ הקיים בעצמו.
    """
    if items is None:
        items = gold_set.load_items()
    return [GoldItem(query=it["query"], relevant=list(it.get("relevant_sections") or [])) for it in items]


# ── אחזור דליל: BM25 על הצ'אנקים ב-Neon ────────────────────────────────


def strip_prefix(token: str) -> str:
    if len(token) > MIN_STEM_LENGTH and token[0] in HEBREW_PREFIXES:
        return token[1:]
    return token


def tokenize(text: str) -> list[str]:
    return [strip_prefix(token) for token in TOKEN_RE.findall(text or "")]


def load_corpus(settings: Settings | None = None) -> list[dict[str, Any]]:
    """קורא את כל הצ'אנקים מ-takanon_sections — אותה טבלה שהפרויקט כולו משתמש בה."""
    return db.get_all_chunks(settings)


def build_bm25(corpus: list[dict[str, Any]]) -> Any:
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(row["content"]) for row in corpus])


def bm25_search(bm25: Any, corpus: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    """מחזירה את הצ'אנקים בעלי הציון הגבוה ביותר, הטוב ביותר קודם."""
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    results = []
    for position in order[:limit]:
        row = corpus[position]
        results.append(
            {
                "section_number": row["section_number"],
                "chapter": row.get("chapter"),
                "page": row.get("page"),
                "content": row.get("content") or "",
                "score": float(scores[position]),
            }
        )
    return results


# ── אחזור צפוף: Pinecone ────────────────────────────────────────────────


def embed_query(client: Any, model: str, dimension: int, query: str) -> list[float]:
    """
    מטמיעה שאלה אחת, עם ניסיון חוזר בכשלים זמניים.

    task_type='RETRIEVAL_QUERY' — שונה מהטמעת מסמכים (add_to_pinecone.py
    משתמש ב-RETRIEVAL_DOCUMENT), כי המודל ממקם שאלות וקטעי טקסט אחרת.
    לוגיקת הניסיון החוזר משותפת עם add_to_pinecone.py, לא משוכפלת.
    """
    from google.genai import types

    last_error: Exception | None = None
    for attempt in range(1, atp.MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=[query],
                config=types.EmbedContentConfig(
                    output_dimensionality=dimension,
                    task_type=QUERY_TASK_TYPE,
                ),
            )
            vector = list(response.embeddings[0].values)
            if len(vector) != dimension:
                raise RuntimeError(
                    f"המודל החזיר {len(vector)} מימדים, האינדקס צריך {dimension}"
                )
            return atp.normalise(vector)
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt < atp.MAX_RETRIES:
                time.sleep(atp.retry_wait(err, attempt))

    raise RuntimeError(f"הטמעת השאלה נכשלה: {atp.describe_error(last_error)}")


def dense_search(embedding: list[float], limit: int) -> list[dict[str, Any]]:
    """
    שואלת את אינדקס Pinecone הקיים ומחזירה פגיעות ברמת הצ'אנק.

    section_number נלקח מה-metadata (לא ממזהה הווקטור, שהוא "section-{id}"
    ולא מספר הסעיף עצמו) — כפי ש-add_to_pinecone.py שמר אותו.
    """
    index = vectors.get_index()
    result = index.query(
        vector=embedding, top_k=limit, include_metadata=True, namespace=vectors.NAMESPACE
    )
    results = []
    for match in result.matches:
        meta = dict(match.metadata or {})
        results.append(
            {
                "section_number": str(meta.get("section_number") or ""),
                "chapter": meta.get("chapter_title"),
                "page": meta.get("source_page"),
                "content": meta.get("text_preview") or "",
                "score": float(match.score),
            }
        )
    return results


# ── ניקוד ─────────────────────────────────────────────────────────────


def collapse_to_sections(hits: list[dict[str, Any]], k: int) -> tuple[list[str], list[dict[str, Any]]]:
    """מצמצמת פגיעות צ'אנק לסעיפים ייחודיים, שומרת את הדירוג הטוב ביותר לכל סעיף."""
    ranked: list[str] = []
    representative: list[dict[str, Any]] = []
    for hit in hits:
        section = str(hit.get("section_number") or "")
        if not section or section in ranked:
            continue
        ranked.append(section)
        representative.append(hit)
        if len(ranked) >= k:
            break
    return ranked, representative


def evaluate_one(item: GoldItem, retriever: str, model: str, hits: list[dict[str, Any]], k: int) -> QueryResult:
    ranked, representative = collapse_to_sections(hits, k)
    relevant = set(item.relevant)
    detail = [
        {
            "rank": position,
            "section_number": hit["section_number"],
            "relevant": hit["section_number"] in relevant,
            "score": round(float(hit.get("score") or 0.0), 4),
            "chapter": hit.get("chapter") or "",
            "page": hit.get("page"),
            "preview": (hit.get("content") or "")[:160],
        }
        for position, hit in enumerate(representative, start=1)
    ]
    return QueryResult(
        query=item.query, retriever=retriever, model=model, k=k,
        ranked=ranked, relevant=item.relevant, hits=detail,
    )


def aggregate(results: list[QueryResult]) -> dict[str, float]:
    if not results:
        return {"mrr": 0.0, "recall": 0.0, "queries": 0, "found": 0}
    return {
        "mrr": sum(r.reciprocal_rank for r in results) / len(results),
        "recall": sum(r.recall for r in results) / len(results),
        "queries": len(results),
        "found": sum(1 for r in results if r.first_relevant_rank),
    }


def csv_rows(results: list[QueryResult]) -> list[dict[str, Any]]:
    """שורה אחת לכל (שאלה, שיטה, דירוג) — לייצוא ה-CSV המפורט."""
    rows: list[dict[str, Any]] = []
    for r in results:
        if not r.hits:
            rows.append(
                {
                    "query": r.query, "method": r.retriever, "model": r.model,
                    "rank": "", "retrieved_section": "", "relevant": "",
                    "score": "", "recall_at_k": round(r.recall, 4),
                    "reciprocal_rank": round(r.reciprocal_rank, 4),
                }
            )
            continue
        for hit in r.hits:
            rows.append(
                {
                    "query": r.query, "method": r.retriever, "model": r.model,
                    "rank": hit["rank"], "retrieved_section": hit["section_number"],
                    "relevant": hit["relevant"], "score": hit["score"],
                    "recall_at_k": round(r.recall, 4),
                    "reciprocal_rank": round(r.reciprocal_rank, 4),
                }
            )
    return rows


def run_retriever(
    items: list[GoldItem],
    retriever: str,
    model: str,
    search: Callable[[str, int], list[dict[str, Any]]],
    k: int,
    progress: Callable[[float, str], None] | None = None,
) -> list[QueryResult]:
    """
    מריצה שיטת אחזור אחת על כל שאלות סט הזהב.

    שאלה שנכשלת (סירוב מכסה, נפילת רשת) מנוקדת כפספוס במקום להפיל את
    כל הריצה — כישלון גלוי בפירוט לפי שאלה, השוואה חלקית עדיין שימושית.
    """
    results: list[QueryResult] = []
    for position, item in enumerate(items, start=1):
        if progress:
            progress(position / max(len(items), 1), f"{retriever}: {item.query[:40]}")
        try:
            hits = search(item.query, k * OVERFETCH)
        except Exception as err:  # noqa: BLE001
            results.append(
                QueryResult(
                    query=item.query, retriever=retriever,
                    model=f"{model} (שגיאה: {str(err)[:60]})",
                    k=k, ranked=[], relevant=item.relevant,
                )
            )
            continue
        results.append(evaluate_one(item, retriever, model, hits, k))
    return results


def missing_sections(items: list[GoldItem], settings: Settings | None = None) -> list[str]:
    """סעיפי זהב שלא קיימים בטבלה כלל — שאלה שמצביעה עליהם לעולם לא תיענה."""
    wanted = {section for item in items for section in item.relevant}
    if not wanted:
        return []
    sql = f"SELECT DISTINCT section_number FROM {db.TABLE} WHERE section_number = ANY(%s);"
    with db.connect(settings) as conn:
        present = {row["section_number"] for row in conn.execute(sql, (list(wanted),))}
    return sorted(wanted - present)
