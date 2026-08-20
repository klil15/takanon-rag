"""
מטמיע (embeds) את הצ'אנקים השמורים ב-Neon ומעדכן אותם באינדקס הקיים של
Pinecone, כדי שהאחזור יוכל לעבוד על אותם נתונים.

Neon הוא מקור האמת לטקסט. Pinecone מחזיק וקטור אחד לכל צ'אנק, עם מזהה
דטרמיניסטי "section-{id}" — המזהה (id) של השורה בטבלת takanon_sections —
כך שתוצאת חיפוש ממופה ישירות בחזרה לשורה שממנה היא הגיעה.

הצינור:
    1. קריאת DATABASE_URL, PINECONE_API_KEY, PINECONE_INDEX ו-EMBEDDING_MODEL
       מקובץ ה-.env (EMBEDDING_MODEL אופציונלי וברירת המחדל שלו תואמת את
       המימד שבו האינדקס כבר נוצר).
    2. שאילת Neon לצ'אנקים שעדיין לא הוטמעו (או שהשתנו מאז ההטמעה האחרונה).
    3. הטמעת כל אחד עם מודל ה-embedding של Google, במימד שבו האינדקס נוצר.
    4. כתיבה ל-Pinecone במנות (batches) של 100.
    5. סימון השורות כמוטמעות, כך שהרצה חוזרת ממשיכה במקום להתחיל מחדש.

על האינדקס: הסקריפט הזה אף פעם לא יוצר אינדקס. הוא מתחבר ל-PINECONE_INDEX,
קורא את המימד שבו האינדקס נבנה בפועל, ומבקש ממודל ה-embedding בדיוק אותו
מספר מימדים. אם אי אפשר ליישב בין השניים, הוא עוצר במקום לכתוב וקטורים
בצורה הלא נכונה.

על עצמאות מהרצה חוזרת (idempotence): מזהה הווקטור נגזר מה-id של השורה
ב-Postgres, כך שהטמעה חוזרת של אותו צ'אנק דורסת ולא משכפלת. סימון שורות
כמוטמעות אומר שהרצה רגילה חוזרת לא עושה כלום; --all מטמיע הכול מחדש,
ועדיין מעדכן במקום ליצור כפילויות.

הרצה:      python add_to_pinecone.py
            python add_to_pinecone.py --dry-run     (בלי הטמעה ובלי כתיבה בכלל)
            python add_to_pinecone.py --all         (הטמעה מחדש של הכול)
            python add_to_pinecone.py --limit 3     (רק 3 השורות הראשונות)
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import db
import vectors
from config import Settings, load_settings_or_exit

# מנת הכתיבה ל-Pinecone, כפי שצוין
UPSERT_BATCH = 100

# ה-API של ה-embedding מקבל כמה טקסטים בכל קריאה. נשמר קטן משמעותית ממנת
# הכתיבה ל-Pinecone, כי בקשה גדולה מדי נכשלת כמכלול, ונסיון חוזר לאחר כשל
# קטן עולה פחות.
EMBED_BATCH = 32

# task type בזמן אינדוקס. שאילתות בהמשך צריכות להיות מוטמעות עם
# RETRIEVAL_QUERY במקום — המודל ממקם שאלות וקטעי טקסט אחרת.
TASK_TYPE = "RETRIEVAL_DOCUMENT"

MAX_RETRIES = 4
RETRY_BACKOFF = 3.0
SLEEP_BETWEEN_BATCHES = 0.2

SOURCE_NAME = "neon"
# תצוגה מקדימה קצרה בלבד ב-metadata — לא מקור האמת לטקסט המלא (זה ב-Neon),
# ונשמרת קצרה כדי להישאר בנוחות מתחת למגבלת ה-metadata של Pinecone.
PREVIEW_CHARS = 300

LOG = logging.getLogger("add_to_pinecone")


@dataclass
class Counters:
    """כל מה שהדוח הסופי מציג."""

    read: int = 0
    valid: int = 0
    upserted: int = 0
    marked: int = 0
    skipped: int = 0
    failed: int = 0


# ── האינדקס הקיים ────────────────────────────────────────────────────


def index_dimension(settings: Settings) -> int:
    """
    מחזירה את המימד של האינדקס הקיים, או זורקת שגיאה ברורה.

    האינדקס אף פעם לא נוצר כאן. אינדקס חסר היא בעיית הקמה שנפתרת עם
    check_vectors.py, לא משהו שמוסתר על ידי יצירה שקטה עם הגדרות מנוחשות.
    """
    vectors.use_settings(settings)
    name = vectors.index_name()
    if not vectors.index_exists():
        raise RuntimeError(
            f"אינדקס Pinecone בשם '{name}' אינו קיים. הסקריפט הזה כותב רק "
            "לאינדקס קיים — צרו אותו קודם עם: python check_vectors.py"
        )
    described = vectors.describe_index()
    return int(described.dimension)


# ── המודל ─────────────────────────────────────────────────────────────


def build_client(settings: Settings) -> Any:
    """יוצרת את לקוח Google לשימוש ב-embeddings."""
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def normalise(values: Sequence[float]) -> list[float]:
    """
    מנרמלת וקטור לאורך יחידה.

    הכרחי, לא קוסמטי. המודל הזה מחזיר 3072 מימדים כשהם כבר מנורמלים, אבל
    בקשת פחות מימדים חותכת (truncates) את הווקטור, ווקטור חתוך כבר לא
    באורך יחידה — נמדד 0.58 עבור 768 מימדים. שמירת אורכים מעורבים הופכת
    את הגדלים השמורים לחסרי משמעות ומשבשת את האינדקס ברגע שמשהו משווה
    לפי מכפלה סקלרית (dot product) במקום cosine.
    """
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return list(values)
    return [value / norm for value in values]


def describe_error(err: Exception) -> str:
    """שורה קצרה אחת במקום בלוק ה-JSON הרב-שורות של הספק."""
    text = str(err)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        quota = re.search(r"quotaValue': '(\d+)'", text)
        detail = f", מגבלה {quota.group(1)}" if quota else ""
        return f"המכסה נגמרה{detail}"
    if "404" in text and "NOT_FOUND" in text:
        return "המודל לא נמצא — בדקו את EMBEDDING_MODEL"
    if "PERMISSION_DENIED" in text or "401" in text:
        return "המפתח נדחה — בדקו את GEMINI_API_KEY"
    return text[:120]


def retry_wait(err: Exception, attempt: int) -> float:
    """
    כמה זמן לחכות לפני ניסיון נוסף.

    תשובת rate limit קובעת כמה לחכות, והמספר הזה שווה יותר מכל ניחוש
    מקומי. מספר השרת מנצח כשהוא ניתן, עם חזרה אקספוננציאלית פשוטה כשלא.
    """
    stated = re.search(r"retryDelay': '(\d+(?:\.\d+)?)s'", str(err))
    if stated:
        return min(float(stated.group(1)) + 2.0, 90.0)
    return RETRY_BACKOFF * attempt


def embed_texts(client: Any, model: str, texts: list[str], dimension: int) -> list[list[float]]:
    """
    מטמיעה מנת טקסטים, עם ניסיון חוזר בכשלים זמניים ב-API.

    זורקת אם המנה לא הצליחה אחרי MAX_RETRIES, כך שהקוד הקורא יכול לדלג על
    המנה הזו ולהמשיך, בלי לסמן אותה כמוטמעת.
    """
    from google.genai import types

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=dimension,
                    task_type=TASK_TYPE,
                ),
            )
            produced = [list(item.values) for item in response.embeddings]
            if len(produced) != len(texts):
                raise RuntimeError(
                    f"התבקשו {len(texts)} embeddings, התקבלו {len(produced)}"
                )
            for vector in produced:
                if len(vector) != dimension:
                    raise RuntimeError(
                        f"המודל החזיר {len(vector)} מימדים, האינדקס צריך {dimension}"
                    )
            return [normalise(vector) for vector in produced]
        except Exception as err:  # noqa: BLE001 — שגיאות מכסה ורשת נופלות לכאן
            last_error = err
            if attempt < MAX_RETRIES:
                wait = retry_wait(err, attempt)
                LOG.warning(
                    "הטמעת מנה נכשלה (ניסיון %d/%d): %s — ניסיון חוזר בעוד %.0f שניות",
                    attempt, MAX_RETRIES, describe_error(err), wait,
                )
                time.sleep(wait)

    raise RuntimeError(f"ההטמעה נכשלה אחרי {MAX_RETRIES} ניסיונות: {describe_error(last_error)}")


# ── כתיבה ל-Pinecone ─────────────────────────────────────────────────


def build_vector(row: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
    """
    בונה רשומת Pinecone אחת.

    המזהה מגיע מהמפתח הראשי (id) ב-Postgres, וזה מה שהופך הרצה חוזרת
    לעדכון: אותה שורה תמיד מייצרת את אותו מזהה.
    """
    metadata: dict[str, Any] = {
        "neon_row_id": int(row["id"]),
        "section_number": row.get("section_number") or "",
        "chapter_title": row.get("chapter") or "",
        "chunk_index": int(row.get("chunk_index") or 0),
        "source": SOURCE_NAME,
    }
    # Pinecone דוחה ערכי None, אז עמוד לא ידוע פשוט לא נכלל
    if row.get("page") is not None:
        metadata["source_page"] = int(row["page"])

    content = str(row.get("content") or "")
    if content:
        metadata["text_preview"] = content[:PREVIEW_CHARS]

    return {
        "id": f"section-{row['id']}",
        "values": embedding,
        "metadata": metadata,
    }


def upsert_batch(records: list[dict[str, Any]]) -> int:
    """כותבת מנה אחת, עם ניסיון חוזר בכשלים זמניים. מחזירה כמה וקטורים נכתבו."""
    if not records:
        return 0

    index = vectors.get_index()
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = index.upsert(vectors=records, namespace=vectors.NAMESPACE)
            if getattr(result, "has_errors", False):
                raise RuntimeError(f"Pinecone דיווח על שגיאות: {result.errors}")
            return len(records)
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt < MAX_RETRIES:
                wait = retry_wait(err, attempt)
                LOG.warning(
                    "כתיבה ל-Pinecone נכשלה (ניסיון %d/%d): %s — ניסיון חוזר בעוד %.0f שניות",
                    attempt, MAX_RETRIES, describe_error(err), wait,
                )
                time.sleep(wait)

    raise RuntimeError(f"הכתיבה נכשלה אחרי {MAX_RETRIES} ניסיונות: {describe_error(last_error)}")


def batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    """מייצרת פרוסות רצופות של עד `size` פריטים."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def flush(records: list[dict[str, Any]], ids: list[int], settings: Settings, counters: Counters) -> int:
    """
    כותבת מנה אחת ומסמנת את השורות שלה, בסדר הזה.

    מנה שנכשלה נספרת ככישלון והשורות שלה נשארות לא מסומנות, כך שההרצה
    הבאה תנסה אותן שוב במקום להשאיר צ'אנק ש-Postgres חושב שהוא ניתן
    לחיפוש אבל מעולם לא נכתב בפועל.
    """
    try:
        written = upsert_batch(records)
    except Exception as err:  # noqa: BLE001
        counters.failed += len(records)
        LOG.error("כתיבת %d וקטורים נכשלה: %s", len(records), describe_error(err))
        return 0

    try:
        db.mark_embedded(ids, settings)
        counters.marked += len(ids)
    except Exception as err:  # noqa: BLE001
        # הווקטורים כבר ב-Pinecone; רק הסימון נכשל. ההרצה הבאה תטמיע מחדש
        # את השורות האלה ותדרוס את אותם מזהים — בזבוז, אבל לא שגיאה.
        LOG.warning("הווקטורים נכתבו אך השורות לא סומנו כמוטמעות: %s", str(err)[:160])
    return written


# ── הרכבת הכול יחד ────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="מטמיע את הצ'אנקים שב-Neon לתוך האינדקס הקיים של Pinecone."
    )
    parser.add_argument(
        "--all", action="store_true", dest="redo_all",
        help="הטמעה מחדש של כל השורות, כולל כאלה שכבר בוצעו (עדיין מעדכן במקום, לא משכפל)",
    )
    parser.add_argument("--limit", type=int, default=0, help="לעבד רק N השורות הראשונות")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="דוח על מה שהיה קורה, בלי הטמעה ובלי כתיבה בכלל",
    )
    parser.add_argument("--model", default="", help="דריסת EMBEDDING_MODEL להרצה הזו")
    parser.add_argument("--verbose", action="store_true", help="רישום כל מנה, לא רק אבני דרך")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for noisy in ("pinecone", "httpx", "httpcore", "google", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    counters = Counters()
    settings = load_settings_or_exit()
    model = args.model.strip() or settings.embedding_model

    LOG.info("מסד נתונים : %s", settings.db_target)
    LOG.info("אינדקס      : %s", settings.pinecone_index)
    LOG.info("מודל        : %s", model)

    # ── האינדקס קובע את המימד, לא הסקריפט הזה ─────────────────────
    try:
        dimension = index_dimension(settings)
    except Exception as err:  # noqa: BLE001
        LOG.error("%s", err)
        return 1
    LOG.info("מימד        : %d (נלקח מהאינדקס הקיים)", dimension)

    try:
        db.init_db(settings)
    except Exception as err:  # noqa: BLE001
        LOG.error("הכנת הטבלה נכשלה: %s", err)
        return 1

    # ── מה צריך לעשות ──────────────────────────────────────────────
    try:
        total_rows = db.count_sections(settings)
        already_embedded = db.count_embedded(settings)
        rows = db.get_chunks_needing_embedding(settings, args.redo_all, args.limit)
    except Exception as err:  # noqa: BLE001
        LOG.error("קריאה מ-Neon נכשלה: %s", err)
        return 1

    counters.read = len(rows)
    outstanding = total_rows - already_embedded
    LOG.info(
        "צ'אנקים     : %d בטבלה, %d כבר מוטמעים, %d ממתינים",
        total_rows, already_embedded, outstanding,
    )
    if args.limit and counters.read < outstanding:
        LOG.warning("--limit פעיל: מטפל ב-%d מתוך %d הצ'אנקים הממתינים", counters.read, outstanding)

    # שורות בלי טקסט לא ניתנות להטמעה ואסור לסמן אותן כמוטמעות
    usable = [row for row in rows if str(row.get("content") or "").strip()]
    counters.skipped += len(rows) - len(usable)
    if len(rows) != len(usable):
        LOG.warning("ל-%d צ'אנקים אין טקסט — דולגו", len(rows) - len(usable))
    counters.valid = len(usable)

    if not usable:
        LOG.info("אין מה לעשות.")
        print_summary(counters, dimension, model, settings)
        return 0

    if args.dry_run:
        LOG.info("--dry-run: עוצר לפני הטמעה או כתיבה כלשהי.")
        example = build_vector(usable[0], [0.0] * dimension)
        LOG.info("הווקטור הראשון היה: id=%s metadata=%s", example["id"], example["metadata"])
        print_summary(counters, dimension, model, settings)
        return 0

    try:
        client = build_client(settings)
    except Exception as err:  # noqa: BLE001
        LOG.error("יצירת לקוח Google נכשלה: %s", err)
        return 1

    # ── הטמעה, כתיבה, ואז סימון ─────────────────────────────────────
    pending: list[dict[str, Any]] = []
    pending_ids: list[int] = []

    for batch_number, batch in enumerate(batched(usable, EMBED_BATCH), start=1):
        texts = [row["content"] for row in batch]
        try:
            embeddings = embed_texts(client, model, texts, dimension)
        except Exception as err:  # noqa: BLE001
            counters.failed += len(batch)
            LOG.error(
                "מנת הטמעה %d דולגה (%d צ'אנקים): %s",
                batch_number, len(batch), describe_error(err),
            )
            continue

        pending.extend(build_vector(row, vec) for row, vec in zip(batch, embeddings))
        pending_ids.extend(row["id"] for row in batch)
        LOG.debug("הוטמעה מנה %d (%d צ'אנקים)", batch_number, len(batch))

        while len(pending) >= UPSERT_BATCH:
            chunk, pending = pending[:UPSERT_BATCH], pending[UPSERT_BATCH:]
            ids, pending_ids = pending_ids[:UPSERT_BATCH], pending_ids[UPSERT_BATCH:]
            counters.upserted += flush(chunk, ids, settings, counters)
            LOG.info(
                "התקדמות    : %d/%d נכתבו, %d סומנו, %d נכשלו",
                counters.upserted, counters.valid, counters.marked, counters.failed,
            )
        time.sleep(SLEEP_BETWEEN_BATCHES)

    if pending:
        counters.upserted += flush(pending, pending_ids, settings, counters)

    LOG.info("סיום.")
    print_summary(counters, dimension, model, settings)
    return 0 if counters.upserted or not counters.read else 1


def print_summary(counters: Counters, dimension: int, model: str, settings: Settings) -> None:
    """הדוח הסופי."""
    line = "─" * 60
    print(f"\n{line}\nסיכום\n{line}")
    print(f"  שורות שנקראו מ-Neon        : {counters.read}")
    print(f"  שורות תקינות שעובדו        : {counters.valid}")
    print(f"  וקטורים שנכתבו ל-Pinecone  : {counters.upserted}")
    print(f"  שורות שסומנו כמוטמעות     : {counters.marked}")
    print(f"  שורות שדולגו               : {counters.skipped}")
    print(f"  שורות שנכשלו               : {counters.failed}")
    print(f"  אינדקס Pinecone            : {settings.pinecone_index}")
    print(f"  מודל embedding             : {model}")
    print(f"  מימד embedding             : {dimension}")
    if counters.failed or counters.skipped:
        print("\n  שורות שנכשלו או דולגו נשארו לא מסומנות — הריצו שוב כדי לנסות אותן.")


if __name__ == "__main__":
    raise SystemExit(main())
