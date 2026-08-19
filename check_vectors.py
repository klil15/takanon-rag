"""
בדיקת תקינות לאינדקס הווקטורים בענן (Pinecone).

מה הסקריפט הזה עושה, שלב אחר שלב:
  1. טוען את ארבעת המפתחות מקובץ ה-.env (בלי להדפיס ערכים).
  2. מתחבר ל-Pinecone ויוצר את האינדקס אם צריך.
  3. סופר כמה וקטורים יש כרגע.
  4. שומר וקטור דמה אחד — וסופר שוב.
  5. קורא את וקטור הדמה חזרה ומשווה אותו למה שנשלח.
  6. מחפש עם אותו וקטור — הוא אמור למצוא את עצמו.
  7. מוחק אותו — וסופר שוב.

יצירת אינדקס serverless בפעם הראשונה יכולה לקחת עד דקה; הסקריפט ממתין לכך.
הרצות הבאות משתמשות באינדקס הקיים והן מהירות.

הרצה:  python check_vectors.py
"""

from __future__ import annotations

import math
import sys
import time

# ב-Windows ה-console ברירת המחדל (cp1252) לא יודע להדפיס עברית — עוברים ל-UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import vectors
from config import REQUIRED_KEYS, load_settings_or_exit

# וקטור הדמה — מזהה שלא יתנגש עם מספרי סעיפים אמיתיים
TEST_ID = "TEST-VEC-000"
TEST_CHAPTER = "פרק בדיקה"
TEST_TEXT = "זהו וקטור דמה זמני שנוצר על ידי check_vectors.py. הוא נמחק בסוף הבדיקה."

# Pinecone שומר ערכים כ-32-bit float, כך שקריאה חוזרת לא תהיה זהה עד הסיבית
FLOAT_TOLERANCE = 1e-6

# המונים מתעדכנים בעיכוב (eventually consistent) — כמה זמן לחכות שיתעדכנו
SETTLE_TIMEOUT = 30.0
SETTLE_INTERVAL = 1.0

LINE = "─" * 60


def step(number: int, title: str) -> None:
    print(f"\n{LINE}\nשלב {number}: {title}\n{LINE}")


def dummy_vector(dimension: int) -> list[float]:
    """
    בונה וקטור קבוע ובר-שחזור באורך יחידה.

    לא אקראי בכוונה: כל הרצה שולחת בדיוק את אותם ערכים, כך שאי-התאמה
    בקריאה חוזרת מעידה על בעיה אמיתית, לא על הגרלה חדשה.
    וקטורים אמיתיים יגיעו ממודל ה-embedding בשיעור הבא.
    """
    raw = [math.sin(i / 10.0) for i in range(dimension)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def show_count(label: str) -> int:
    n = vectors.count_vectors()
    print(f"  📊 {label}: {n} וקטורים באינדקס.")
    return n


def wait_for_count(expected: int) -> int:
    """
    מחכה שמונה הווקטורים יגיע למספר הצפוי.

    Pinecone מעדכן את המונה הזה זמן קצר אחרי הכתיבה עצמה, כך שקריאה מיידית
    שלו יכולה עדיין להראות את הערך הישן. מחזירה את הערך האחרון שנצפה,
    בין אם הגיע לצפוי ובין אם לא.
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT
    n = vectors.count_vectors()
    while n != expected and time.monotonic() < deadline:
        time.sleep(SETTLE_INTERVAL)
        n = vectors.count_vectors()
    return n


def wait_for_vector(should_exist: bool) -> dict | None:
    """
    מחכה עד שווקטור הדמה מופיע באינדקס, או נעלם ממנו.

    Pinecone הוא eventually consistent: כתיבה או מחיקה מתקבלות מיד, אבל
    קריאה שבריר שנייה אחר כך עדיין יכולה להראות את המצב הקודם. קריאה בודדת
    מיד אחרי השינוי היא מירוץ — היא בדרך כלל עוברת ולפעמים מדווחת את ההפך.

    מחזירה את התוצאה האחרונה שנצפתה, בין אם הגיעה למצב הרצוי ובין אם לא,
    כדי שהקוד הקורא יוכל להחליט שטיימאאוט אמיתי הוא כשל של ממש.
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT
    found = vectors.fetch_section_vector(TEST_ID)
    while (found is not None) != should_exist and time.monotonic() < deadline:
        time.sleep(SETTLE_INTERVAL)
        found = vectors.fetch_section_vector(TEST_ID)
    return found


def wait_for_search_hit(embedding: list[float]) -> list[dict]:
    """
    מחכה עד שהחיפוש מוצא את וקטור הדמה.

    לחיפוש יש עיכוב משלו, נפרד מזה של הקריאה: וקטור יכול כבר להיות קריא
    לפי מזהה בעוד שצד החיפוש עדיין לא קלט אותו. אז נראות בשלב 5 לא מבטיחה
    שהוא ניתן למציאה כאן.
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while True:
        matches = vectors.query_similar(embedding, top_k=3)
        if any(match["section_number"] == TEST_ID for match in matches):
            return matches
        if time.monotonic() >= deadline:
            return matches
        time.sleep(SETTLE_INTERVAL)


def main() -> int:
    print(f"{LINE}\nבדיקת אינדקס הווקטורים בענן — takanon-rag\n{LINE}")

    # ── שלב 1: מפתחות ────────────────────────────────────────────
    step(1, "טעינת המפתחות מקובץ ה-.env")
    settings = load_settings_or_exit()
    for key in REQUIRED_KEYS:
        print(f"  ✓ {key} — נטען בהצלחה (הערך מוסתר)")

    # מעבירים את ההגדרות שנטענו למודול vectors, כדי שכל קריאה למטה
    # תשתמש באותו לקוח Pinecone במקום ליצור חדש בכל פעם
    vectors.use_settings(settings)

    # ── שלב 2: התחברות ויצירת האינדקס ──────────────────────────
    step(2, "התחברות ל-Pinecone ויצירת האינדקס אם אינו קיים")
    name = vectors.index_name()
    print(f"  🌐 שם האינדקס: '{name}'  (מתוך PINECONE_INDEX)")
    try:
        if vectors.index_exists():
            print("  · האינדקס כבר קיים — משתמשים בו.")
        else:
            print(f"  · האינדקס לא קיים — יוצרים אותו "
                  f"({vectors.CLOUD}/{vectors.REGION}, serverless). זה יכול לקחת עד דקה...")
        described = vectors.ensure_index()
    except Exception as err:  # noqa: BLE001 — רוצים הודעת שגיאה ברורה למשתמש
        print("\n  ❌ החיבור ל-Pinecone נכשל.", file=sys.stderr)
        print(f"     פרטי השגיאה: {err}", file=sys.stderr)
        print(
            "     ודאו ש-PINECONE_API_KEY ו-PINECONE_INDEX בקובץ ה-.env נכונים,\n"
            "     ושם האינדקס תקין (אותיות קטנות באנגלית, ספרות ומקפים בלבד).",
            file=sys.stderr,
        )
        return 1

    print(f"  ✓ האינדקס מוכן.")
    print(f"     מימד  : {described.dimension}")
    print(f"     מטריקה: {described.metric}")
    print(f"     Host  : {described.host}")

    if described.dimension != vectors.DIMENSION:
        print(
            f"  ❌ אי-התאמה במימד: האינדקס הוא {described.dimension}, "
            f"הקוד מצפה ל-{vectors.DIMENSION}.",
            file=sys.stderr,
        )
        return 1

    # ── שלב 3: ספירה לפני ────────────────────────────────────────
    step(3, "ספירה לפני ההוספה")
    before = show_count("לפני")

    # ── שלב 4: שמירת וקטור הדמה ──────────────────────────────────
    step(4, "שמירת וקטור דמה אחד")
    sent = dummy_vector(vectors.DIMENSION)
    written = vectors.upsert_section_vector(
        TEST_ID, sent, TEST_CHAPTER, TEST_TEXT
    )
    print(f"  ✓ נכתב {written} וקטור עם מזהה '{TEST_ID}' ({len(sent)} ערכים).")
    print(f"     4 הערכים הראשונים שנשלחו: {[round(v, 6) for v in sent[:4]]}")
    after_insert = wait_for_count(before + 1)
    print(f"  📊 אחרי ההוספה: {after_insert} וקטורים באינדקס.")
    if after_insert != before + 1:
        print("  ⚠ המונה עדיין לא התעדכן — Pinecone מעדכן אותו בעיכוב.")
        print("     זה כשלעצמו אינו כשל; הקריאה למטה היא זו שמוכיחה את הכתיבה.")

    # ── שלב 5: קריאה חזרה ────────────────────────────────────────
    step(5, "קריאת וקטור הדמה חזרה מהאינדקס")
    stored = wait_for_vector(should_exist=True)
    if stored is None:
        print(
            f"  ❌ הווקטור לא נמצא {SETTLE_TIMEOUT:.0f} שניות אחרי ההוספה — "
            "משהו השתבש.",
            file=sys.stderr,
        )
        return 1
    print("  ✓ הווקטור נקרא בהצלחה:")
    print(f"     מזהה     : {stored['section_number']}")
    print(f"     ערכים    : {len(stored['values'])} מספרים")
    print(f"     מטא-דאטה : {stored['metadata']}")

    if len(stored["values"]) != len(sent):
        print(
            f"  ❌ אורך שגוי: נשלחו {len(sent)} ערכים, התקבלו חזרה "
            f"{len(stored['values'])}.",
            file=sys.stderr,
        )
        return 1

    worst = max(abs(a - b) for a, b in zip(sent, stored["values"]))
    if worst > FLOAT_TOLERANCE:
        print(
            f"  ❌ הערכים שחזרו שונים מאלה שנשלחו "
            f"(ההפרש הגדול ביותר {worst}).",
            file=sys.stderr,
        )
        return 1
    print(f"  ✓ הערכים תואמים למה שנשלח (ההפרש הגדול ביותר {worst:.2e},")
    print("    שהוא עיגול רגיל של 32-bit float, לא שינוי אמיתי).")

    # ── שלב 6: חיפוש עם אותו וקטור ────────────────────────────────
    step(6, "חיפוש עם אותו וקטור — הוא אמור למצוא את עצמו")
    matches = wait_for_search_hit(sent)
    if not matches:
        print(
            f"  ❌ החיפוש עדיין לא החזיר כלום {SETTLE_TIMEOUT:.0f} שניות "
            "אחרי ההוספה.",
            file=sys.stderr,
        )
        return 1
    for position, match in enumerate(matches, start=1):
        marker = "←" if match["section_number"] == TEST_ID else " "
        print(f"     {position}. {match['section_number']:<16} ציון {match['score']:.6f} {marker}")

    top = matches[0]
    if top["section_number"] != TEST_ID:
        print(
            f"  ❌ ההתאמה הקרובה ביותר היא '{top['section_number']}', ציפינו ל-'{TEST_ID}'.",
            file=sys.stderr,
        )
        return 1
    print(f"  ✓ הוא מצא את עצמו כהתאמה הקרובה ביותר, ציון {top['score']:.6f}")
    print("    (במטריקת cosine, וקטור מול עצמו מקבל ציון סביב 1.0).")

    # ── שלב 7: מחיקה ──────────────────────────────────────────────
    step(7, "מחיקת וקטור הדמה")
    vectors.delete_section_vector(TEST_ID)
    still_there = wait_for_vector(should_exist=False)
    if still_there is not None:
        print(
            f"  ❌ הווקטור עדיין באינדקס {SETTLE_TIMEOUT:.0f} שניות אחרי המחיקה — "
            "זה יותר ממה שעיכוב עקביות רגיל מסביר.",
            file=sys.stderr,
        )
        return 1
    print("  ✓ הווקטור נמחק.")
    after_delete = wait_for_count(before)
    print(f"  📊 אחרי המחיקה: {after_delete} וקטורים באינדקס.")

    # ── סיכום ────────────────────────────────────────────────────
    print(f"\n{LINE}\nסיכום\n{LINE}")
    print(f"  אינדקס        : {name} ({described.dimension} מימדים, {described.metric})")
    print(f"  לפני ההוספה   : {before}")
    print(f"  אחרי ההוספה   : {after_insert}")
    print(f"  אחרי המחיקה   : {after_delete}")

    print("\n✅ הבדיקה עברה בהצלחה: יצירה, כתיבה, קריאה, חיפוש ומחיקה")
    print("   באינדקס הווקטורים בענן — הכול עובד.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
