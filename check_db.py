"""
בדיקת תקינות למסד הנתונים בענן (Neon).

מה הסקריפט הזה עושה, שלב אחר שלב:
  1. טוען את ארבעת המפתחות מקובץ ה-.env (בלי להדפיס ערכים).
  2. מתחבר ל-Neon ויוצר את הטבלה אם צריך.
  3. סופר כמה סעיפים יש כרגע.
  4. מוסיף סעיף בדיקה אחד — וסופר שוב.
  5. קורא את סעיף הבדיקה חזרה ומציג אותו.
  6. מעדכן אותו כדי להראות שמספר הגרסה עולה.
  7. מוחק אותו — וסופר שוב.

הרצה:  python check_db.py
"""

from __future__ import annotations

import sys

# ב-Windows ה-console ברירת המחדל (cp1252) לא יודע להדפיס עברית — עוברים ל-UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import db
from config import REQUIRED_KEYS, load_settings_or_exit

# סעיף הבדיקה — מספר שלא יתנגש עם סעיפי תקנון אמיתיים
TEST_NUMBER = "TEST-000"
TEST_CHAPTER = "פרק בדיקה"
TEST_TEXT = "זהו סעיף בדיקה זמני שנוצר על ידי check_db.py. הוא נמחק בסוף הבדיקה."
TEST_TEXT_UPDATED = "זהו סעיף הבדיקה לאחר עדכון — מספר הגרסה אמור לעלות."

LINE = "─" * 60


def step(number: int, title: str) -> None:
    print(f"\n{LINE}\nשלב {number}: {title}\n{LINE}")


def show_count(label: str) -> int:
    n = db.count_sections()
    print(f"  📊 {label}: {n} סעיפים במסד הנתונים.")
    return n


def show_section(row: dict) -> None:
    print(f"     מספר סעיף  : {row['section_number']}")
    print(f"     פרק        : {row['chapter']}")
    print(f"     גרסה       : {row['version']}")
    print(f"     עודכן ב-   : {row['updated_at']}")
    print(f"     תוכן       : {row['content']}")


def main() -> int:
    print(f"{LINE}\nבדיקת חיבור למסד הנתונים בענן — takanon-rag\n{LINE}")

    # ── שלב 1: מפתחות ────────────────────────────────────────────
    step(1, "טעינת המפתחות מקובץ ה-.env")
    settings = load_settings_or_exit()
    for key in REQUIRED_KEYS:
        print(f"  ✓ {key} — נטען בהצלחה (הערך מוסתר)")

    # ── שלב 2: התחברות ויצירת הטבלה ────────────────────────────
    step(2, "התחברות ל-Neon ויצירת הטבלה אם אינה קיימת")
    print(f"  🌐 מתחבר אל: {settings.db_target}  (חיבור מוצפן, sslmode=require)")
    try:
        db.init_db(settings)
        version_text = db.server_version(settings).split(",")[0]
    except Exception as err:  # noqa: BLE001 — רוצים הודעת שגיאה ברורה למשתמש
        print("\n  ❌ החיבור למסד הנתונים נכשל.", file=sys.stderr)
        print(f"     פרטי השגיאה: {err}", file=sys.stderr)
        print(
            "     ודאו ש-DATABASE_URL בקובץ ה-.env נכון ומסתיים ב-?sslmode=require,\n"
            "     ושהפרויקט ב-Neon פעיל.",
            file=sys.stderr,
        )
        return 1
    print(f"  ✓ החיבור הצליח. שרת: {version_text}")
    print(f"  ✓ הטבלה '{db.TABLE}' מוכנה.")

    # ── שלב 3: ספירה לפני ────────────────────────────────────────
    step(3, "ספירה לפני ההוספה")
    before = show_count("לפני")

    # ── שלב 4: הוספת סעיף בדיקה ──────────────────────────────────
    step(4, "הוספת סעיף בדיקה אחד")
    saved = db.upsert_section(TEST_NUMBER, TEST_CHAPTER, TEST_TEXT, settings)
    print(f"  ✓ נשמר סעיף '{saved['section_number']}' בגרסה {saved['version']}.")
    after_insert = show_count("אחרי ההוספה")

    # ── שלב 5: קריאה חזרה ────────────────────────────────────────
    step(5, "קריאת הסעיף חזרה ממסד הנתונים")
    row = db.get_section(TEST_NUMBER, settings)
    if row is None:
        print("  ❌ הסעיף לא נמצא אחרי ההוספה — משהו השתבש.", file=sys.stderr)
        return 1
    print("  ✓ הסעיף נקרא בהצלחה:")
    show_section(row)
    if row["content"] != TEST_TEXT:
        print("  ❌ התוכן שנקרא שונה מהתוכן שנשמר.", file=sys.stderr)
        return 1
    print("  ✓ התוכן שנקרא זהה לתוכן שנשמר.")
    show_count("ללא שינוי")

    # ── שלב 6: עדכון והעלאת הגרסה ────────────────────────────────
    step(6, "עדכון אותו סעיף — מספר הגרסה אמור לעלות")
    updated = db.upsert_section(TEST_NUMBER, TEST_CHAPTER, TEST_TEXT_UPDATED, settings)
    print(f"  ✓ הגרסה עלתה מ-{row['version']} ל-{updated['version']}.")
    show_section(updated)
    show_count("אחרי העדכון (לא אמור להשתנות)")

    # ── שלב 7: מחיקה ──────────────────────────────────────────────
    step(7, "מחיקת סעיף הבדיקה")
    deleted = db.delete_section(TEST_NUMBER, settings)
    print("  ✓ הסעיף נמחק." if deleted else "  ⚠ לא נמחק דבר (הסעיף כבר לא היה קיים).")
    after_delete = show_count("אחרי המחיקה")

    # ── סיכום ────────────────────────────────────────────────────
    print(f"\n{LINE}\nסיכום\n{LINE}")
    print(f"  לפני ההוספה  : {before}")
    print(f"  אחרי ההוספה  : {after_insert}")
    print(f"  אחרי המחיקה  : {after_delete}")

    if after_insert == before + 1 and after_delete == before:
        print("\n✅ הבדיקה עברה בהצלחה: קריאה, כתיבה ומחיקה במסד הנתונים בענן — הכול עובד.")
        return 0

    print("\n❌ הבדיקה נכשלה: הספירות אינן מסתדרות.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
