"""
טוען את קובץ ה-PDF של התקנון לתוך מסד הנתונים בענן (Neon).

צינור העיבוד, לפי הסדר:
    1. קריאת data/takanon.pdf עמוד אחר עמוד (pypdf).
    2. פיצול הטקסט לסעיפים לפי המספור של התקנון עצמו
       (1, 1.1, 2.3.4 ...), תוך שמירת שם הפרק הנוכחי.
    3. פיצול כל סעיף שאורכו מעל כ-800 תווים לצ'אנקים חופפים,
       כשהחיתוך נעשה רק בין משפטים.
    4. שאילת Gemini לקבלת אובייקט JSON לכל צ'אנק: section_number,
       chapter_title, summary. כל תשובה שאינה JSON נקי מדולגת ונרשמת ביומן.
    5. אימות כל צ'אנק, ולאחר מכן כתיבתו (upsert) לטבלת takanon_sections.

הרצה חוזרת בטוחה: המפתח הטבעי הוא (section_number, chunk_index), כך
שהרצה שנייה מעדכנת שורות קיימות במקום ליצור כפילויות.

הערה על הטבלה: הגרסה המקורית של takanon_sections שמרה שורה אחת לכל סעיף
עם UNIQUE (section_number). חלוקה לצ'אנקים משמעה כמה שורות לכל סעיף, כך
שהסכמה ב-db.py מבצעת מיגרציה לטבלה — מוסיפה chunk_index, page ו-summary,
ומעבירה את הייחודיות ל-(section_number, chunk_index). המיגרציה אידמפוטנטית
ובטוחה להרצה חוזרת.

הרצה:      python load_takanon.py
            python load_takanon.py --dry-run          (בלי קריאות ל-LLM ובלי כתיבה)
            python load_takanon.py --limit 5          (רק 5 הצ'אנקים הראשונים)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import db
from config import Settings, load_settings_or_exit

DEFAULT_PDF = Path(__file__).resolve().parent / "data" / "takanon.pdf"
# מאומת מול המפתח של הפרויקט הזה. שני דברים לדעת לפני שמשנים את זה:
#   * גם models.list() מחזירה מודלים שנופלים ב-404 בשימוש בפועל ("לא זמין
#     יותר למשתמשים חדשים"), כך שהופעה ברשימה אינה הוכחה שאפשר לקרוא לו.
#   * מגבלת הבקשות היומית בשכבה החינמית שונה מאוד בין מודל למודל.
#     gemini-3.5-flash מאפשר רק 20 בקשות ביום, לא מספיק לטעינת מסמך בגודל
#     הזה בהרצה אחת; gemini-3.1-flash-lite נדיב הרבה יותר ומספיק ליכולת
#     לסכם סעיף בודד.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# בקשת JSON אינה מבטיחה קבלת JSON. בלי סכמה, המודל לפעמים מחזיר אובייקט
# תקין ואחריו '}' מיותר, מה שאינו ניתן לפענוח ועולה את כל הצ'אנק. העברת
# הסכמה מכריחה את הפלט למבנה מסוים, כך שהתשובה חוזרת נקייה.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "section_number": {"type": "STRING"},
        "chapter_title": {"type": "STRING"},
        "summary": {"type": "STRING"},
    },
    "required": ["section_number", "chapter_title", "summary"],
}

# השפה שבה המודל כותב את הסיכומים. התקנון עצמו בעברית והסיכומים נשמרים
# לחיפוש ולהצגה לקוראים דוברי עברית, לכן הם נשארים בעברית גם אם הקוד
# והפלט שלו באנגלית. לסיכומים באנגלית — לשנות ל-"English".
SUMMARY_LANGUAGE = "Hebrew"

# חלוקה לצ'אנקים. MAX_CHARS הוא יעד, לא חיתוך מוחלט: משפט ארוך ממנו
# נשמר שלם במקום להיחתך באמצע.
MAX_CHARS = 800
OVERLAP_CHARS = 120

# נימוס כלפי השכבה החינמית של ה-API, ומה לעשות כשהיא דוחה בקשה
SLEEP_BETWEEN_CALLS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF = 4.0
# בשימוש רק בניסיון חוזר. הניסיון הראשון רץ בטמפרטורה 0 לשחזוריות;
# ניסיון חוזר דטרמיניסטי היה רק משחזר את אותה תשובה פגומה.
RETRY_TEMPERATURE = 0.3

# ── תבניות כותרות ──────────────────────────────────────────────────
#
# מילות המפתח בעברית נשארות בעברית בכוונה: הן נבדקות מול מסמך המקור
# בעברית, ותרגום שלהן היה מפסיק למצוא התאמות. סעיף ופרק הן מילות המפתח.
#
# כותרת יכולה להגיע בכמה צורות, כי PDF בעברית עשוי להיחלץ בסדר לוגי או
# בסדר חזותי, ומספר הוא LTR בתוך שורה RTL. בקובץ takanon.pdf האמיתי
# הכותרות יוצאות כך:
#
#     '1.1סעיף'          סעיף 1.1 — המספר והמילה צמודים
#     '— כללי1פרק'       פרק 1 — כללי
#
# המספר מוביל את המחרוזת גם אם המילה נקראת קודם, ואין רווח ביניהם.
# חילוצים אחרים מציבים את המילה קודם, או משתמשים במספר בודד עם הטקסט
# באותה שורה. כל ארבע הצורות נתמכות, והצורה הפעילה מזוהה פעם אחת לכל
# מסמך במקום להינחש שורה-שורה — ראו detect_heading_style(). ההחלטה החד-
# פעמית הזו חשובה: הצורות של מספר בודד רופפות מספיק כדי להתאים גם
# למשפטים רגילים, ולכן אסור לנסות אותן על מסמך שבו הצורות עם מילת
# המפתח כבר עובדות.

_SECTION_WORD = r"(?:סעיף|סע׳|סע'|ס׳)"
_CHAPTER_WORD = r"(?:פרק|חלק|שער)"
_NUM = r"\d+(?:\.\d+)*"

# כל רשומה ממפה שם צורה ל-(תבנית, אינדקס-קבוצת-מספר, אינדקס-קבוצת-טקסט).
# text_group הוא 0 כשהכותרת עומדת לבד בשורה בלי גוף טקסט.
SECTION_STYLES: dict[str, tuple[re.Pattern[str], int, int]] = {
    # '1.1סעיף' — מספר ואז מילת מפתח, בלי רווח. takanon.pdf האמיתי.
    "keyword_after": (
        re.compile(rf"^[\s‏‎]*({_NUM})\s*{_SECTION_WORD}[\s‏‎.:)]*$"),
        1,
        0,
    ),
    # 'סעיף 1.1' — מילת מפתח ואז מספר, סדר לוגי
    "keyword_before": (
        re.compile(rf"^[\s‏‎]*{_SECTION_WORD}\s*({_NUM})[\s‏‎.:)]*$"),
        1,
        0,
    ),
    # '1.1 טקסט הסעיף...' — מספר בודד בתחילת השורה
    "number_start": (
        re.compile(rf"^[\s‏‎]*({_NUM})[.)]?\s+(.*)$"),
        1,
        2,
    ),
    # '...טקסט הסעיף 1.1' — מספר בודד בסוף השורה
    "number_end": (
        re.compile(rf"^(.*?)\s+({_NUM})[.)]?[\s‏‎]*$"),
        2,
        1,
    ),
}

# כותרות פרקים, אותו רעיון. קבוצת המספר אופציונלית (0 = "אין מספר בצורה הזו").
CHAPTER_STYLES: dict[str, tuple[re.Pattern[str], int, int]] = {
    # '— כללי1פרק' — כותרת, מספר, מילת מפתח, הכול צמוד
    "keyword_after": (
        re.compile(rf"^[\s‏‎]*[—–-]?\s*(.*?)\s*(\d+)\s*{_CHAPTER_WORD}[\s‏‎]*$"),
        2,
        1,
    ),
    # 'פרק 1 — כללי' — מילת מפתח, מספר, כותרת
    "keyword_before": (
        re.compile(rf"^[\s‏‎]*{_CHAPTER_WORD}\s*(\d+)\s*[—–-]?\s*(.*)$"),
        1,
        2,
    ),
    # 'פרק א׳ — כללי' — מילת מפתח קודם, בלי ספרות (מספור באותיות)
    "plain_start": (
        re.compile(rf"^[\s‏‎]*({_CHAPTER_WORD}\s+[^\n]{{0,80}})$"),
        0,
        1,
    ),
    # 'כללי - א פרק' — מילת מפתח אחרונה, בלי ספרות
    "plain_end": (
        re.compile(rf"^([^\n]{{0,80}}\s+{_CHAPTER_WORD})[\s‏‎]*$"),
        0,
        1,
    ),
}

# גבולות משפט. עברית משתמשת באותם סימני עצירה כמו אנגלית, בתוספת נקודתיים
# שלעיתים קרובות מסיימות פסוקית בטקסט משפטי.
SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

LINE = "─" * 64

# ההנחיה המדויקת שנשלחת למודל. מכוונת בכוונה: מבקשים מהמודל לסכם,
# לא לגלות מחדש את המספור שכבר ידוע לנו.
PROMPT_TEMPLATE = """You are given an excerpt from the regulations of an
academic institution. The excerpt is in Hebrew.

Return one single JSON object and nothing else: no text before or after it, no
explanations, and no markdown fences. It must have exactly these three fields:

{{"section_number": "...", "chapter_title": "...", "summary": "..."}}

section_number  the number of the section this excerpt belongs to. It is
                "{section_number}". Return exactly that. The excerpt does not
                contain its own heading, and any number you see inside the text
                is a cross-reference to a different section — never return one.
chapter_title   the title of the chapter this excerpt belongs to. It is
                "{chapter_title}". Return exactly that unless the excerpt
                clearly shows otherwise.
summary         a short summary in {language}, one or two sentences,
                of what the section states.

The excerpt:
---
{text}
---"""


# ── הנתונים שעוברים בצינור העיבוד ─────────────────────────────────────


@dataclass(frozen=True)
class Chunk:
    """חתיכה אחת של סעיף אחד, מוכנה לסיכום ולשמירה."""

    section_number: str
    chapter_title: str
    text: str
    page: int
    chunk_index: int


@dataclass
class Counters:
    """כל מה שהדוח הסופי צריך."""

    pages: int = 0
    chunks: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    already_present: int = 0


# ── שלב 1: קריאת ה-PDF ──────────────────────────────────────────────


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """
    מחזירה [(מספר עמוד, טקסט)] כשמספרי העמודים מתחילים מ-1.

    עמודים שלא מניבים טקסט נשארים ברשימה — עמוד סרוק שדורש OCR צריך
    להיראות במספרים, לא להיעלם בשקט.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as err:  # noqa: BLE001 — עמוד בעייתי אחד לא יעצור את הריצה
            print(f"  ⚠ עמוד {number}: חילוץ הטקסט נכשל ({err})")
            text = ""
        pages.append((number, text))
    return pages


# ── שלב 2: פיצול לסעיפים ───────────────────────────────────────────


def looks_like_chapter_heading(title: str, number: str, seen: set[str]) -> bool:
    """
    מגן מפני טקסט גוף שבמקרה נראה כמו כותרת פרק.

    הפניה צולבת בתוך משפט עטוף יכולה להיראות זהה לכותרת אמיתית ברגע
    שהשורה נחלצת בסדר חזותי. התקנון מכיל בדיוק את המצב הזה, בעמוד 7:

        שורה 1: '...מועד מיוחד שאושר מכוח זכאות שבסעיף'      (המשפט ממשיך)
        שורה 2: '— פטור מאגרה.14פרק'                          ("...או פרק 14 —
                                                               פטור מאגרה.")

    השורה השנייה מתאימה לתבנית הכותרת בול. שני דברים חושפים את זה,
    ושניהם תכונות של המסמך ולא של השורה הבודדת:

    * כותרת אמיתית לא מסתיימת בסימן פיסוק של משפט; השורה הזו מסתיימת
      בנקודה, כי המשפט שהיא סוגרת הסתיים שם.
    * מספר פרק מוכרז פעם אחת. לראות 14 שוב, לא לפי הסדר ולפני שפרק 11
      הופיע, פירושו שמדובר בהפניה, לא בהכרזה.

    החזרת False שולחת את השורה בחזרה להיחשב טקסט גוף רגיל, כך שההפניה
    הצולבת נשארת חלק מהסעיף שהיא שייכת אליו במקום להיזרק.
    """
    if not title:
        return False
    if title[-1] in ".,:;":
        return False
    if number and number in seen:
        return False
    return True


def iter_lines(pages: list[tuple[int, str]]) -> Iterator[tuple[int, str]]:
    """מייצרת (מספר עמוד, שורה מנוקה) לכל שורה לא ריקה."""
    for page_number, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if line:
                yield page_number, line


def count_style_matches(
    pages: list[tuple[int, str]], styles: dict[str, tuple[re.Pattern[str], int, int]]
) -> dict[str, int]:
    """כמה שורות כל צורת כותרת הייתה מתאימה בכל המסמך."""
    counts = {name: 0 for name in styles}
    for _, line in iter_lines(pages):
        for name, (pattern, _, _) in styles.items():
            if pattern.match(line):
                counts[name] += 1
    return counts


def detect_heading_style(
    pages: list[tuple[int, str]],
    styles: dict[str, tuple[re.Pattern[str], int, int]],
    preferred: tuple[str, ...],
) -> str:
    """
    בוחרת את צורת הכותרות שבה המסמך הזה בפועל משתמש.

    צורות שמופיעות ב-`preferred` מנצחות בכל פעם שהן מתאימות למשהו, גם אם
    צורה רופפת יותר מתאימה ליותר שורות. הסדר הזה חשוב: צורות מילת המפתח
    ('1.1סעיף') חד-משמעיות, בעוד שהצורות עם מספר בודד מתאימות לכל שורה
    שבמקרה מתחילה או מסתיימת במספר — במסמך אמיתי הצורות הרופפות היו
    קוברות את התשובה הנכונה תחת התאמות שווא.
    """
    counts = count_style_matches(pages, styles)
    for name in preferred:
        if counts.get(name, 0) > 0:
            return name
    return max(counts, key=lambda name: counts[name])


def split_into_sections(
    pages: list[tuple[int, str]],
    section_style: str = "keyword_after",
    chapter_style: str = "keyword_after",
) -> list[tuple[str, str, str, int]]:
    """
    מפצלת את העמודים לסעיפים לפי המספור של התקנון עצמו.

    מחזירה [(section_number, chapter_title, text, page)], כש-page הוא
    העמוד שבו הסעיף התחיל. סעיף שנמשך על פני שבירת עמוד נשאר סעיף אחד,
    משויך לעמוד שבו הופיע המספר שלו.

    הצורות מגיעות מ-detect_heading_style().
    """
    section_re, section_num_group, section_text_group = SECTION_STYLES[section_style]
    chapter_re, chapter_num_group, chapter_text_group = CHAPTER_STYLES[chapter_style]

    sections: list[tuple[str, str, str, int]] = []

    chapter_title = ""
    current_number: str | None = None
    current_lines: list[str] = []
    current_page = 1

    def flush() -> None:
        if current_number is None:
            return
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_number, chapter_title, text, current_page))

    seen_chapters: set[str] = set()

    for page_number, line in iter_lines(pages):
        chapter_match = chapter_re.match(line)
        if chapter_match:
            title = (
                chapter_match.group(chapter_text_group).strip()
                if chapter_text_group
                else ""
            )
            number = (
                chapter_match.group(chapter_num_group).strip()
                if chapter_num_group
                else ""
            )
            if looks_like_chapter_heading(title, number, seen_chapters):
                flush()
                current_number, current_lines = None, []
                if number:
                    seen_chapters.add(number)
                # בונים מחדש את הכותרת בסדר קריאה, איך שלא הגיעה
                chapter_title = (
                    f"פרק {number} — {title}".strip(" —") if number else title
                )
                continue
            # בסופו של דבר לא כותרת — ממשיכים הלאה ומשאירים כטקסט גוף

        section_match = section_re.match(line)
        if section_match:
            flush()
            current_number = section_match.group(section_num_group).strip()
            remainder = (
                section_match.group(section_text_group).strip()
                if section_text_group
                else ""
            )
            current_lines = [remainder] if remainder else []
            current_page = page_number
            continue

        if current_number is not None:
            current_lines.append(line)

    flush()
    return sections


# ── שלב 3: חלוקת סעיפים ארוכים לצ'אנקים ────────────────────────────


def split_sentences(text: str) -> list[str]:
    """מפצלת טקסט למשפטים, ומשמיטה ריקים."""
    return [part.strip() for part in SENTENCE_RE.split(text) if part and part.strip()]


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    מפצלת טקסט לצ'אנקים של עד ~max_chars תווים, כשהחיתוך נעשה רק בין
    משפטים, עם כ-overlap תווים חוזרים בכל תפר.

    משפט בודד הארוך מ-max_chars הופך לצ'אנק גדול משלו: עדיף צ'אנק ארוך
    אחד מאשר משפט חתוך לשניים.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and length + addition > max_chars:
            chunks.append(" ".join(current))

            # מעבירים משפטים שלמים כחפיפה, החדש ביותר קודם
            carried: list[str] = []
            carried_len = 0
            for previous in reversed(current):
                if carried_len + len(previous) > overlap:
                    break
                carried.insert(0, previous)
                carried_len += len(previous) + 1
            current = carried
            length = carried_len

        current.append(sentence)
        length += addition

    if current:
        chunks.append(" ".join(current))

    # החפיפה יכולה לגרום לזנב להיות כפילות של מה שהיה קודם
    return [c for i, c in enumerate(chunks) if i == 0 or c != chunks[i - 1]]


def build_chunks(
    sections: list[tuple[str, str, str, int]], max_chars: int, overlap: int
) -> list[Chunk]:
    """הופכת סעיפים לרשימה השטוחה של הצ'אנקים שיישמרו."""
    chunks: list[Chunk] = []
    for section_number, chapter_title, text, page in sections:
        for index, piece in enumerate(chunk_text(text, max_chars, overlap)):
            chunks.append(
                Chunk(
                    section_number=section_number,
                    chapter_title=chapter_title,
                    text=piece,
                    page=page,
                    chunk_index=index,
                )
            )
    return chunks


# ── שלב 4: המודל ────────────────────────────────────────────────────


def build_client(settings: Settings) -> Any:
    """יוצרת את לקוח Gemini."""
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def describe_api_error(err: Exception) -> str:
    """
    הופכת חריגה מהספק לשורה קצרה אחת.

    שגיאת המכסה הגולמית היא בלוק JSON רב-שורות באורך כמה מאות תווים.
    הדפסה שלה פעם אחת לכל צ'אנק שנכשל קוברת את הדוח, אז המקרים
    השכיחים מזוהים ומוצגים בפשטות במקום זאת.
    """
    text = str(err)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        quota = re.search(r"quotaValue': '(\d+)'", text)
        retry = re.search(r"retryDelay': '([^']+)'", text)
        detail = f", מגבלת השכבה החינמית היומית היא {quota.group(1)} בקשות" if quota else ""
        wait = f", ניתן לנסות שוב בעוד {retry.group(1)}" if retry else ""
        return f"מכסת Gemini נגמרה{detail}{wait}"
    if "404" in text and "NOT_FOUND" in text:
        return "המודל לא נמצא — בדקו את --model מול models.list()"
    if "PERMISSION_DENIED" in text or "401" in text:
        return "מפתח Gemini נדחה — בדקו את GEMINI_API_KEY"
    return f"קריאה למודל נכשלה: {text[:120]}"


def parse_response(raw: str) -> tuple[dict[str, Any] | None, str]:
    """
    הופכת את הטקסט הגולמי מהמודל לאובייקט JSON, או מסבירה מה לא בסדר.

    קפדנית בכוונה: התשובה חייבת להיות אובייקט JSON נקי אחד עם שלושת
    השדות הצפויים. גדרות markdown, טקסט מסביב וגרר בסוף — כולם נדחים
    ולא מנוקים או מתוקנים, כך שדבר שלא הוחזר בפועל לא ימצא את דרכו
    למסד הנתונים.
    """
    raw = raw.strip()
    if not raw:
        return None, "המודל החזיר תשובה ריקה"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        return None, f"התשובה אינה JSON תקין ({err.msg}, בתו {err.pos})"

    if not isinstance(parsed, dict):
        return None, f"התשובה אינה אובייקט JSON אלא {type(parsed).__name__}"

    missing = [
        field
        for field in ("section_number", "chapter_title", "summary")
        if field not in parsed
    ]
    if missing:
        return None, f"שדות חסרים בתשובת המודל: {', '.join(missing)}"

    return parsed, ""


def ask_model(client: Any, model: str, chunk: Chunk) -> tuple[dict[str, Any] | None, str]:
    """
    שולחת צ'אנק אחד למודל ומחזירה (parsed_json, reason).

    בהצלחה reason הוא "". בכישלון parsed_json הוא None ו-reason מסביר
    מה השתבש, כדי שהקוד הקורא יוכל לרשום זאת ולדלג על הצ'אנק.

    תשובה פגומה מנוסה שוב, ולא מתוקנת. זה משנה בפועל: לפעמים המודל
    מחזיר אובייקט תקין ואחריו '}' מיותר, שאינו JSON תקין ונדחה כראוי —
    אבל אותה בקשה בדרך כלל חוזרת נקייה בניסיון שני, וזריקת הסעיף בגלל
    פיסוק שגוי הייתה מאבדת תוכן אמיתי בלי סיבה טובה.

    הניסיון החוזר רץ בטמפרטורה מעט גבוהה יותר מהניסיון הראשון. בטמפרטורה
    0 הקריאה דטרמיניסטית, כך שניסיון חוזר זהה היה משחזר את אותה תשובה שבורה.
    """
    from google.genai import types

    prompt = PROMPT_TEMPLATE.format(
        section_number=chunk.section_number,
        chapter_title=chunk.chapter_title or "unknown",
        language=SUMMARY_LANGUAGE,
        text=chunk.text,
    )

    last_reason = "המודל לא נענה מעולם"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.0 if attempt == 1 else RETRY_TEMPERATURE,
                ),
            )
        except Exception as err:  # noqa: BLE001 — שגיאות מכסה ורשת נופלות לכאן
            last_reason = describe_api_error(err)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
            continue

        parsed, reason = parse_response(response.text or "")
        if parsed is not None:
            return parsed, ""

        # תשובה גרועה שווה ניסיון נוסף, אבל לא המתנה ארוכה
        last_reason = reason
        if attempt < MAX_RETRIES:
            time.sleep(SLEEP_BETWEEN_CALLS)

    return None, last_reason


# ── שלב 5: אימות ─────────────────────────────────────────────────────


def validate(chunk: Chunk, seen: set[tuple[str, int]]) -> str | None:
    """
    בודקת צ'אנק אחד לפני שהוא מגיע למסד הנתונים.
    מחזירה None אם הכול תקין, או סיבה לדלג עליו.
    """
    if not chunk.text.strip():
        return "הטקסט ריק"
    if not chunk.section_number.strip():
        return "אין מספר סעיף"
    if (chunk.section_number, chunk.chunk_index) in seen:
        return (
            f"כפילות: סעיף {chunk.section_number} צ'אנק "
            f"{chunk.chunk_index} כבר טופל"
        )
    return None


# ── שלב 6: מסד הנתונים ───────────────────────────────────────────────

UPSERT_SQL = f"""
INSERT INTO {db.TABLE}
    (section_number, chunk_index, chapter, content, summary, page)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (section_number, chunk_index) DO UPDATE
SET chapter    = EXCLUDED.chapter,
    content    = EXCLUDED.content,
    summary    = EXCLUDED.summary,
    page       = EXCLUDED.page,
    version    = {db.TABLE}.version + 1,
    updated_at = now()
RETURNING (xmax = 0) AS inserted;
"""


def existing_keys(settings: Settings) -> set[tuple[str, int]]:
    """
    מחזירה את צמדי (section_number, chunk_index) שכבר נשמרו עם סיכום.
    בשימוש על ידי --skip-existing כדי להמשיך ריצה בזול.
    """
    sql = (
        f"SELECT section_number, chunk_index FROM {db.TABLE} "
        "WHERE summary IS NOT NULL AND summary <> '';"
    )
    with db.connect(settings) as conn:
        return {(r["section_number"], r["chunk_index"]) for r in conn.execute(sql)}


def ensure_schema(settings: Settings) -> None:
    """
    מביאה את הטבלה לצורה שהסקריפט הזה צריך. אידמפוטנטי.

    הסכמה והמיגרציה שלה חיות שתיהן ב-db.SCHEMA_SQL, כך שיש הגדרה אחת
    לטבלה במקום שתיים שיכולות להתרחק זו מזו.
    """
    db.init_db(settings)


def upsert_chunk(
    conn: Any, chunk: Chunk, chapter_title: str, summary: str
) -> bool:
    """
    כותבת צ'אנק אחד. מחזירה True אם נוצרה שורה חדשה, False אם עודכנה
    שורה קיימת.

    התרגיל `xmax = 0` הוא הדרך של Postgres לדווח איזה ענף של ה-upsert
    רץ: בהוספה טרייה xmax הוא 0, בעדכון הוא מחזיק מזהה טרנזקציה.
    """
    row = conn.execute(
        UPSERT_SQL,
        (
            chunk.section_number,
            chunk.chunk_index,
            chapter_title or None,
            chunk.text,
            summary or None,
            chunk.page,
        ),
    ).fetchone()
    return bool(row["inserted"])


# ── הרכבת הכול יחד ────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="טוען את תקנון המכללה מקובץ PDF למסד הנתונים בענן."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="נתיב לקובץ ה-PDF")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="שם מודל Gemini")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS, help="אורך צ'אנק מרבי")
    parser.add_argument("--overlap", type=int, default=OVERLAP_CHARS, help="חפיפה בין צ'אנקים")
    parser.add_argument("--limit", type=int, default=0, help="לעבד רק N הצ'אנקים הראשונים")
    parser.add_argument(
        "--section-style",
        choices=("auto",) + tuple(SECTION_STYLES),
        default="auto",
        help="צורת כותרות הסעיפים (ברירת מחדל: זיהוי אוטומטי)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ניתוח הקובץ בלבד — בלי קריאות למודל ובלי כתיבה למסד הנתונים",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="להשאיר צ'אנקים שכבר יש להם סיכום כמו שהם — ממשיך ריצה שנעצרה "
        "באמצע בלי לבזבז מכסה על עבודה שכבר בוצעה",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counters = Counters()
    # מופרדים בכוונה: דילוג פירושו שצ'אנק לא הגיע למסד הנתונים ודורש
    # תשומת לב, בעוד שהערה היא משהו שכדאי לדעת על צ'אנק שכן נשמר.
    # ערבוב ביניהם מסתיר את הכשלים האמיתיים.
    skipped_reasons: list[str] = []
    notes: list[str] = []

    print(f"{LINE}\nטעינת התקנון למסד הנתונים\n{LINE}")

    # ── קריאת הקובץ ──────────────────────────────────────────────
    if not args.pdf.exists():
        print(f"❌ קובץ ה-PDF לא נמצא ב: {args.pdf}", file=sys.stderr)
        print("   הניחו את הקובץ שם, או ציינו נתיב אחר עם --pdf.", file=sys.stderr)
        return 1

    print(f"\n[1/5] קריאת הקובץ: {args.pdf.name}")
    try:
        pages = read_pdf_pages(args.pdf)
    except Exception as err:  # noqa: BLE001
        print(f"❌ קריאת ה-PDF נכשלה: {err}", file=sys.stderr)
        return 1
    counters.pages = len(pages)
    empty_pages = sum(1 for _, text in pages if not text.strip())
    print(f"      נקראו {counters.pages} עמודים.")
    if empty_pages:
        print(
            f"      ⚠ ל-{empty_pages} עמודים אין טקסט — ייתכן שהם סרוקים וזקוקים ל-OCR."
        )

    # ── פיצול לסעיפים ────────────────────────────────────────────
    print("\n[2/5] פיצול לסעיפים לפי המספור של התקנון")
    section_style = args.section_style
    if section_style == "auto":
        section_style = detect_heading_style(
            pages, SECTION_STYLES, ("keyword_after", "keyword_before")
        )
    chapter_style = detect_heading_style(
        pages, CHAPTER_STYLES, ("keyword_after", "keyword_before")
    )
    section_hits = count_style_matches(pages, SECTION_STYLES)[section_style]
    print(
        f"      צורת כותרות: סעיפים '{section_style}' ({section_hits} כותרות), "
        f"פרקים '{chapter_style}'."
    )

    sections = split_into_sections(pages, section_style, chapter_style)
    print(f"      נמצאו {len(sections)} סעיפים.")
    if not sections:
        print(
            "❌ לא נמצא אף סעיף. ייתכן שהכותרות בקובץ נראות שונה מכל צורה\n"
            "   שמוכרת ל-SECTION_STYLES — הדפיסו כמה שורות עם --dry-run והוסיפו\n"
            "   תבנית, או כפו צורה עם --section-style.",
            file=sys.stderr,
        )
        return 1
    if len(sections) < 5:
        print("      ⚠ מעט מאוד סעיפים — ייתכן שתבנית המספור זקוקה לכיוון.")

    # ── חלוקה לצ'אנקים ───────────────────────────────────────────
    print("\n[3/5] חלוקת סעיפים ארוכים לצ'אנקים")
    chunks = build_chunks(sections, args.max_chars, args.overlap)
    counters.chunks = len(chunks)
    longest = max((len(c.text) for c in chunks), default=0)
    print(f"      נוצרו {counters.chunks} צ'אנקים (הארוך ביותר: {longest} תווים).")

    if args.limit:
        chunks = chunks[: args.limit]
        print(f"      ⚠ --limit פעיל: מעבדים רק את {len(chunks)} הצ'אנקים הראשונים.")

    if args.dry_run:
        print("\n[4/5] --dry-run: מדלגים על המודל ועל מסד הנתונים.")
        print(f"\n{LINE}\nסיכום (dry run)\n{LINE}")
        print(f"  עמודים שנקראו  : {counters.pages}")
        print(f"  צ'אנקים שנוצרו : {counters.chunks}")
        for chunk in chunks[:3]:
            print(
                f"\n  · סעיף {chunk.section_number} (צ'אנק {chunk.chunk_index}, "
                f"עמוד {chunk.page}, פרק: {chunk.chapter_title or '—'})\n"
                f"    {chunk.text[:120]}..."
            )
        return 0

    # ── המודל ומסד הנתונים ───────────────────────────────────────
    settings = load_settings_or_exit()

    print("\n[4/5] הכנת מסד הנתונים")
    try:
        ensure_schema(settings)
    except Exception as err:  # noqa: BLE001
        print(f"❌ הכנת הטבלה נכשלה: {err}", file=sys.stderr)
        return 1
    print(f"      הטבלה '{db.TABLE}' מוכנה (מפתח: section_number + chunk_index).")

    try:
        client = build_client(settings)
    except Exception as err:  # noqa: BLE001
        print(f"❌ יצירת לקוח Gemini נכשלה: {err}", file=sys.stderr)
        return 1

    already: set[tuple[str, int]] = set()
    if args.skip_existing:
        already = existing_keys(settings)
        pending = [c for c in chunks if (c.section_number, c.chunk_index) not in already]
        counters.already_present = len(chunks) - len(pending)
        print(
            f"      --skip-existing: ל-{counters.already_present} צ'אנקים כבר יש סיכום, "
            f"נותרו {len(pending)}."
        )
        chunks = pending

    print(f"\n[5/5] סיכום ושמירה של {len(chunks)} צ'אנקים (מודל: {args.model})")
    seen: set[tuple[str, int]] = set()

    try:
        with db.connect(settings) as conn:
            for position_index, chunk in enumerate(chunks, start=1):
                reason = validate(chunk, seen)
                if reason:
                    counters.skipped += 1
                    skipped_reasons.append(
                        f"סעיף {chunk.section_number}/{chunk.chunk_index}: {reason}"
                    )
                    continue

                parsed, failure = ask_model(client, args.model, chunk)
                if parsed is None:
                    counters.skipped += 1
                    skipped_reasons.append(
                        f"סעיף {chunk.section_number}/{chunk.chunk_index}: {failure}"
                    )
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                # המספור מגיע מהמסמך, לא מהמודל: המנתח ראה את הכותרת
                # האמיתית. הערך של המודל בשימוש רק כשלמנתח לא היה כלום,
                # וסתירה שווה תיעוד.
                model_number = str(parsed.get("section_number") or "").strip()
                if model_number and model_number != chunk.section_number:
                    notes.append(
                        f"סעיף {chunk.section_number}/{chunk.chunk_index}: "
                        f"המודל דיווח מספר שונה ({model_number}) — "
                        "נשמר לפי הקובץ"
                    )

                chapter_title = (
                    str(parsed.get("chapter_title") or "").strip()
                    or chunk.chapter_title
                )
                summary = str(parsed.get("summary") or "").strip()

                try:
                    inserted = upsert_chunk(conn, chunk, chapter_title, summary)
                except Exception as err:  # noqa: BLE001
                    counters.skipped += 1
                    skipped_reasons.append(
                        f"סעיף {chunk.section_number}/{chunk.chunk_index}: "
                        f"כתיבה למסד הנתונים נכשלה ({str(err)[:100]})"
                    )
                    continue

                seen.add((chunk.section_number, chunk.chunk_index))
                if inserted:
                    counters.inserted += 1
                else:
                    counters.updated += 1

                if position_index % 10 == 0 or position_index == len(chunks):
                    print(
                        f"      {position_index}/{len(chunks)} — "
                        f"נוספו {counters.inserted}, עודכנו {counters.updated}, "
                        f"דולגו {counters.skipped}"
                    )
                time.sleep(SLEEP_BETWEEN_CALLS)
    except Exception as err:  # noqa: BLE001
        print(f"\n❌ הריצה נעצרה: {err}", file=sys.stderr)
        return 1

    # ── הדוח ──────────────────────────────────────────────────────
    print(f"\n{LINE}\nסיכום\n{LINE}")
    print(f"  עמודים שנקראו   : {counters.pages}")
    print(f"  צ'אנקים שנוצרו  : {counters.chunks}")
    print(f"  צ'אנקים נוספו   : {counters.inserted}")
    print(f"  צ'אנקים עודכנו  : {counters.updated}")
    print(f"  צ'אנקים דולגו   : {counters.skipped}")
    if counters.already_present:
        print(f"  לא נגעו בהם     : {counters.already_present} (--skip-existing)")

    if skipped_reasons:
        print(f"\n  ❌ דולגו — הצ'אנקים האלה אינם במסד הנתונים ({len(skipped_reasons)}):")
        for reason in skipped_reasons[:20]:
            print(f"    • {reason}")
        if len(skipped_reasons) > 20:
            print(f"    ... ועוד {len(skipped_reasons) - 20}")
        print(
            "\n     הריצו שוב עם --skip-existing כדי לנסות רק אלה מחדש, בלי\n"
            "     לבזבז מכסה על הצ'אנקים שכבר הצליחו."
        )

    if notes:
        print(f"\n  ℹ הערות — הצ'אנקים האלה כן נשמרו ({len(notes)}):")
        for note in notes[:5]:
            print(f"    • {note}")
        if len(notes) > 5:
            print(f"    ... ועוד {len(notes) - 5}")

    print(
        "\n✅ הטעינה הסתיימה."
        if counters.inserted or counters.updated
        else "\n⚠ לא נשמר אף צ'אנק."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
