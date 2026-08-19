"""
Loading secret settings from the .env file.

Hard rules:
1. The .env file is never committed to git / uploaded to the cloud (see .gitignore).
2. The values themselves are never printed to the screen — only the key names.
3. If a key is missing, the program stops with a clear message stating which key is missing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

# Project directory = the directory this file lives in
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# The four required keys, each with a description
REQUIRED_KEYS: dict[str, str] = {
    "GEMINI_API_KEY": "Gemini API key (the language model)",
    "PINECONE_API_KEY": "Pinecone API key (the vector database)",
    "PINECONE_INDEX": "Pinecone index name",
    "DATABASE_URL": "Connection string for the cloud database (Neon)",
}

# Keys that may be set but are not required. They stay out of REQUIRED_KEYS on
# purpose: adding one there would stop every existing .env file from loading,
# and the whole point of a default is that an old file keeps working.
OPTIONAL_KEYS: dict[str, str] = {
    "EMBEDDING_MODEL": "Google embedding model used to build the vectors",
}

# Produces 3072 dimensions natively and can be asked for 768, which is what the
# existing Pinecone index was created with.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


class ConfigError(Exception):
    """Configuration error — a key is missing, empty or invalid."""


@dataclass(frozen=True)
class Settings:
    """All project settings in one place."""

    gemini_api_key: str
    pinecone_api_key: str
    pinecone_index: str
    database_url: str
    # Optional, with a default, so a .env written before this existed still loads
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    def __repr__(self) -> str:  # Safeguard: even an accidental print won't leak secrets
        return "Settings(gemini_api_key=***, pinecone_api_key=***, " \
               f"pinecone_index={self.pinecone_index!r}, database_url=***, " \
               f"embedding_model={self.embedding_model!r})"

    @property
    def db_target(self) -> str:
        """Display-safe description: host and database name only — no username or password."""
        return safe_db_target(self.database_url)


def safe_db_target(database_url: str) -> str:
    """Returns 'host/dbname' from the connection string, without credentials."""
    try:
        parts = urlsplit(database_url)
        host = parts.hostname or "?"
        dbname = (parts.path or "/").lstrip("/") or "?"
        return f"{host}/{dbname}"
    except Exception:
        return "?"


def _load_raw_values(path: Path) -> dict[str, str | None]:
    """
    Reads the raw key/value pairs, from whichever source is available.

    Locally, that's the .env file. On Streamlit Community Cloud there is no
    .env file — secrets pasted into the app's Secrets box are exposed
    through st.secrets instead, so we fall back to that when the file is
    missing.
    """
    if path.exists():
        # Read directly from the file, without polluting global environment variables
        return dict(dotenv_values(path))

    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:
        return {}


def load_settings(env_path: Path | None = None) -> Settings:
    """
    Reads the four keys from the .env file (or, on Streamlit Cloud, from
    st.secrets) and returns a Settings object.
    Raises ConfigError with a clear message if anything is missing.
    """
    path = env_path or ENV_PATH
    values = _load_raw_values(path)

    if not values:
        raise ConfigError(
            f"Error: the .env secrets file was not found at: {path}\n"
            "Create it based on the .env.example sample file "
            "(or, on Streamlit Cloud, fill in the app's Secrets box)."
        )

    missing: list[str] = []
    for key, description in REQUIRED_KEYS.items():
        value = (values.get(key) or "").strip()
        if not value:
            missing.append(f"  • {key} — {description}")

    if missing:
        raise ConfigError(
            "Error: keys are missing from the .env file — the program is stopping.\n"
            "The missing (or empty) keys are:\n"
            + "\n".join(missing)
            + f"\n\nFile checked: {path}\n"
            "Add the missing values and try again. (The values themselves are never printed to the screen.)"
        )

    database_url = (values["DATABASE_URL"] or "").strip()

    # A Neon URL must include sslmode=require — we keep it exactly as is, unchanged
    if "sslmode=require" not in database_url:
        raise ConfigError(
            "Error: DATABASE_URL must include sslmode=require "
            "(Neon requires an encrypted connection).\n"
            "Keep the end of the URL exactly like this: ?sslmode=require"
        )

    return Settings(
        gemini_api_key=(values["GEMINI_API_KEY"] or "").strip(),
        pinecone_api_key=(values["PINECONE_API_KEY"] or "").strip(),
        pinecone_index=(values["PINECONE_INDEX"] or "").strip(),
        database_url=database_url,
        embedding_model=(values.get("EMBEDDING_MODEL") or "").strip()
        or DEFAULT_EMBEDDING_MODEL,
    )


def load_settings_or_exit(env_path: Path | None = None) -> Settings:
    """Like load_settings, but exits the program with a clear message instead of raising."""
    try:
        return load_settings(env_path)
    except ConfigError as err:
        print(str(err), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    settings = load_settings_or_exit()
    print("All four keys were loaded successfully from the .env file:")
    for key in REQUIRED_KEYS:
        print(f"  ✓ {key} — loaded (value hidden)")
    print(f"Database target: {settings.db_target}")
