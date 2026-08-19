"""
Vector index layer — Pinecone in the cloud (not a local file on the machine).

The connection is always made through PINECONE_API_KEY / PINECONE_INDEX
coming from the .env file (via config.Settings).

Each vector represents one regulation section:
    id (section_number)  The section number — the vector's id in the index
    values                The embedding (DIMENSION numbers)
    metadata.chapter      The chapter the section belongs to
    metadata.content      The text of the section
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config import Settings, load_settings

# Matches the existing Pinecone index and EMBEDDING_MODEL default (see config.py)
DIMENSION = 768
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

# How long to wait for a freshly created serverless index to become ready
READY_TIMEOUT = 60.0
READY_INTERVAL = 1.0

_settings: Settings | None = None
_client: Pinecone | None = None


@dataclass(frozen=True)
class IndexInfo:
    dimension: int
    metric: str
    host: str


def use_settings(settings: Settings) -> None:
    """Lets a caller hand in already-loaded settings, so every call below shares one client."""
    global _settings, _client
    _settings = settings
    _client = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def get_client() -> Pinecone:
    """Creates the Pinecone client once and reuses it."""
    global _client
    if _client is None:
        _client = Pinecone(api_key=get_settings().pinecone_api_key)
    return _client


def index_name() -> str:
    return get_settings().pinecone_index


def index_exists() -> bool:
    client = get_client()
    return client.has_index(index_name())


def ensure_index() -> IndexInfo:
    """Creates the index if it does not exist yet, waits until it is ready, and describes it."""
    client = get_client()
    name = index_name()

    if not client.has_index(name):
        client.create_index(
            name=name,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )

    deadline = time.monotonic() + READY_TIMEOUT
    described = client.describe_index(name)
    while not described.status.ready and time.monotonic() < deadline:
        time.sleep(READY_INTERVAL)
        described = client.describe_index(name)

    return IndexInfo(
        dimension=described.dimension,
        metric=described.metric,
        host=described.host,
    )


def _index():
    described = get_client().describe_index(index_name())
    return get_client().Index(host=described.host)


def count_vectors() -> int:
    """How many vectors are currently stored in the index."""
    stats = _index().describe_index_stats()
    return int(stats.total_vector_count or 0)


def upsert_section_vector(
    section_number: str,
    values: list[float],
    chapter: str | None,
    content: str,
) -> int:
    """Inserts or overwrites one vector. Returns the number of vectors written."""
    metadata: dict[str, Any] = {"content": content}
    if chapter is not None:
        metadata["chapter"] = chapter
    result = _index().upsert(
        vectors=[{"id": section_number, "values": values, "metadata": metadata}]
    )
    return int(result.upserted_count)


def fetch_section_vector(section_number: str) -> dict[str, Any] | None:
    """Returns one vector by id, or None if it does not exist."""
    result = _index().fetch(ids=[section_number])
    vector = result.vectors.get(section_number)
    if vector is None:
        return None
    return {
        "section_number": vector.id,
        "values": list(vector.values),
        "metadata": dict(vector.metadata or {}),
    }


def query_similar(embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
    """Returns the top_k vectors closest to the given embedding."""
    result = _index().query(vector=embedding, top_k=top_k, include_metadata=True)
    return [
        {
            "section_number": match.id,
            "score": float(match.score),
            "metadata": dict(match.metadata or {}),
        }
        for match in result.matches
    ]


def delete_section_vector(section_number: str) -> None:
    """Deletes one vector by id."""
    _index().delete(ids=[section_number])
