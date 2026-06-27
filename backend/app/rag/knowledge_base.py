"""RAG knowledge-base ingestion, tagging, and retrieval (Task 15.1).

This module is the storage + retrieval foundation that the grounded
visa/safety and guide answering tasks (15.2-15.4) build on. It provides:

* :class:`Document` - an ingestible knowledge-base document that is **always**
  tagged with a ``source`` name and a ``last_updated`` date. These tags are
  required by **Property 24** ("knowledge-base documents are tagged") and by
  the citation requirements (Requirement 11.2/11.5) that later tasks enforce.
* :class:`KnowledgeBase` - the ingestion + retrieval API: ``ingest(documents)``
  stores documents in a vector store, and ``search(query, k)`` returns the most
  similar documents paired with a similarity score.

Design notes (see design.md "RAG Service" and Requirement 20.3):

* The design lists the vector store as "pgvector | Chroma". Both require infra
  (a Postgres+pgvector instance, or the optional ``chromadb`` package). To keep
  the system runnable with **zero infrastructure and no paid keys** - matching
  the mock-first provider pattern in :mod:`app.providers` - the default backend
  is a deterministic in-memory store with a deterministic hashing embedding
  (no network, no extra dependency).
* The embedding and vector-store backends are **swappable** via
  :class:`app.config.Settings` (``kb_embedding_backend`` /
  ``vector_store_backend``) or by passing instances directly to
  :class:`KnowledgeBase`. Optional ``chroma``/``pgvector`` backends are imported
  **lazily** inside the factory so this module stays import-safe even when those
  packages / services are unavailable.

The module performs no I/O at import time and depends only on the standard
library, ``pydantic``, and :mod:`app.config`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from datetime import date
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings

# =============================================================================
# Models
# =============================================================================


class Document(BaseModel):
    """A knowledge-base document, tagged with its provenance.

    ``source`` and ``last_updated`` are **required** fields: a document cannot
    be constructed (and therefore cannot be ingested) without them, which is
    how this layer guarantees **Property 24** - every ingested document carries
    a source name and a last-updated date.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, description="Document body text.")
    source: str = Field(..., min_length=1, description="Source name tag (e.g. publisher/site).")
    last_updated: date = Field(..., description="Date the source content was last updated.")
    doc_id: str | None = Field(
        default=None,
        description="Stable identifier; derived deterministically when omitted.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Optional free-form tags (e.g. topic, country)."
    )


class SearchResult(BaseModel):
    """A retrieved document paired with its similarity score.

    ``score`` is a cosine similarity in ``[0.0, 1.0]`` (negative similarities
    are clamped to ``0.0``), where higher means more similar to the query.
    """

    model_config = ConfigDict(extra="forbid")

    document: Document
    score: float = Field(..., ge=0.0, le=1.0)


# =============================================================================
# Embedding backends (swappable; deterministic default)
# =============================================================================


@runtime_checkable
class Embedding(Protocol):
    """Turns text into a fixed-length vector."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer used by the hashing embedding."""
    return _TOKEN_RE.findall(text.lower())


class HashEmbedding:
    """Deterministic, no-network feature-hashing (bag-of-words) embedding.

    Each token is hashed into a bucket with a stable sign (the "hashing trick").
    The resulting vector is L2-normalized so dot products are cosine
    similarities. Documents sharing tokens land in the same buckets and score
    highly; this is sufficient for grounded retrieval over a curated KB while
    requiring no model download, network call, or paid key.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError(f"embedding dim must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokenize(text):
            digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            bucket = digest % self._dim
            sign = 1.0 if (digest >> 8) & 1 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(component * component for component in vec))
        if norm > 0.0:
            vec = [component / norm for component in vec]
        return vec


# =============================================================================
# Vector-store backends (swappable; in-memory default)
# =============================================================================


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Inputs from :class:`HashEmbedding` are already L2-normalized, but the dot
    product is normalized here too so the store works with any embedding.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@runtime_checkable
class VectorStore(Protocol):
    """Stores document vectors and answers nearest-neighbour queries."""

    def add(self, doc_id: str, vector: list[float], document: Document) -> None: ...

    def query(self, vector: list[float], k: int) -> list[tuple[Document, float]]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    """Deterministic in-process vector store backed by a list.

    Suitable for the zero-infra default and for tests. Adding a document with a
    ``doc_id`` that already exists replaces the prior entry (idempotent
    re-ingestion). Queries return documents sorted by descending cosine
    similarity, clamped to ``[0.0, 1.0]``.
    """

    def __init__(self) -> None:
        # Ordered list of (doc_id, vector, document) for stable, deterministic
        # iteration. An index by id supports replace-on-reingest.
        self._items: list[tuple[str, list[float], Document]] = []
        self._index: dict[str, int] = {}

    def add(self, doc_id: str, vector: list[float], document: Document) -> None:
        existing = self._index.get(doc_id)
        if existing is not None:
            self._items[existing] = (doc_id, vector, document)
        else:
            self._index[doc_id] = len(self._items)
            self._items.append((doc_id, vector, document))

    def query(self, vector: list[float], k: int) -> list[tuple[Document, float]]:
        if k <= 0:
            return []
        scored = [
            # Clamp to [0.0, 1.0]: cosine is mathematically bounded by 1.0, but
            # floating-point rounding can yield e.g. 1.0000000000000002, which
            # would violate the SearchResult score bound.
            (document, min(1.0, max(0.0, _cosine(vector, stored_vector))))
            for _doc_id, stored_vector, document in self._items
        ]
        # Stable sort by descending score; ties keep insertion order.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def clear(self) -> None:
        self._items.clear()
        self._index.clear()

    def __len__(self) -> int:
        return len(self._items)


# =============================================================================
# Backend factories (lazy, config-driven)
# =============================================================================


def make_embedding(settings: Settings | None = None) -> Embedding:
    """Build the configured embedding backend.

    Defaults to the deterministic :class:`HashEmbedding`. ``sentence-transformers``
    is imported lazily only when explicitly selected.
    """
    settings = settings if settings is not None else get_settings()
    backend = settings.kb_embedding_backend
    if backend == "hash":
        return HashEmbedding(dim=settings.kb_embedding_dim)
    if backend == "sentence-transformers":  # pragma: no cover - optional dep
        try:
            from app.rag.embeddings_st import SentenceTransformerEmbedding
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "kb_embedding_backend='sentence-transformers' requires the optional "
                "embeddings backend and the 'sentence-transformers' package."
            ) from exc
        return SentenceTransformerEmbedding(dim=settings.kb_embedding_dim)
    raise ValueError(f"Unknown kb_embedding_backend {backend!r}.")


def make_vector_store(settings: Settings | None = None) -> VectorStore:
    """Build the configured vector-store backend.

    Defaults to the zero-infra :class:`InMemoryVectorStore`. ``chroma`` and
    ``pgvector`` are imported lazily only when explicitly selected, so this
    module imports cleanly without those packages/services installed.
    """
    settings = settings if settings is not None else get_settings()
    backend = settings.vector_store_backend
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "chroma":  # pragma: no cover - optional infra
        try:
            from app.rag.vectorstores_chroma import ChromaVectorStore
        except ImportError as exc:  # pragma: no cover - optional infra
            raise RuntimeError(
                "vector_store_backend='chroma' requires the optional Chroma backend "
                "and the 'chromadb' package."
            ) from exc
        return ChromaVectorStore(settings)
    if backend == "pgvector":  # pragma: no cover - optional infra
        try:
            from app.rag.vectorstores_pgvector import PgVectorStore
        except ImportError as exc:  # pragma: no cover - optional infra
            raise RuntimeError(
                "vector_store_backend='pgvector' requires the optional pgvector backend "
                "and a reachable Postgres+pgvector database."
            ) from exc
        return PgVectorStore(settings)
    raise ValueError(f"Unknown vector_store_backend {backend!r}.")


# =============================================================================
# Knowledge base
# =============================================================================


def _stable_doc_id(document: Document) -> str:
    """Derive a deterministic id from a document's source + content.

    Lets re-ingesting identical content be idempotent without callers having to
    manage ids themselves.
    """
    basis = f"{document.source}\x00{document.last_updated.isoformat()}\x00{document.content}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def _embed_text(document: Document) -> str:
    """Text used to embed a document.

    The source name is included so provenance terms contribute to retrieval,
    while the body dominates the signal.
    """
    return f"{document.source}\n{document.content}"


class KnowledgeBase:
    """Ingests tagged documents and retrieves them by similarity.

    Both the embedding and vector-store backends are injectable for tests and
    swapping; when omitted they are built from :class:`app.config.Settings`.
    """

    def __init__(
        self,
        *,
        embedding: Embedding | None = None,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings if settings is not None else get_settings()
        self._embedding: Embedding = embedding if embedding is not None else make_embedding(resolved)
        self._store: VectorStore = store if store is not None else make_vector_store(resolved)

    def ingest(self, documents: Iterable[Document | dict]) -> list[str]:
        """Ingest documents into the vector store, returning their ids.

        Each item may be a :class:`Document` or a mapping (validated into one).
        Because ``Document`` requires ``source`` and ``last_updated``, every
        ingested document is guaranteed to be tagged (Property 24). A document
        without an explicit ``doc_id`` is assigned a deterministic id derived
        from its source + content, making re-ingestion idempotent.
        """
        ingested_ids: list[str] = []
        for item in documents:
            document = item if isinstance(item, Document) else Document.model_validate(item)
            doc_id = document.doc_id or _stable_doc_id(document)
            # Persist the resolved id back onto the stored document so retrieval
            # results expose a stable identifier.
            stored = document if document.doc_id == doc_id else document.model_copy(
                update={"doc_id": doc_id}
            )
            vector = self._embedding.embed(_embed_text(stored))
            self._store.add(doc_id, vector, stored)
            ingested_ids.append(doc_id)
        return ingested_ids

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the ``k`` most similar documents to ``query`` with scores.

        Results are ordered by descending similarity. An empty/whitespace query
        or a non-positive ``k`` yields an empty list.
        """
        if not query or not query.strip() or k <= 0:
            return []
        query_vector = self._embedding.embed(query)
        hits = self._store.query(query_vector, k)
        return [SearchResult(document=document, score=score) for document, score in hits]

    def clear(self) -> None:
        """Remove all ingested documents (primarily for tests)."""
        self._store.clear()

    def __len__(self) -> int:
        """Number of documents currently stored."""
        return len(self._store)


__all__ = [
    "Document",
    "SearchResult",
    "Embedding",
    "HashEmbedding",
    "VectorStore",
    "InMemoryVectorStore",
    "KnowledgeBase",
    "make_embedding",
    "make_vector_store",
]
