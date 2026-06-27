"""Tests for the RAG knowledge base ingestion + tagging + retrieval (Task 15.1).

Covers:
- Documents are always tagged with a source name and last-updated date
  (Property 24 surface; Requirement 11.5).
- ``ingest`` stores documents and ``search`` returns them with similarity
  scores ordered by relevance (foundation for Requirement 11.1 retrieval).
- The default backend is the deterministic, no-network in-memory/hash combo
  (zero-infra default; Requirement 20.3).

Validates: Requirements 11.5, 20.3
"""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.config import Settings
from app.rag.knowledge_base import (
    Document,
    HashEmbedding,
    InMemoryVectorStore,
    KnowledgeBase,
    SearchResult,
    make_embedding,
    make_vector_store,
)


def _kb() -> KnowledgeBase:
    """A knowledge base on the deterministic default backends."""
    return KnowledgeBase(embedding=HashEmbedding(dim=128), store=InMemoryVectorStore())


# --- Tagging (Property 24 / Requirement 11.5) -------------------------------


def test_document_requires_source_and_last_updated() -> None:
    # Missing source.
    with pytest.raises(ValidationError):
        Document(content="visa rules", last_updated=date(2024, 1, 1))  # type: ignore[call-arg]
    # Missing last_updated.
    with pytest.raises(ValidationError):
        Document(content="visa rules", source="gov.example")  # type: ignore[call-arg]


def test_ingested_documents_carry_source_and_last_updated() -> None:
    kb = _kb()
    kb.ingest(
        [
            Document(
                content="Citizens of country X need a visa for country Y.",
                source="travel.gov.example",
                last_updated=date(2024, 6, 1),
            )
        ]
    )
    results = kb.search("visa for country Y", k=1)
    assert len(results) == 1
    doc = results[0].document
    assert doc.source == "travel.gov.example"
    assert doc.last_updated == date(2024, 6, 1)


def test_ingest_accepts_mapping_and_validates_tags() -> None:
    kb = _kb()
    ids = kb.ingest(
        [{"content": "Safety advisory text", "source": "safety.example", "last_updated": "2023-12-31"}]
    )
    assert len(ids) == 1
    # A mapping missing the required tags is rejected at ingest time.
    with pytest.raises(ValidationError):
        kb.ingest([{"content": "no tags here"}])


# --- Ingestion + retrieval primitive ----------------------------------------


def test_search_ranks_relevant_document_highest() -> None:
    kb = _kb()
    kb.ingest(
        [
            Document(
                content="Tokyo offers world-class sushi and ramen food experiences.",
                source="food.example",
                last_updated=date(2024, 1, 1),
            ),
            Document(
                content="Visa requirements and passport rules for entering Japan.",
                source="visa.example",
                last_updated=date(2024, 2, 1),
            ),
        ]
    )
    results = kb.search("passport and visa rules for Japan", k=2)
    assert len(results) == 2
    # The visa document should outrank the food document for this query.
    assert results[0].document.source == "visa.example"
    # Scores are sorted descending and within [0, 1].
    assert results[0].score >= results[1].score
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_returns_scores_in_unit_range_and_descending() -> None:
    kb = _kb()
    kb.ingest(
        [
            Document(content=f"document number {i} about travel topic {i}", source=f"s{i}",
                     last_updated=date(2024, 1, 1))
            for i in range(5)
        ]
    )
    results = kb.search("travel topic 3", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_search_respects_k_and_empty_query() -> None:
    kb = _kb()
    kb.ingest(
        [
            Document(content="alpha beta gamma", source="a", last_updated=date(2024, 1, 1)),
            Document(content="delta epsilon zeta", source="b", last_updated=date(2024, 1, 1)),
        ]
    )
    assert len(kb.search("alpha", k=1)) == 1
    assert kb.search("   ", k=3) == []
    assert kb.search("alpha", k=0) == []


def test_reingesting_same_document_is_idempotent() -> None:
    kb = _kb()
    doc = Document(content="same content", source="same.source", last_updated=date(2024, 1, 1))
    first = kb.ingest([doc])
    second = kb.ingest([doc])
    assert first == second  # stable deterministic id
    assert len(kb) == 1  # not duplicated


def test_search_results_are_search_result_models() -> None:
    kb = _kb()
    kb.ingest([Document(content="hello world", source="s", last_updated=date(2024, 1, 1))])
    results = kb.search("hello", k=1)
    assert isinstance(results[0], SearchResult)


# --- Default backends are the zero-infra deterministic combo (Req 20.3) -----


def test_default_backends_are_memory_and_hash() -> None:
    settings = Settings(_env_file=None)
    assert isinstance(make_embedding(settings), HashEmbedding)
    assert isinstance(make_vector_store(settings), InMemoryVectorStore)


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    emb = HashEmbedding(dim=64)
    v1 = emb.embed("repeatable text")
    v2 = emb.embed("repeatable text")
    assert v1 == v2
    norm = sum(c * c for c in v1) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


# --- Property-based test: ingested documents are always tagged --------------
# Feature: wayfinder-travel-planner, Property 24
# Validates: Requirements 11.5

_text = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")


@hyp_settings(max_examples=150)
@given(
    content=_text,
    source=_text,
    year=st.integers(min_value=1990, max_value=2035),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),
)
def test_property_ingested_documents_are_tagged(
    content: str, source: str, year: int, month: int, day: int
) -> None:
    """For any ingested document, retrieval exposes a source name and date."""
    kb = KnowledgeBase(embedding=HashEmbedding(dim=64), store=InMemoryVectorStore())
    updated = date(year, month, day)
    kb.ingest([Document(content=content, source=source, last_updated=updated)])
    results = kb.search(content, k=1)
    assert len(results) == 1
    doc = results[0].document
    assert doc.source == source and doc.source != ""
    assert doc.last_updated == updated
