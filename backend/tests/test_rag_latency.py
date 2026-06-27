"""Unit test for the RAG visa/safety latency budget (Task 15.8).

Requirement 11.6: WHEN the RAG_Service answers a visa or safety query, THE
RAG_Service SHALL return the response within 5 seconds of receiving the query.

This test asserts the budget two ways for both result shapes
(:class:`GroundedAnswer` and :class:`NoVerifiedInfo`):

* the **measured wall-clock time** around ``answer_visa_safety`` stays under
  the 5-second budget, and
* the ``elapsed_seconds`` the service surfaces on its result is within the
  budget (and ``within_latency_budget`` reports ``True``).

The realistic zero-key :class:`MockLLMProvider` is used (rather than a no-op
spy) so the grounded path exercises real prompt construction + narration, while
a controlled :class:`StubRetriever` keeps the retrieval deterministic without
standing up an embedding/vector store.

Validates: Requirements 11.6
"""

from __future__ import annotations

import time
from datetime import date

from app.config import Settings
from app.llm.base import MockLLMProvider
from app.rag.knowledge_base import Document, SearchResult
from app.rag.rag_service import (
    VISA_SAFETY_LATENCY_BUDGET_SECONDS,
    GroundedAnswer,
    NoVerifiedInfo,
    RAGService,
)


class StubRetriever:
    """A retriever returning a fixed list of ``SearchResult`` for any query."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return list(self._results)


def _doc(content: str, source: str = "gov.example") -> Document:
    return Document(content=content, source=source, last_updated=date(2024, 1, 1))


def _settings() -> Settings:
    # rag_similarity_threshold defaults to 0.7.
    return Settings(_env_file=None)


def _service(results: list[SearchResult]) -> RAGService:
    return RAGService(
        knowledge_base=StubRetriever(results),
        llm_provider=MockLLMProvider(),
        settings=_settings(),
    )


def test_grounded_visa_safety_answer_returns_within_5_seconds() -> None:
    # A document above the 0.7 threshold -> grounded answer (LLM is invoked).
    service = _service(
        [SearchResult(document=_doc("Citizens of X are visa-free for 90 days."), score=0.88)]
    )

    started = time.perf_counter()
    result = service.answer_visa_safety("Do I need a visa for country X?")
    measured = time.perf_counter() - started

    assert isinstance(result, GroundedAnswer)
    # Wall-clock latency is within the 5-second budget (Requirement 11.6).
    assert measured < VISA_SAFETY_LATENCY_BUDGET_SECONDS
    # The latency the service reports is also within budget.
    assert result.elapsed_seconds <= VISA_SAFETY_LATENCY_BUDGET_SECONDS
    assert result.within_latency_budget is True


def test_no_verified_info_response_returns_within_5_seconds() -> None:
    # Nothing clears the 0.7 threshold -> NoVerifiedInfo (LLM not invoked).
    service = _service([SearchResult(document=_doc("weakly related note."), score=0.3)])

    started = time.perf_counter()
    result = service.answer_visa_safety("Do I need a visa for country Z?")
    measured = time.perf_counter() - started

    assert isinstance(result, NoVerifiedInfo)
    assert measured < VISA_SAFETY_LATENCY_BUDGET_SECONDS
    assert result.elapsed_seconds <= VISA_SAFETY_LATENCY_BUDGET_SECONDS
    assert result.within_latency_budget is True
