"""Property-based test for citations and disclaimer on grounded answers.

Feature: wayfinder-travel-planner, Property 23: Citations and disclaimer on
grounded answers.

*For any* grounded visa or safety answer, each cited document SHALL include its
source name and last-updated date, and the response SHALL include a disclaimer
to verify with official sources; *for any* guide answer, at least one source
citation SHALL be present.

Validates: Requirements 11.2, 11.4

This test drives the already-implemented RAG service (``app/rag/rag_service.py``,
tasks 15.2-15.4) through a controlled in-memory retriever so similarity scores
are deterministic. Generators constrain to the input space that actually
produces a grounded answer:

* visa/safety answers are gated by the 0.7 threshold, so grounding documents
  are generated with scores in ``[0.7, 1.0]``;
* guide answers are not threshold-gated but require a positive similarity, so
  guide documents are generated with scores in ``(0.0, 1.0]``.
"""

from __future__ import annotations

from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st

from app.config import Settings
from app.llm.base import LLMResponse
from app.rag.knowledge_base import Document, SearchResult
from app.rag.rag_service import (
    VERIFY_SOURCES_DISCLAIMER,
    GroundedAnswer,
    GuideAnswer,
    RAGService,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubLLM:
    """An ``LLMProvider`` that returns fixed narration without any I/O."""

    model_name = "stub-llm"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return LLMResponse(text="grounded narration", model=self.model_name)


class StubRetriever:
    """Returns a fixed list of ``SearchResult`` regardless of the query."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return list(self._results)


def _settings() -> Settings:
    # rag_similarity_threshold defaults to 0.7.
    return Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Generators (constrained to the grounded input space)
# ---------------------------------------------------------------------------

# Source names and content always have at least one non-whitespace character so
# the rendered citation line is meaningful and Document's min_length holds.
_nonblank_text = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
)

# A non-blank query so retrieval is not short-circuited by the blank-query guard.
_query = _nonblank_text

_last_updated = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))


def _search_result(score: st.SearchStrategy[float]) -> st.SearchStrategy[SearchResult]:
    return st.builds(
        lambda content, source, last_updated, sc: SearchResult(
            document=Document(content=content, source=source, last_updated=last_updated),
            score=sc,
        ),
        content=_nonblank_text,
        source=_nonblank_text,
        last_updated=_last_updated,
        sc=score,
    )


# Visa/safety grounding: at least one document at or above the 0.7 threshold.
_grounding_results = st.lists(
    _search_result(st.floats(min_value=0.7, max_value=1.0)),
    min_size=1,
    max_size=5,
)

# Guide content: at least one document with a positive similarity score.
_guide_results = st.lists(
    _search_result(st.floats(min_value=0.01, max_value=1.0)),
    min_size=1,
    max_size=5,
)


# ---------------------------------------------------------------------------
# Property 23: Citations and disclaimer on grounded answers
# Feature: wayfinder-travel-planner, Property 23
# Validates: Requirements 11.2, 11.4
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(query=_query, results=_grounding_results)
def test_grounded_visa_safety_answer_carries_citations_and_disclaimer(
    query: str, results: list[SearchResult]
) -> None:
    """Every grounded visa/safety answer cites source+date and warns to verify.

    Feature: wayfinder-travel-planner, Property 23
    Validates: Requirements 11.2
    """
    service = RAGService(
        knowledge_base=StubRetriever(results),
        llm_provider=StubLLM(),
        settings=_settings(),
    )

    result = service.answer_visa_safety(query)

    # A document cleared the 0.7 threshold, so the answer must be grounded.
    assert isinstance(result, GroundedAnswer)

    # Every cited document carries a source name and a last-updated date that
    # trace back to a real grounding document, and the rendered citation line
    # surfaces both (Requirement 11.2).
    input_sources = {r.document.source for r in results}
    input_dates = {r.document.last_updated.isoformat() for r in results}
    assert len(result.citations) >= 1
    for citation in result.citations:
        assert citation.source
        assert citation.source in input_sources
        assert citation.last_updated in input_dates
        # last_updated is a valid ISO date string.
        date.fromisoformat(citation.last_updated)
        assert citation.citation_line == (
            f"{citation.source} (last updated {citation.last_updated})"
        )

    # The response appends the verify-with-official-sources disclaimer (11.2).
    assert result.disclaimer == VERIFY_SOURCES_DISCLAIMER
    assert "verify" in result.disclaimer.lower()
    assert "official sources" in result.disclaimer.lower()


@settings(max_examples=200, deadline=None)
@given(query=_query, results=_guide_results)
def test_grounded_guide_answer_has_at_least_one_citation(
    query: str, results: list[SearchResult]
) -> None:
    """Every grounded guide answer carries at least one source citation.

    Feature: wayfinder-travel-planner, Property 23
    Validates: Requirements 11.4
    """
    service = RAGService(
        knowledge_base=StubRetriever(results),
        llm_provider=StubLLM(),
        settings=_settings(),
    )

    result = service.answer_guide(query)

    assert isinstance(result, GuideAnswer)
    # At least one source citation per answer (Requirement 11.4), each carrying
    # a source name and a last-updated date.
    assert len(result.citations) >= 1
    for citation in result.citations:
        assert citation.source
        date.fromisoformat(citation.last_updated)
        assert citation.citation_line == (
            f"{citation.source} (last updated {citation.last_updated})"
        )
