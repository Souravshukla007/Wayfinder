"""Tests for grounded visa/safety answering (Task 15.2).

Focus of this task: the retrieval-threshold *gate* and grounded answering vs
``NoVerifiedInfo``.

(a) When a document at similarity >= 0.7 exists, a grounded answer is returned
    and the (mock) LLM is invoked **only** with the retrieved document content.
(b) When nothing clears 0.7, ``NoVerifiedInfo`` is returned and the LLM is
    **not** invoked at all (asserted via a spy provider).

Validates: Requirements 11.1, 11.3, 19.3
"""

from __future__ import annotations

from datetime import date

from app.config import Settings
from app.llm.base import LLMResponse
from app.rag.knowledge_base import Document, SearchResult
from app.rag.rag_service import (
    GroundedAnswer,
    NoVerifiedInfo,
    RAGService,
)


# --- Test doubles -----------------------------------------------------------


class SpyLLM:
    """An ``LLMProvider`` that records every call and the prompts it received."""

    model_name = "spy-llm"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system})
        return LLMResponse(text="grounded narration", model=self.model_name)


class StubRetriever:
    """A retriever returning a fixed list of ``SearchResult`` regardless of query."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        return list(self._results)


def _doc(content: str, source: str = "gov.example") -> Document:
    return Document(content=content, source=source, last_updated=date(2024, 1, 1))


def _settings() -> Settings:
    # rag_similarity_threshold defaults to 0.7.
    return Settings(_env_file=None)


# --- (a) Grounded answer when a >= 0.7 document exists -----------------------


def test_returns_grounded_answer_when_document_clears_threshold() -> None:
    grounding_text = "Citizens of country X do not need a visa for stays under 90 days."
    retriever = StubRetriever([SearchResult(document=_doc(grounding_text), score=0.82)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_visa_safety("Do I need a visa for country X?")

    assert isinstance(result, GroundedAnswer)
    assert result.grounded is True
    assert result.answer == "grounded narration"
    # The grounding document is cited with its provenance.
    assert len(result.citations) == 1
    assert result.citations[0].source == "gov.example"
    assert result.citations[0].last_updated == "2024-01-01"
    assert result.citations[0].score == 0.82


def test_llm_invoked_only_with_retrieved_content() -> None:
    grounding_text = "Yellow fever vaccination is required for entry to country Y."
    retriever = StubRetriever([SearchResult(document=_doc(grounding_text), score=0.75)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    service.answer_visa_safety("Vaccination rules for country Y?")

    assert len(spy.calls) == 1
    prompt = spy.calls[0]["prompt"]
    assert isinstance(prompt, str)
    # The retrieved document text is present in the prompt (the answer is
    # grounded in it) and the provenance tag is included for grounding.
    assert grounding_text in prompt
    assert "gov.example" in prompt


def test_only_threshold_clearing_documents_ground_the_answer() -> None:
    high = SearchResult(document=_doc("Visa-free for 30 days.", source="high.src"), score=0.91)
    low = SearchResult(document=_doc("Unrelated low-similarity note.", source="low.src"), score=0.4)
    retriever = StubRetriever([high, low])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_visa_safety("Visa rules?")

    assert isinstance(result, GroundedAnswer)
    cited_sources = {c.source for c in result.citations}
    assert cited_sources == {"high.src"}
    # The below-threshold document must not leak into the grounding prompt.
    assert "Unrelated low-similarity note." not in spy.calls[0]["prompt"]


def test_boundary_score_exactly_threshold_is_grounded() -> None:
    retriever = StubRetriever([SearchResult(document=_doc("Exactly at threshold."), score=0.7)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_visa_safety("edge case query")

    assert isinstance(result, GroundedAnswer)
    assert len(spy.calls) == 1


# --- (b) NoVerifiedInfo + LLM NOT invoked when nothing clears 0.7 ------------


def test_no_verified_info_when_nothing_clears_threshold() -> None:
    retriever = StubRetriever(
        [
            SearchResult(document=_doc("weakly related"), score=0.69),
            SearchResult(document=_doc("also weak"), score=0.3),
        ]
    )
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_visa_safety("Do I need a visa for country Z?")

    assert isinstance(result, NoVerifiedInfo)
    assert result.grounded is False
    assert result.threshold == 0.7
    # CRITICAL: the LLM must not be invoked at all.
    assert spy.calls == []


def test_no_verified_info_when_no_documents_retrieved() -> None:
    retriever = StubRetriever([])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_visa_safety("anything")

    assert isinstance(result, NoVerifiedInfo)
    assert spy.calls == []


def test_retrieval_error_fails_safe_to_no_verified_info() -> None:
    class BrokenRetriever:
        def search(self, query: str, k: int = 5) -> list[SearchResult]:
            raise RuntimeError("vector store down")

    spy = SpyLLM()
    service = RAGService(knowledge_base=BrokenRetriever(), llm_provider=spy, settings=_settings())

    result = service.answer_visa_safety("visa rules")

    assert isinstance(result, NoVerifiedInfo)
    assert spy.calls == []


def test_blank_query_returns_no_verified_info_without_calling_llm() -> None:
    retriever = StubRetriever([SearchResult(document=_doc("something"), score=0.95)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_visa_safety("   ")

    assert isinstance(result, NoVerifiedInfo)
    assert spy.calls == []


# --- Task 15.3: citations, disclaimer, and latency budget -------------------
# Verifies that grounded answers carry per-citation source name + last-updated
# date, append the verify-with-official-sources disclaimer, and surface the
# measured elapsed time within the 5-second budget.
# Validates: Requirements 11.2, 11.6


def test_grounded_answer_appends_verify_sources_disclaimer() -> None:
    from app.rag.rag_service import VERIFY_SOURCES_DISCLAIMER

    retriever = StubRetriever([SearchResult(document=_doc("Visa-free for 90 days."), score=0.9)])
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_visa_safety("visa rules?")

    assert isinstance(result, GroundedAnswer)
    assert result.disclaimer == VERIFY_SOURCES_DISCLAIMER
    # The disclaimer must direct the user to verify with official sources.
    assert "verify" in result.disclaimer.lower()
    assert "official sources" in result.disclaimer.lower()


def test_each_citation_includes_source_name_and_last_updated_date() -> None:
    retriever = StubRetriever(
        [SearchResult(document=_doc("Passport valid 6 months.", source="embassy.gov"), score=0.88)]
    )
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_visa_safety("passport rules?")

    assert isinstance(result, GroundedAnswer)
    citation = result.citations[0]
    assert citation.source == "embassy.gov"
    assert citation.last_updated == "2024-01-01"
    # The rendered citation line surfaces both source name and last-updated date.
    assert citation.citation_line == "embassy.gov (last updated 2024-01-01)"
    assert result.citation_lines == ["embassy.gov (last updated 2024-01-01)"]


def test_grounded_answer_responds_within_latency_budget() -> None:
    from app.rag.rag_service import VISA_SAFETY_LATENCY_BUDGET_SECONDS

    retriever = StubRetriever([SearchResult(document=_doc("Entry allowed."), score=0.95)])
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_visa_safety("entry rules?")

    assert isinstance(result, GroundedAnswer)
    assert result.elapsed_seconds >= 0.0
    assert result.elapsed_seconds <= VISA_SAFETY_LATENCY_BUDGET_SECONDS
    assert result.within_latency_budget is True


def test_no_verified_info_also_surfaces_elapsed_seconds() -> None:
    retriever = StubRetriever([SearchResult(document=_doc("weak"), score=0.2)])
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_visa_safety("unknown country visa?")

    assert isinstance(result, NoVerifiedInfo)
    assert result.elapsed_seconds >= 0.0
    assert result.within_latency_budget is True


# --- Task 15.4: grounded guide answering ------------------------------------
# Verifies that guide queries (neighborhood/photography/food/themed-interest/
# transport) retrieve and present KB content with >= 1 source citation per
# answer, ground narration only in retrieved documents, and decline (without
# calling the LLM) when nothing relevant is retrieved.
# Validates: Requirements 11.4


def test_guide_answer_includes_at_least_one_citation() -> None:
    from app.rag.rag_service import GuideAnswer

    retriever = StubRetriever(
        [
            SearchResult(
                document=_doc("Shimokitazawa is full of vintage shops.", source="guide.example"),
                score=0.83,
            )
        ]
    )
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_guide("Best neighborhoods in Tokyo?", topic="neighborhood")

    assert isinstance(result, GuideAnswer)
    assert result.grounded is True
    assert result.topic == "neighborhood"
    # At least one source citation per answer (Requirement 11.4).
    assert len(result.citations) >= 1
    assert result.citations[0].source == "guide.example"
    assert result.citations[0].last_updated == "2024-01-01"
    assert result.citation_lines == ["guide.example (last updated 2024-01-01)"]


def test_guide_answer_grounds_llm_only_in_retrieved_content() -> None:
    grounding_text = "The fish market opens at dawn for the freshest sushi."
    retriever = StubRetriever([SearchResult(document=_doc(grounding_text), score=0.9)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    service.answer_guide("Where to eat sushi?", topic="food")

    assert len(spy.calls) == 1
    prompt = spy.calls[0]["prompt"]
    assert isinstance(prompt, str)
    assert grounding_text in prompt
    # The guide topic is carried into the prompt to keep narration on-subject.
    assert "food" in prompt


def test_guide_answer_does_not_append_visa_safety_disclaimer() -> None:
    from app.rag.rag_service import GuideAnswer

    retriever = StubRetriever([SearchResult(document=_doc("Great photo spots at sunset."), score=0.8)])
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_guide("Best photography spots?", topic="photography")

    assert isinstance(result, GuideAnswer)
    # The disclaimer obligation (11.2) is specific to visa/safety; guide answers
    # must not carry a disclaimer field.
    assert not hasattr(result, "disclaimer")


def test_guide_answer_works_without_an_explicit_topic() -> None:
    from app.rag.rag_service import GuideAnswer

    retriever = StubRetriever([SearchResult(document=_doc("Trains run every 5 minutes."), score=0.77)])
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_guide("How does the metro work?")

    assert isinstance(result, GuideAnswer)
    assert result.topic is None
    assert len(result.citations) >= 1


def test_guide_answer_is_not_gated_by_visa_safety_threshold() -> None:
    from app.rag.rag_service import GuideAnswer

    # Score below the 0.7 visa/safety threshold still grounds a guide answer.
    retriever = StubRetriever([SearchResult(document=_doc("Quiet riverside walk."), score=0.45)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_guide("Where can I take a quiet walk?", topic="themed-interest")

    assert isinstance(result, GuideAnswer)
    assert len(result.citations) == 1
    assert len(spy.calls) == 1


def test_guide_answer_cites_multiple_retrieved_documents() -> None:
    from app.rag.rag_service import GuideAnswer

    retriever = StubRetriever(
        [
            SearchResult(document=_doc("Spot A is scenic.", source="a.src"), score=0.9),
            SearchResult(document=_doc("Spot B has murals.", source="b.src"), score=0.6),
        ]
    )
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_guide("Photo walk ideas?", topic="photography")

    assert isinstance(result, GuideAnswer)
    cited = {c.source for c in result.citations}
    assert cited == {"a.src", "b.src"}
    # Ordered by descending similarity.
    assert result.citations[0].source == "a.src"


def test_guide_returns_no_content_when_nothing_retrieved() -> None:
    from app.rag.rag_service import NoGuideContent

    retriever = StubRetriever([])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_guide("Neighborhoods in an unknown city?", topic="neighborhood")

    assert isinstance(result, NoGuideContent)
    assert result.grounded is False
    # No content to cite -> the LLM is not invoked.
    assert spy.calls == []


def test_guide_returns_no_content_when_only_zero_score_matches() -> None:
    from app.rag.rag_service import NoGuideContent

    retriever = StubRetriever([SearchResult(document=_doc("irrelevant"), score=0.0)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_guide("anything")

    assert isinstance(result, NoGuideContent)
    assert spy.calls == []


def test_guide_blank_query_returns_no_content_without_calling_llm() -> None:
    from app.rag.rag_service import NoGuideContent

    retriever = StubRetriever([SearchResult(document=_doc("something"), score=0.95)])
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    result = service.answer_guide("   ", topic="food")

    assert isinstance(result, NoGuideContent)
    assert spy.calls == []


def test_guide_retrieval_error_fails_safe_to_no_content() -> None:
    from app.rag.rag_service import NoGuideContent

    class BrokenRetriever:
        def search(self, query: str, k: int = 5) -> list[SearchResult]:
            raise RuntimeError("vector store down")

    spy = SpyLLM()
    service = RAGService(knowledge_base=BrokenRetriever(), llm_provider=spy, settings=_settings())

    result = service.answer_guide("transport options?", topic="transport")

    assert isinstance(result, NoGuideContent)
    assert spy.calls == []


def test_guide_unknown_topic_is_rejected() -> None:
    import pytest

    retriever = StubRetriever([SearchResult(document=_doc("content"), score=0.9)])
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    with pytest.raises(ValueError):
        service.answer_guide("query", topic="not-a-real-topic")


def test_guide_answer_surfaces_elapsed_seconds() -> None:
    from app.rag.rag_service import GuideAnswer

    retriever = StubRetriever([SearchResult(document=_doc("Open-air food stalls."), score=0.85)])
    service = RAGService(knowledge_base=retriever, llm_provider=SpyLLM(), settings=_settings())

    result = service.answer_guide("street food?", topic="food")

    assert isinstance(result, GuideAnswer)
    assert result.elapsed_seconds >= 0.0
