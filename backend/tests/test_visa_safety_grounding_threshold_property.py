"""Property-based test for the visa/safety grounding threshold (Task 15.5).

Feature: wayfinder-travel-planner, Property 22: Visa/safety grounding threshold.

*For any* visa or safety query, when the RAG service returns an answer it SHALL
have retrieved at least one document with similarity at or above 0.7 and SHALL
construct the answer using only claims present in retrieved documents; when no
document reaches similarity 0.7, it SHALL return a no-verified-information
response and SHALL NOT invoke the language model to generate an answer.

Validates: Requirements 11.1, 11.3, 19.3

The property is exercised against :meth:`RAGService.answer_visa_safety` with a
controlled :class:`StubRetriever` (so similarity scores can be set exactly) and
a :class:`SpyLLM` that records whether the language model was invoked. Each
generated retrieval result set carries scores deliberately drawn to straddle
the 0.7 threshold so every run exercises both the grounded and the
no-verified-information branch boundary.
"""

from __future__ import annotations

from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st

from app.config import Settings
from app.llm.base import LLMResponse
from app.rag.knowledge_base import Document, SearchResult
from app.rag.rag_service import GroundedAnswer, NoVerifiedInfo, RAGService

THRESHOLD = 0.7


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

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return list(self._results)


def _settings() -> Settings:
    # rag_similarity_threshold defaults to 0.7.
    return Settings(_env_file=None)


# --- Generators -------------------------------------------------------------

# Scores are drawn to straddle the 0.7 gate: a uniform spread across the full
# [0, 1] range, plus values sampled right at and adjacent to the boundary so the
# >= 0.7 edge (inclusive) is exercised on most runs.
_score = st.one_of(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, 0.5, 0.69, 0.6999999, 0.7, 0.70000001, 0.71, 0.9, 1.0]),
)


@st.composite
def _result_sets(draw: st.DrawFn) -> list[SearchResult]:
    """Draw a retrieval result set with varied, boundary-straddling scores.

    Each document gets a distinct source name (``src{i}``) so the cited sources
    can be matched back to the documents that cleared the threshold.
    """
    scores = draw(st.lists(_score, min_size=0, max_size=8))
    return [
        SearchResult(
            document=Document(
                content=f"Visa/safety fact {index}.",
                source=f"src{index}",
                last_updated=date(2024, 1, 1),
            ),
            score=score,
        )
        for index, score in enumerate(scores)
    ]


# --- Property 22: Visa/safety grounding threshold ---------------------------
# Feature: wayfinder-travel-planner, Property 22
# Validates: Requirements 11.1, 11.3, 19.3


@settings(max_examples=200)
@given(results=_result_sets())
def test_visa_safety_grounding_threshold(results: list[SearchResult]) -> None:
    """The 0.7 gate decides grounded-answer vs no-verified-info and LLM use.

    For any retrieval result set:

    * If at least one document has similarity ``>= 0.7``, ``answer_visa_safety``
      returns a :class:`GroundedAnswer` grounded **only** in the ``>= 0.7``
      documents, the LLM is invoked exactly once, and only the clearing
      documents' content appears in the grounding prompt (Req 11.1, 11.3).
    * If no document reaches ``0.7``, it returns :class:`NoVerifiedInfo` and the
      LLM is **not** invoked at all (Req 11.3, 19.3).
    """
    retriever = StubRetriever(results)
    spy = SpyLLM()
    service = RAGService(knowledge_base=retriever, llm_provider=spy, settings=_settings())

    clearing = [r for r in results if r.score >= THRESHOLD]
    answer = service.answer_visa_safety("Do I need a visa or vaccination for this country?")

    if clearing:
        # Grounded branch: an answer is produced from only the clearing docs.
        assert isinstance(answer, GroundedAnswer)
        assert answer.grounded is True

        # The LLM was invoked exactly once for synthesis.
        assert len(spy.calls) == 1
        prompt = spy.calls[0]["prompt"]
        assert isinstance(prompt, str)

        # Citations are exactly the threshold-clearing documents.
        clearing_sources = {r.document.source for r in clearing}
        cited_sources = {c.source for c in answer.citations}
        assert cited_sources == clearing_sources
        assert len(answer.citations) == len(clearing)

        # Every cited document cleared the threshold and only their content
        # (not the below-threshold docs') grounds the prompt.
        for citation in answer.citations:
            assert citation.score >= THRESHOLD
        for result in results:
            if result.score >= THRESHOLD:
                assert result.document.content in prompt
            else:
                assert result.document.content not in prompt
    else:
        # No-verified-info branch: declined, and the LLM is never invoked.
        assert isinstance(answer, NoVerifiedInfo)
        assert answer.grounded is False
        assert answer.threshold == THRESHOLD
        assert spy.calls == []
