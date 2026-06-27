"""Grounded visa/safety answering over the RAG knowledge base (Task 15.2).

This module turns the retrieval foundation built in Task 15.1
(:mod:`app.rag.knowledge_base`) into **grounded** visa/safety answers. Its one
defining rule (Requirements 11.1, 11.3, 19.3 / Property 22):

* A visa/safety query is answered **only** when retrieval surfaces at least one
  document whose similarity score is **at or above** the configured threshold
  (``Settings.rag_similarity_threshold``, default ``0.7``). The answer is
  constructed using **only** claims present in those retrieved documents.
* When **no** retrieved document reaches the threshold, the service returns a
  structured :class:`NoVerifiedInfo` result and **does not call the LLM at
  all** — preventing ungrounded generation of visa/safety facts.

Scope of this task: retrieval-threshold *gating* and grounded answering vs
``NoVerifiedInfo``. The structure is deliberately left open so the sibling
tasks can extend it without reshaping the core:

* Task 15.3 (this task) adds the per-citation ``source`` + ``last_updated``
  rendering (via :attr:`Citation.citation_line`), the
  verify-with-official-sources disclaimer (:data:`VERIFY_SOURCES_DISCLAIMER`,
  appended to every :class:`GroundedAnswer`), and the 5-second latency budget
  (:data:`VISA_SAFETY_LATENCY_BUDGET_SECONDS`). The measured wall-clock
  ``elapsed_seconds`` is surfaced on both result types so the budget is
  testable (Requirements 11.2, 11.6).
* Task 15.4 adds ``answer_guide`` for neighbourhood/photography/food/themed/
  transport content, reusing the same retrieval + citation machinery. Every
  guide answer carries at least one source citation (Requirement 11.4); guide
  answering is not gated by the 0.7 visa/safety threshold and does not append
  the visa/safety disclaimer.

The LLM (resolved via :func:`app.llm.base.get_llm_provider`, mock-first) is used
for **narration/synthesis only**: it is prompted strictly with the retrieved
document text and never invoked when nothing clears the threshold.

The module performs no I/O at import time.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.llm.base import LLMProvider, get_llm_provider
from app.rag.knowledge_base import KnowledgeBase, SearchResult

# =============================================================================
# Task 15.3 constants: disclaimer + latency budget
# =============================================================================

# Verify-with-official-sources disclaimer appended to every grounded visa/safety
# answer (Requirement 11.2). The wording explicitly states the information may be
# outdated and directs the user to confirm with official sources before relying
# on it.
VERIFY_SOURCES_DISCLAIMER = (
    "Disclaimer: This information may be outdated. Visa and safety rules change "
    "frequently, so verify these details with official sources (such as the "
    "destination country's embassy, consulate, or official government website) "
    "before you travel."
)

# Latency budget for answering a visa/safety query (Requirement 11.6). Mirrors
# the memory layer's budget constants (e.g. ``LOAD_CONTEXT_BUDGET_SECONDS``):
# the operation is structured to respond within this many seconds, and the
# measured ``elapsed_seconds`` is surfaced on the result so the budget can be
# asserted in tests and monitored in production.
VISA_SAFETY_LATENCY_BUDGET_SECONDS = 5.0

# =============================================================================
# Task 15.4 constants: guide content categories
# =============================================================================

# The guide content categories supported by :meth:`RAGService.answer_guide`
# (Requirement 11.4). A guide query may optionally be tagged with one of these
# topics; the value is surfaced on the answer and used to bias retrieval, but it
# is not required (an untagged guide query is still answered from the KB).
GUIDE_TOPICS = (
    "neighborhood",
    "photography",
    "food",
    "themed-interest",
    "transport",
)

# =============================================================================
# Retrieval port (duck-typed so tests can inject a controlled retriever)
# =============================================================================


@runtime_checkable
class Retriever(Protocol):
    """The retrieval surface the RAG service depends on.

    :class:`app.rag.knowledge_base.KnowledgeBase` satisfies this structurally.
    Depending on the protocol (rather than the concrete class) lets the
    threshold-gating logic be exercised with deterministic, controlled scores
    in tests without standing up an embedding/vector store.
    """

    def search(self, query: str, k: int = ...) -> list[SearchResult]: ...


# =============================================================================
# Result models
# =============================================================================


class Citation(BaseModel):
    """Provenance for a single document that grounded an answer.

    Carries the document's ``source`` name and ``last_updated`` date (always
    present because :class:`app.rag.knowledge_base.Document` requires them) plus
    the similarity ``score`` that cleared the threshold. Task 15.3 renders these
    into user-facing citations; this task only needs them present as grounding
    evidence.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    last_updated: str
    score: float = Field(..., ge=0.0, le=1.0)
    snippet: str = Field(..., description="The grounding document text used for the answer.")

    @property
    def citation_line(self) -> str:
        """Render this citation as ``"<source> (last updated <date>)"``.

        Surfaces the document's source name and last-updated date together so a
        caller can present the provenance of each cited document directly in the
        answer payload (Requirement 11.2) without re-deriving the format.
        """
        return f"{self.source} (last updated {self.last_updated})"


class GroundedAnswer(BaseModel):
    """A visa/safety answer constructed only from retrieved documents.

    ``answer`` is narration synthesised by the LLM from the retrieved document
    text alone. ``citations`` lists every document that grounded the answer (all
    at or above the similarity threshold), each carrying its source name and
    last-updated date (Requirement 11.2). ``disclaimer`` is the
    verify-with-official-sources notice appended to **every** grounded answer
    (Requirement 11.2); it defaults to :data:`VERIFY_SOURCES_DISCLAIMER` and is
    always present. ``elapsed_seconds`` is the measured wall-clock time taken to
    produce the answer, surfaced so the 5-second latency budget
    (Requirement 11.6 / :data:`VISA_SAFETY_LATENCY_BUDGET_SECONDS`) is testable.
    ``grounded`` is always ``True`` and exists so callers can branch on the
    result type without ``isinstance``.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool = True
    query: str
    answer: str
    citations: list[Citation] = Field(..., min_length=1)
    disclaimer: str = VERIFY_SOURCES_DISCLAIMER
    elapsed_seconds: float = Field(default=0.0, ge=0.0)

    @property
    def citation_lines(self) -> list[str]:
        """Each cited document rendered as ``"<source> (last updated <date>)"``.

        Convenience accessor that surfaces the per-citation source name and
        last-updated date (Requirement 11.2) for direct presentation.
        """
        return [citation.citation_line for citation in self.citations]

    @property
    def within_latency_budget(self) -> bool:
        """Whether the answer was produced within the 5-second budget (11.6)."""
        return self.elapsed_seconds <= VISA_SAFETY_LATENCY_BUDGET_SECONDS


class NoVerifiedInfo(BaseModel):
    """Returned when no retrieved document reaches the similarity threshold.

    Signals that the service declined to answer rather than generating an
    ungrounded response. The LLM is **not** invoked when this is returned.
    ``elapsed_seconds`` is surfaced for the same latency-budget observability as
    :class:`GroundedAnswer`.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool = False
    query: str
    threshold: float
    message: str = (
        "No verified information is available for this query. "
        "No knowledge-base source met the similarity threshold required to "
        "ground a visa/safety answer."
    )
    elapsed_seconds: float = Field(default=0.0, ge=0.0)

    @property
    def within_latency_budget(self) -> bool:
        """Whether the response was produced within the 5-second budget (11.6)."""
        return self.elapsed_seconds <= VISA_SAFETY_LATENCY_BUDGET_SECONDS


VisaSafetyResult = GroundedAnswer | NoVerifiedInfo


class GuideAnswer(BaseModel):
    """A guide answer (neighborhood/photography/food/themed/transport).

    Produced for guide queries (Requirement 11.4). Like
    :class:`GroundedAnswer`, ``answer`` is narration synthesised by the LLM from
    the retrieved document text alone, and ``citations`` lists every document
    that grounded the answer — guaranteed non-empty so **every** guide answer
    carries at least one source citation (source name + last-updated date,
    Requirement 11.4). ``topic`` echoes the optional guide category when one was
    supplied. Unlike visa/safety answers, guide answers do **not** append the
    verify-with-official-sources disclaimer (that obligation, Requirement 11.2,
    is specific to visa/safety). ``elapsed_seconds`` is surfaced for the same
    observability as the visa/safety results.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool = True
    query: str
    topic: str | None = None
    answer: str
    citations: list[Citation] = Field(..., min_length=1)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)

    @property
    def citation_lines(self) -> list[str]:
        """Each cited document rendered as ``"<source> (last updated <date>)"``."""
        return [citation.citation_line for citation in self.citations]


class NoGuideContent(BaseModel):
    """Returned when retrieval surfaces no usable content for a guide query.

    A guide answer must carry at least one source citation (Requirement 11.4);
    when nothing is retrieved there is no content to present and no source to
    cite, so this structured result is returned instead of an uncited answer.
    The LLM is **not** invoked in this case.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool = False
    query: str
    topic: str | None = None
    message: str = (
        "No knowledge-base content is available for this guide query, "
        "so no answer could be grounded with a source citation."
    )
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


GuideResult = GuideAnswer | NoGuideContent


# =============================================================================
# Prompt construction (grounding)
# =============================================================================

_GROUNDING_SYSTEM = (
    "You are a travel information assistant. Answer the user's visa or safety "
    "question using ONLY the claims contained in the provided sources below. "
    "Do not add facts that are not supported by the sources. If the sources do "
    "not contain enough information, say so plainly."
)


def _format_sources(results: list[SearchResult]) -> str:
    """Render retrieved documents into a numbered, source-tagged context block.

    The returned text contains **only** retrieved document content (plus its
    provenance tag) so the grounding prompt cannot leak ungrounded material.
    """
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        document = result.document
        blocks.append(
            f"[{index}] source={document.source} "
            f"last_updated={document.last_updated.isoformat()}\n{document.content}"
        )
    return "\n\n".join(blocks)


def _build_grounding_prompt(query: str, results: list[SearchResult]) -> str:
    """Build the LLM prompt from the query and retrieved documents only."""
    return (
        f"Question: {query}\n\n"
        f"Sources:\n{_format_sources(results)}\n\n"
        "Answer using only the sources above:"
    )


_GUIDE_SYSTEM = (
    "You are a travel guide assistant. Answer the user's question about a "
    "destination's neighborhoods, photography spots, food, themed interests, or "
    "transport using ONLY the content contained in the provided sources below. "
    "Do not add facts that are not supported by the sources. If the sources do "
    "not contain enough information, say so plainly."
)


def _build_guide_prompt(query: str, topic: str | None, results: list[SearchResult]) -> str:
    """Build the guide LLM prompt from the query, optional topic, and docs only.

    Mirrors :func:`_build_grounding_prompt` but includes the guide topic (when
    supplied) so narration stays on-subject. The context block contains only
    retrieved document content, so the answer cannot be grounded in anything the
    KB did not surface.
    """
    topic_line = f"Guide topic: {topic}\n" if topic else ""
    return (
        f"{topic_line}Question: {query}\n\n"
        f"Sources:\n{_format_sources(results)}\n\n"
        "Answer using only the sources above:"
    )


# =============================================================================
# RAG service
# =============================================================================


class RAGService:
    """Answers visa/safety queries grounded strictly in retrieved documents.

    The retriever and LLM provider are injectable (for tests and swapping); when
    omitted they default to a :class:`KnowledgeBase` and the config-driven
    :func:`get_llm_provider` (mock-first). ``min_similarity`` defaults to
    ``Settings.rag_similarity_threshold`` (0.7).
    """

    def __init__(
        self,
        *,
        knowledge_base: Retriever | None = None,
        llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
        retrieve_k: int = 5,
    ) -> None:
        resolved = settings if settings is not None else get_settings()
        self._settings = resolved
        self._kb: Retriever = (
            knowledge_base if knowledge_base is not None else KnowledgeBase(settings=resolved)
        )
        # The LLM is resolved lazily-by-default but stored eagerly here; it is
        # only *invoked* when at least one document clears the threshold.
        self._llm: LLMProvider = (
            llm_provider if llm_provider is not None else get_llm_provider(resolved)
        )
        self._min_similarity: float = resolved.rag_similarity_threshold
        self._retrieve_k = retrieve_k

    @property
    def min_similarity(self) -> float:
        """The similarity threshold a document must reach to ground an answer."""
        return self._min_similarity

    def answer_visa_safety(self, query: str) -> VisaSafetyResult:
        """Answer a visa/safety query, grounded only in retrieved documents.

        Returns a :class:`GroundedAnswer` when at least one retrieved document
        has similarity ``>= min_similarity`` (default 0.7), constructing the
        answer using only those documents. Each citation carries the document's
        source name and last-updated date, and the verify-with-official-sources
        :data:`VERIFY_SOURCES_DISCLAIMER` is appended to the answer
        (Requirement 11.2). Otherwise returns :class:`NoVerifiedInfo`
        **without invoking the LLM**.

        The whole operation is timed with a monotonic clock and the measured
        ``elapsed_seconds`` is surfaced on the result so the 5-second latency
        budget (Requirement 11.6) can be asserted and monitored.

        Retrieval-backend errors fail safe to :class:`NoVerifiedInfo` so a
        broken store can never cause ungrounded visa/safety generation.
        """
        started = time.perf_counter()
        grounded = self._retrieve_grounding(query)
        if not grounded:
            # Threshold not met (or retrieval unavailable): decline to answer and
            # never call the LLM.
            return self._no_verified_info(query, elapsed_seconds=time.perf_counter() - started)

        # At least one document cleared the threshold -> ground an answer using
        # ONLY the retrieved document text. The LLM narrates the retrieved
        # claims; it is prompted strictly with retrieved content.
        prompt = _build_grounding_prompt(query, grounded)
        response = self._llm.complete(prompt, system=_GROUNDING_SYSTEM)
        return GroundedAnswer(
            query=query,
            answer=response.text,
            citations=[_to_citation(result) for result in grounded],
            disclaimer=VERIFY_SOURCES_DISCLAIMER,
            elapsed_seconds=time.perf_counter() - started,
        )

    def answer_guide(self, query: str, topic: str | None = None) -> GuideResult:
        """Answer a guide query grounded in retrieved knowledge-base content.

        Handles neighborhood, photography, food, themed-interest, and transport
        guide queries (Requirement 11.4). Retrieves the most relevant documents
        from the knowledge base and synthesises an answer using **only** the
        retrieved document text. Every :class:`GuideAnswer` carries at least one
        :class:`Citation` (source name + last-updated date) so the
        "≥1 source citation per answer" obligation is structurally guaranteed.

        Unlike :meth:`answer_visa_safety`, guide answering is **not** gated by
        the 0.7 visa/safety similarity threshold and does **not** append the
        verify-with-official-sources disclaimer — those obligations are specific
        to visa/safety content (Requirements 11.1-11.3). A guide query is
        answered from whatever relevant content the KB surfaces.

        When retrieval surfaces no usable content (empty result, only zero-score
        matches, a blank query, or a retrieval-backend error), a structured
        :class:`NoGuideContent` is returned and the LLM is **not** invoked,
        because an answer with no source to cite would violate Requirement 11.4.

        ``topic``, when provided, is echoed on the result and used to keep the
        narration on-subject; it is validated against :data:`GUIDE_TOPICS`.
        """
        started = time.perf_counter()
        normalized_topic = self._normalize_guide_topic(topic)
        grounding = self._retrieve_guide_content(query)
        if not grounding:
            # Nothing relevant to present -> cannot cite a source, so decline to
            # answer rather than emit an uncited (ungrounded) guide answer.
            return NoGuideContent(
                query=query,
                topic=normalized_topic,
                elapsed_seconds=time.perf_counter() - started,
            )

        prompt = _build_guide_prompt(query, normalized_topic, grounding)
        response = self._llm.complete(prompt, system=_GUIDE_SYSTEM)
        return GuideAnswer(
            query=query,
            topic=normalized_topic,
            answer=response.text,
            citations=[_to_citation(result) for result in grounding],
            elapsed_seconds=time.perf_counter() - started,
        )

    # -- internals -----------------------------------------------------------

    def _retrieve_grounding(self, query: str) -> list[SearchResult]:
        """Retrieve documents and keep only those at or above the threshold.

        Returns the threshold-clearing results ordered by descending score.
        Any retrieval error degrades to an empty list (fail safe).
        """
        if not query or not query.strip():
            return []
        try:
            results = self._kb.search(query, self._retrieve_k)
        except Exception:
            # Fail safe: a retrieval-backend error must not lead to ungrounded
            # generation; treat it as "nothing met the threshold".
            return []
        threshold = self._min_similarity
        clearing = [result for result in results if result.score >= threshold]
        clearing.sort(key=lambda result: result.score, reverse=True)
        return clearing

    def _no_verified_info(self, query: str, *, elapsed_seconds: float = 0.0) -> NoVerifiedInfo:
        return NoVerifiedInfo(
            query=query,
            threshold=self._min_similarity,
            elapsed_seconds=elapsed_seconds,
        )

    def _retrieve_guide_content(self, query: str) -> list[SearchResult]:
        """Retrieve guide content, keeping only documents with a real match.

        Unlike visa/safety retrieval, this is **not** gated by the 0.7
        threshold (Requirement 11.4 imposes no minimum similarity). It keeps
        every retrieved document with a positive similarity score — a zero-score
        document shares no terms with the query and would be a meaningless
        citation. Results are ordered by descending score. Any retrieval error
        degrades to an empty list (fail safe), yielding :class:`NoGuideContent`.
        """
        if not query or not query.strip():
            return []
        try:
            results = self._kb.search(query, self._retrieve_k)
        except Exception:
            return []
        relevant = [result for result in results if result.score > 0.0]
        relevant.sort(key=lambda result: result.score, reverse=True)
        return relevant

    @staticmethod
    def _normalize_guide_topic(topic: str | None) -> str | None:
        """Normalize and validate an optional guide topic against ``GUIDE_TOPICS``.

        Returns ``None`` for an absent/blank topic. A recognised topic is
        returned lower-cased; an unrecognised one raises ``ValueError`` so
        callers cannot silently mis-tag a guide answer.
        """
        if topic is None:
            return None
        cleaned = topic.strip().lower()
        if not cleaned:
            return None
        if cleaned not in GUIDE_TOPICS:
            raise ValueError(
                f"Unknown guide topic {topic!r}; expected one of {GUIDE_TOPICS}."
            )
        return cleaned


def _to_citation(result: SearchResult) -> Citation:
    document = result.document
    return Citation(
        source=document.source,
        last_updated=document.last_updated.isoformat(),
        score=result.score,
        snippet=document.content,
    )


__all__ = [
    "Retriever",
    "Citation",
    "GroundedAnswer",
    "NoVerifiedInfo",
    "VisaSafetyResult",
    "GuideAnswer",
    "NoGuideContent",
    "GuideResult",
    "GUIDE_TOPICS",
    "RAGService",
]
