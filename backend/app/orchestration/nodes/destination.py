"""The Destination agent node (task 13.3).

The Destination agent has two responsibilities, both implemented here:

1. **Propose candidate cities.** Given the trip ``constraints`` (and the
   personalization ``preference_vector``) on the shared :class:`GraphState`, it
   proposes a deterministic set of candidate destinations for the deterministic
   Decision Engine to score. Proposal is mock-first and uses no LLM, no
   randomness, no clock, and no network, so a planning run is reproducible with
   zero paid keys.

2. **Narrate the ranking using Decision Engine scores ONLY.** Once the Decision
   Engine has populated ``state.scored``, this node renders a human-readable
   narration of *why* one destination outranks another. The narration is built
   strictly from the engine's scores and their order; the agent (and the LLM it
   may call for prose) **never generates, alters, or reorders scores**
   (Requirements 3.3, 5.10, 19.4 / Property 7).

The guardrail is structural, not advisory: ``narrate_ranking`` reads the score
and rank of every option directly from the engine output and never parses a
number back out of any LLM text, so no LLM behavior can change the ranking that
is presented. :func:`destination_node` returns ``state.scored`` unchanged to
make non-alteration explicit and testable.

Requirements: 3.3, 5.10, 19.4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.llm.base import LLMProvider, get_llm_provider
from app.models.schemas import FEATURES, TripConstraints
from app.orchestration.state import Candidate, GraphState, ScoredCandidate

# Maximum number of candidate cities the agent proposes for scoring. At least
# three are needed so the system can present the top-three A/B/C options
# (Requirement 7.2); a small cap keeps the deterministic core fast.
MAX_PROPOSED_CANDIDATES = 6

# System prompt that constrains the LLM to narration only. The LLM is never
# given the ability to produce numeric scores or rankings (design: "The LLM is
# restricted to narration prompts").
NARRATION_SYSTEM_PROMPT = (
    "You are a travel consultant narrating a ranking that has ALREADY been "
    "computed by a deterministic scoring engine. Explain in plain language why "
    "higher-ranked destinations outrank lower-ranked ones. You MUST NOT invent, "
    "change, or reorder any score or rank; treat the provided scores and order "
    "as fixed facts."
)


@dataclass(frozen=True)
class CatalogEntry:
    """A curated candidate city with raw 0-100 feature signals.

    ``features`` are raw signals on the engine's 0-100 input scale (clamped and
    normalized later by the Decision Engine). ``interests`` are free-form tags
    used only to order proposals against the user's stated interests; they are
    not part of the scored feature set.
    """

    name: str
    country: str
    features: Mapping[str, float]
    interests: tuple[str, ...] = field(default_factory=tuple)


# A deterministic, mock-first catalog. Feature signals are on the 0-100 scale
# the Decision Engine expects. This is intentionally fixed data (not generated)
# so proposals are reproducible without any external provider.
DESTINATION_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "Tokyo", "Japan",
        {"budget_fit": 60, "weather_fit": 80, "crowd_score": 45,
         "food_score": 95, "photography_score": 90, "travel_efficiency": 92},
        ("anime", "food", "photography", "city", "shopping", "technology"),
    ),
    CatalogEntry(
        "Kyoto", "Japan",
        {"budget_fit": 65, "weather_fit": 82, "crowd_score": 55,
         "food_score": 88, "photography_score": 95, "travel_efficiency": 80},
        ("temples", "photography", "food", "history", "less crowded", "nature"),
    ),
    CatalogEntry(
        "Osaka", "Japan",
        {"budget_fit": 70, "weather_fit": 80, "crowd_score": 50,
         "food_score": 96, "photography_score": 78, "travel_efficiency": 85},
        ("food", "nightlife", "city", "anime", "shopping"),
    ),
    CatalogEntry(
        "Sapporo", "Japan",
        {"budget_fit": 72, "weather_fit": 70, "crowd_score": 75,
         "food_score": 85, "photography_score": 82, "travel_efficiency": 68},
        ("nature", "food", "less crowded", "photography", "winter"),
    ),
    CatalogEntry(
        "Kanazawa", "Japan",
        {"budget_fit": 75, "weather_fit": 78, "crowd_score": 80,
         "food_score": 80, "photography_score": 84, "travel_efficiency": 62},
        ("history", "less crowded", "photography", "nature", "art"),
    ),
    CatalogEntry(
        "Hiroshima", "Japan",
        {"budget_fit": 74, "weather_fit": 79, "crowd_score": 70,
         "food_score": 82, "photography_score": 80, "travel_efficiency": 66},
        ("history", "food", "less crowded", "photography"),
    ),
    CatalogEntry(
        "Nara", "Japan",
        {"budget_fit": 76, "weather_fit": 80, "crowd_score": 60,
         "food_score": 75, "photography_score": 86, "travel_efficiency": 70},
        ("nature", "history", "photography", "less crowded", "temples"),
    ),
    CatalogEntry(
        "Fukuoka", "Japan",
        {"budget_fit": 78, "weather_fit": 81, "crowd_score": 72,
         "food_score": 90, "photography_score": 74, "travel_efficiency": 64},
        ("food", "less crowded", "nightlife", "city"),
    ),
)


def _normalize_interest(text: str) -> str:
    """Lowercase/trim an interest tag for order-insensitive comparison."""
    return text.strip().lower()


def _interest_match_count(entry: CatalogEntry, interests: Sequence[str]) -> int:
    """Count how many of the user's interests the catalog entry advertises.

    Matching is substring-aware in both directions (e.g. user "local food"
    matches tag "food") and case-insensitive, so it is robust to phrasing.
    """
    if not interests:
        return 0
    tags = [_normalize_interest(tag) for tag in entry.interests]
    count = 0
    for raw in interests:
        wanted = _normalize_interest(raw)
        if not wanted:
            continue
        if any(wanted in tag or tag in wanted for tag in tags):
            count += 1
    return count


def propose_candidates(
    state: GraphState,
    *,
    catalog: Sequence[CatalogEntry] = DESTINATION_CATALOG,
    limit: int = MAX_PROPOSED_CANDIDATES,
) -> list[Candidate]:
    """Propose candidate cities for the Decision Engine to score.

    Selection is deterministic: catalog entries are ordered by how many of the
    user's stated interests they match (descending), with ties broken by the
    fixed catalog order. This is a pure function of ``state`` — no LLM, no
    randomness, no clock, no network — so the same trip always proposes the same
    candidates.

    The proposal does **not** score candidates; it only chooses which cities to
    forward. Scoring (and therefore the ranking) is the Decision Engine's job.

    Args:
        state: shared graph state; ``constraints.interests`` steers ordering.
        catalog: candidate source (overridable for tests); defaults to the
            built-in mock-first :data:`DESTINATION_CATALOG`.
        limit: maximum number of candidates to propose.

    Returns:
        A list of :class:`Candidate` carrying raw feature signals, ordered most-
        to least-relevant to the user's interests.
    """
    constraints: TripConstraints | None = state.constraints
    interests = list(constraints.interests) if constraints else []

    # Decorate-sort-undecorate with a stable key. Python's sort is stable, so
    # entries with an equal match count keep their catalog order, keeping the
    # whole proposal deterministic for a fixed input.
    indexed = list(enumerate(catalog))
    indexed.sort(key=lambda pair: (-_interest_match_count(pair[1], interests), pair[0]))

    proposed: list[Candidate] = []
    for _, entry in indexed[: max(0, limit)]:
        proposed.append(
            Candidate(
                name=entry.name,
                country=entry.country,
                features={feature: float(entry.features[feature]) for feature in FEATURES},
            )
        )
    return proposed


def _format_score(score: float) -> str:
    """Render a final score for display without changing its value."""
    return f"{score:.1f}"


def _ranking_lines(scored: Sequence[ScoredCandidate]) -> list[str]:
    """Build one display line per option, in the engine's given order.

    Scores and order are read verbatim from the engine output; this function
    never sorts, recomputes, or otherwise alters them.
    """
    lines: list[str] = []
    for position, option in enumerate(scored, start=1):
        place = f" ({option.country})" if option.country else ""
        lines.append(
            f"{position}. {option.name}{place} — score {_format_score(option.score)}"
        )
    return lines


def _comparison_lines(scored: Sequence[ScoredCandidate]) -> list[str]:
    """Explain why each option outranks the next, using engine scores only.

    Produces a comparative sentence for each adjacent pair (Requirement 7.3 /
    1.2), citing the two final scores exactly as the engine reported them.
    """
    comparisons: list[str] = []
    for higher, lower in zip(scored, scored[1:]):
        comparisons.append(
            f"{higher.name} outranks {lower.name} because its final score "
            f"({_format_score(higher.score)}) is higher than {lower.name}'s "
            f"({_format_score(lower.score)})."
        )
    return comparisons


def narrate_ranking(
    scored: Sequence[ScoredCandidate],
    *,
    llm: LLMProvider | None = None,
    include_llm_prose: bool = True,
) -> str:
    """Narrate the destination ranking using Decision Engine scores ONLY.

    The authoritative ranking — which destinations, their final scores, and the
    descending order — is taken **verbatim** from ``scored`` (the Decision
    Engine's output). The narration never derives a score or an order from
    anything else, so it cannot generate, alter, or reorder scores
    (Requirements 3.3, 5.10, 19.4 / Property 7).

    An LLM may be invoked to add friendly prose, but it is given the already-
    decided ranking as fixed context and its text is only ever *appended*; no
    number is parsed back out of it. With the default mock provider the prose is
    deterministic, keeping whole runs reproducible.

    Args:
        scored: the engine's scored candidates, already ranked descending.
        llm: optional narration provider; defaults to the config-selected
            provider (mock when no key is configured).
        include_llm_prose: when ``True`` (default) append LLM prose after the
            deterministic ranking facts; when ``False`` return only the facts.

    Returns:
        A narration string. Empty input yields a short, explicit notice.
    """
    if not scored:
        return "No destinations were available to rank."

    facts = ["Ranked by the Decision Engine (scores fixed by the engine):"]
    facts.extend(_ranking_lines(scored))
    comparisons = _comparison_lines(scored)
    if comparisons:
        facts.append("")
        facts.extend(comparisons)
    narration = "\n".join(facts)

    if not include_llm_prose:
        return narration

    provider = llm if llm is not None else get_llm_provider()
    prompt = (
        "Narrate the following destination ranking for the traveler. "
        "Do not change any score or the order.\n\n" + narration
    )
    prose = provider.complete(prompt, system=NARRATION_SYSTEM_PROMPT, temperature=0.0).text
    if prose and prose.strip():
        return f"{narration}\n\n{prose.strip()}"
    return narration


def destination_node(
    state: GraphState,
    *,
    llm: LLMProvider | None = None,
) -> dict[str, Any]:
    """LangGraph node for the Destination agent.

    Behavior depends on how far the pipeline has progressed:

    * **Before scoring** (no ``state.scored`` yet): propose candidate cities so
      the Decision Engine has something to score, returning them under
      ``"candidates"``. If candidates were already proposed, they are left as-is.
    * **After scoring** (``state.scored`` populated): narrate the ranking using
      the engine's scores only and return ``state.scored`` **unchanged**, making
      the never-alter/never-reorder guarantee explicit (Requirements 5.10, 19.4).

    The node returns a partial-state update dict (the LangGraph node contract).
    The ``"destination_narration"`` key carries the rendered narration for the
    Coordinator's final merge; it is additive and never replaces engine data.

    Args:
        state: the shared graph state.
        llm: optional narration provider (defaults to the config-selected one).

    Returns:
        A dict of state updates: ``candidates`` and/or the unchanged ``scored``
        list plus a ``destination_narration`` string when scores are present.
    """
    update: dict[str, Any] = {}

    if not state.candidates:
        update["candidates"] = propose_candidates(state)

    if state.scored:
        # Pass the engine's ranking straight through. We copy the list container
        # but keep every ScoredCandidate identical and in the same order, so the
        # LLM/narration step provably cannot change scores or ordering.
        update["scored"] = list(state.scored)
        update["destination_narration"] = narrate_ranking(state.scored, llm=llm)

    return update


__all__ = [
    "MAX_PROPOSED_CANDIDATES",
    "NARRATION_SYSTEM_PROMPT",
    "CatalogEntry",
    "DESTINATION_CATALOG",
    "propose_candidates",
    "narrate_ranking",
    "destination_node",
]
