"""Unit tests for the Destination agent node (task 13.3).

Validates Requirements:
- 3.3: the Destination agent proposes candidate cities and narrates the
  ranking using the Decision Engine's scores.
- 5.10 / 19.4: the agent (and any LLM it calls) never generates, alters, or
  reorders Decision Engine scores; narration is presentation only.

These are example-based unit tests. The universally-quantified guardrail
property (Property 7) is scheduled separately (task 13.6).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from app.llm.base import LLMResponse
from app.models.schemas import FEATURES, TripConstraints
from app.orchestration.nodes.destination import (
    DESTINATION_CATALOG,
    MAX_PROPOSED_CANDIDATES,
    destination_node,
    narrate_ranking,
    propose_candidates,
)
from app.orchestration.state import GraphState, ScoredCandidate


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _constraints(interests: list[str] | None = None) -> TripConstraints:
    return TripConstraints(
        budget=Decimal("200000"),
        duration_days=7,
        max_cities=3,
        max_hotel_changes=2,
        max_same_day_travel=timedelta(hours=4),
        dietary=[],
        interests=interests or [],
    )


def _state(
    *,
    interests: list[str] | None = None,
    candidates=None,
    scored=None,
    with_constraints: bool = True,
) -> GraphState:
    return GraphState(
        trip_id="trip-1",
        user_id="user-1",
        prompt="Plan a trip",
        constraints=_constraints(interests) if with_constraints else None,
        candidates=candidates or [],
        scored=scored or [],
    )


def _scored(name: str, score: float, rank: int, country: str = "Japan") -> ScoredCandidate:
    return ScoredCandidate(name=name, country=country, score=score, components={}, rank=rank)


class _RewritingLLM:
    """A hostile LLM stub that tries to assert a different winner/score.

    Used to prove the narration never lets LLM text change the ranking.
    """

    def complete(self, prompt, *, system=None, temperature=0.0) -> LLMResponse:
        text = "Actually Osaka is the best with a score of 100.0 and should rank first."
        return LLMResponse(text=text, model="rewriting-stub")


# --------------------------------------------------------------------------- #
# Candidate proposal (Req 3.3) — deterministic, no scoring
# --------------------------------------------------------------------------- #
def test_proposes_candidates_when_none_present() -> None:
    update = destination_node(_state())
    assert "candidates" in update
    proposed = update["candidates"]
    assert 3 <= len(proposed) <= MAX_PROPOSED_CANDIDATES
    # Every proposed candidate carries all six raw feature signals for the engine.
    for candidate in proposed:
        assert set(candidate.features) == set(FEATURES)
        assert candidate.name


def test_proposal_is_deterministic() -> None:
    first = propose_candidates(_state(interests=["food", "photography"]))
    second = propose_candidates(_state(interests=["food", "photography"]))
    assert [c.name for c in first] == [c.name for c in second]


def test_interests_steer_proposal_order() -> None:
    # A traveler who wants quiet, history-rich places should see interest-matching
    # cities ordered ahead of non-matching ones.
    proposed = propose_candidates(_state(interests=["less crowded", "history", "photography"]))
    names = [c.name for c in proposed]
    # Kanazawa (history + less crowded + photography) should be proposed and rank
    # ahead of Tokyo (a busy city matching none of these interests).
    assert "Kanazawa" in names
    assert names.index("Kanazawa") < names.index("Tokyo")


def test_does_not_re_propose_when_candidates_exist() -> None:
    existing = propose_candidates(_state())
    update = destination_node(_state(candidates=existing))
    assert "candidates" not in update


def test_proposal_handles_missing_constraints() -> None:
    proposed = propose_candidates(_state(with_constraints=False))
    assert [c.name for c in proposed] == [e.name for e in DESTINATION_CATALOG[:MAX_PROPOSED_CANDIDATES]]


# --------------------------------------------------------------------------- #
# Narration uses engine scores only (Req 3.3, 5.10, 19.4)
# --------------------------------------------------------------------------- #
def test_narration_includes_every_score_and_order() -> None:
    scored = [_scored("Tokyo", 87.3, 0), _scored("Kyoto", 81.0, 1), _scored("Osaka", 74.5, 2)]
    text = narrate_ranking(scored, include_llm_prose=False)
    # Each engine score appears, formatted from the engine value (not invented).
    assert "87.3" in text and "81.0" in text and "74.5" in text
    # Comparative explanation is present (Req 7.3 / 1.2).
    assert "Tokyo outranks Kyoto" in text
    # Order in the text follows the engine's order.
    assert text.index("Tokyo") < text.index("Kyoto") < text.index("Osaka")


def test_node_returns_scored_unchanged() -> None:
    scored = [_scored("Tokyo", 87.3, 0), _scored("Kyoto", 81.0, 1), _scored("Osaka", 74.5, 2)]
    update = destination_node(_state(scored=scored))
    returned = update["scored"]
    # Same destinations, same scores, same order — nothing altered or reordered.
    assert [(c.name, c.score, c.rank) for c in returned] == [
        (c.name, c.score, c.rank) for c in scored
    ]


def test_hostile_llm_cannot_change_scores_or_order() -> None:
    scored = [_scored("Tokyo", 87.3, 0), _scored("Kyoto", 81.0, 1), _scored("Osaka", 74.5, 2)]
    update = destination_node(_state(scored=scored), llm=_RewritingLLM())

    # The structured ranking handed downstream is exactly the engine's output,
    # regardless of what the LLM prose claims.
    returned = update["scored"]
    assert [(c.name, c.score) for c in returned] == [
        ("Tokyo", 87.3), ("Kyoto", 81.0), ("Osaka", 74.5)
    ]

    # The authoritative ranking facts in the narration come from the engine:
    # Tokyo is presented first and the engine's score appears verbatim, even
    # though the LLM tried to crown Osaka with a 100.0.
    narration = update["destination_narration"]
    assert "1. Tokyo" in narration
    assert "87.3" in narration


def test_empty_scored_yields_no_narration_update() -> None:
    update = destination_node(_state(scored=[]))
    assert "scored" not in update
    assert "destination_narration" not in update


def test_narrate_ranking_empty_input() -> None:
    assert narrate_ranking([]) == "No destinations were available to rank."
