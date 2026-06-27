"""Property-based test for solver infeasibility completeness (task 9.3).

Feature: wayfinder-travel-planner, Property 21: Infeasibility completeness.

*For any* trip whose hard constraints cannot be simultaneously satisfied, the
solver SHALL NOT emit a skeleton, SHALL report the specific unsatisfied
constraints, and SHALL prevent the Itinerary Agent from running.

Validates: Requirements 9.8

This is a property test only. It exercises
``app.solver.cp_sat.solve_skeleton`` on inputs that are *constructed to be
provably infeasible* — at least one hard constraint cannot be satisfied no
matter which cities are chosen — and asserts the solver returns a
``SolverRejection`` (never a skeleton) that names the responsible constraint and
keeps the Itinerary Agent from running. Emitting a skeleton for a
provably-infeasible input is a violation.

The generators isolate a single, independently-verifiable cause of
infeasibility per case so the expected unsatisfied-constraint label is known
ahead of time:

* ``budget_cap``        — every candidate (and therefore the cheapest one)
                          costs strictly more than the budget.
* ``dietary_exclusions``— every candidate conflicts with a trip dietary
                          constraint, so none is eligible.
* ``interest_consistency`` — every candidate carries interest tags disjoint
                          from the trip's interests, so none is consistent.
* ``no_candidate_cities`` — no candidates are supplied at all.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("ortools")

from app.models.schemas import TripConstraints
from app.solver.cp_sat import CandidateCity, solve_skeleton

# Constraint reason labels reported in SolverRejection.unsatisfied_constraints.
_BUDGET = "budget_cap"
_DIETARY = "dietary_exclusions"
_INTERESTS = "interest_consistency"
_NO_CANDIDATES = "no_candidate_cities"

# A relaxed, never-binding travel limit, paired with no travel_times, so the
# same-day-travel and (generous) hotel/city-count constraints never become the
# accidental cause of infeasibility — the constructed cause is the only one.
_GENEROUS_TRAVEL = timedelta(hours=24)

_DIETARY_VOCAB = ["vegetarian", "vegan", "halal", "kosher", "gluten_free"]
_INTEREST_VOCAB = ["anime", "food", "photography", "beach", "history", "nightlife"]

_city_names = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8
)


def _constraints(
    *,
    budget: Decimal,
    max_cities: int,
    max_hotel_changes: int,
    dietary: list[str],
    interests: list[str],
) -> TripConstraints:
    return TripConstraints(
        budget=budget,
        duration_days=6,
        max_cities=max_cities,
        max_hotel_changes=max_hotel_changes,
        max_same_day_travel=_GENEROUS_TRAVEL,
        dietary=dietary,
        interests=interests,
        solver_timeout=timedelta(seconds=10),
    )


# --------------------------------------------------------------------------- #
# Generators — each yields (candidates, constraints, expected_reason)
# --------------------------------------------------------------------------- #
@st.composite
def _over_budget_cases(draw) -> tuple[list[CandidateCity], TripConstraints, str]:
    """Every candidate costs strictly more than the budget (budget_cap).

    With at least one eligible city whose cheapest cost exceeds the budget, even
    a single-city plan cannot fit, so the problem is provably infeasible on the
    budget cap regardless of city-count / hotel-change headroom.
    """
    budget = draw(st.integers(min_value=1_000, max_value=500_000))
    n = draw(st.integers(min_value=1, max_value=5))
    names = draw(
        st.lists(_city_names, min_size=n, max_size=n, unique=True)
    )
    candidates: list[CandidateCity] = []
    for i, name in enumerate(names):
        # Strictly greater than budget => infeasible even for one city.
        overshoot = draw(st.integers(min_value=1, max_value=400_000))
        candidates.append(
            CandidateCity(
                name=name,
                cost=Decimal(budget + overshoot),
                days=1,
                score=float(i),
            )
        )
    constraints = _constraints(
        budget=Decimal(budget),
        # Generous headroom so only the budget can be the blocker.
        max_cities=n,
        max_hotel_changes=n,
        dietary=[],
        interests=[],
    )
    return candidates, constraints, _BUDGET


@st.composite
def _all_dietary_conflict_cases(
    draw,
) -> tuple[list[CandidateCity], TripConstraints, str]:
    """Every candidate conflicts with a trip dietary constraint (dietary_exclusions)."""
    dietary = draw(
        st.lists(st.sampled_from(_DIETARY_VOCAB), min_size=1, max_size=3, unique=True)
    )
    n = draw(st.integers(min_value=1, max_value=5))
    names = draw(st.lists(_city_names, min_size=n, max_size=n, unique=True))
    budget = draw(st.integers(min_value=10_000, max_value=300_000))
    candidates: list[CandidateCity] = []
    for i, name in enumerate(names):
        # Each city conflicts with at least one specified dietary constraint.
        conflict = draw(st.sampled_from(dietary))
        # Affordable: budget is never the blocker here.
        cost = draw(st.integers(min_value=1, max_value=budget))
        candidates.append(
            CandidateCity(
                name=name,
                cost=Decimal(cost),
                days=1,
                score=float(i),
                dietary_conflicts=[conflict],
            )
        )
    constraints = _constraints(
        budget=Decimal(budget),
        max_cities=n,
        max_hotel_changes=n,
        dietary=dietary,
        interests=[],
    )
    return candidates, constraints, _DIETARY


@st.composite
def _all_interest_inconsistent_cases(
    draw,
) -> tuple[list[CandidateCity], TripConstraints, str]:
    """Every candidate's interest tags are disjoint from the trip's (interest_consistency).

    Cities carry non-empty tags drawn from a pool that excludes the trip's
    interests, so none is consistent (an untagged city would be treated as
    neutral, so tags are required to force the violation).
    """
    interests = draw(
        st.lists(st.sampled_from(_INTEREST_VOCAB), min_size=1, max_size=3, unique=True)
    )
    disjoint_pool = [t for t in _INTEREST_VOCAB if t not in interests]
    # The vocab is large enough that a non-empty disjoint pool always remains.
    assert disjoint_pool
    n = draw(st.integers(min_value=1, max_value=5))
    names = draw(st.lists(_city_names, min_size=n, max_size=n, unique=True))
    budget = draw(st.integers(min_value=10_000, max_value=300_000))
    candidates: list[CandidateCity] = []
    for i, name in enumerate(names):
        tags = draw(
            st.lists(st.sampled_from(disjoint_pool), min_size=1, max_size=2, unique=True)
        )
        cost = draw(st.integers(min_value=1, max_value=budget))
        candidates.append(
            CandidateCity(
                name=name,
                cost=Decimal(cost),
                days=1,
                score=float(i),
                interest_tags=tags,
            )
        )
    constraints = _constraints(
        budget=Decimal(budget),
        max_cities=n,
        max_hotel_changes=n,
        dietary=[],
        interests=interests,
    )
    return candidates, constraints, _INTERESTS


@st.composite
def _no_candidate_cases(
    draw,
) -> tuple[list[CandidateCity], TripConstraints, str]:
    """No candidate cities supplied at all (no_candidate_cities)."""
    budget = draw(st.integers(min_value=1_000, max_value=500_000))
    max_cities = draw(st.integers(min_value=1, max_value=5))
    constraints = _constraints(
        budget=Decimal(budget),
        max_cities=max_cities,
        max_hotel_changes=max_cities,
        dietary=[],
        interests=[],
    )
    return [], constraints, _NO_CANDIDATES


def _infeasible_cases() -> st.SearchStrategy:
    return st.one_of(
        _over_budget_cases(),
        _all_dietary_conflict_cases(),
        _all_interest_inconsistent_cases(),
        _no_candidate_cases(),
    )


# --------------------------------------------------------------------------- #
# Property 21: Infeasibility completeness
# Feature: wayfinder-travel-planner, Property 21
# Validates: Requirements 9.8
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(case=_infeasible_cases())
def test_property_21_infeasibility_completeness(
    case: tuple[list[CandidateCity], TripConstraints, str],
) -> None:
    """A provably-infeasible input is rejected, names the cause, and blocks the agent.

    **Validates: Requirements 9.8**

    For inputs constructed so at least one hard constraint cannot be satisfied:
    ``solve_skeleton`` MUST return a rejection (no skeleton), the rejection MUST
    report the specific unsatisfied constraint, and the Itinerary Agent MUST be
    prevented from running.
    """
    candidates, constraints, expected_reason = case

    result = solve_skeleton(candidates, constraints)

    # SHALL NOT emit a skeleton.
    assert result.skeleton is None
    assert result.feasible is False
    # SHALL prevent the Itinerary Agent from running.
    assert result.itinerary_agent_should_run is False
    # SHALL report the specific unsatisfied constraint(s).
    assert result.rejection is not None
    assert result.rejection.unsatisfied_constraints, (
        "an infeasible input must report at least one unsatisfied constraint"
    )
    assert expected_reason in result.rejection.unsatisfied_constraints
