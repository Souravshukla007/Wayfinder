"""Property-based test for solver feasibility soundness (task 9.2).

Covers:
- **Property 20: Feasibility soundness of emitted skeletons** (Task 9.2) - for
  any trip whose hard constraints are simultaneously satisfiable, when the
  solver emits a feasible skeleton, that skeleton satisfies *every* enforced
  hard constraint:

    * total cost <= budget (equality accepted, any overshoot excluded),
    * no same-day inter-city hop exceeding the maximum same-day travel duration,
    * hotel-change count <= limit,
    * distinct city count <= limit,
    * no option violating any dietary constraint,
    * only options consistent with the specified interests.

  Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7.
  Tagged: Feature: wayfinder-travel-planner, Property 20.

The skeleton is validated by an **independent constraint checker** implemented
locally in this test module. The checker re-derives constraint satisfaction
straight from the generated inputs and the emitted skeleton; it never calls the
solver or imports the solver's private helpers. A counterexample here therefore
indicates a genuine soundness bug to report rather than mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytest.importorskip("ortools")

from app.models.schemas import TripConstraints
from app.orchestration.state import ItinerarySkeleton
from app.solver.cp_sat import CandidateCity, solve_skeleton

# ---------------------------------------------------------------------------
# Input vocabulary
# ---------------------------------------------------------------------------
# Fixed pools keep dietary/interest tags overlapping the trip constraints often
# enough that the solver actually emits feasible skeletons (so the soundness
# assertions are exercised, not skipped).
_DIETARY_VOCAB = ["vegetarian", "vegan", "halal", "kosher", "gluten_free"]
_INTEREST_VOCAB = ["anime", "food", "photography", "beach", "history", "nightlife"]
# Distinct city names; drawn `unique=True` so each leg maps to exactly one city.
_CITY_POOL = [
    "Tokyo", "Kyoto", "Osaka", "Nara", "Sapporo",
    "Fukuoka", "Nagoya", "Hiroshima", "Sendai", "Kobe",
]


# ---------------------------------------------------------------------------
# Independent constraint checker (does NOT use the solver)
# ---------------------------------------------------------------------------
def _is_dietary_ok(city: CandidateCity, dietary: list[str]) -> bool:
    """A city is dietary-eligible when it conflicts with none of the trip's
    dietary constraints (Requirement 9.5)."""
    if not dietary:
        return True
    return set(city.dietary_conflicts).isdisjoint(dietary)


def _is_interest_ok(city: CandidateCity, interests: list[str]) -> bool:
    """A city is interest-consistent when the trip specifies no interests, the
    city carries no interest tags (neutral), or its tags intersect the trip's
    interests (Requirement 9.6)."""
    if not interests:
        return True
    if not city.interest_tags:
        return True
    return not set(city.interest_tags).isdisjoint(interests)


def _travel_seconds(
    travel_times: Mapping[tuple[str, str], timedelta], a: str, b: str
) -> float:
    """Inter-city travel duration (seconds) for hop a -> b, either ordering;
    an unknown pair imposes no restriction (0 seconds)."""
    td = travel_times.get((a, b)) or travel_times.get((b, a))
    return td.total_seconds() if td is not None else 0.0


def _assert_skeleton_satisfies_all_constraints(
    skeleton: ItinerarySkeleton,
    candidates: list[CandidateCity],
    constraints: TripConstraints,
    travel_times: Mapping[tuple[str, str], timedelta],
) -> None:
    """Independently confirm the emitted skeleton honours every hard constraint."""
    by_name = {c.name: c for c in candidates}
    leg_cities = [leg.city for leg in skeleton.legs]

    # Every leg references a real candidate, and a skeleton always has >= 1 leg
    # (a feasible plan visits at least one city; Requirement 9.7).
    assert leg_cities, "feasible skeleton must contain at least one city"
    chosen = [by_name[name] for name in leg_cities]

    # 9.4 - distinct city-count limit.
    distinct_cities = set(leg_cities)
    assert len(distinct_cities) == len(leg_cities), "a city was visited twice"
    assert len(distinct_cities) <= constraints.max_cities

    # 9.3 - hotel-change limit (changes == cities - 1).
    assert skeleton.hotel_changes <= constraints.max_hotel_changes
    assert skeleton.hotel_changes == len(leg_cities) - 1

    # 9.1 - budget cap (<=, equality accepted). Recompute cost independently and
    # confirm both the recomputed sum and the emitted total respect the budget.
    recomputed_cost = sum((Decimal(c.cost) for c in chosen), Decimal(0))
    assert recomputed_cost == skeleton.total_cost, "emitted total_cost is inconsistent"
    assert skeleton.total_cost <= constraints.budget

    # 9.5 - dietary exclusions; 9.6 - interest consistency.
    for city in chosen:
        assert _is_dietary_ok(city, constraints.dietary), (
            f"selected city '{city.name}' violates a dietary constraint"
        )
        assert _is_interest_ok(city, constraints.interests), (
            f"selected city '{city.name}' is inconsistent with trip interests"
        )

    # 9.2 - same-day inter-city travel feasibility for every adjacent hop.
    max_travel = constraints.max_same_day_travel.total_seconds()
    for first, second in zip(leg_cities, leg_cities[1:]):
        assert _travel_seconds(travel_times, first, second) <= max_travel, (
            f"same-day hop {first} -> {second} exceeds the max travel duration"
        )


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
@st.composite
def _candidate(draw: st.DrawFn, name: str) -> CandidateCity:
    return CandidateCity(
        name=name,
        cost=Decimal(draw(st.integers(min_value=0, max_value=120_000))),
        days=draw(st.integers(min_value=1, max_value=5)),
        score=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        dietary_conflicts=draw(
            st.lists(st.sampled_from(_DIETARY_VOCAB), max_size=2, unique=True)
        ),
        interest_tags=draw(
            st.lists(st.sampled_from(_INTEREST_VOCAB), max_size=3, unique=True)
        ),
    )


@st.composite
def _problem(
    draw: st.DrawFn,
) -> tuple[list[CandidateCity], TripConstraints, dict[tuple[str, str], timedelta]]:
    names = draw(
        st.lists(st.sampled_from(_CITY_POOL), min_size=1, max_size=5, unique=True)
    )
    candidates = [draw(_candidate(name)) for name in names]

    # Inter-city travel durations for every unordered pair.
    travel_times: dict[tuple[str, str], timedelta] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            travel_times[(a, b)] = timedelta(hours=draw(st.integers(min_value=0, max_value=30)))

    constraints = TripConstraints(
        # Budget band overlaps the candidate cost range so feasible skeletons
        # are emitted frequently.
        budget=Decimal(draw(st.integers(min_value=10_000, max_value=600_000))),
        duration_days=draw(st.integers(min_value=1, max_value=14)),
        max_cities=draw(st.integers(min_value=1, max_value=5)),
        max_hotel_changes=draw(st.integers(min_value=0, max_value=5)),
        max_same_day_travel=timedelta(hours=draw(st.integers(min_value=1, max_value=24))),
        dietary=draw(st.lists(st.sampled_from(_DIETARY_VOCAB), max_size=2, unique=True)),
        interests=draw(st.lists(st.sampled_from(_INTEREST_VOCAB), max_size=3, unique=True)),
        # Generous, fixed timeout so the solver concludes rather than timing out.
        solver_timeout=timedelta(seconds=10),
    )
    return candidates, constraints, travel_times


# ---------------------------------------------------------------------------
# Property 20: Feasibility soundness of emitted skeletons
# Feature: wayfinder-travel-planner, Property 20
# Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
# ---------------------------------------------------------------------------
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(problem=_problem())
def test_emitted_skeleton_satisfies_every_hard_constraint(
    problem: tuple[list[CandidateCity], TripConstraints, dict[tuple[str, str], timedelta]],
) -> None:
    """Whenever ``solve_skeleton`` emits a feasible skeleton, an independent
    checker confirms it satisfies budget, same-day travel-time, hotel-change,
    city-count, dietary, and interest-consistency constraints simultaneously.

    Rejections/timeouts carry no skeleton and impose no obligation here; the
    soundness invariant is asserted only on emitted skeletons (Property 20).
    """
    candidates, constraints, travel_times = problem

    result = solve_skeleton(candidates, constraints, travel_times=travel_times)

    if not result.feasible:
        return  # No skeleton emitted -> Property 20 makes no claim.

    assert result.skeleton is not None
    _assert_skeleton_satisfies_all_constraints(
        result.skeleton, candidates, constraints, travel_times
    )
