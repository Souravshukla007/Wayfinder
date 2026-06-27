"""Unit tests for the CP-SAT constraint solver (task 9.1).

Validates Requirements:
- 9.1: hard budget cap (<=, equality accepted; overshoot rejected).
- 9.2: same-day inter-city travel-time feasibility.
- 9.3: max hotel-change limit.
- 9.4: city-count limit.
- 9.5: dietary exclusions.
- 9.6: interest consistency.
- 9.7: emit a feasible skeleton when all hard constraints are satisfiable.
- 9.8: reject with the specific unsatisfied constraints and emit no skeleton
       (so the Itinerary Agent does not run).
- 9.9: timeout rejection carries no skeleton.

Example-based unit tests; the universally-quantified property tests for
feasibility soundness / infeasibility completeness are scheduled separately
(tasks 9.2-9.4).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

pytest.importorskip("ortools")

from app.models.schemas import TripConstraints
from app.solver.cp_sat import (
    CandidateCity,
    SolverRejection,
    SolverResult,
    SolverTimeout,
    solve_skeleton,
)


def _constraints(**overrides) -> TripConstraints:
    base = dict(
        budget=Decimal(200000),
        duration_days=6,
        max_cities=3,
        max_hotel_changes=2,
        max_same_day_travel=timedelta(hours=6),
        dietary=[],
        interests=[],
        solver_timeout=timedelta(seconds=5),
    )
    base.update(overrides)
    return TripConstraints(**base)


def _cities() -> list[CandidateCity]:
    return [
        CandidateCity(name="Tokyo", cost=Decimal(60000), days=2, score=90.0,
                      interest_tags=["anime", "food"]),
        CandidateCity(name="Kyoto", cost=Decimal(50000), days=2, score=80.0,
                      interest_tags=["food", "photography"]),
        CandidateCity(name="Osaka", cost=Decimal(40000), days=2, score=70.0,
                      interest_tags=["food"]),
    ]


# --------------------------------------------------------------------------- #
# Feasible cases (Requirements 9.1-9.7)
# --------------------------------------------------------------------------- #
def test_feasible_case_emits_skeleton() -> None:
    result = solve_skeleton(_cities(), _constraints())

    assert result.feasible is True
    assert result.itinerary_agent_should_run is True
    assert result.rejection is None
    skeleton = result.skeleton
    assert skeleton is not None
    assert len(skeleton.legs) >= 1
    # Budget cap respected (9.1).
    assert skeleton.total_cost <= Decimal(200000)
    # City-count limit respected (9.4).
    assert len({leg.city for leg in skeleton.legs}) <= 3
    # Hotel-change limit respected (9.3).
    assert skeleton.hotel_changes <= 2
    assert skeleton.hotel_changes == len(skeleton.legs) - 1
    # Days laid out within the trip duration.
    assert all(1 <= leg.arrival_day <= leg.departure_day <= 6 for leg in skeleton.legs)


def test_budget_equality_is_accepted() -> None:
    # Single city whose cost exactly equals the budget is feasible (9.1).
    cities = [CandidateCity(name="Tokyo", cost=Decimal(100000), days=3, score=50.0)]
    result = solve_skeleton(cities, _constraints(budget=Decimal(100000), max_cities=1))

    assert result.feasible is True
    assert result.skeleton is not None
    assert result.skeleton.total_cost == Decimal(100000)


def test_city_count_limit_respected() -> None:
    result = solve_skeleton(_cities(), _constraints(max_cities=1, max_hotel_changes=0))

    assert result.feasible is True
    assert result.skeleton is not None
    assert len(result.skeleton.legs) == 1
    assert result.skeleton.hotel_changes == 0


def test_same_day_travel_excludes_long_hop_pairs() -> None:
    # Tokyo<->Kyoto hop is too long to be adjacent; only one of them can appear
    # alongside the short-hop Osaka, so the skeleton never places the long pair
    # back-to-back (9.2).
    travel = {
        ("Tokyo", "Kyoto"): timedelta(hours=12),
        ("Tokyo", "Osaka"): timedelta(hours=1),
        ("Kyoto", "Osaka"): timedelta(hours=1),
    }
    result = solve_skeleton(
        _cities(),
        _constraints(max_same_day_travel=timedelta(hours=6)),
        travel_times=travel,
    )

    assert result.feasible is True
    skeleton = result.skeleton
    assert skeleton is not None
    cities_in_order = [leg.city for leg in skeleton.legs]
    for first, second in zip(cities_in_order, cities_in_order[1:]):
        pair = {first, second}
        assert pair != {"Tokyo", "Kyoto"}, "long hop scheduled same-day"


def test_dietary_eligible_city_is_selected_over_excluded() -> None:
    cities = [
        CandidateCity(name="MeatTown", cost=Decimal(30000), days=3, score=99.0,
                      dietary_conflicts=["vegetarian"]),
        CandidateCity(name="VeggieCity", cost=Decimal(40000), days=3, score=50.0),
    ]
    result = solve_skeleton(cities, _constraints(dietary=["vegetarian"], max_cities=2))

    assert result.feasible is True
    assert result.skeleton is not None
    chosen = {leg.city for leg in result.skeleton.legs}
    assert "MeatTown" not in chosen  # excluded despite higher score (9.5)
    assert "VeggieCity" in chosen


# --------------------------------------------------------------------------- #
# Infeasible / rejection cases (Requirements 9.8, 9.9)
# --------------------------------------------------------------------------- #
def test_over_budget_yields_rejection_without_skeleton() -> None:
    cities = [CandidateCity(name="Tokyo", cost=Decimal(500000), days=3, score=90.0)]
    result = solve_skeleton(cities, _constraints(budget=Decimal(100000), max_cities=1))

    assert result.feasible is False
    assert result.itinerary_agent_should_run is False
    # No skeleton present => Itinerary Agent must not run (9.8).
    assert result.skeleton is None
    assert result.rejection is not None
    assert "budget_cap" in result.rejection.unsatisfied_constraints


def test_all_candidates_violate_dietary_yields_rejection() -> None:
    cities = [
        CandidateCity(name="MeatTown", cost=Decimal(10000), days=3,
                      dietary_conflicts=["vegan"]),
        CandidateCity(name="SteakCity", cost=Decimal(10000), days=3,
                      dietary_conflicts=["vegan"]),
    ]
    result = solve_skeleton(cities, _constraints(dietary=["vegan"]))

    assert result.skeleton is None
    assert result.rejection is not None
    assert "dietary_exclusions" in result.rejection.unsatisfied_constraints


def test_no_candidates_yields_rejection() -> None:
    result = solve_skeleton([], _constraints())

    assert result.skeleton is None
    assert result.rejection is not None
    assert "no_candidate_cities" in result.rejection.unsatisfied_constraints


def test_interest_inconsistent_candidates_yield_rejection() -> None:
    cities = [
        CandidateCity(name="BeachTown", cost=Decimal(10000), days=3,
                      interest_tags=["beach"]),
    ]
    result = solve_skeleton(cities, _constraints(interests=["anime"]))

    assert result.skeleton is None
    assert result.rejection is not None
    assert "interest_consistency" in result.rejection.unsatisfied_constraints


def test_solver_timeout_type_carries_no_skeleton() -> None:
    # The timeout result is a rejection subtype with no skeleton, so the
    # Itinerary Agent cannot run (9.9).
    timeout = SolverTimeout(message="exceeded")
    result = SolverResult.rejected(timeout)

    assert isinstance(result.rejection, SolverRejection)
    assert result.rejection.timed_out is True
    assert result.skeleton is None
    assert result.itinerary_agent_should_run is False
