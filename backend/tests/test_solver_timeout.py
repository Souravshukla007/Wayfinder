"""Unit test for the CP-SAT solver timeout path (task 9.4).

Validates Requirements:
- 9.9: IF the solver does not reach a satisfiable or proven-infeasible result
       within the trip's maximum solving time, THEN it SHALL reject the plan,
       report a timeout indication, and prevent the Itinerary Agent from
       running (rejection carries no skeleton; itinerary_agent_should_run is
       False).

Unlike the synthetic type check in ``test_solver_cp_sat.py``, this test drives
``solve_skeleton`` itself: a near-zero ``solver_timeout`` over a large, hard
problem forces the solver to stop without a conclusive result, exercising the
real timeout branch.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

pytest.importorskip("ortools")

from app.models.schemas import TripConstraints
from app.solver.cp_sat import (
    CandidateCity,
    SolverTimeout,
    solve_skeleton,
)


def _hard_problem() -> tuple[list[CandidateCity], TripConstraints, dict]:
    """Construct a large, combinatorially hard solve.

    Many candidate cities plus dense long-hop travel times generate a large
    model so that, with an effectively zero time budget, the solver cannot
    reach a conclusive (feasible or proven-infeasible) result and must time
    out.
    """
    n = 60
    cities = [
        CandidateCity(
            name=f"City{i}",
            # Costs cluster just under the budget so the budget cap behaves
            # like a hard knapsack rather than being trivially satisfiable.
            cost=Decimal(9000 + (i % 7) * 250),
            days=2,
            score=float(i % 11),
        )
        for i in range(n)
    ]

    # Every distinct pair is a long hop that exceeds the same-day travel limit,
    # so the adjacency constraint emits O(n^2) clauses per slot transition.
    travel_times: dict[tuple[str, str], timedelta] = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                travel_times[(f"City{i}", f"City{j}")] = timedelta(hours=12)

    constraints = TripConstraints(
        budget=Decimal(60000),
        duration_days=40,
        max_cities=n,
        max_hotel_changes=n,
        max_same_day_travel=timedelta(hours=6),
        dietary=[],
        interests=[],
        # Effectively zero solving time forces the timeout path.
        solver_timeout=timedelta(0),
    )
    return cities, constraints, travel_times


def test_solver_timeout_rejects_without_skeleton_and_blocks_itinerary_agent() -> None:
    """A timed-out solve returns a SolverTimeout rejection with no skeleton,
    and the Itinerary Agent is gated off (Requirement 9.9)."""
    cities, constraints, travel_times = _hard_problem()

    result = solve_skeleton(cities, constraints, travel_times=travel_times)

    # Timeout is reported as the SolverTimeout rejection subtype.
    assert result.rejection is not None
    assert isinstance(result.rejection, SolverTimeout)
    assert result.rejection.timed_out is True

    # No skeleton is emitted on timeout.
    assert result.skeleton is None
    assert result.feasible is False

    # The Itinerary Agent must not run without a skeleton.
    assert result.itinerary_agent_should_run is False
