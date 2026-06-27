"""CP-SAT Constraint Solver (Google OR-Tools).

The solver takes the ranked destination candidates plus the trip's hard
constraints and either:

* emits a **provably feasible** :class:`ItinerarySkeleton` that satisfies every
  enforced hard constraint, or
* returns a :class:`SolverRejection` listing the specific unsatisfied
  constraints, or
* returns a :class:`SolverTimeout` when the solver exceeds the configured
  ``solver_timeout``.

In **every** rejection/timeout path the result carries **no skeleton**, which is
the signal the orchestrator uses to keep the Itinerary Agent from running
(Requirements 9.8, 9.9). Use :meth:`SolverResult.itinerary_agent_should_run`
to gate that stage.

Hard constraints modeled (Requirement 9.1-9.6):

1. **Budget cap** — total cost ``<=`` budget (equality accepted; any overshoot
   rejected).
2. **Same-day travel-time feasibility** — the emitted skeleton schedules city
   stays back-to-back, so each inter-city hop happens same-day; therefore two
   cities whose inter-city travel time exceeds ``max_same_day_travel`` may not
   be adjacent.
3. **Max hotel changes** — count of hotel changes ``<=`` limit. Each additional
   city stay after the first is one hotel change, so ``changes = cities - 1``.
4. **City-count limit** — number of distinct cities ``<=`` limit.
5. **Dietary exclusions** — every city that violates any specified dietary
   constraint is excluded.
6. **Interest consistency** — only cities consistent with the specified
   interests are included.

Determinism: the model is solved single-threaded with a fixed random seed so
the same inputs always yield the same skeleton.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from pydantic import BaseModel, Field

from app.models.schemas import TripConstraints
from app.orchestration.state import (
    ItineraryLeg,
    ItinerarySkeleton,
    SolverRejection,
)

try:  # OR-Tools is an optional-at-import-time dependency in some environments.
    from ortools.sat.python import cp_model

    _ORTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without OR-Tools.
    cp_model = None  # type: ignore[assignment]
    _ORTOOLS_AVAILABLE = False


__all__ = [
    "CandidateCity",
    "SolverRejection",
    "SolverTimeout",
    "SolverResult",
    "solve_skeleton",
]

# Costs/budgets are scaled to integer minor units (paise) for the integer-only
# CP-SAT model, preserving two decimal places.
_MONEY_SCALE = 100

# Constraint reason labels used in SolverRejection.unsatisfied_constraints.
_BUDGET = "budget_cap"
_TRAVEL = "same_day_travel_time"
_HOTEL = "max_hotel_changes"
_CITIES = "city_count_limit"
_DIETARY = "dietary_exclusions"
_INTERESTS = "interest_consistency"
_NO_CANDIDATES = "no_candidate_cities"


# --------------------------------------------------------------------------- #
# Solver input
# --------------------------------------------------------------------------- #
class CandidateCity(BaseModel):
    """A candidate city the solver may place into the skeleton.

    Carries the planning data the CP-SAT model needs but that the ranked
    :class:`~app.orchestration.state.ScoredCandidate` does not hold: the cost of
    including the city, how many days it occupies, and the dietary/interest tags
    used for the exclusion and consistency constraints. The Destination Agent /
    Decision Engine populate these from the provider/tool layer.
    """

    name: str
    # Cost of including this city in the trip (flights + lodging + activities
    # estimate). Compared against the trip budget for the hard budget cap.
    cost: Decimal = Decimal(0)
    # Number of trip days this city occupies (used to lay out arrival/departure
    # days in the emitted skeleton). At least 1 when the city is selected.
    days: int = Field(default=1, ge=1)
    # Decision Engine score (0-100); used only as the optimization objective so
    # the solver prefers higher-ranked cities. Never affects feasibility.
    score: float = 0.0
    # Dietary constraints this city cannot satisfy. If any intersects the trip's
    # dietary constraints, the city is excluded.
    dietary_conflicts: list[str] = Field(default_factory=list)
    # Interests this city satisfies. When the trip specifies interests, a city
    # with tags must intersect them; an untagged city is treated as neutral
    # (consistent).
    interest_tags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Solver outputs
# --------------------------------------------------------------------------- #
class SolverTimeout(SolverRejection):
    """A rejection produced when the solver exceeds its time budget.

    A specialization of :class:`SolverRejection` with ``timed_out`` set, kept as
    a distinct type so callers can branch on a timeout specifically while still
    treating it as a rejection (no skeleton; Itinerary Agent must not run).
    """

    timed_out: bool = True


class SolverResult(BaseModel):
    """Outcome of a solve: a feasible skeleton XOR a rejection.

    Exactly one of ``skeleton`` / ``rejection`` is set. The Itinerary Agent runs
    only when a skeleton is present (Requirements 9.8, 9.9).
    """

    skeleton: Optional[ItinerarySkeleton] = None
    rejection: Optional[SolverRejection] = None

    @property
    def feasible(self) -> bool:
        """True only when a feasible skeleton was emitted."""
        return self.skeleton is not None and self.rejection is None

    @property
    def itinerary_agent_should_run(self) -> bool:
        """Gate for the Itinerary Agent: run only on a feasible skeleton."""
        return self.feasible

    @classmethod
    def feasible_result(cls, skeleton: ItinerarySkeleton) -> "SolverResult":
        return cls(skeleton=skeleton, rejection=None)

    @classmethod
    def rejected(cls, rejection: SolverRejection) -> "SolverResult":
        # Defensive: a rejection NEVER carries a skeleton, so the Itinerary
        # Agent cannot run off a rejected plan.
        return cls(skeleton=None, rejection=rejection)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scale_money(value: Decimal) -> int:
    """Scale a money value to integer minor units (paise), half-up rounded."""
    return int((Decimal(value) * _MONEY_SCALE).to_integral_value(rounding=ROUND_HALF_UP))


def _seconds(value: timedelta) -> float:
    return value.total_seconds()


def _is_dietary_ok(city: CandidateCity, dietary: list[str]) -> bool:
    """A city is dietary-eligible when it conflicts with none of the trip's
    dietary constraints (Requirement 9.5)."""
    if not dietary:
        return True
    conflicts = set(city.dietary_conflicts)
    return conflicts.isdisjoint(dietary)


def _is_interest_ok(city: CandidateCity, interests: list[str]) -> bool:
    """A city is interest-consistent when the trip specifies no interests, the
    city carries no interest tags (neutral), or its tags intersect the trip's
    interests (Requirement 9.6)."""
    if not interests:
        return True
    if not city.interest_tags:
        return True
    return not set(city.interest_tags).isdisjoint(interests)


def _distribute_days(total_days: int, num_cities: int) -> list[int]:
    """Split ``total_days`` across ``num_cities`` as evenly as possible.

    Earlier cities receive the extra day when the split is uneven. Guarantees
    every city gets at least one day when ``total_days >= num_cities``.
    """
    if num_cities <= 0:
        return []
    base = total_days // num_cities
    remainder = total_days % num_cities
    days = [base + (1 if i < remainder else 0) for i in range(num_cities)]
    # Ensure each selected city has at least one day even when the trip is very
    # short relative to the number of cities.
    return [max(1, d) for d in days]


def _build_skeleton(
    ordered_cities: list[CandidateCity],
    total_days: int,
) -> ItinerarySkeleton:
    """Construct an :class:`ItinerarySkeleton` from the chosen city order."""
    per_city_days = _distribute_days(total_days, len(ordered_cities))
    legs: list[ItineraryLeg] = []
    cursor = 1  # 1-indexed trip day.
    total_cost = Decimal(0)
    for index, city in enumerate(ordered_cities):
        stay = per_city_days[index]
        arrival = cursor
        departure = cursor + stay - 1
        legs.append(
            ItineraryLeg(
                city=city.name,
                arrival_day=arrival,
                departure_day=departure,
                # Each city after the first is reached by changing hotels.
                hotel_change=index > 0,
            )
        )
        total_cost += Decimal(city.cost)
        cursor = departure + 1
    hotel_changes = max(0, len(ordered_cities) - 1)
    return ItinerarySkeleton(
        legs=legs,
        total_cost=total_cost,
        hotel_changes=hotel_changes,
    )


def _travel_seconds(
    travel_times: Mapping[tuple[str, str], timedelta],
    a: str,
    b: str,
) -> float:
    """Inter-city travel duration (seconds) for the hop a -> b.

    Looks up both orderings; an unknown pair defaults to 0, imposing no
    same-day-travel restriction on that hop.
    """
    td = travel_times.get((a, b))
    if td is None:
        td = travel_times.get((b, a))
    return _seconds(td) if td is not None else 0.0


def _diagnose(
    candidates: list[CandidateCity],
    constraints: TripConstraints,
) -> list[str]:
    """Best-effort identification of which hard constraints block a plan.

    Used to populate ``SolverRejection.unsatisfied_constraints`` when the model
    is infeasible. Always returns at least one reason.
    """
    reasons: list[str] = []
    if not candidates:
        return [_NO_CANDIDATES]

    if constraints.max_cities < 1:
        reasons.append(_CITIES)

    dietary_ok = [c for c in candidates if _is_dietary_ok(c, constraints.dietary)]
    interest_ok = [c for c in candidates if _is_interest_ok(c, constraints.interests)]
    eligible = [
        c
        for c in candidates
        if _is_dietary_ok(c, constraints.dietary)
        and _is_interest_ok(c, constraints.interests)
    ]

    if not eligible:
        # Attribute the empty eligible set to the filter(s) responsible.
        if not dietary_ok:
            reasons.append(_DIETARY)
        if not interest_ok:
            reasons.append(_INTERESTS)
        if dietary_ok and interest_ok:
            # Each filter alone leaves options, but their intersection is empty.
            reasons.append(_DIETARY)
            reasons.append(_INTERESTS)
        return reasons or [_DIETARY, _INTERESTS]

    # With at least one eligible city, a single-city plan needs the cheapest
    # eligible city to fit the budget.
    budget_scaled = _scale_money(constraints.budget)
    cheapest = min(_scale_money(c.cost) for c in eligible)
    if cheapest > budget_scaled:
        reasons.append(_BUDGET)

    if not reasons:
        # Fallback: the combination of constraints is jointly unsatisfiable for
        # a reason not isolated above (e.g. travel-time interactions across the
        # required multi-city structure).
        reasons.append(_TRAVEL)
    return reasons


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def solve_skeleton(
    candidates: list[CandidateCity],
    constraints: TripConstraints,
    travel_times: Optional[Mapping[tuple[str, str], timedelta]] = None,
    solver_timeout: Optional[timedelta] = None,
) -> SolverResult:
    """Solve for a feasible itinerary skeleton under the trip's hard constraints.

    Args:
        candidates: Candidate cities (with cost/days/dietary/interest data).
        constraints: The trip's hard constraints (budget, max cities, max hotel
            changes, max same-day travel, dietary, interests, solver timeout).
        travel_times: Optional inter-city travel durations keyed by
            ``(city_a, city_b)``. Missing pairs impose no same-day restriction.
        solver_timeout: Optional override for the solver time budget; defaults
            to ``constraints.solver_timeout``.

    Returns:
        A :class:`SolverResult` carrying either a feasible skeleton or a
        rejection/timeout. A rejection/timeout never carries a skeleton, which
        keeps the Itinerary Agent from running.

    Raises:
        RuntimeError: If OR-Tools is not importable in this environment.
    """
    if not _ORTOOLS_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "OR-Tools (ortools) is required for the CP-SAT solver but is not "
            "installed in this environment."
        )

    travel_times = travel_times or {}
    timeout = solver_timeout or constraints.solver_timeout

    # No candidates => nothing to place; reject (no skeleton).
    if not candidates:
        return SolverResult.rejected(
            SolverRejection(
                unsatisfied_constraints=[_NO_CANDIDATES],
                message="No candidate cities were supplied to the solver.",
            )
        )

    n = len(candidates)
    # Slots are filled contiguously from 0; cap at the city-count limit so the
    # model can never select more than allowed.
    num_slots = max(1, min(n, constraints.max_cities)) if constraints.max_cities >= 1 else 0

    if num_slots == 0:
        # The city-count limit forbids selecting any city.
        return SolverResult.rejected(
            SolverRejection(
                unsatisfied_constraints=[_CITIES],
                message="city-count limit is below 1; no city may be selected.",
            )
        )

    model = cp_model.CpModel()

    # assign[s][c] == 1 iff slot s holds candidate c.
    assign: list[list[cp_model.IntVar]] = [
        [model.NewBoolVar(f"assign_s{s}_c{c}") for c in range(n)] for s in range(num_slots)
    ]

    # Each slot holds at most one city.
    for s in range(num_slots):
        model.Add(sum(assign[s][c] for c in range(n)) <= 1)

    # Each city occupies at most one slot.
    for c in range(n):
        model.Add(sum(assign[s][c] for s in range(num_slots)) <= 1)

    # Per-slot occupancy expressions.
    slot_filled = [sum(assign[s][c] for c in range(n)) for s in range(num_slots)]

    # Fill slots contiguously from slot 0 (no gaps): slot s used >= slot s+1.
    for s in range(num_slots - 1):
        model.Add(slot_filled[s] >= slot_filled[s + 1])

    # Total number of selected cities.
    total_filled = sum(slot_filled)

    # At least one city in any emitted skeleton.
    model.Add(total_filled >= 1)

    # City-count limit (Requirement 9.4).
    model.Add(total_filled <= constraints.max_cities)

    # Max hotel changes (Requirement 9.3): changes == cities - 1.
    model.Add(total_filled <= constraints.max_hotel_changes + 1)

    # Dietary exclusions (9.5) and interest consistency (9.6): forbid ineligible
    # cities from being selected at all.
    for c, city in enumerate(candidates):
        if not _is_dietary_ok(city, constraints.dietary) or not _is_interest_ok(
            city, constraints.interests
        ):
            model.Add(sum(assign[s][c] for s in range(num_slots)) == 0)

    # Budget cap (Requirement 9.1): total cost <= budget (equality accepted).
    budget_scaled = _scale_money(constraints.budget)
    cost_terms = [
        assign[s][c] * _scale_money(candidates[c].cost)
        for s in range(num_slots)
        for c in range(n)
    ]
    if cost_terms:
        model.Add(sum(cost_terms) <= budget_scaled)

    # Same-day travel-time feasibility (Requirement 9.2): adjacent slots form a
    # same-day hop, so cities whose travel time exceeds the limit may not be
    # adjacent in either order.
    max_travel_seconds = _seconds(constraints.max_same_day_travel)
    for s in range(num_slots - 1):
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if _travel_seconds(travel_times, candidates[a].name, candidates[b].name) > max_travel_seconds:
                    model.Add(assign[s][a] + assign[s + 1][b] <= 1)

    # Objective: prefer higher-scored cities (deterministic; never affects
    # feasibility). Scores scaled to integers.
    score_terms = [
        assign[s][c] * int(round(candidates[c].score * 1000))
        for s in range(num_slots)
        for c in range(n)
    ]
    if score_terms:
        model.Maximize(sum(score_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.0, _seconds(timeout))
    # Deterministic, reproducible solves.
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Recover the chosen cities in slot order.
        ordered: list[CandidateCity] = []
        for s in range(num_slots):
            for c in range(n):
                if solver.Value(assign[s][c]) == 1:
                    ordered.append(candidates[c])
                    break
        skeleton = _build_skeleton(ordered, constraints.duration_days)
        return SolverResult.feasible_result(skeleton)

    if status == cp_model.INFEASIBLE:
        reasons = _diagnose(candidates, constraints)
        return SolverResult.rejected(
            SolverRejection(
                unsatisfied_constraints=reasons,
                message="No itinerary satisfies all hard constraints simultaneously.",
            )
        )

    # UNKNOWN (time limit hit before a result) or MODEL_INVALID => no result.
    if status == cp_model.UNKNOWN:
        return SolverResult.rejected(
            SolverTimeout(
                unsatisfied_constraints=[],
                message=(
                    "Solver exceeded the configured solver_timeout of "
                    f"{_seconds(timeout):.3f}s without a conclusive result."
                ),
            )
        )

    # MODEL_INVALID or any other unexpected status: reject without a skeleton.
    return SolverResult.rejected(
        SolverRejection(
            unsatisfied_constraints=[],
            message=f"Solver returned an unusable status ({solver.StatusName(status)}).",
        )
    )
