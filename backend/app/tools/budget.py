"""Budget tool (Task 5.5, Requirement 3.5).

Pure arithmetic over trip costs - no providers, no LLM, no I/O. All monetary
math uses :class:`decimal.Decimal` for exactness. Functions are total and
deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal


def _to_decimal(value: Decimal | int | float | str) -> Decimal:
    """Coerce a numeric value to ``Decimal`` (via ``str`` for float safety)."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def total_cost(items: Iterable[Decimal | int | float | str]) -> Decimal:
    """Sum a collection of line-item costs."""
    return sum((_to_decimal(item) for item in items), Decimal("0"))


def remaining_budget(
    budget: Decimal | int | float | str, spent: Decimal | int | float | str
) -> Decimal:
    """Return ``budget - spent`` (may be negative when over budget)."""
    return _to_decimal(budget) - _to_decimal(spent)


def within_budget(
    budget: Decimal | int | float | str, spent: Decimal | int | float | str
) -> bool:
    """True when total spend does not exceed the budget (equality allowed)."""
    return _to_decimal(spent) <= _to_decimal(budget)


def per_day_budget(budget: Decimal | int | float | str, days: int) -> Decimal:
    """Return the per-day budget allowance. Raises on non-positive ``days``."""
    if days <= 0:
        raise ValueError("days must be positive")
    return _to_decimal(budget) / Decimal(days)


def budget_fit_ratio(
    budget: Decimal | int | float | str, spent: Decimal | int | float | str
) -> float:
    """Fraction of budget consumed as a float (``spent / budget``).

    A ratio <= 1.0 means within budget; > 1.0 means over. Raises on a
    non-positive budget where the ratio would be undefined.
    """
    budget_d = _to_decimal(budget)
    if budget_d <= 0:
        raise ValueError("budget must be positive")
    return float(_to_decimal(spent) / budget_d)
