"""The pure Travel Decision Engine scoring function (task 7.2).

This module is the deterministic heart of the Decision Engine. ``score_candidates``
is a **pure function** of ``(candidates, weights, constraints)``:

- No LLM, no randomness, no system clock, no network, no global mutable state.
- Same inputs always produce bit-for-bit identical outputs (Requirements 5.1, 5.7).

Pipeline per candidate (Requirements 5.2-5.5):
1. Reject invalid candidates -- any candidate missing or non-numeric on any of
   the six features is excluded, while every remaining valid candidate is still
   scored (Requirement 5.4).
2. Clamp each raw feature into its defined input bounds (Requirement 5.3).
3. Normalize each clamped feature to the inclusive range [0.0, 1.0]
   (Requirement 5.2).
4. Compute the final score as the weighted sum of the normalized features,
   scaled to the inclusive 0-100 range (Requirement 5.5).
5. Rank surviving candidates by final score in descending order (Requirement 7.1).

The weight configuration is validated up front (Requirement 5.6): a negative
weight or weights that do not sum to 1.0 within tolerance 0.001 raise before any
score is computed, so the engine never produces scores from an invalid config.

The per-feature normalized values **and** weighted contributions are returned on
each :class:`ScoredCandidate` so the explainability ledger (task 7.3) and
persistence (task 7.4) can build/store decision traces without recomputing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

from app.decision.weights import validate_weights
from app.models.schemas import FEATURES, FeatureWeights, TripConstraints

# Defined input bounds for each raw feature. Raw feature signals are expressed
# on a 0-100 scale; values outside the band are clamped to the nearest bound
# (Requirement 5.3) before being normalized to [0, 1] (Requirement 5.2). Bounds
# are centralized here so all features share one documented convention.
FEATURE_INPUT_MIN = 0.0
FEATURE_INPUT_MAX = 100.0
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    feature: (FEATURE_INPUT_MIN, FEATURE_INPUT_MAX) for feature in FEATURES
}

# Final scores live on an inclusive 0-100 scale (Requirement 5.5). Because the
# weights sum to 1.0 and each normalized feature lies in [0, 1], the weighted
# sum lies in [0, 1]; multiplying by this scale maps it to [0, 100].
FINAL_SCORE_SCALE = 100.0

# A weights argument may be a validated ``FeatureWeights`` or a raw mapping; it
# is validated via ``validate_weights`` before any scoring happens.
WeightsArg = Union[FeatureWeights, Mapping[str, float]]


@dataclass(frozen=True)
class RawCandidate:
    """An unscored destination candidate with raw feature signals.

    ``features`` maps feature name -> raw value. Values may be absent or
    non-numeric; such candidates are rejected by ``score_candidates`` while the
    valid ones are preserved (Requirement 5.4). ``metadata`` carries opaque
    fields (country, flag, etc.) through to the scored result untouched.
    """

    destination: str
    features: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredCandidate:
    """A scored destination with everything the ledger/persistence layers need.

    - ``score``: final score on the inclusive 0-100 scale.
    - ``normalized``: per-feature normalized value in [0, 1].
    - ``contributions``: per-feature weighted contribution to the final score;
      the contributions sum to ``score`` (up to float rounding), which the
      ledger asserts within tolerance 0.001 (task 7.3 / Requirement 8.2).
    """

    destination: str
    score: float
    normalized: Mapping[str, float]
    contributions: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RejectedCandidate:
    """A candidate excluded from scoring, with the reason it was rejected."""

    destination: str
    reason: str


@dataclass(frozen=True)
class ScoringResult:
    """The full, deterministic outcome of a scoring run.

    ``ranked`` is ordered by final score in descending order (Requirement 7.1);
    ``rejected`` lists invalid candidates that were excluded (Requirement 5.4).
    """

    ranked: list[ScoredCandidate]
    rejected: list[RejectedCandidate]


def _coerce_numeric(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if it is not numeric.

    Booleans are treated as non-numeric: a feature value of ``True``/``False``
    is not a valid measurement. ``NaN`` and infinities are rejected as well so
    they can never poison a score (Requirement 5.4).
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _extract_features(candidate: RawCandidate) -> dict[str, float] | str:
    """Validate and extract the six numeric features for a candidate.

    Returns a ``{feature: value}`` dict when every feature is present and
    numeric, otherwise a human-readable rejection reason naming the first
    offending feature (Requirement 5.4).
    """
    features = candidate.features
    extracted: dict[str, float] = {}
    for feature in FEATURES:
        if feature not in features:
            return f"missing feature '{feature}'"
        numeric = _coerce_numeric(features[feature])
        if numeric is None:
            return f"non-numeric value for feature '{feature}'"
        extracted[feature] = numeric
    return extracted


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]`` (Requirement 5.3)."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _normalize(value: float, low: float, high: float) -> float:
    """Normalize a clamped ``value`` from ``[low, high]`` to ``[0, 1]``.

    The input is clamped first, so the result is always within [0, 1]
    (Requirement 5.2). A degenerate (zero-width) band normalizes to 0.0.
    """
    span = high - low
    if span <= 0.0:
        return 0.0
    return (_clamp(value, low, high) - low) / span


def _score_one(
    candidate: RawCandidate,
    raw_features: Mapping[str, float],
    weights: FeatureWeights,
) -> ScoredCandidate:
    """Clamp, normalize, weight, and total one valid candidate's features."""
    normalized: dict[str, float] = {}
    contributions: dict[str, float] = {}
    for feature in FEATURES:
        low, high = FEATURE_BOUNDS[feature]
        norm = _normalize(raw_features[feature], low, high)
        normalized[feature] = norm
        # Weighted contribution on the final 0-100 scale; summing these over all
        # features yields the final score, which is what the ledger relies on.
        contributions[feature] = getattr(weights, feature) * norm * FINAL_SCORE_SCALE

    # The weighted sum lies in [0, 100] by construction, but floating-point
    # rounding in the weights/normalized values can nudge it a hair outside
    # (e.g. 100.00000000000001). Clamp to the documented inclusive 0-100 scale
    # so the engine's contract holds for every downstream consumer (Requirement
    # 5.5). The deviation is ~1e-14, far inside the ledger's 0.001 tolerance.
    score = _clamp(math.fsum(contributions.values()), 0.0, FINAL_SCORE_SCALE)
    return ScoredCandidate(
        destination=candidate.destination,
        score=score,
        normalized=normalized,
        contributions=contributions,
        metadata=candidate.metadata,
    )


def score_candidates(
    candidates: Sequence[RawCandidate],
    weights: WeightsArg,
    constraints: TripConstraints | None = None,
) -> ScoringResult:
    """Score and rank destination candidates deterministically.

    Args:
        candidates: raw candidates to score; invalid ones are excluded while
            valid ones are preserved (Requirement 5.4).
        weights: feature weights, validated up front -- a negative weight or a
            non-unit sum (outside tolerance 0.001) raises ``ValidationError``
            and no scores are produced (Requirement 5.6).
        constraints: trip constraints, accepted for interface completeness and
            future use; scoring itself does not depend on them, preserving
            purity and determinism (Requirements 5.1, 5.7).

    Returns:
        A :class:`ScoringResult` whose ``ranked`` list is ordered by final score
        in descending order (Requirement 7.1) and whose ``rejected`` list names
        every excluded candidate.

    The function is pure: no LLM, randomness, system time, or network access, so
    identical inputs yield identical outputs (Requirements 5.1, 5.7).
    """
    # Requirement 5.6: reject an invalid weight configuration before scoring.
    valid_weights = validate_weights(weights)

    scored: list[ScoredCandidate] = []
    rejected: list[RejectedCandidate] = []

    for candidate in candidates:
        extracted = _extract_features(candidate)
        if isinstance(extracted, str):
            rejected.append(RejectedCandidate(candidate.destination, extracted))
            continue
        scored.append(_score_one(candidate, extracted, valid_weights))

    # Requirement 7.1: rank descending by final score. Python's sort is stable,
    # so candidates with equal scores keep their input order -- deterministic for
    # a fixed input (Requirements 5.1, 5.7).
    ranked = sorted(scored, key=lambda c: c.score, reverse=True)

    return ScoringResult(ranked=ranked, rejected=rejected)


__all__ = [
    "FEATURE_BOUNDS",
    "FEATURE_INPUT_MIN",
    "FEATURE_INPUT_MAX",
    "FINAL_SCORE_SCALE",
    "RawCandidate",
    "ScoredCandidate",
    "RejectedCandidate",
    "ScoringResult",
    "score_candidates",
]
