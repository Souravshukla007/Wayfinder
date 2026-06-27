"""Feature weights: base defaults, validation, and personalization blend.

This is part of the deterministic Decision Engine core. Every function here is
**pure**: no LLM, no randomness, no system time, no network. Given the same
inputs (and the same configured base weights) the outputs are identical.

Responsibilities (task 7.1):
- Load base default :class:`FeatureWeights` from configuration (Requirement
  5.9 — weights come from configurable data, not hardcoded in the engine).
- Validate a weight configuration: non-negative and summing to 1.0 within a
  tolerance of 0.001. Validation reuses the ``FeatureWeights`` Pydantic
  validator from ``app.models.schemas`` so behavior stays consistent
  (Requirement 5.6).
- ``personalized_weights``: derive per-feature signals from a user preference
  vector, renormalize to 1.0, blend 70/30 with the base weights, then
  renormalize again to 1.0 (Requirements 6.1, 6.2, 6.3).
- Cold-start: when no usable personalization signal exists, fall back to the
  base default weights (Requirement 6.4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Union

from app.config import Settings, get_settings
from app.models.schemas import (
    FEATURES,
    WEIGHT_SUM_TOLERANCE,
    FeatureWeights,
    PreferenceEntry,
)

# Fraction of the blended weights taken from the personalized signal; the
# remainder (1 - ratio) comes from the base default weights (Requirement 6.3).
DEFAULT_BLEND_RATIO = 0.70

# Maps user preference topics onto Decision Engine features. Topics that match
# a feature name map to themselves; human-friendly aliases map to the closest
# feature. Topics not present here contribute no personalization signal.
TOPIC_TO_FEATURE: dict[str, str] = {feature: feature for feature in FEATURES}
TOPIC_TO_FEATURE.update(
    {
        "budget": "budget_fit",
        "cost": "budget_fit",
        "affordability": "budget_fit",
        "weather": "weather_fit",
        "climate": "weather_fit",
        "crowd": "crowd_score",
        "crowds": "crowd_score",
        "quiet": "crowd_score",
        "food": "food_score",
        "cuisine": "food_score",
        "dining": "food_score",
        "photography": "photography_score",
        "photos": "photography_score",
        "scenery": "photography_score",
        "travel_efficiency": "travel_efficiency",
        "efficiency": "travel_efficiency",
        "convenience": "travel_efficiency",
        "transit": "travel_efficiency",
    }
)

# A preference vector may be supplied as a sequence of PreferenceEntry objects
# or as a plain {topic: weight} mapping. ``None``/empty triggers cold-start.
PreferenceVector = Union[Sequence[PreferenceEntry], Mapping[str, float], None]


def base_weights(settings: Settings | None = None) -> FeatureWeights:
    """Return the base default feature weights from configuration.

    Reads configurable data (Requirement 5.9). The returned object is validated
    by the ``FeatureWeights`` schema, so a misconfigured base set (negative or
    not summing to 1.0) fails fast here rather than silently skewing scores.
    """
    settings = settings or get_settings()
    return FeatureWeights(**settings.base_feature_weights())


def validate_weights(weights: FeatureWeights | Mapping[str, float]) -> FeatureWeights:
    """Validate a weight configuration, reusing the ``FeatureWeights`` rules.

    Accepts either an existing ``FeatureWeights`` (re-validated) or a raw
    ``{feature: weight}`` mapping. Raises ``pydantic.ValidationError`` when any
    weight is negative or the weights do not sum to 1.0 within tolerance 0.001
    (Requirement 5.6).
    """
    if isinstance(weights, FeatureWeights):
        # Re-run validators by reconstructing from the current values.
        return FeatureWeights(**{f: getattr(weights, f) for f in FEATURES})
    return FeatureWeights(**dict(weights))


def _as_topic_weights(pref_vector: PreferenceVector) -> dict[str, float]:
    """Normalize the accepted preference-vector shapes to a {topic: weight} map."""
    if pref_vector is None:
        return {}
    if isinstance(pref_vector, Mapping):
        return {str(topic): float(weight) for topic, weight in pref_vector.items()}
    topic_weights: dict[str, float] = {}
    for entry in pref_vector:
        # Combine duplicate topics additively (last-writer-free, order-independent).
        topic_weights[entry.topic] = topic_weights.get(entry.topic, 0.0) + float(entry.weight)
    return topic_weights


def map_preferences_to_features(pref_vector: PreferenceVector) -> dict[str, float]:
    """Derive a per-feature signal from a preference vector (Requirement 6.1).

    Each preference topic is mapped to its feature via ``TOPIC_TO_FEATURE`` and
    its (non-negative) weight accumulated onto that feature. Unknown topics are
    ignored. Features with no contributing signal default to 0.0.
    """
    derived: dict[str, float] = {feature: 0.0 for feature in FEATURES}
    for topic, weight in _as_topic_weights(pref_vector).items():
        feature = TOPIC_TO_FEATURE.get(topic)
        if feature is None:
            continue
        # Guard against negative signals so derived weights stay non-negative.
        derived[feature] += max(0.0, weight)
    return derived


def renormalize_to_one(weights: Mapping[str, float]) -> dict[str, float] | None:
    """Scale weights so they sum to 1.0 (Requirement 6.2).

    Returns ``None`` when the total is not positive (no usable signal), letting
    callers fall back to cold-start defaults rather than dividing by zero.
    """
    total = sum(max(0.0, weights.get(feature, 0.0)) for feature in FEATURES)
    if total <= 0.0:
        return None
    return {feature: max(0.0, weights.get(feature, 0.0)) / total for feature in FEATURES}


def blend(
    personalized: Mapping[str, float],
    base: Mapping[str, float],
    ratio: float = DEFAULT_BLEND_RATIO,
) -> dict[str, float]:
    """Blend personalized and base weights at ``ratio``/``1 - ratio`` (Req 6.3)."""
    return {
        feature: ratio * personalized.get(feature, 0.0)
        + (1.0 - ratio) * base.get(feature, 0.0)
        for feature in FEATURES
    }


def personalized_weights(
    pref_vector: PreferenceVector,
    base: FeatureWeights | None = None,
    *,
    ratio: float = DEFAULT_BLEND_RATIO,
    settings: Settings | None = None,
) -> FeatureWeights:
    """Compute personalized feature weights for a user.

    Pipeline (Requirements 6.1–6.3):
        derive signals -> renormalize to 1.0 -> blend 70/30 with base ->
        renormalize to 1.0.

    Cold-start (Requirement 6.4): when ``pref_vector`` yields no usable signal
    (absent, empty, all-zero, or only unknown topics), the base default weights
    are returned unchanged.

    The result is always a validated ``FeatureWeights`` (non-negative, summing
    to 1.0 within tolerance), satisfying Property 14.
    """
    base = base or base_weights(settings)
    base_map = {feature: getattr(base, feature) for feature in FEATURES}

    derived = renormalize_to_one(map_preferences_to_features(pref_vector))
    if derived is None:
        # Cold-start: no personalization signal -> base defaults.
        return base

    blended = blend(derived, base_map, ratio)
    renormalized = renormalize_to_one(blended)
    if renormalized is None:
        # Defensive: a positive-signal derived vector blended with a valid base
        # cannot sum to zero, but fall back to base rather than fail.
        return base

    return FeatureWeights(**renormalized)


__all__ = [
    "DEFAULT_BLEND_RATIO",
    "TOPIC_TO_FEATURE",
    "WEIGHT_SUM_TOLERANCE",
    "PreferenceVector",
    "base_weights",
    "validate_weights",
    "map_preferences_to_features",
    "renormalize_to_one",
    "blend",
    "personalized_weights",
]
