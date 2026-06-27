"""Property-based test for personalized scoring weights (task 7.13).

Feature: wayfinder-travel-planner, Property 14: Personalized weights are
derived, renormalized, and blended.

*For any* user preference vector, the engine SHALL derive feature weights from
it, renormalize the derived weights to sum to 1.0 (100%), and blend them with
base default weights at a 70/30 ratio, such that the resulting weights are
non-negative and sum to 1.0 within tolerance 0.001.

Validates: Requirements 6.1, 6.2, 6.3

This is the universally-quantified counterpart to the example-based
personalization tests in ``test_decision_weights.py``. It lives in its own
module so the single Property-14 test sits next to the weights code it
validates without colliding with the other weight tests.

The output of ``personalized_weights`` is always a ``FeatureWeights``; that
Pydantic model's own validators enforce non-negativity and the unit-sum
tolerance. We assert the invariant directly here as well, so a regression that
weakened or bypassed the model validation would still be caught by this
property test rather than silently relying on construction-time validation.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.config import Settings
from app.decision.weights import (
    DEFAULT_BLEND_RATIO,
    TOPIC_TO_FEATURE,
    base_weights,
    map_preferences_to_features,
    personalized_weights,
    renormalize_to_one,
)
from app.models.schemas import (
    FEATURES,
    WEIGHT_SUM_TOLERANCE,
    FeatureWeights,
    PreferenceEntry,
)


def _settings() -> Settings:
    return Settings(_env_file=None)


def _sum(weights: FeatureWeights) -> float:
    return sum(getattr(weights, f) for f in FEATURES)


# Topics the engine recognizes (feature names + human-friendly aliases) plus a
# pool of unknown topics that contribute no signal. Mixing both stresses the
# derive -> renormalize -> blend pipeline across recognized and ignored input.
_known_topics = sorted(TOPIC_TO_FEATURE.keys())
_unknown_topics = ["nightlife", "shopping", "nightclubs", "spa", "casino", "zzz"]
_topic = st.sampled_from(_known_topics + _unknown_topics)

# PreferenceEntry constrains weight to [0.0, 1.0]; honor that here so generated
# vectors are themselves valid domain inputs.
_pref_weight = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# A preference vector may be supplied as a list of PreferenceEntry objects...
_pref_entry = st.builds(
    PreferenceEntry,
    topic=_topic,
    weight=_pref_weight,
    source=st.sampled_from(["explicit", "implicit"]),
)
_pref_entry_list = st.lists(_pref_entry, min_size=0, max_size=12)

# ...or as a plain {topic: weight} mapping.
_pref_mapping = st.dictionaries(keys=_topic, values=_pref_weight, max_size=12)

# ``None`` triggers the documented cold-start path. All three shapes are valid
# "user preference vectors" for the purposes of Property 14, and the invariant
# must hold for every one of them.
_pref_vector = st.one_of(st.none(), _pref_entry_list, _pref_mapping)


# ---------------------------------------------------------------------------
# Property 14: Personalized weights are derived, renormalized, and blended
# Feature: wayfinder-travel-planner, Property 14
# Validates: Requirements 6.1, 6.2, 6.3
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(pref_vector=_pref_vector)
def test_personalized_weights_are_non_negative_and_sum_to_one(pref_vector) -> None:
    """The blended personalized weights are non-negative and sum to 1.0.

    **Validates: Requirements 6.1, 6.2, 6.3**

    For any preference vector (list, mapping, or absent), the result is a fully
    valid ``FeatureWeights``: exactly the six features, each non-negative, and
    summing to 1.0 within the 0.001 tolerance.
    """
    base = base_weights(_settings())
    weights = personalized_weights(pref_vector, base=base, settings=_settings())

    assert isinstance(weights, FeatureWeights)
    # Every feature is present and non-negative (Requirement 6.1/6.2 outputs).
    for feature in FEATURES:
        value = getattr(weights, feature)
        assert value >= 0.0, f"{feature} is negative: {value}"
    # The blended weights renormalize back to 1.0 within tolerance (Req 6.2/6.3).
    assert abs(_sum(weights) - 1.0) <= WEIGHT_SUM_TOLERANCE, (
        f"weights sum to {_sum(weights)}, outside 1.0 +/- {WEIGHT_SUM_TOLERANCE}"
    )


@settings(max_examples=200)
@given(pref_vector=st.one_of(_pref_entry_list, _pref_mapping))
def test_personalized_weights_blend_is_70_30_of_renormalized_signals(
    pref_vector,
) -> None:
    """When a usable signal exists, the result is the renormalized 70/30 blend.

    **Validates: Requirements 6.1, 6.2, 6.3**

    Reconstructs the pipeline independently (derive -> renormalize -> blend
    70/30 -> renormalize) and asserts ``personalized_weights`` matches it. This
    pins down *how* the weights are produced, not merely that they are valid.
    Vectors with no recognized signal fall back to base and are skipped here;
    they are covered by the cold-start case in the invariant test above.
    """
    base = base_weights(_settings())
    base_map = {feature: getattr(base, feature) for feature in FEATURES}

    derived = renormalize_to_one(map_preferences_to_features(pref_vector))
    # No usable signal -> documented cold-start fallback (covered elsewhere).
    if derived is None:
        result = personalized_weights(pref_vector, base=base, settings=_settings())
        assert result == base
        return

    blended = {
        feature: DEFAULT_BLEND_RATIO * derived[feature]
        + (1.0 - DEFAULT_BLEND_RATIO) * base_map[feature]
        for feature in FEATURES
    }
    expected = renormalize_to_one(blended)
    assert expected is not None  # positive signal blended with valid base

    result = personalized_weights(pref_vector, base=base, settings=_settings())
    for feature in FEATURES:
        assert getattr(result, feature) == _approx(expected[feature]), (
            f"{feature}: {getattr(result, feature)} != expected {expected[feature]}"
        )


def _approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-9)
