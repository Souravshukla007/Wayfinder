"""Property-based test for weight-configuration validation (task 7.10).

Feature: wayfinder-travel-planner, Property 6: Weight-configuration validation.

*For any* weight configuration containing a negative weight or whose weights do
not sum to 1.0 within tolerance 0.001, the engine SHALL reject the configuration
and compute no scores; *for any* configuration with non-negative weights summing
to 1.0 within tolerance, the engine SHALL accept it.

Validates: Requirements 5.6

This is a property test only. It exercises ``app.decision.weights.validate_weights``
(which reuses the ``FeatureWeights`` Pydantic validators) without modifying the
implementation. "Reject" means a ``pydantic.ValidationError`` is raised so no
scores can be computed from the configuration; "accept" means a validated
``FeatureWeights`` is returned whose values round-trip the input.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.decision.weights import validate_weights
from app.models.schemas import FEATURES, WEIGHT_SUM_TOLERANCE, FeatureWeights

# A finite, well-bounded float. We keep magnitudes modest so that sums stay in
# a range where the 0.001 tolerance is meaningful and not lost to float noise.
_finite = st.floats(
    min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False
)


def _as_config(values: list[float]) -> dict[str, float]:
    """Pair six raw weight values with the six feature names."""
    return {feature: value for feature, value in zip(FEATURES, values)}


def _sum(config: dict[str, float]) -> float:
    return sum(config[feature] for feature in FEATURES)


# --- Generators -------------------------------------------------------------


@st.composite
def valid_configs(draw) -> dict[str, float]:
    """Non-negative weights that sum to 1.0 within tolerance.

    Six non-negative values are drawn and renormalized so their total is 1.0.
    Renormalization can leave a sub-tolerance float residue, which is exactly
    the accepted band Property 6 cares about.
    """
    raw = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
            min_size=len(FEATURES),
            max_size=len(FEATURES),
        )
    )
    total = sum(raw)
    # Need a positive total to renormalize; reject the all-zero draw.
    assume(total > 1e-6)
    normalized = [value / total for value in raw]
    # Guard against any pathological residue exceeding tolerance.
    assume(abs(sum(normalized) - 1.0) <= WEIGHT_SUM_TOLERANCE)
    return _as_config(normalized)


@st.composite
def negative_weight_configs(draw) -> dict[str, float]:
    """Configs with at least one strictly-negative weight.

    The sum is left unconstrained: a negative weight alone must trigger
    rejection regardless of whether the total happens to land near 1.0.
    """
    values = draw(
        st.lists(_finite, min_size=len(FEATURES), max_size=len(FEATURES))
    )
    # Force at least one strictly-negative weight at a random position.
    index = draw(st.integers(min_value=0, max_value=len(FEATURES) - 1))
    magnitude = draw(
        st.floats(min_value=1e-3, max_value=1_000.0, allow_nan=False, allow_infinity=False)
    )
    values[index] = -magnitude
    return _as_config(values)


@st.composite
def wrong_sum_configs(draw) -> dict[str, float]:
    """Non-negative weights whose total is clearly outside 1.0 +/- 0.001.

    A target total is drawn from a band that excludes [1-tol, 1+tol] with a
    safety margin, then non-negative values are scaled to hit that total. This
    isolates the "does not sum to 1.0" rejection cause from the negative cause.
    """
    margin = 0.01  # comfortably outside the 0.001 tolerance
    target = draw(
        st.one_of(
            st.floats(min_value=0.0, max_value=1.0 - margin, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0 + margin, max_value=50.0, allow_nan=False, allow_infinity=False),
        )
    )
    raw = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
            min_size=len(FEATURES),
            max_size=len(FEATURES),
        )
    )
    base = sum(raw)
    if base <= 1e-6:
        # All-zero draw: distribute the target evenly so it is reachable.
        values = [target / len(FEATURES)] * len(FEATURES)
    else:
        values = [value / base * target for value in raw]
    # Confirm the realized sum is genuinely outside the accepted band.
    assume(abs(sum(values) - 1.0) > WEIGHT_SUM_TOLERANCE)
    return _as_config(values)


# ---------------------------------------------------------------------------
# Property 6: Weight-configuration validation
# Feature: wayfinder-travel-planner, Property 6
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(config=valid_configs())
def test_valid_weight_configs_are_accepted(config: dict[str, float]) -> None:
    """Non-negative weights summing to 1.0 within tolerance are accepted.

    **Validates: Requirements 5.6**

    Acceptance returns a ``FeatureWeights`` whose six values equal the input,
    confirming the configuration is usable for scoring (no rejection).
    """
    result = validate_weights(config)

    assert isinstance(result, FeatureWeights)
    for feature in FEATURES:
        assert getattr(result, feature) == config[feature]
    # Sanity: the accepted config really satisfies the documented bounds.
    assert all(config[feature] >= 0.0 for feature in FEATURES)
    assert abs(_sum(config) - 1.0) <= WEIGHT_SUM_TOLERANCE


@settings(max_examples=200)
@given(config=negative_weight_configs())
def test_negative_weight_configs_are_rejected(config: dict[str, float]) -> None:
    """Any negative weight causes rejection; no scores are computed.

    **Validates: Requirements 5.6**
    """
    assert any(config[feature] < 0.0 for feature in FEATURES)
    with pytest.raises(ValidationError):
        validate_weights(config)


@settings(max_examples=200)
@given(config=wrong_sum_configs())
def test_wrong_sum_configs_are_rejected(config: dict[str, float]) -> None:
    """Weights not summing to 1.0 within tolerance are rejected.

    **Validates: Requirements 5.6**
    """
    assert abs(_sum(config) - 1.0) > WEIGHT_SUM_TOLERANCE
    with pytest.raises(ValidationError):
        validate_weights(config)


@settings(max_examples=300)
@given(
    values=st.lists(_finite, min_size=len(FEATURES), max_size=len(FEATURES))
)
def test_validation_decision_matches_definition(values: list[float]) -> None:
    """Accept iff non-negative AND sum within tolerance; reject otherwise.

    **Validates: Requirements 5.6**

    This is the full biconditional over arbitrary finite inputs: the validator's
    accept/reject decision must agree with the Requirement-5.6 predicate on
    every configuration, not just the curated valid/invalid generators above.
    """
    config = _as_config(values)
    non_negative = all(value >= 0.0 for value in values)
    sums_to_one = abs(sum(values) - 1.0) <= WEIGHT_SUM_TOLERANCE
    should_accept = non_negative and sums_to_one

    if should_accept:
        result = validate_weights(config)
        assert isinstance(result, FeatureWeights)
        for feature in FEATURES:
            assert getattr(result, feature) == config[feature]
    else:
        with pytest.raises(ValidationError):
            validate_weights(config)


@settings(max_examples=200)
@given(config=valid_configs())
def test_accepted_featureweights_revalidate(config: dict[str, float]) -> None:
    """A returned ``FeatureWeights`` re-validates (idempotent acceptance).

    **Validates: Requirements 5.6**

    Passing an already-accepted ``FeatureWeights`` back through ``validate_weights``
    must also be accepted and preserve every value.
    """
    first = validate_weights(config)
    second = validate_weights(first)
    assert isinstance(second, FeatureWeights)
    for feature in FEATURES:
        assert math.isclose(
            getattr(second, feature), getattr(first, feature), abs_tol=0.0, rel_tol=0.0
        )
