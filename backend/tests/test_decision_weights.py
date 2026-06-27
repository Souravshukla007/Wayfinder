"""Unit tests for the Decision Engine feature weights (task 7.1).

Validates Requirements:
- 5.6: weight-config validation (non-negative, sum 1.0 within 0.001).
- 5.9: base weights read from configurable data.
- 6.1: derive personalized weights from the preference vector.
- 6.2: renormalize derived weights to sum to 1.0.
- 6.3: blend personalized/base at a 70/30 ratio.
- 6.4: cold-start falls back to base default weights.

These tests exercise the pure deterministic core: no LLM, randomness, time, or
network. They complement the property-based tests scheduled in tasks 7.10/7.13.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.decision.weights import (
    DEFAULT_BLEND_RATIO,
    base_weights,
    blend,
    map_preferences_to_features,
    personalized_weights,
    renormalize_to_one,
    validate_weights,
)
from app.models.schemas import FEATURES, WEIGHT_SUM_TOLERANCE, FeatureWeights, PreferenceEntry


def _settings() -> Settings:
    return Settings(_env_file=None)


def _sum(weights: FeatureWeights) -> float:
    return sum(getattr(weights, f) for f in FEATURES)


# --- Base weights from config (Req 5.9) -------------------------------------

def test_base_weights_loaded_from_config_defaults() -> None:
    weights = base_weights(_settings())
    assert weights.budget_fit == pytest.approx(0.25)
    assert weights.weather_fit == pytest.approx(0.20)
    assert weights.crowd_score == pytest.approx(0.15)
    assert weights.food_score == pytest.approx(0.15)
    assert weights.photography_score == pytest.approx(0.15)
    assert weights.travel_efficiency == pytest.approx(0.10)
    assert _sum(weights) == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)


def test_base_weights_are_configurable() -> None:
    settings = _settings()
    settings.base_weight_budget_fit = 0.50
    settings.base_weight_weather_fit = 0.10
    settings.base_weight_crowd_score = 0.10
    settings.base_weight_food_score = 0.10
    settings.base_weight_photography_score = 0.10
    settings.base_weight_travel_efficiency = 0.10
    weights = base_weights(settings)
    assert weights.budget_fit == pytest.approx(0.50)
    assert _sum(weights) == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)


# --- Weight-config validation (Req 5.6) -------------------------------------

def test_validate_weights_accepts_valid_mapping() -> None:
    valid = {
        "budget_fit": 0.25,
        "weather_fit": 0.20,
        "crowd_score": 0.15,
        "food_score": 0.15,
        "photography_score": 0.15,
        "travel_efficiency": 0.10,
    }
    result = validate_weights(valid)
    assert isinstance(result, FeatureWeights)
    assert _sum(result) == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)


def test_validate_weights_rejects_negative() -> None:
    bad = {
        "budget_fit": -0.05,
        "weather_fit": 0.25,
        "crowd_score": 0.20,
        "food_score": 0.15,
        "photography_score": 0.15,
        "travel_efficiency": 0.30,
    }
    with pytest.raises(ValidationError):
        validate_weights(bad)


def test_validate_weights_rejects_non_unit_sum() -> None:
    bad = {f: 0.5 for f in FEATURES}  # sums to 3.0
    with pytest.raises(ValidationError):
        validate_weights(bad)


def test_validate_weights_within_tolerance_accepted() -> None:
    # Sum = 1.0005, inside the 0.001 tolerance band.
    near = {
        "budget_fit": 0.2505,
        "weather_fit": 0.20,
        "crowd_score": 0.15,
        "food_score": 0.15,
        "photography_score": 0.15,
        "travel_efficiency": 0.10,
    }
    assert _sum(validate_weights(near)) == pytest.approx(1.0005)


def test_validate_weights_revalidates_existing_object() -> None:
    weights = base_weights(_settings())
    assert validate_weights(weights) == weights


# --- Derivation + renormalization (Req 6.1, 6.2) ----------------------------

def test_map_preferences_to_features_accumulates_known_topics() -> None:
    prefs = [
        PreferenceEntry(topic="food", weight=0.8),
        PreferenceEntry(topic="cuisine", weight=0.2),  # alias -> food_score
        PreferenceEntry(topic="budget", weight=0.5),
    ]
    derived = map_preferences_to_features(prefs)
    assert derived["food_score"] == pytest.approx(1.0)
    assert derived["budget_fit"] == pytest.approx(0.5)
    assert derived["weather_fit"] == 0.0


def test_map_preferences_ignores_unknown_topics() -> None:
    derived = map_preferences_to_features({"nightlife": 0.9, "budget": 0.3})
    assert derived["budget_fit"] == pytest.approx(0.3)
    assert sum(derived.values()) == pytest.approx(0.3)


def test_renormalize_to_one_sums_to_unit() -> None:
    out = renormalize_to_one({f: 2.0 for f in FEATURES})
    assert out is not None
    assert sum(out.values()) == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)


def test_renormalize_to_one_returns_none_for_zero_total() -> None:
    assert renormalize_to_one({f: 0.0 for f in FEATURES}) is None


# --- Blend (Req 6.3) --------------------------------------------------------

def test_blend_uses_70_30_ratio() -> None:
    personalized = {f: (1.0 if f == "food_score" else 0.0) for f in FEATURES}
    base = {f: (1.0 if f == "budget_fit" else 0.0) for f in FEATURES}
    blended = blend(personalized, base, DEFAULT_BLEND_RATIO)
    assert blended["food_score"] == pytest.approx(0.70)
    assert blended["budget_fit"] == pytest.approx(0.30)


# --- personalized_weights pipeline (Req 6.1-6.3, Property 14) ----------------

def test_personalized_weights_are_valid_and_unit_sum() -> None:
    prefs = [
        PreferenceEntry(topic="food", weight=0.9),
        PreferenceEntry(topic="photography", weight=0.6),
        PreferenceEntry(topic="budget", weight=0.3),
    ]
    weights = personalized_weights(prefs, settings=_settings())
    assert isinstance(weights, FeatureWeights)  # construction => validated
    assert _sum(weights) == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)
    assert all(getattr(weights, f) >= 0.0 for f in FEATURES)


def test_personalized_weights_shift_toward_preferred_feature() -> None:
    base = base_weights(_settings())
    prefs = [PreferenceEntry(topic="food", weight=1.0)]
    weights = personalized_weights(prefs, base=base, settings=_settings())
    # A strong single-topic signal should raise that feature above its base.
    assert weights.food_score > base.food_score


def test_personalized_weights_deterministic() -> None:
    prefs = [PreferenceEntry(topic="weather", weight=0.7), PreferenceEntry(topic="food", weight=0.4)]
    first = personalized_weights(prefs, settings=_settings())
    second = personalized_weights(prefs, settings=_settings())
    assert first == second


# --- Cold-start fallback (Req 6.4) ------------------------------------------

def test_cold_start_none_falls_back_to_base() -> None:
    base = base_weights(_settings())
    assert personalized_weights(None, base=base, settings=_settings()) == base


def test_cold_start_empty_vector_falls_back_to_base() -> None:
    base = base_weights(_settings())
    assert personalized_weights([], base=base, settings=_settings()) == base
    assert personalized_weights({}, base=base, settings=_settings()) == base


def test_cold_start_only_unknown_topics_falls_back_to_base() -> None:
    base = base_weights(_settings())
    weights = personalized_weights({"nightlife": 0.9}, base=base, settings=_settings())
    assert weights == base


def test_cold_start_all_zero_signal_falls_back_to_base() -> None:
    base = base_weights(_settings())
    prefs = [PreferenceEntry(topic="food", weight=0.0)]
    assert personalized_weights(prefs, base=base, settings=_settings()) == base
