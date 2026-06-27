"""Unit tests for config-driven settings loading (task 1.1).

Validates Requirements:
- 2.5: provider selection driven from configuration without code changes.
- 4.2: LLM vendor selected from configuration.

Core rule under test: *absent provider credentials default to mock.*

Tests construct ``Settings`` with ``_env_file=None`` so they are isolated from
any on-disk ``.env`` file and depend only on the environment that the
autouse ``_clean_provider_env`` fixture controls.
"""

from __future__ import annotations

import pytest

from app.config import PROVIDER_DOMAINS, ProviderKind, Settings, get_settings


def _settings(**env: str) -> Settings:
    """Build Settings from explicit env vars, ignoring any .env file."""
    import os

    for key, value in env.items():
        os.environ[key] = value
    try:
        return Settings(_env_file=None)
    finally:
        for key in env:
            os.environ.pop(key, None)


# --- Default behaviour: no credentials => everything mock -------------------

def test_absent_credentials_default_all_domains_to_mock() -> None:
    settings = _settings()
    selection = settings.provider_selection()
    assert set(selection) == set(PROVIDER_DOMAINS)
    assert all(kind is ProviderKind.MOCK for kind in selection.values())
    for domain in PROVIDER_DOMAINS:
        assert settings.uses_mock(domain) is True


# --- Amadeus needs BOTH key and secret --------------------------------------

def test_amadeus_partial_credentials_stay_mock() -> None:
    settings = _settings(AMADEUS_API_KEY="key-only")
    assert settings.amadeus_configured is False
    assert settings.flight_provider_kind() is ProviderKind.MOCK
    assert settings.hotel_provider_kind() is ProviderKind.MOCK


def test_amadeus_full_credentials_select_real_for_flights_and_hotels() -> None:
    settings = _settings(AMADEUS_API_KEY="k", AMADEUS_API_SECRET="s")
    assert settings.amadeus_configured is True
    assert settings.flight_provider_kind() is ProviderKind.REAL
    assert settings.hotel_provider_kind() is ProviderKind.REAL
    # Other domains remain mock since their credentials are absent.
    assert settings.weather_provider_kind() is ProviderKind.MOCK
    assert settings.routes_provider_kind() is ProviderKind.MOCK
    assert settings.events_provider_kind() is ProviderKind.MOCK


# --- Per-domain real selection ----------------------------------------------

def test_openweathermap_key_selects_real_weather() -> None:
    settings = _settings(OPENWEATHERMAP_API_KEY="owm")
    assert settings.weather_provider_kind() is ProviderKind.REAL
    assert settings.uses_mock("weather") is False
    assert settings.uses_mock("flights") is True


def test_mapbox_token_selects_real_routes() -> None:
    settings = _settings(MAPBOX_ACCESS_TOKEN="mb")
    assert settings.routes_provider_kind() is ProviderKind.REAL


def test_ticketmaster_key_selects_real_events() -> None:
    settings = _settings(TICKETMASTER_API_KEY="tm")
    assert settings.events_provider_kind() is ProviderKind.REAL


def test_all_real_credentials_select_real_everywhere() -> None:
    settings = _settings(
        AMADEUS_API_KEY="k",
        AMADEUS_API_SECRET="s",
        OPENWEATHERMAP_API_KEY="owm",
        MAPBOX_ACCESS_TOKEN="mb",
        TICKETMASTER_API_KEY="tm",
    )
    selection = settings.provider_selection()
    assert all(kind is ProviderKind.REAL for kind in selection.values())


# --- LLM vendor selection (Req 4.2) -----------------------------------------

def test_llm_vendor_defaults_to_gemini() -> None:
    assert _settings().llm_vendor == "gemini"


@pytest.mark.parametrize("vendor", ["gemini", "claude", "gpt"])
def test_llm_vendor_selected_from_env(vendor: str) -> None:
    assert _settings(LLM_VENDOR=vendor).llm_vendor == vendor


def test_llm_vendor_is_case_insensitive() -> None:
    assert _settings(LLM_VENDOR="Claude").llm_vendor == "claude"
    assert _settings(LLM_VENDOR="GPT").llm_vendor == "gpt"


def test_invalid_llm_vendor_rejected() -> None:
    with pytest.raises(Exception):
        _settings(LLM_VENDOR="llama")


# --- Domain validation ------------------------------------------------------

def test_provider_kind_unknown_domain_raises() -> None:
    with pytest.raises(ValueError):
        _settings().provider_kind("spaceflights")


# --- get_settings caching ---------------------------------------------------

def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
