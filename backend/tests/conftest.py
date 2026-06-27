"""Shared pytest fixtures and import-path setup for the backend test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the backend/ directory is importable as the project root regardless of
# the working directory pytest is invoked from.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings, get_settings  # noqa: E402 - after sys.path setup

# Disable the ``backend/.env`` source for the ENTIRE test session, at import
# time — before any fixture (including module/session-scoped ones) constructs
# Settings. A function-scoped fixture is too late for higher-scoped fixtures
# (pytest builds those first), which is why this is done at module load. Tests
# are hermetic and match CI, which runs with no .env file present.
Settings.model_config["env_file"] = None


# Provider credential env vars that influence Settings selection. Cleared
# before each test so cases start from a known "no credentials" baseline.
_PROVIDER_ENV_VARS = (
    "AMADEUS_API_KEY",
    "AMADEUS_API_SECRET",
    "OPENWEATHERMAP_API_KEY",
    "MAPBOX_ACCESS_TOKEN",
    "OPENROUTESERVICE_API_KEY",
    "TICKETMASTER_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "LLM_VENDOR",
    "SUPABASE_JWKS_URL",
    "SUPABASE_JWT_SECRET",
    "DATABASE_URL",
    "REDIS_URL",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate Settings from the environment AND the developer's local .env.

    Two sources can leak real credentials into a test run:

    * OS environment variables — cleared here so each test starts from a
      known "no credentials" baseline and controls them explicitly.
    * The ``backend/.env`` file — pydantic-settings reads it at construction, so
      a developer with live keys (Gemini, OpenWeatherMap, OpenRouteService,
      Ticketmaster, ...) would otherwise flip providers to "real" and try live
      network/SDK calls mid-test. We disable the dotenv source for the duration
      of each test so the suite is hermetic and matches CI (which has no .env).

    The cached ``get_settings`` singleton is cleared around each test so neither
    a stale nor a .env-populated instance bleeds across cases.
    """
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    os.environ.pop("LLM_VENDOR", None)

    # .env is already disabled session-wide at import; clear the cached
    # singleton around each test so no populated instance bleeds across cases.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
