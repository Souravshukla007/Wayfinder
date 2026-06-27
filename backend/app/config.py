"""Application configuration.

Environment-driven settings that select providers and the LLM vendor. When a
real provider's credentials are absent, the provider registry falls back to the
mock provider — so the whole system runs with zero paid API keys.

Requirements: 20.2 (FastAPI + Pydantic v2), 2.5 (config-driven provider
selection), 4.2 (config-driven LLM vendor).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMVendor = Literal["gemini", "claude", "gpt"]


class ProviderKind(str, Enum):
    """Which implementation backs a provider domain.

    ``MOCK`` is the zero-key default; ``REAL`` is selected only when the
    domain's credentials are present in configuration (Requirements 2.4, 2.5).
    """

    MOCK = "mock"
    REAL = "real"


# The provider domains the registry resolves. Ordered for stable iteration.
PROVIDER_DOMAINS: tuple[str, ...] = (
    "flights",
    "hotels",
    "weather",
    "routes",
    "events",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "Wayfinder"
    environment: Literal["development", "production", "test"] = "development"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8080"])

    # --- Database / cache / vectors ---
    database_url: str = "postgresql+psycopg://wayfinder:wayfinder@localhost:5432/wayfinder"
    redis_url: str = "redis://localhost:6379/0"

    # --- Supabase JWT verification ---
    supabase_jwks_url: str | None = None
    supabase_jwt_secret: str | None = None  # HS256 fallback
    supabase_jwt_audience: str = "authenticated"

    # --- LLM ---
    llm_vendor: LLMVendor = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    @field_validator("llm_vendor", mode="before")
    @classmethod
    def _normalize_llm_vendor(cls, value: object) -> object:
        """Accept the LLM vendor case-insensitively (Requirement 4.2).

        ``case_sensitive=False`` only affects env-var *key* matching, not the
        *value*. Normalizing to lower-case here lets ``LLM_VENDOR=Claude`` /
        ``GPT`` select a vendor while still rejecting unknown vendors via the
        ``LLMVendor`` literal.
        """
        if isinstance(value, str):
            return value.lower()
        return value

    # --- Real provider credentials (absent => mock provider) ---
    amadeus_api_key: str | None = None
    amadeus_api_secret: str | None = None
    openweathermap_api_key: str | None = None
    mapbox_access_token: str | None = None
    openrouteservice_api_key: str | None = None
    ticketmaster_api_key: str | None = None

    # --- Caching ---
    price_snapshot_ttl_seconds: int = 900

    # --- Constraint Solver (CP-SAT) ---
    # Default maximum wall-clock time, in seconds, the CP-SAT solver may spend
    # before it must return a satisfiable / proven-infeasible result. Used when
    # a trip's ``TripConstraints.solver_timeout`` is not supplied. When the
    # solver exceeds this budget it returns a timeout rejection and the
    # Itinerary Agent does not run (Requirement 9.9).
    solver_timeout_seconds: float = 10.0

    # --- RAG knowledge base / vector store (Requirements 11.x, 20.3) ---
    # The vector + embedding backends are swappable via config. Defaults are a
    # zero-infra, no-network, deterministic combination so the RAG knowledge
    # base runs with no paid key — consistent with the mock-first provider
    # pattern. "chroma"/"pgvector" and "sentence-transformers" are optional and
    # imported lazily only when selected.
    vector_store_backend: Literal["memory", "chroma", "pgvector"] = "memory"
    kb_embedding_backend: Literal["hash", "sentence-transformers"] = "hash"
    kb_embedding_dim: int = 256
    # Minimum cosine similarity a visa/safety document must reach to ground an
    # answer (design RAG Service; used by later RAG tasks 15.2-15.3).
    rag_similarity_threshold: float = 0.7

    # --- Decision Engine base feature weights (Requirement 5.9) ---
    # Weights are read from configurable data rather than hardcoded in the
    # engine. Defaults match the design's base weight set and sum to 1.0.
    base_weight_budget_fit: float = 0.25
    base_weight_weather_fit: float = 0.20
    base_weight_crowd_score: float = 0.15
    base_weight_food_score: float = 0.15
    base_weight_photography_score: float = 0.15
    base_weight_travel_efficiency: float = 0.10

    def base_feature_weights(self) -> dict[str, float]:
        """Return the configured base default feature weights.

        Keyed by the six Decision Engine feature names. Sourced from settings
        so the engine reads weights from configurable data (Requirement 5.9).
        """
        return {
            "budget_fit": self.base_weight_budget_fit,
            "weather_fit": self.base_weight_weather_fit,
            "crowd_score": self.base_weight_crowd_score,
            "food_score": self.base_weight_food_score,
            "photography_score": self.base_weight_photography_score,
            "travel_efficiency": self.base_weight_travel_efficiency,
        }

    @property
    def amadeus_configured(self) -> bool:
        """True only when BOTH the Amadeus key and secret are present.

        Amadeus backs flights and hotels; either domain falls back to mock
        unless the full credential pair is configured (Requirement 2.5).
        """
        return bool(self.amadeus_api_key and self.amadeus_api_secret)

    # --- Per-domain provider-kind selection (Requirements 2.4, 2.5) ---------

    def flight_provider_kind(self) -> ProviderKind:
        return ProviderKind.REAL if self.amadeus_configured else ProviderKind.MOCK

    def hotel_provider_kind(self) -> ProviderKind:
        return ProviderKind.REAL if self.amadeus_configured else ProviderKind.MOCK

    def weather_provider_kind(self) -> ProviderKind:
        return (
            ProviderKind.REAL
            if self.has_weather_provider_creds()
            else ProviderKind.MOCK
        )

    def routes_provider_kind(self) -> ProviderKind:
        return (
            ProviderKind.REAL
            if self.has_routes_provider_creds()
            else ProviderKind.MOCK
        )

    def events_provider_kind(self) -> ProviderKind:
        return (
            ProviderKind.REAL
            if self.has_events_provider_creds()
            else ProviderKind.MOCK
        )

    def provider_kind(self, domain: str) -> ProviderKind:
        """Return the configured :class:`ProviderKind` for ``domain``.

        Raises ``ValueError`` for an unknown domain so misconfiguration surfaces
        loudly rather than silently selecting a mock.
        """
        dispatch = {
            "flights": self.flight_provider_kind,
            "hotels": self.hotel_provider_kind,
            "weather": self.weather_provider_kind,
            "routes": self.routes_provider_kind,
            "events": self.events_provider_kind,
        }
        selector = dispatch.get(domain)
        if selector is None:
            raise ValueError(
                f"Unknown provider domain {domain!r}; expected one of {PROVIDER_DOMAINS}."
            )
        return selector()

    def provider_selection(self) -> dict[str, ProviderKind]:
        """Map every provider domain to its selected :class:`ProviderKind`."""
        return {domain: self.provider_kind(domain) for domain in PROVIDER_DOMAINS}

    def uses_mock(self, domain: str) -> bool:
        """True when ``domain`` resolves to the mock provider."""
        return self.provider_kind(domain) is ProviderKind.MOCK

    def has_flight_provider_creds(self) -> bool:
        return bool(self.amadeus_api_key and self.amadeus_api_secret)

    def has_hotel_provider_creds(self) -> bool:
        return bool(self.amadeus_api_key and self.amadeus_api_secret)

    def has_weather_provider_creds(self) -> bool:
        return bool(self.openweathermap_api_key)

    def has_routes_provider_creds(self) -> bool:
        return bool(self.mapbox_access_token or self.openrouteservice_api_key)

    def has_events_provider_creds(self) -> bool:
        return bool(self.ticketmaster_api_key)

    def has_llm_creds(self) -> bool:
        """True when an API key exists for the configured LLM vendor.

        When this is False the LLM selector falls back to the mock/stub
        provider so the system runs with zero paid keys (Requirement 4.2).
        """
        key_by_vendor: dict[LLMVendor, str | None] = {
            "gemini": self.gemini_api_key,
            "claude": self.anthropic_api_key,
            "gpt": self.openai_api_key,
        }
        return bool(key_by_vendor.get(self.llm_vendor))


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (single instance per process)."""
    return Settings()
