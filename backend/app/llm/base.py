"""Pluggable LLM provider interface and config-driven selector (Task 12.1).

A single :class:`typing.Protocol` (:class:`LLMProvider`) defines the only LLM
contract agent code is allowed to depend on. Agents call ``complete(...)`` and
never import a concrete vendor, so switching between Gemini / Claude / GPT is a
configuration change with no agent-code edits (Requirements 4.1, 4.2).

``get_llm_provider`` is the config-driven factory. It mirrors the mock-first
pattern already used by :mod:`app.providers.registry`:

* When no API key is configured for the selected vendor
  (``Settings.has_llm_creds()`` is ``False``) it returns :class:`MockLLMProvider`
  so the whole system runs end-to-end with zero paid keys.
* When credentials are present it lazily imports the matching vendor adapter
  (implemented in Task 12.2) so this module stays import-safe even when the
  optional vendor SDKs are not installed, and importing it never pulls in a
  network client.

This module performs no I/O and the protocol is ``@runtime_checkable`` so tests
can assert a resolved provider structurally implements the interface.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """The result of a single LLM completion call.

    Captures the generated ``text`` plus lightweight usage metadata so callers
    (e.g. the observability layer's ``agent_runs`` tracing) can record token
    usage without depending on any vendor-specific response shape.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """The single LLM interface that all agent code depends on.

    Concrete vendor adapters (Gemini / Claude / GPT) and the mock provider all
    implement this method. The LLM is used for narration only; it never
    produces numeric scores or rankings.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class MockLLMProvider:
    """Zero-key default LLM provider (mock-first pattern).

    Returns a deterministic, network-free response so the system runs with no
    paid API key. The output is stable for a given input, which keeps mock
    runs reproducible for tests and the evaluation harness.
    """

    model_name = "mock-llm"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Deterministic stub narration: echo a bounded preview of the prompt so
        # downstream code has non-empty, repeatable text to work with.
        preview = " ".join(prompt.split())[:240]
        text = f"[mock-llm] {preview}" if preview else "[mock-llm]"
        return LLMResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )


def _resilient_llm_cls(base_cls: type) -> type:
    """Build (once per vendor class) a fallback-wrapped LLM provider subclass.

    The wrapper subclasses the vendor adapter so it still satisfies
    ``isinstance(provider, GeminiProvider/ClaudeProvider/OpenAIProvider)`` (the
    selector's contract: the configured vendor is chosen when a key is present)
    while guaranteeing the app never crashes on narration. If a completion fails
    for any reason — the vendor SDK is not installed, the key is invalid, the
    network is down — it logs a warning and returns the deterministic mock
    narration instead. Narration is non-numeric, so this never affects scores.
    """
    cached = _resilient_llm_cache.get(base_cls)
    if cached is not None:
        return cached

    class _ResilientLLMProvider(base_cls):  # type: ignore[valid-type, misc]
        """Vendor LLM provider that degrades to mock narration on failure."""

        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self._fallback = MockLLMProvider()

        def complete(
            self,
            prompt: str,
            *,
            system: str | None = None,
            temperature: float = 0.0,
        ) -> LLMResponse:
            try:
                return super().complete(
                    prompt, system=system, temperature=temperature
                )
            except Exception as exc:  # noqa: BLE001 - resilience boundary
                logger.warning(
                    "LLM provider %s failed (%s); falling back to mock narration.",
                    base_cls.__name__,
                    exc,
                )
                return self._fallback.complete(
                    prompt, system=system, temperature=temperature
                )

    _resilient_llm_cache[base_cls] = _ResilientLLMProvider
    return _ResilientLLMProvider


# Cache of vendor adapter class -> its resilient subclass (built on first use).
_resilient_llm_cache: dict[type, type] = {}


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Resolve the configured LLM provider, defaulting to the mock provider.

    Selection is driven entirely by :class:`app.config.Settings`:

    * no credentials for the configured vendor -> :class:`MockLLMProvider`;
    * credentials present -> the vendor adapter named by ``settings.llm_vendor``,
      wrapped so any live failure (missing SDK, invalid key, network error)
      degrades to mock narration rather than crashing the pipeline.

    Vendor adapters are imported lazily so this factory remains import-safe
    when their optional SDKs are absent (the default zero-key path never
    imports them).
    """
    settings = settings if settings is not None else get_settings()

    if not settings.has_llm_creds():
        return MockLLMProvider()

    vendor = settings.llm_vendor
    if vendor == "gemini":
        from app.llm.gemini import GeminiProvider

        base_cls: type = GeminiProvider
    elif vendor == "claude":
        from app.llm.claude import ClaudeProvider

        base_cls = ClaudeProvider
    elif vendor == "gpt":
        from app.llm.openai import OpenAIProvider

        base_cls = OpenAIProvider
    else:
        raise ValueError(
            f"Unknown LLM vendor {vendor!r}; expected one of 'gemini', "
            "'claude', 'gpt'."
        )

    return _resilient_llm_cls(base_cls)(settings)


__all__ = [
    "LLMResponse",
    "LLMProvider",
    "MockLLMProvider",
    "get_llm_provider",
]
