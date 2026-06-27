"""Google Gemini adapter implementing the shared ``LLMProvider`` interface.

Wraps the modern ``google-genai`` SDK (``from google import genai``) behind
:class:`app.llm.base.LLMProvider` so agent code never imports the vendor
directly (Requirements 4.1, 4.3). The SDK is imported lazily so this module
stays import-safe when the optional dependency is not installed; a clear error
is raised only when the provider is actually used without the SDK or an API key.

Note: this uses ``google-genai`` (the supported SDK), not the deprecated
``google-generativeai`` package — the latter pins ``protobuf < 6`` which
conflicts with OR-Tools (the CP-SAT solver) requiring ``protobuf >= 6.33``.
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMResponse


class GeminiProvider:
    """LLM provider backed by Google Gemini (``google-genai``)."""

    #: Default model used when no override is supplied via Settings.
    default_model = "gemini-2.5-flash"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.gemini_api_key
        # Settings has no dedicated model field; allow an optional attribute
        # override but fall back to a sensible default.
        self.model_name = getattr(settings, "gemini_model", None) or self.default_model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self._api_key:
            raise RuntimeError(
                "GeminiProvider requires 'gemini_api_key' to be configured."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised only without SDK
            raise RuntimeError(
                "The 'google-genai' package is required to use GeminiProvider. "
                "Install it with 'pip install google-genai'."
            ) from exc

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

        return LLMResponse(
            text=getattr(response, "text", "") or "",
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = ["GeminiProvider"]
