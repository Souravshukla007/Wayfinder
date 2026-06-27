"""OpenAI GPT adapter implementing the shared ``LLMProvider`` interface.

Wraps the ``openai`` SDK behind :class:`app.llm.base.LLMProvider` so agent code
never imports the vendor directly (Requirements 4.1, 4.3). The SDK is imported
lazily so this module stays import-safe when the optional dependency is not
installed; a clear error is raised only when the provider is actually used
without the SDK or an API key.
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMResponse


class OpenAIProvider:
    """LLM provider backed by OpenAI GPT (``openai``)."""

    #: Default model used when no override is supplied via Settings.
    default_model = "gpt-4o-mini"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.openai_api_key
        self.model_name = getattr(settings, "openai_model", None) or self.default_model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self._api_key:
            raise RuntimeError(
                "OpenAIProvider requires 'openai_api_key' to be configured."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised only without SDK
            raise RuntimeError(
                "The 'openai' package is required to use OpenAIProvider. "
                "Install it with 'pip install openai'."
            ) from exc

        client = openai.OpenAI(api_key=self._api_key)
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )

        choice = response.choices[0] if getattr(response, "choices", None) else None
        text = ""
        if choice is not None:
            text = getattr(getattr(choice, "message", None), "content", "") or ""

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return LLMResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = ["OpenAIProvider"]
