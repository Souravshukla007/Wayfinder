"""Anthropic Claude adapter implementing the shared ``LLMProvider`` interface.

Wraps the ``anthropic`` SDK behind :class:`app.llm.base.LLMProvider` so agent
code never imports the vendor directly (Requirements 4.1, 4.3). The SDK is
imported lazily so this module stays import-safe when the optional dependency
is not installed; a clear error is raised only when the provider is actually
used without the SDK or an API key.
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMResponse


class ClaudeProvider:
    """LLM provider backed by Anthropic Claude (``anthropic``)."""

    #: Default model used when no override is supplied via Settings.
    default_model = "claude-3-5-sonnet-latest"
    #: Anthropic requires an explicit max_tokens; narration prompts are short.
    default_max_tokens = 1024

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.anthropic_api_key
        self.model_name = (
            getattr(settings, "anthropic_model", None) or self.default_model
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self._api_key:
            raise RuntimeError(
                "ClaudeProvider requires 'anthropic_api_key' to be configured."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without SDK
            raise RuntimeError(
                "The 'anthropic' package is required to use ClaudeProvider. "
                "Install it with 'pip install anthropic'."
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        create_kwargs: dict = {
            "model": self.model_name,
            "max_tokens": self.default_max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            create_kwargs["system"] = system

        message = client.messages.create(**create_kwargs)

        # Concatenate any text blocks from the structured content response.
        text_parts = [
            getattr(block, "text", "")
            for block in (getattr(message, "content", None) or [])
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(text_parts)

        usage = getattr(message, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        return LLMResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = ["ClaudeProvider"]
