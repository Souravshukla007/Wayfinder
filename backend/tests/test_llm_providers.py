"""Tests for the pluggable LLM provider adapters (Task 12.2).

Validates Requirement 4.3: the system uses a single LLM interface; concrete
vendor adapters (Gemini / Claude / GPT) are selectable purely by config and
all structurally implement ``LLMProvider``. SDKs are stubbed so these tests run
with no network access and without the optional vendor packages installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.config import Settings
from app.llm.base import LLMProvider, LLMResponse, MockLLMProvider, get_llm_provider
from app.llm.claude import ClaudeProvider
from app.llm.gemini import GeminiProvider
from app.llm.openai import OpenAIProvider


# --- Factory routing (Req 4.3: config-driven selection) ----------------------


def test_no_credentials_defaults_to_mock() -> None:
    provider = get_llm_provider(Settings(llm_vendor="gemini"))
    assert isinstance(provider, MockLLMProvider)


@pytest.mark.parametrize(
    ("vendor", "key_field", "expected_type", "expected_model"),
    [
        ("gemini", "gemini_api_key", GeminiProvider, "gemini-2.5-flash"),
        ("claude", "anthropic_api_key", ClaudeProvider, "claude-3-5-sonnet-latest"),
        ("gpt", "openai_api_key", OpenAIProvider, "gpt-4o-mini"),
    ],
)
def test_vendor_selected_by_config(
    vendor: str, key_field: str, expected_type: type, expected_model: str
) -> None:
    settings = Settings(llm_vendor=vendor, **{key_field: "test-key"})
    provider = get_llm_provider(settings)
    assert isinstance(provider, expected_type)
    # All adapters structurally satisfy the single shared interface.
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == expected_model


# --- Missing-key error path --------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "key_field"),
    [
        (GeminiProvider, "gemini_api_key"),
        (ClaudeProvider, "anthropic_api_key"),
        (OpenAIProvider, "openai_api_key"),
    ],
)
def test_complete_without_key_raises(provider_cls: type, key_field: str) -> None:
    provider = provider_cls(Settings())
    with pytest.raises(RuntimeError, match=key_field):
        provider.complete("hello")


# --- Response mapping with stubbed SDKs (no network) -------------------------


def test_gemini_maps_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _Usage:
        prompt_token_count = 5
        candidates_token_count = 7

    class _Resp:
        text = "narration"
        usage_metadata = _Usage()

    class _Models:
        def generate_content(self, model, contents, config=None):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return _Resp()

    class _Client:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.models = _Models()

    class _GenerateContentConfig:
        def __init__(self, system_instruction=None, temperature=0.0):
            captured["system"] = system_instruction
            captured["temperature"] = temperature

    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _Client
    types_mod = types.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = _GenerateContentConfig
    genai_mod.types = types_mod
    google_pkg = types.ModuleType("google")
    google_pkg.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)

    provider = GeminiProvider(Settings(llm_vendor="gemini", gemini_api_key="k"))
    result = provider.complete("hi", system="be brief", temperature=0.0)

    assert isinstance(result, LLMResponse)
    assert result.text == "narration"
    assert result.model == "gemini-2.5-flash"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 7
    assert captured["api_key"] == "k"
    assert captured["system"] == "be brief"


def test_claude_maps_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Block:
        type = "text"
        text = "claude narration"

    class _Usage:
        input_tokens = 3
        output_tokens = 9

    class _Message:
        content = [_Block()]
        usage = _Usage()

    class _Messages:
        def create(self, **kwargs):
            _Messages.kwargs = kwargs
            return _Message()

    class _Client:
        def __init__(self, api_key=None):
            _Client.api_key = api_key
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    provider = ClaudeProvider(Settings(llm_vendor="claude", anthropic_api_key="k"))
    result = provider.complete("hi", system="be brief", temperature=0.2)

    assert result.text == "claude narration"
    assert result.model == "claude-3-5-sonnet-latest"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 9
    assert _Messages.kwargs["system"] == "be brief"
    assert _Messages.kwargs["temperature"] == 0.2


def test_openai_maps_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Msg:
        content = "gpt narration"

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 4
        completion_tokens = 6

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            _Completions.kwargs = kwargs
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, api_key=None):
            _Client.api_key = api_key
            self.chat = _Chat()

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    provider = OpenAIProvider(Settings(llm_vendor="gpt", openai_api_key="k"))
    result = provider.complete("hi", system="be brief", temperature=0.5)

    assert result.text == "gpt narration"
    assert result.model == "gpt-4o-mini"
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 6
    messages = _Completions.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "user", "content": "hi"}
