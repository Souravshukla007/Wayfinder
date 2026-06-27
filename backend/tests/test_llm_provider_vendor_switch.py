"""Integration test for single-interface usage and config-driven vendor switch.

Task 12.3. Validates Requirements 4.1 and 4.3:

* 4.1 - The system accesses all LLM functionality through a *single*
  ``LLMProvider`` interface.
* 4.3 - The provider is configurable as Gemini, Claude, or GPT, and switching
  the configured vendor routes calls to that vendor *without changing agent
  code*.

The test models a vendor-agnostic "agent": a function that depends only on the
``LLMProvider`` Protocol and calls ``complete(...)``. The exact same agent code
is exercised across every vendor configuration; only ``Settings.llm_vendor``
(plus the matching API key) changes between runs. This demonstrates that a
vendor switch is a pure configuration change.

Mock-first: vendor SDKs are stubbed via ``sys.modules`` so the test needs no
real API keys and makes no network calls.
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


# --- A vendor-agnostic "agent" that only knows the single interface ----------


def run_agent(provider: LLMProvider) -> LLMResponse:
    """Stand-in for agent code.

    It depends solely on the ``LLMProvider`` interface: it never imports a
    concrete vendor and never branches on which vendor it received. The body is
    identical no matter which vendor backs ``provider`` — that is the property
    Requirement 4.2/4.3 require. If switching vendors needed agent-code changes,
    this single function could not serve all of them.
    """
    return provider.complete(
        "Narrate why Kyoto outranks Osaka for a photography trip.",
        system="You narrate pre-computed scores; never invent numbers.",
        temperature=0.0,
    )


# --- SDK stubs (no network, no real keys) ------------------------------------


def _stub_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Usage:
        prompt_token_count = 5
        candidates_token_count = 7

    class _Resp:
        text = "gemini narration"
        usage_metadata = _Usage()

    class _Models:
        def generate_content(self, model, contents, config=None):
            return _Resp()

    class _Client:
        def __init__(self, api_key=None):
            self.models = _Models()

    class _GenerateContentConfig:
        def __init__(self, system_instruction=None, temperature=0.0):
            pass

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


def _stub_claude(monkeypatch: pytest.MonkeyPatch) -> None:
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
            return _Message()

    class _Client:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def _stub_openai(monkeypatch: pytest.MonkeyPatch) -> None:
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
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake)


# (vendor, key_field, expected adapter type, SDK stub, expected narration text)
_VENDOR_CASES = [
    ("gemini", "gemini_api_key", GeminiProvider, _stub_gemini, "gemini narration"),
    ("claude", "anthropic_api_key", ClaudeProvider, _stub_claude, "claude narration"),
    ("gpt", "openai_api_key", OpenAIProvider, _stub_openai, "gpt narration"),
]


# --- Tests -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vendor", "key_field", "expected_type", "stub", "expected_text"),
    _VENDOR_CASES,
)
def test_config_switch_routes_through_single_interface(
    monkeypatch: pytest.MonkeyPatch,
    vendor: str,
    key_field: str,
    expected_type: type,
    stub,
    expected_text: str,
) -> None:
    """Switching the configured vendor reroutes the unchanged agent.

    For each vendor, configuration alone (``llm_vendor`` + its key) selects the
    matching adapter; the *same* ``run_agent`` then drives it through the single
    ``LLMProvider.complete`` contract and gets that vendor's output back.
    """
    stub(monkeypatch)
    settings = Settings(llm_vendor=vendor, **{key_field: "test-key"})

    provider = get_llm_provider(settings)

    # Config selected the right vendor adapter ...
    assert isinstance(provider, expected_type)
    # ... and it conforms to the single shared interface (Req 4.1).
    assert isinstance(provider, LLMProvider)

    # Identical agent code runs against every vendor (Req 4.3: no code change).
    result = run_agent(provider)
    assert isinstance(result, LLMResponse)
    assert result.text == expected_text


def test_same_agent_code_serves_every_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """One agent call path produces a valid response for all three vendors.

    Iterating configuration over every vendor and invoking the *same*
    ``run_agent`` proves the interface is the only coupling point: a vendor
    switch never requires touching agent code.
    """
    outputs: dict[str, str] = {}
    for vendor, key_field, _type, stub, _text in _VENDOR_CASES:
        stub(monkeypatch)
        settings = Settings(llm_vendor=vendor, **{key_field: "test-key"})
        provider = get_llm_provider(settings)

        result = run_agent(provider)  # unchanged across vendors

        assert isinstance(result, LLMResponse)
        assert result.text
        outputs[vendor] = result.text

    # Each vendor was actually exercised and produced its own narration.
    assert set(outputs) == {"gemini", "claude", "gpt"}
    assert len(set(outputs.values())) == 3


def test_zero_key_config_routes_to_mock_through_same_interface() -> None:
    """With no credentials, the same agent runs on the mock provider.

    The zero-paid-key default also flows through the single interface, so the
    agent code path is identical whether a real vendor or the mock backs it.
    """
    provider = get_llm_provider(Settings(llm_vendor="gemini"))

    assert isinstance(provider, MockLLMProvider)
    assert isinstance(provider, LLMProvider)

    result = run_agent(provider)
    assert isinstance(result, LLMResponse)
    assert result.text  # deterministic, non-empty mock narration
    assert result.model == "mock-llm"
