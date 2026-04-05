from __future__ import annotations

import sys
import types

import pytest

from llm4fairrouting.llm import client_utils


class _FakeResponse:
    def __init__(self, content: str):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


class _FlakyCompletions:
    def __init__(self, failures: int, content: str):
        self._failures = failures
        self._content = content
        self.calls = 0

    def create(self, **_: object):
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError("temporary failure")
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, failures: int = 0, content: str = '{"ok": true}'):
        completions = _FlakyCompletions(failures, content)
        self.chat = type("Chat", (), {"completions": completions})()
        self.completions = completions


def test_parse_json_response_accepts_markdown_fence():
    parsed = client_utils.parse_json_response('```json\n{"dialogues": []}\n```')

    assert parsed == {"dialogues": []}


def test_call_llm_retries_before_succeeding(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(client_utils.time, "sleep", sleep_calls.append)
    client = _FakeClient(failures=2, content='{"status": "ok"}')

    result = client_utils.call_llm(client, "demo-model", "system", "user", temperature=0.2)

    assert result == '{"status": "ok"}'
    assert client.completions.calls == 3
    assert sleep_calls == [2.0, 2.0]


def test_call_llm_does_not_retry_non_retryable_context_error(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(client_utils.time, "sleep", sleep_calls.append)

    class _ContextCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **_: object):
            self.calls += 1
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': "
                "\"'max_tokens' or 'max_completion_tokens' is too large: 1200. "
                "This model's maximum context length is 16384 tokens and your request "
                "has 16263 input tokens\"}}"
            )

    completions = _ContextCompletions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()

    with pytest.raises(RuntimeError) as exc_info:
        client_utils.call_llm(client, "demo-model", "system", "user")

    assert "without retry" in str(exc_info.value)
    assert completions.calls == 1
    assert sleep_calls == []


class _FakeOpenAI:
    def __init__(self, *, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key


def _install_fake_openai(monkeypatch):
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def test_create_openai_client_accepts_openai_env_aliases(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-compatible.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    client = client_utils.create_openai_client()

    assert client.base_url == "https://openai-compatible.example/v1"
    assert client.api_key == "sk-openai"


def test_create_openai_client_error_lists_supported_env_names(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        client_utils.create_openai_client()

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
