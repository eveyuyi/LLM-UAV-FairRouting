from __future__ import annotations

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
