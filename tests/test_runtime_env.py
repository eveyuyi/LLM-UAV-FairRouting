from __future__ import annotations

from pathlib import Path

from llm4fairrouting.config.runtime_env import (
    env_bool,
    env_int_list,
    load_env_file,
)


def test_load_env_file_parses_quotes_and_export_prefix(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'export LLM4FAIRROUTING_API_KEY="sk-test"\n'
        "LLM4FAIRROUTING_MODEL=gpt-4o-mini\n"
        "LLM4FAIRROUTING_TIME_SLOTS=0, 1 2\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("LLM4FAIRROUTING_API_KEY", raising=False)
    loaded = load_env_file(env_file)

    assert loaded["LLM4FAIRROUTING_API_KEY"] == "sk-test"
    assert loaded["LLM4FAIRROUTING_MODEL"] == "gpt-4o-mini"
    assert env_int_list("LLM4FAIRROUTING_TIME_SLOTS") == [0, 1, 2]


def test_env_bool_accepts_true_and_false_variants(monkeypatch):
    monkeypatch.setenv("LLM4FAIRROUTING_OFFLINE", "yes")
    assert env_bool("LLM4FAIRROUTING_OFFLINE") is True

    monkeypatch.setenv("LLM4FAIRROUTING_OFFLINE", "off")
    assert env_bool("LLM4FAIRROUTING_OFFLINE") is False
