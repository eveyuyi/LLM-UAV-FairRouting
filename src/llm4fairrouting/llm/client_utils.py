"""Shared OpenAI client helpers for LLM-facing modules."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI


DEFAULT_API_BASE = "http://35.220.164.252:3888/v1/"
API_BASE_ENV = "OPENAI_BASE_URL"
API_KEY_ENV = "OPENAI_API_KEY"


def create_openai_client(
    api_base: str | None = None,
    api_key: str | None = None,
) -> "OpenAI":
    """Create an OpenAI-compatible client from explicit args or env vars."""
    from openai import OpenAI

    base = api_base or os.getenv(API_BASE_ENV) or DEFAULT_API_BASE
    key = api_key or os.getenv(API_KEY_ENV)
    if not key:
        raise ValueError(f"Missing API key. Set {API_KEY_ENV}, or pass --api-key.")
    return OpenAI(base_url=base, api_key=key)


def call_llm(
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Call the chat-completions API with a small retry loop."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            print(f"  [LLM] attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(2.0)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON payload while tolerating Markdown code fences."""
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return json.loads(cleaned.strip())
