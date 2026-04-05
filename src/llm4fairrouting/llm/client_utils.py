"""Shared OpenAI client helpers for LLM-facing modules."""

from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI


DEFAULT_API_BASE = "http://35.220.164.252:3888/v1/"
API_BASE_ENV = "OPENAI_BASE_URL"
API_KEY_ENV = "OPENAI_API_KEY"
TIMEOUT_ENV = "LLM4FAIRROUTING_OPENAI_TIMEOUT_S"
MAX_OUTPUT_TOKENS_ENV = "LLM4FAIRROUTING_MAX_OUTPUT_TOKENS"


def _is_non_retryable_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "maximum context length" in text
        or "reduce the length of the input messages" in text
        or ("max_tokens" in text and "too large" in text)
        or ("max_completion_tokens" in text and "too large" in text)
    )


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
    timeout_s = float(os.getenv(TIMEOUT_ENV, "300"))
    return OpenAI(base_url=base, api_key=key, timeout=timeout_s, max_retries=0)


def call_llm(
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    max_tokens: int | None = None,
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
                max_tokens=max_tokens if max_tokens is not None else int(os.getenv(MAX_OUTPUT_TOKENS_ENV, "1200")),
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            if _is_non_retryable_llm_error(exc):
                print(f"  [LLM] request too large for one shot, skip retries: {exc}")
                raise RuntimeError(f"LLM call failed without retry: {exc}") from exc
            print(f"  [LLM] attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(min(3.0 * attempt, 12.0))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON payload while tolerating Markdown code fences."""
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    cleaned = cleaned.strip()

    candidates: list[str] = [cleaned]
    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
    if starts:
        start = min(starts)
        ends = [idx for idx in (cleaned.rfind("}"), cleaned.rfind("]")) if idx >= 0]
        if ends:
            end = max(ends)
            if end >= start:
                candidates.append(cleaned[start : end + 1].strip())

    # Common JSON formatting issue from LLM output: trailing commas before ] or }.
    candidates.extend(
        re.sub(r",\s*([}\]])", r"\1", candidate).strip()
        for candidate in list(candidates)
    )

    last_exc: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    raise json.JSONDecodeError("Empty JSON response", cleaned, 0)
