"""Thin LLM client shared by every agent.

Each agent calls `chat_json(agent, system, user)` and gets back a parsed dict.
The client handles the two failures that actually happen on free tiers:
rate limits (back off and retry) and malformed JSON (ask the model to fix it).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

from lib.config import AGENT_TEMPERATURE, Settings


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable answer after retries."""


class SupportsChatJSON(Protocol):
    def chat_json(self, agent: str, system: str, user: str, max_tokens: int = 2500) -> dict[str, Any]: ...


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Models often wrap JSON in code fences or add a sentence before it, so we
    try the fenced block first, then the outermost {...} span.
    """
    if not text or not text.strip():
        raise ValueError("empty model reply")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in model reply")


class LLMClient:
    def __init__(self, settings: Settings, max_retries: int = 3):
        if not settings.api_key:
            raise LLMError(
                "No API key configured. Set GEMINI_API_KEY (or GROQ_API_KEY with "
                "HIRELENS_PROVIDER=groq) in .env or the deployment's secrets."
            )
        from openai import OpenAI  # imported lazily so tests don't need network setup

        self.settings = settings
        self.max_retries = max_retries
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=90)
        self._last_call = 0.0
        self.calls = 0

    def _respect_rate_limit(self) -> None:
        wait = self.settings.min_call_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def chat_json(self, agent: str, system: str, user: str, max_tokens: int = 2500) -> dict[str, Any]:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        messages = [
            {"role": "system", "content": system + "\n\nRespond with a single JSON object only."},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        text = ""
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            try:
                self.calls += 1
                response = self._client.chat.completions.create(
                    model=self.settings.model_for(agent),
                    messages=messages,
                    temperature=AGENT_TEMPERATURE.get(agent, 0.2),
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""
                return extract_json(text)
            except ValueError as exc:  # malformed JSON: ask the model to repair it
                last_error = exc
                messages.append({"role": "assistant", "content": text[:4000]})
                messages.append(
                    {"role": "user", "content": "That was not valid JSON. Reply again with only the JSON object."}
                )
            except RateLimitError as exc:
                last_error = exc
                time.sleep(min(60, 8 * (2**attempt)))
            except APIConnectionError as exc:
                last_error = exc
                time.sleep(3 * (attempt + 1))
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code and exc.status_code < 500:
                    break  # bad request / auth: retrying will not help
                time.sleep(3 * (attempt + 1))
        raise LLMError(f"{agent}: model call failed after retries ({last_error})")
