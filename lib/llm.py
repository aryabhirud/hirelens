"""Thin LLM client shared by every agent.

Each agent calls `chat_json(agent, system, user)` and gets back a parsed dict.
The client handles the two failures that actually happen on free tiers:
rate limits (back off and retry) and malformed JSON (ask the model to fix it).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Protocol

from lib.config import AGENT_TEMPERATURE, Settings


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable answer after retries."""


class QuotaExhausted(LLMError):
    """The provider's daily quota is used up. Retrying within this run cannot help, so the run stops."""


QUOTA_MESSAGE = (
    "The AI model's free daily quota is used up on every configured key, so HireLens cannot analyse right now. "
    "It resets once a day (Gemini: midnight Pacific time, about 12:30 PM IST). "
    "Try again after that, or configure a different API key or model."
)

# Daily limits: Gemini "GenerateRequestsPerDayPerProjectPerModel", Groq "requests per day (RPD)" / "(TPD)".
_DAILY_LIMIT = re.compile(r"per[\s_-]?day|\b(?:rpd|tpd)\b", re.IGNORECASE)


def is_daily_quota_error(exc: Exception) -> bool:
    return bool(_DAILY_LIMIT.search(str(exc)))


def _route_is_unusable(exc) -> bool:
    """A key that is wrong or expired, or a model this key cannot use. (Gemini reports a bad key as 400.)"""
    code = getattr(exc, "status_code", None)
    return code in (401, 403, 404) or (code == 400 and "api key" in str(exc).lower().replace("_", " "))


class SupportsChatJSON(Protocol):
    def chat_json(self, agent: str, system: str, user: str, max_tokens: int = 2500) -> dict[str, Any]: ...


MAX_TOKEN_BUDGET = 8000

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


# Routes (key, model) whose daily quota ran out, with the time it happened. Shared by every
# client in this process, so the next analysis starts on a route that still works.
_EXHAUSTED: dict[tuple[str, str], float] = {}
EXHAUSTED_RECHECK_S = 30 * 60  # quotas reset once a day; look again after half an hour


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


class LLMClient:
    def __init__(self, settings: Settings, max_retries: int = 3):
        if not settings.api_keys:
            raise LLMError(
                "No API key configured. Set GEMINI_API_KEY (or GROQ_API_KEY with "
                "HIRELENS_PROVIDER=groq) in .env or the deployment's secrets."
            )
        from openai import OpenAI  # imported lazily so tests don't need network setup

        self.settings = settings
        self.max_retries = max_retries
        # max_retries=0: this class does its own retrying and must react at once to a daily quota.
        self._clients = {key: OpenAI(api_key=key, base_url=settings.base_url, timeout=90, max_retries=0)
                         for key in settings.api_keys}
        # Every key on the configured model(s) first, then every key on each fallback model.
        self.routes: list[tuple[str, str | None]] = [(k, None) for k in settings.api_keys] + [
            (k, m) for m in settings.fallback_models for k in settings.api_keys]
        self._route = 0
        self._last_call = 0.0
        self.calls = 0
        self.switches: list[str] = []  # human-readable record of fallbacks, for logs and tests

    # The transport for the current key. Tests replace it with a fake for every key at once.
    @property
    def _client(self):
        return self._clients[self.routes[self._route][0]]

    @_client.setter
    def _client(self, value) -> None:
        self._clients = {key: value for key in self._clients}

    def _model(self, agent: str) -> str:
        return self.routes[self._route][1] or self.settings.model_for(agent)

    def _route_id(self, agent: str) -> tuple[str, str]:
        return _fingerprint(self.routes[self._route][0]), self._model(agent)

    def _skip_exhausted(self, agent: str) -> None:
        """Start on the first route not known to be used up (another run may have found out already)."""
        now = time.time()
        while self._route < len(self.routes) - 1:
            when = _EXHAUSTED.get(self._route_id(agent))
            if when is None or now - when > EXHAUSTED_RECHECK_S:
                return
            self._route += 1

    def _fall_back(self, agent: str) -> bool:
        """Mark the current route as unusable for now and move to the next one. False if none is left."""
        _EXHAUSTED[self._route_id(agent)] = time.time()
        if self._route >= len(self.routes) - 1:
            return False
        self._route += 1
        self._skip_exhausted(agent)
        key_no = self.settings.api_keys.index(self.routes[self._route][0]) + 1
        self.switches.append(f"daily quota used up; switched to key #{key_no} on {self._model(agent)}")
        return True

    def _respect_rate_limit(self) -> None:
        wait = self.settings.min_call_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def chat_json(self, agent: str, system: str, user: str, max_tokens: int = 2500) -> dict[str, Any]:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        messages = [
            {"role": "system", "content": system + "\n\nRespond with a single JSON object only. Keep every text field brief."},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        text = ""
        cap = self.settings.max_output_tokens or MAX_TOKEN_BUDGET
        budget = min(max_tokens, cap)
        extra = dict(self.settings.extra_body)
        if self.settings.reasoning_effort:
            extra["reasoning_effort"] = self.settings.reasoning_effort
        options = {"response_format": {"type": "json_object"}} if self.settings.json_mode else {}
        if extra:
            options["extra_body"] = extra
        self._skip_exhausted(agent)
        attempt = 0
        while attempt <= self.max_retries:
            attempt += 1
            self._respect_rate_limit()
            try:
                self.calls += 1
                response = self._client.chat.completions.create(
                    model=self._model(agent),
                    messages=messages,
                    temperature=AGENT_TEMPERATURE.get(agent, 0.2),
                    max_tokens=budget,
                    **options,
                )
                choice = response.choices[0]
                text = choice.message.content or ""
                if not text.strip():
                    # Reasoning models can spend the whole token budget thinking and return no answer.
                    # Give the next attempt more room instead of asking the model to "repair" nothing.
                    last_error = ValueError(f"empty model reply (finish_reason={choice.finish_reason})")
                    budget = min(budget * 2, cap)
                    continue
                return extract_json(text)
            except ValueError as exc:  # malformed JSON: ask the model to repair it
                last_error = exc
                messages.append({"role": "assistant", "content": text[:4000]})
                messages.append(
                    {"role": "user", "content": "That was not valid JSON. Reply again with only the JSON object."}
                )
            except RateLimitError as exc:
                if is_daily_quota_error(exc):
                    if self._fall_back(agent):
                        attempt -= 1  # a fresh key or model is not a retry of the same call
                        continue
                    raise QuotaExhausted(QUOTA_MESSAGE) from exc
                last_error = exc
                time.sleep(min(60, 8 * (2 ** (attempt - 1))))
            except APIConnectionError as exc:
                last_error = exc
                time.sleep(3 * attempt)
            except APIStatusError as exc:
                last_error = exc
                if _route_is_unusable(exc) and self._fall_back(agent):
                    attempt -= 1  # a wrong or expired key, or an unknown model: try the next one
                    continue
                if exc.status_code and exc.status_code < 500:
                    break  # bad request / auth: retrying will not help
                time.sleep(3 * attempt)
        raise LLMError(f"{agent}: model call failed after retries ({last_error})")
