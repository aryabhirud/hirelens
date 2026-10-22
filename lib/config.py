"""Runtime configuration for HireLens.

Settings come from environment variables (or a local .env file). On Hugging
Face Spaces or Render, set the same keys as secrets / environment variables.

Any OpenAI-compatible endpoint works. Two presets are provided:
  - gemini (default): Google AI Studio free tier, OpenAI-compatible endpoint
  - groq: Groq free tier
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

try:  # optional, only for local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROVIDER_PRESETS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.8-flash",
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
}

# Per-agent temperature. Judging agents (matcher, critic) run cold so their
# scores are repeatable; the coach runs warmer so its writing is less generic.
AGENT_TEMPERATURE = {
    "orchestrator": 0.0,
    "resume_parser": 0.0,
    "job_analyst": 0.1,
    "matcher": 0.0,
    "critic": 0.0,
    "career_coach": 0.6,
}


@dataclass
class Settings:
    provider: str = "gemini"
    api_key: str | None = None
    base_url: str = ""
    default_model: str = ""
    agent_models: dict[str, str] = field(default_factory=dict)
    min_call_interval: float = 0.0  # seconds between LLM calls (free-tier RPM)
    max_critic_revisions: int = 2
    max_analyst_steps: int = 4
    reasoning_effort: str | None = None  # for reasoning models (e.g. gpt-oss on Groq): low | medium | high
    max_output_tokens: int | None = None  # hard cap on reply length (some free tiers limit output tokens per minute)
    json_mode: bool = False               # ask the provider for a guaranteed JSON object reply
    extra_body: dict = field(default_factory=dict)  # provider-specific options, e.g. {"reasoning_format": "hidden"}

    def model_for(self, agent: str) -> str:
        return self.agent_models.get(agent) or self.default_model


def _json_env(name: str) -> dict:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object, e.g. {{\"reasoning_format\": \"hidden\"}}") from exc
    return value if isinstance(value, dict) else {}


def load_settings(api_key_override: str | None = None, provider_override: str | None = None) -> Settings:
    provider = (provider_override or os.getenv("HIRELENS_PROVIDER", "gemini")).lower()
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["gemini"])
    api_key = api_key_override or os.getenv("HIRELENS_API_KEY") or os.getenv(preset["key_env"])

    # Optional per-agent model override, e.g. HIRELENS_MODEL_CRITIC=gemini-...-pro
    agent_models = {}
    for agent in AGENT_TEMPERATURE:
        value = os.getenv(f"HIRELENS_MODEL_{agent.upper()}")
        if value:
            agent_models[agent] = value

    return Settings(
        provider=provider,
        api_key=api_key,
        base_url=os.getenv("HIRELENS_BASE_URL", preset["base_url"]),
        default_model=os.getenv("HIRELENS_MODEL", preset["model"]),
        agent_models=agent_models,
        min_call_interval=float(os.getenv("HIRELENS_MIN_CALL_INTERVAL", "0")),
        max_critic_revisions=int(os.getenv("HIRELENS_MAX_CRITIC_REVISIONS", "2")),
        max_analyst_steps=int(os.getenv("HIRELENS_MAX_ANALYST_STEPS", "4")),
        reasoning_effort=os.getenv("HIRELENS_REASONING_EFFORT") or None,
        max_output_tokens=int(os.getenv("HIRELENS_MAX_OUTPUT_TOKENS", "0")) or None,
        json_mode=os.getenv("HIRELENS_JSON_MODE", "").strip().lower() in {"1", "true", "yes"},
        extra_body=_json_env("HIRELENS_EXTRA_BODY"),
    )
