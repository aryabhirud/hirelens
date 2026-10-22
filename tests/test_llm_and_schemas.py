"""LLM client helpers, provider settings and the schemas that guard every hand-off."""
import pytest
from pydantic import ValidationError

from lib.config import load_settings
from lib.llm import LLMClient, LLMError, QuotaExhausted, extract_json, is_daily_quota_error
from lib.schemas import Critique, HireLensInput, JobProfile, MatchResult, ResumeProfile, SectionScore


# ------------------------------------------------------------- extract_json
def test_extract_json_from_fenced_reply():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_preamble():
    assert extract_json('Here you go: {"verdict": "approve"} hope it helps') == {"verdict": "approve"}


def test_extract_json_rejects_non_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


# ------------------------------------------------------------------ config
def test_gemini_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("HIRELENS_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    settings = load_settings()
    assert settings.provider == "gemini" and settings.api_key == "g-key"
    assert "generativelanguage.googleapis.com" in settings.base_url


def test_groq_preset_and_per_agent_model(monkeypatch):
    monkeypatch.setenv("HIRELENS_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "q-key")
    monkeypatch.setenv("HIRELENS_MODEL_CRITIC", "bigger-model")
    settings = load_settings()
    assert settings.api_key == "q-key" and "groq.com" in settings.base_url
    assert settings.model_for("critic") == "bigger-model"
    assert settings.model_for("matcher") == settings.default_model


def test_client_refuses_to_start_without_key(monkeypatch):
    for var in ("HIRELENS_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMError, match="No API key"):
        LLMClient(load_settings())


# ----------------------------------------------------------------- schemas
def test_section_score_is_clamped():
    assert SectionScore(score=140).score == 100
    assert SectionScore(score="72.6").score == 73


def test_match_result_requires_all_sections():
    with pytest.raises(ValidationError):
        MatchResult(sections={"skills": {"score": 50}})


def test_critique_verdict_normalised():
    assert Critique(verdict="Approved").verdict == "approve"
    assert Critique(verdict="needs revision").verdict == "revise"


def test_job_profile_seniority_aliases_and_nulls():
    job = JobProfile.model_validate({"seniority": "Internship", "must_have": None})
    assert job.seniority == "intern" and job.must_have == []


def test_resume_profile_confidence_must_be_a_probability():
    with pytest.raises(ValidationError):
        ResumeProfile(parse_confidence=1.7)


def test_input_requires_resume_text():
    with pytest.raises(ValidationError):
        HireLensInput(resume_filename="r.pdf", resume_text="")


# ------------------------------------------------- client retry behaviour
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason="stop"):
        self.message, self.finish_reason = _Msg(content), finish_reason


class _FakeCompletions:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content, reason = self.replies.pop(0)
        return type("R", (), {"choices": [_Choice(content, reason)]})()


def _client_with(replies, **settings):
    from lib.config import Settings

    client = LLMClient(Settings(api_key="k", base_url="http://unused", default_model="m", **settings), max_retries=2)
    fake = _FakeCompletions(replies)
    client._client = type("C", (), {"chat": type("Ch", (), {"completions": fake})()})()
    return client, fake


def test_empty_reply_retries_with_a_bigger_token_budget():
    client, fake = _client_with([("", "length"), ('{"ok": true}', "stop")])
    assert client.chat_json("matcher", "sys", "user", max_tokens=1000) == {"ok": True}
    assert [r["max_tokens"] for r in fake.requests] == [1000, 2000]


def test_reasoning_effort_is_sent_only_when_configured():
    client, fake = _client_with([('{"a": 1}', "stop")], reasoning_effort="low")
    client.chat_json("critic", "sys", "user")
    assert fake.requests[0]["extra_body"] == {"reasoning_effort": "low"}

    client, fake = _client_with([('{"a": 1}', "stop")])
    client.chat_json("critic", "sys", "user")
    assert "extra_body" not in fake.requests[0]


def test_json_mode_and_extra_options_are_passed_through():
    client, fake = _client_with([('{"a": 1}', "stop")], json_mode=True,
                                extra_body={"reasoning_format": "hidden"})
    client.chat_json("matcher", "sys", "user")
    request = fake.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"reasoning_format": "hidden"}


def test_extra_body_env_must_be_json(monkeypatch):
    monkeypatch.setenv("HIRELENS_EXTRA_BODY", "not json")
    with pytest.raises(ValueError, match="JSON object"):
        load_settings()


def test_output_token_cap_limits_every_request():
    client, fake = _client_with([("", "length"), ('{"ok": 1}', "stop")], max_output_tokens=900)
    client.chat_json("matcher", "sys", "user", max_tokens=3500)
    assert [r["max_tokens"] for r in fake.requests] == [900, 900]


# ------------------------------------------------- rate limits: per-minute vs per-day
def _rate_limit(message):
    import httpx
    from openai import RateLimitError

    request = httpx.Request("POST", "http://unused/chat/completions")
    return RateLimitError(message, response=httpx.Response(429, request=request), body=None)


class _Raising:
    def __init__(self, errors_then_reply):
        self.items, self.requests = list(errors_then_reply), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return type("R", (), {"choices": [_Choice(item)]})()


def _client_raising(items):
    client, _ = _client_with([])
    fake = _Raising(items)
    client._client = type("C", (), {"chat": type("Ch", (), {"completions": fake})()})()
    return client, fake


def test_daily_quota_stops_at_once_without_waiting(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lib.llm.time.sleep", sleeps.append)
    gemini = _rate_limit("Error code: 429 - quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    client, fake = _client_raising([gemini, '{"never": "reached"}'])
    with pytest.raises(QuotaExhausted, match="daily quota"):
        client.chat_json("orchestrator", "sys", "user")
    assert len(fake.requests) == 1 and not [s for s in sleeps if s > 1]


def test_per_minute_limit_is_still_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lib.llm.time.sleep", sleeps.append)
    groq_tpm = _rate_limit("Rate limit reached on tokens per minute (TPM): Limit 1000, Requested 1217")
    client, fake = _client_raising([groq_tpm, '{"ok": 1}'])
    assert client.chat_json("matcher", "sys", "user") == {"ok": 1}
    assert len(fake.requests) == 2 and 8 in sleeps


def test_daily_quota_detection_covers_groq_and_gemini():
    assert is_daily_quota_error(Exception("Limit reached on requests per day (RPD)"))
    assert is_daily_quota_error(Exception("tokens per day (TPD): Limit 100000"))
    assert is_daily_quota_error(Exception("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert not is_daily_quota_error(Exception("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))



# ------------------------------------------------- backup keys and fallback models
GEMINI_DAILY = "Error code: 429 - quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def _multi_key_client(transports, **settings):
    """One fake transport per key, so each test can see which key every request went to."""
    from lib.config import Settings

    keys = list(transports)
    client = LLMClient(Settings(api_key=keys[0], backup_api_keys=keys[1:], base_url="http://unused",
                                default_model="m", **settings), max_retries=2)
    client._clients = {k: type("C", (), {"chat": type("Ch", (), {"completions": t})()})()
                       for k, t in transports.items()}
    return client


def test_used_up_key_hands_over_to_the_next_key(monkeypatch):
    monkeypatch.setattr("lib.llm.time.sleep", lambda s: None)
    first, second = _Raising([_rate_limit(GEMINI_DAILY)]), _Raising(['{"ok": 2}'])
    client = _multi_key_client({"key-1": first, "key-2": second})
    assert client.chat_json("orchestrator", "sys", "user") == {"ok": 2}
    assert len(first.requests) == 1 and len(second.requests) == 1
    assert client.switches == ["daily quota used up; switched to key #2 on m"]


def test_later_runs_start_on_the_key_that_still_works(monkeypatch):
    monkeypatch.setattr("lib.llm.time.sleep", lambda s: None)
    first, second = _Raising([_rate_limit(GEMINI_DAILY)]), _Raising(['{"run": 1}', '{"run": 2}'])
    _multi_key_client({"key-1": first, "key-2": second}).chat_json("matcher", "sys", "user")
    next_run = _multi_key_client({"key-1": first, "key-2": second})
    assert next_run.chat_json("matcher", "sys", "user") == {"run": 2}
    assert len(first.requests) == 1  # the used-up key was not tried again


def test_fallback_model_is_tried_after_every_key(monkeypatch):
    monkeypatch.setattr("lib.llm.time.sleep", lambda s: None)
    first = _Raising([_rate_limit(GEMINI_DAILY), '{"model": "lite"}'])
    second = _Raising([_rate_limit(GEMINI_DAILY)])
    client = _multi_key_client({"key-1": first, "key-2": second}, fallback_models=["m-lite"])
    assert client.chat_json("critic", "sys", "user") == {"model": "lite"}
    assert [r["model"] for r in first.requests] == ["m", "m-lite"]


def test_all_keys_used_up_stops_with_a_clear_message(monkeypatch):
    monkeypatch.setattr("lib.llm.time.sleep", lambda s: None)
    first, second = _Raising([_rate_limit(GEMINI_DAILY)]), _Raising([_rate_limit(GEMINI_DAILY)])
    client = _multi_key_client({"key-1": first, "key-2": second})
    with pytest.raises(QuotaExhausted, match="every configured key"):
        client.chat_json("orchestrator", "sys", "user")


def test_wrong_backup_key_is_skipped(monkeypatch):
    import httpx
    from openai import AuthenticationError

    monkeypatch.setattr("lib.llm.time.sleep", lambda s: None)
    request = httpx.Request("POST", "http://unused/chat/completions")
    bad = AuthenticationError("invalid key", response=httpx.Response(401, request=request), body=None)
    client = _multi_key_client({"key-1": _Raising([_rate_limit(GEMINI_DAILY)]), "key-2": _Raising([bad]),
                                "key-3": _Raising(['{"ok": 3}'])})
    assert client.chat_json("matcher", "sys", "user") == {"ok": 3}


def test_backup_keys_and_fallback_models_are_read_from_env(monkeypatch):
    monkeypatch.delenv("HIRELENS_PROVIDER", raising=False)
    monkeypatch.delenv("HIRELENS_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "main")
    monkeypatch.setenv("GEMINI_API_KEY_2", "aryan")
    monkeypatch.setenv("GEMINI_API_KEY_3", " arun ")
    monkeypatch.setenv("GEMINI_API_KEYS", "extra, main")
    monkeypatch.setenv("HIRELENS_FALLBACK_MODELS", "flash-lite, ")
    settings = load_settings()
    assert settings.api_keys == ["main", "aryan", "arun", "extra"]
    assert settings.fallback_models == ["flash-lite"]
