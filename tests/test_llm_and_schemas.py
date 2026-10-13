"""LLM client helpers, provider settings and the schemas that guard every hand-off."""
import pytest
from pydantic import ValidationError

from lib.config import load_settings
from lib.llm import LLMClient, LLMError, extract_json
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
