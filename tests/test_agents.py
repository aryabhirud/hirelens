"""Each worker agent on its own, with a scripted model (no graph, no network)."""
import copy

import pytest

from agents import career_coach, critic, job_analyst, matcher, resume_parser
from lib.llm import LLMError
from tests.conftest import COACHING, INFLATED_MATCH, JOB_FINISH, RESUME_PROFILE, REVISED_MATCH

REVISE = {"verdict": "revise", "summary": "Kafka is not in the resume.",
          "issues": [{"section": "skills", "claim": "Kafka", "problem": "not in resume", "suggested_score": 70}]}
APPROVE = {"verdict": "approve", "summary": "Supported.", "issues": []}


def _state(resume_text, **extra):
    state = {"resume_text": resume_text, "resume_pages": 1, "job_description": "", "job_url": "",
             "agent_notes": {}, "attempts": {}, "match_history": [], "critique_history": [], "critic_rounds": 0}
    state.update(extra)
    return state


# ------------------------------------------------------------ resume parser
def test_parser_returns_validated_profile_and_passes_hint(make_ctx, resume_text):
    ctx, llm = make_ctx({"resume_parser": [RESUME_PROFILE]})
    out = resume_parser.run(ctx, _state(resume_text, agent_notes={"resume_parser": "skills are at the end"}))
    assert out["resume_profile"]["name"] == "Riya Deshmukh"
    assert "skills are at the end" in llm.calls[0][1]


def test_parser_quality_signals():
    assert resume_parser.quality_problems(RESUME_PROFILE) == []
    weak = {"is_resume": False, "skills": ["Python"], "parse_confidence": 0.2}
    problems = resume_parser.quality_problems(weak)
    assert "document does not look like a resume" in problems and len(problems) == 4


# -------------------------------------------------------------- job analyst
def test_analyst_reads_the_given_url_then_finishes(make_ctx, resume_text):
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return "Backend intern. Must have: Python, SQL, REST APIs. " * 10

    steps = [{"thought": "read it", "action": "fetch_url", "url": "https://jobs.example.com/9"}, JOB_FINISH]
    ctx, _ = make_ctx({"job_analyst": steps}, tools={"fetch_url": fake_fetch})
    job = job_analyst.run(ctx, _state(resume_text, job_url="https://jobs.example.com/9"))["job_profile"]
    assert fetched == ["https://jobs.example.com/9"]
    assert "https://jobs.example.com/9" in job["sources"]
    assert job["enrichment_used"] is False  # reading the user's own link is not enrichment


def test_analyst_searches_when_the_post_is_vague(make_ctx, resume_text):
    def fake_search(query, max_results=5):
        return [{"title": "Careers", "url": "https://careers.example.com/x", "snippet": "Python, C++"}]

    steps = [{"thought": "too vague", "action": "web_search", "query": "Qualys SWE intern requirements"}, JOB_FINISH]
    ctx, _ = make_ctx({"job_analyst": steps}, tools={"web_search": fake_search})
    job = job_analyst.run(ctx, _state(resume_text, job_description="SWE Intern at Qualys. Apply now!"))["job_profile"]
    assert job["enrichment_used"] is True
    assert job["keywords"][:1] and set(job["must_have"]) <= set(job["keywords"])


def test_analyst_recovers_from_a_failed_fetch(make_ctx, resume_text, job_text):
    from tools.errors import ToolError

    def broken_fetch(url):
        raise ToolError("HTTP 403")

    steps = [{"thought": "read it", "action": "fetch_url", "url": "https://jobs.example.com/9"}, JOB_FINISH]
    ctx, llm = make_ctx({"job_analyst": steps}, tools={"fetch_url": broken_fetch})
    job = job_analyst.run(ctx, _state(resume_text, job_description=job_text, job_url="https://jobs.example.com/9"))
    assert job["job_profile"]["title"] == "Backend Engineering Intern"
    assert "FAILED: HTTP 403" in llm.calls[1][1]  # the failure was shown to the model as an observation


# ------------------------------------------------------------------ matcher
def test_matcher_measures_keywords_and_hands_off_to_critic(make_ctx, resume_text):
    ctx, _ = make_ctx({"matcher": [REVISED_MATCH]})
    out = matcher.run(ctx, _state(resume_text, resume_profile=RESUME_PROFILE, job_profile=JOB_FINISH["job_profile"]))
    assert 0 < out["keyword_coverage"]["coverage"] <= 1
    assert len(out["match_history"]) == 1
    assert ctx.trace.events[-1].data == {"from": "matcher", "to": "critic"}


def test_matcher_sees_critic_issues_on_revision(make_ctx, resume_text):
    ctx, llm = make_ctx({"matcher": [REVISED_MATCH]})
    state = _state(resume_text, resume_profile=RESUME_PROFILE, job_profile=JOB_FINISH["job_profile"],
                   last="critic", critique=REVISE, match=INFLATED_MATCH, match_history=[INFLATED_MATCH],
                   keyword_coverage={"found": [], "missing": [], "coverage": 0.7})
    matcher.run(ctx, state)
    prompt = llm.calls[0][1]
    assert "THE CRITIC REJECTED IT" in prompt and "not in resume" in prompt


# ------------------------------------------------------------------- critic
def _critic_state(resume_text, match):
    return _state(resume_text, resume_profile=RESUME_PROFILE, job_profile=JOB_FINISH["job_profile"],
                  match=copy.deepcopy(match), match_history=[match])


def test_critic_sends_invented_evidence_back(make_ctx, resume_text):
    ctx, _ = make_ctx({"critic": [REVISE]})
    out = critic.run(ctx, _critic_state(resume_text, INFLATED_MATCH))
    assert out["critic_approved"] is False and out["critic_rounds"] == 1
    assert critic.route_after({**out, "match_history": [INFLATED_MATCH]}, max_revisions=2) == "matcher"


def test_critic_policy_overrides_an_unjustified_approval(make_ctx, resume_text):
    ctx, _ = make_ctx({"critic": [APPROVE]})
    out = critic.run(ctx, _critic_state(resume_text, INFLATED_MATCH))
    assert out["critique"]["verdict"] == "revise"
    assert any(i["claim"] == "Kafka" for i in out["critique"]["issues"])


def test_critic_approves_supported_scores(make_ctx, resume_text):
    ctx, _ = make_ctx({"critic": [APPROVE]})
    out = critic.run(ctx, _critic_state(resume_text, REVISED_MATCH))
    assert out["critic_approved"] is True
    assert critic.route_after({**out, "match_history": [REVISED_MATCH]}, max_revisions=2) == "orchestrator"


def test_critic_stops_after_the_revision_budget():
    state = {"critic_approved": False, "match_history": [INFLATED_MATCH] * 3}
    assert critic.route_after(state, max_revisions=2) == "orchestrator"


# ------------------------------------------------------------- career coach
def test_coach_targets_verified_gaps(make_ctx, resume_text):
    ctx, llm = make_ctx({"career_coach": [COACHING]})
    match = {**REVISED_MATCH, "missing_must_have": ["Kafka"]}
    out = career_coach.run(ctx, _state(resume_text, resume_profile=RESUME_PROFILE,
                                       job_profile=JOB_FINISH["job_profile"], match=match, critique=APPROVE))
    assert out["coaching"]["gap_actions"][0]["gap"] == "Kafka"
    assert "Kafka" in llm.calls[0][1] and "Built REST APIs" in llm.calls[0][1]


def test_coach_rejects_invalid_response(make_ctx, resume_text):
    ctx, _ = make_ctx({"career_coach": [{"bullet_suggestions": [{"original": "Built APIs"}]}]})
    with pytest.raises(LLMError, match="career coach returned invalid structure"):
        career_coach.run(ctx, _state(resume_text, resume_profile=RESUME_PROFILE,
                                     job_profile=JOB_FINISH["job_profile"], match=REVISED_MATCH, critique=APPROVE))
