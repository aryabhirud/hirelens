"""End-to-end behaviour of the multi-agent pipeline with a scripted model.

These tests pin down the agentic behaviour the design relies on: the critic
loop, policy override, arbitration, re-planning and graceful failure.
"""
import copy
import json

import pytest
from tests.conftest import (COACHING, INFLATED_MATCH, JOB_FINISH, PLAN, RESUME_PROFILE, REVISED_MATCH)

from agents.graph import run_pipeline
from lib.llm import LLMError
from lib.schemas import HireLensInput
from tools.errors import ToolError

APPROVE = {"verdict": "approve", "summary": "Scores are supported by the resume.", "issues": []}
REVISE_KAFKA = {"verdict": "revise", "summary": "Kafka claim is not supported.", "issues": [
    {"section": "skills", "claim": "Kafka", "problem": "Kafka does not appear in the resume", "suggested_score": 60}]}


def _input(resume_text, jd="", url=""):
    return HireLensInput(resume_filename="sample_resume.pdf", resume_text=resume_text,
                         job_description=jd, job_url=url)


def _events(ctx, kind=None, agent=None):
    return [e for e in ctx.trace.events if (kind is None or e.event == kind) and (agent is None or e.agent == agent)]


def _happy_script(**overrides):
    script = {
        "orchestrator": [PLAN], "resume_parser": [RESUME_PROFILE], "job_analyst": [JOB_FINISH],
        "matcher": [REVISED_MATCH], "critic": [APPROVE], "career_coach": [COACHING],
    }
    script.update(overrides)
    return copy.deepcopy(script)


def test_happy_path_uses_all_agents(make_ctx, resume_text, job_text):
    ctx, llm = make_ctx(_happy_script())
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert report.status == "completed"
    assert report.critic_approved and report.critic_rounds == 1
    # 0.35*78 + 0.2*70 + 0.2*72 + 0.1*90 + 0.15*60 = 73.7
    assert report.overall_score == 74 and report.verdict == "Moderate fit"
    assert set(ctx.trace.agents_seen()) >= {"orchestrator", "resume_parser", "job_analyst",
                                            "matcher", "critic", "career_coach"}
    assert report.coaching and report.coaching.interview_questions


def test_critic_rejects_inflated_score_and_matcher_revises(make_ctx, resume_text, job_text):
    ctx, llm = make_ctx(_happy_script(matcher=[INFLATED_MATCH, REVISED_MATCH], critic=[REVISE_KAFKA, APPROVE]))
    report = run_pipeline(_input(resume_text, job_text), ctx)

    assert llm.count("matcher") == 2 and report.critic_rounds == 2 and report.critic_approved
    assert report.section_scores["skills"].score == 78
    assert any(e.data.get("to") == "matcher" for e in _events(ctx, "handoff", "critic"))
    # the second matcher call must have been shown the critic's issues
    second_matcher_prompt = [u for a, u in llm.calls if a == "matcher"][1]
    assert "THE CRITIC REJECTED IT" in second_matcher_prompt and "Kafka" in second_matcher_prompt
    unsupported = _events(ctx, "tool_result", "critic")[0].data["unsupported"]
    assert any("Kafka" in u["quote"] for u in unsupported)


def test_policy_overrides_llm_approval_of_unverified_evidence(make_ctx, resume_text, job_text):
    ctx, llm = make_ctx(_happy_script(matcher=[INFLATED_MATCH, REVISED_MATCH], critic=[APPROVE, APPROVE]))
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert llm.count("matcher") == 2  # policy forced a revision even though the LLM approved
    assert any("Policy override" in e.summary for e in _events(ctx, "decision", "critic"))
    assert report.critic_approved


def test_unresolved_disagreement_is_arbitrated_by_orchestrator(make_ctx, resume_text, job_text):
    arbitration = {"reasoning": "Critic's evidence check is objective.", "action": "apply_critic_scores"}
    ctx, llm = make_ctx(_happy_script(orchestrator=[PLAN, arbitration], matcher=[INFLATED_MATCH] * 3,
                                      critic=[REVISE_KAFKA] * 3))
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert llm.count("matcher") == 3 and report.critic_rounds == 3  # 2 revisions max
    assert not report.critic_approved
    assert report.section_scores["skills"].score == 60
    assert "Kafka" in report.skill_gaps and "Kafka" not in report.matched_must_have
    assert "resolved" in report.message


def test_job_url_failure_falls_back_to_pasted_text(make_ctx, resume_text, job_text):
    def broken_fetch(url):
        raise ToolError("https://jobs.example.com/123 returned HTTP 403.")

    fetch_then_finish = [{"thought": "Read the posting first", "action": "fetch_url",
                          "url": "https://jobs.example.com/123"}, JOB_FINISH]
    ctx, _ = make_ctx(_happy_script(job_analyst=fetch_then_finish), tools={"fetch_url": broken_fetch})
    report = run_pipeline(_input(resume_text, job_text, url="https://jobs.example.com/123"), ctx)
    assert report.status == "completed"
    failed = [e for e in _events(ctx, "tool_result", "job_analyst") if not e.data["ok"]]
    assert failed and "403" in failed[0].summary


def test_vague_job_triggers_careers_page_search(make_ctx, resume_text):
    searches = []

    def fake_search(query, max_results=5):
        searches.append(query)
        return [{"title": "Qualys Careers - SWE Intern", "url": "https://careers.example.com/swe-intern",
                 "snippet": "Python, C++, data structures"}]

    def fake_fetch(url):
        return "Software Engineer Intern. Requirements: Python or C++, data structures, Linux, Git. " * 5

    enriched = copy.deepcopy(JOB_FINISH)
    enriched["job_profile"].update(company="Qualys", title="Software Engineer Intern", must_have=["Python", "C++", "Linux"])
    steps = [{"thought": "Too vague, no skills listed", "action": "web_search", "query": "Qualys software engineer intern careers"},
             {"thought": "Open the careers page", "action": "fetch_url", "url": "https://careers.example.com/swe-intern"},
             enriched]
    ctx, _ = make_ctx(_happy_script(job_analyst=steps), tools={"web_search": fake_search, "fetch_url": fake_fetch})
    report = run_pipeline(_input(resume_text, "We're hiring a Software Engineer Intern at Qualys, Pune. Apply now!"), ctx)

    assert searches == ["Qualys software engineer intern careers"]
    assert "https://careers.example.com/swe-intern" in report.job_sources
    job_msg = _events(ctx, "message", "job_analyst")[0]
    assert job_msg.data["job_profile"]["enrichment_used"] is True


def test_not_a_resume_aborts_gracefully(make_ctx, resume_text, job_text):
    not_resume = {"is_resume": False, "skills": [], "parse_confidence": 0.1, "missing_sections": ["experience"]}
    ctx, llm = make_ctx(_happy_script(resume_parser=[not_resume],
                                      orchestrator=[PLAN, {"reasoning": "Looks like an essay", "action": "abort"}]))
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert report.status == "aborted" and "resume" in report.message
    assert report.overall_score is None and llm.count("matcher") == 0


def test_weak_parse_is_retried_with_orchestrator_hint(make_ctx, resume_text, job_text):
    weak = {"is_resume": True, "skills": ["Python"], "parse_confidence": 0.3, "missing_sections": ["skills", "projects"]}
    decision = {"reasoning": "Skills section probably at the end", "action": "retry_resume_parser",
                "note_for_agent": "The skills list is near the end of the document; extract it."}
    ctx, llm = make_ctx(_happy_script(resume_parser=[weak, RESUME_PROFILE], orchestrator=[PLAN, decision]))
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert llm.count("resume_parser") == 2 and report.status == "completed"
    assert "skills list is near the end" in [u for a, u in llm.calls if a == "resume_parser"][1]


def test_coach_failure_still_returns_scores(make_ctx, resume_text, job_text):
    ctx, _ = make_ctx(_happy_script(career_coach=[LLMError("rate limited")]))
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert report.status == "completed" and report.overall_score is not None
    assert report.coaching is None
    assert any("Coaching suggestions could not be generated" in w for w in report.warnings)


def test_matcher_failure_is_retried_then_aborts_cleanly(make_ctx, resume_text, job_text):
    ctx, llm = make_ctx(_happy_script(matcher=[LLMError("timeout"), LLMError("timeout")]))
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert llm.count("matcher") == 2
    assert report.status == "aborted" and "matcher" in report.message


def test_invalid_llm_plan_falls_back_to_default(make_ctx, resume_text, job_text):
    bad_plan = {"reasoning": "skip stuff", "plan": ["matcher", "critic"]}
    ctx, _ = make_ctx(_happy_script(orchestrator=[bad_plan]))
    report = run_pipeline(_input(resume_text, job_text), ctx)
    assert report.status == "completed"
    assert any("invalid" in e.summary for e in _events(ctx, "warning", "orchestrator"))


def test_missing_job_input_is_rejected(make_ctx, resume_text):
    ctx, _ = make_ctx(_happy_script())
    with pytest.raises(ValueError, match="job description"):
        run_pipeline(_input(resume_text), ctx)


def test_trace_exports_as_json_and_markdown(make_ctx, resume_text, job_text):
    ctx, _ = make_ctx(_happy_script())
    run_pipeline(_input(resume_text, job_text), ctx)
    data = json.loads(ctx.trace.to_json())
    assert data["event_count"] == len(data["events"]) > 10
    assert data["events"][-1]["event"] == "final"
    assert "| orchestrator |" in ctx.trace.to_markdown()
