"""Shared fixtures. Tests never call a real LLM or the network.

`ScriptedLLM` stands in for the model: each agent gets a queue of replies.
The pipeline, routing, tools, critic policy and scoring all run for real.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pdf_tool import extract_pdf_text  # noqa: E402

# lib.* and agents.* are imported inside the helpers below, so this file (and the
# tests that only need the sample resume) work before those modules are merged.

SAMPLE_PDF = ROOT / "samples" / "inputs" / "sample_resume.pdf"
SAMPLE_JD = ROOT / "samples" / "inputs" / "job_backend_intern.txt"


class ScriptedLLM:
    def __init__(self, script: dict[str, list[Any]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, agent: str, system: str, user: str, max_tokens: int = 2500) -> dict:
        self.calls.append((agent, user))
        queue = self.script.get(agent)
        if not queue:
            from lib.llm import LLMError

            raise LLMError(f"no scripted reply left for {agent}")
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply(user) if callable(reply) else reply

    def count(self, agent: str) -> int:
        return sum(1 for a, _ in self.calls if a == agent)


@pytest.fixture(scope="session")
def resume_text() -> str:
    if not SAMPLE_PDF.exists():
        from scripts.make_sample_resume import build

        build()
    text, _ = extract_pdf_text(SAMPLE_PDF.read_bytes())
    return text


@pytest.fixture(scope="session")
def job_text() -> str:
    return SAMPLE_JD.read_text()


@pytest.fixture
def make_ctx():
    def _make(script: dict[str, list[Any]], tools: dict | None = None):
        from agents.base import AgentContext
        from lib.config import Settings
        from lib.trace import Trace

        llm = ScriptedLLM(script)
        settings = Settings(api_key="test", default_model="test-model", max_critic_revisions=2, max_analyst_steps=4)
        ctx = AgentContext(llm=llm, trace=Trace(), settings=settings, tools=tools or {})
        return ctx, llm

    return _make


# ---------------------------------------------------------------- replies --
PLAN = {"reasoning": "Full job text pasted; standard order.",
        "plan": ["resume_parser", "job_analyst", "matcher", "critic", "career_coach"],
        "notes_for_agents": {}}

RESUME_PROFILE = {
    "is_resume": True, "name": "Riya Deshmukh", "headline": "Final-year CS student",
    "skills": ["Python", "C++", "JavaScript", "SQL", "FastAPI", "Flask", "PostgreSQL", "MySQL", "Git",
               "Docker", "GitHub Actions", "pandas", "scikit-learn", "Linux"],
    "experience": [{"title": "Software Development Intern", "organization": "Brightpath Labs",
                    "duration": "May 2025 - July 2025", "months": 3,
                    "highlights": ["Built REST APIs in Python with FastAPI for an internal inventory tool used by 40 staff."]}],
    "total_experience_months": 3,
    "projects": [{"name": "CampusEats", "description": "Food pre-ordering app",
                  "technologies": ["Python", "FastAPI", "React", "PostgreSQL"]}],
    "education": [{"degree": "B.Tech CSE", "institution": "Example Institute of Technology", "year": "2026", "score": "8.4"}],
    "certifications": ["AWS Cloud Practitioner"], "parse_confidence": 0.92, "missing_sections": [],
}

JOB_FINISH = {"thought": "Requirements are explicit in the pasted text.", "action": "finish", "job_profile": {
    "title": "Backend Engineering Intern", "company": "Nimbus Analytics", "seniority": "intern",
    "min_years_experience": None,
    "must_have": ["Python", "REST APIs", "SQL", "Git", "B.Tech in Computer Science"],
    "nice_to_have": ["Docker", "Kubernetes", "Kafka", "AWS", "Redis"],
    "responsibilities": ["Build REST APIs", "Optimise SQL"], "education_requirement": "B.Tech CS",
    "keywords": ["Python", "REST APIs", "PostgreSQL", "Docker", "Kafka", "Redis", "Git"],
    "completeness": 0.9, "sources": ["user description"]}}


def section(score, claim, quote):
    return {"score": score, "reasoning": "r", "evidence": [{"claim": claim, "resume_quote": quote}]}


GOOD_SECTIONS = {
    "skills": section(78, "Python REST APIs", "Built REST APIs in Python with FastAPI"),
    "experience": section(70, "Relevant internship", "Software Development Intern, Brightpath Labs"),
    "projects": section(72, "API project", "Designed the database schema and order APIs"),
    "education": section(90, "CS degree", "B.Tech in Computer Science and Engineering"),
    "keywords": section(60, "Keyword overlap", "Containerised the service with Docker"),
}

INFLATED_MATCH = {
    "sections": {**GOOD_SECTIONS,
                 "skills": section(95, "Kafka experience", "Built event streaming pipelines with Kafka and Redis")},
    "matched_must_have": ["Python", "REST APIs", "SQL", "Git", "B.Tech in Computer Science", "Kafka"],
    "missing_must_have": [], "matched_nice_to_have": ["Docker", "Kafka"],
    "missing_nice_to_have": ["Kubernetes", "AWS", "Redis"], "strengths": ["APIs"],
}

REVISED_MATCH = {
    "sections": GOOD_SECTIONS,
    "matched_must_have": ["Python", "REST APIs", "SQL", "Git"],
    "missing_must_have": [], "matched_nice_to_have": ["Docker"],
    "missing_nice_to_have": ["Kubernetes", "Kafka", "Redis"], "strengths": ["FastAPI internship"],
    "revision_notes": ["Removed Kafka claim: not in resume; skills lowered to 78"],
}

COACHING = {
    "bullet_suggestions": [{"original": "Built REST APIs...", "improved": "Designed FastAPI REST endpoints...",
                            "targets": "REST APIs"}],
    "interview_questions": [{"question": "How did you index the slow query?", "category": "technical",
                             "why_asked": "SQL must-have"}],
    "gap_actions": [{"gap": "Kafka", "action": "Add a small producer/consumer to CampusEats"}],
}
