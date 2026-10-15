"""Evidence verification, keyword coverage and scoring (the Critic's ground truth)."""
import pytest

from lib.schemas import SectionScore
from lib.scoring import MUST_HAVE_CAP_SCORE, WEIGHTS, overall_score, verdict_label
from lib.trace import Trace
from tools.text_tools import keyword_coverage, verify_evidence


# ---------------------------------------------------------- verify_evidence
def test_verify_evidence_exact(resume_text):
    assert verify_evidence("Built REST APIs in Python with FastAPI", resume_text)["supported"]


def test_verify_evidence_tolerates_line_breaks(resume_text):
    # PDF extraction splits lines; the quote should still verify
    quote = "Comfortable building REST APIs in Python and deploying them with Docker"
    assert verify_evidence(quote, resume_text)["supported"]


def test_verify_evidence_rejects_invented_claim(resume_text):
    result = verify_evidence("Built event streaming pipelines with Kafka and Redis", resume_text)
    assert not result["supported"]


def test_verify_evidence_rejects_too_short(resume_text):
    assert not verify_evidence("Python", resume_text)["supported"]


# --------------------------------------------------------- keyword_coverage
def test_keyword_coverage_handles_symbols():
    text = "Skills: C++, Node.js, CI/CD pipelines, Python and JavaScript."
    result = keyword_coverage(["C++", "Node.js", "CI/CD", "Java", "Go"], text)
    assert result["found"] == ["C++", "Node.js", "CI/CD"]
    assert set(result["missing"]) == {"Java", "Go"}  # "Java" must not match inside "JavaScript"
    assert result["coverage"] == 0.6


def test_keyword_coverage_empty():
    assert keyword_coverage([], "anything")["coverage"] == 0.0


# ------------------------------------------------------------------ scoring
def _sections(**scores):
    return {k: SectionScore(score=scores.get(k, 80)) for k in WEIGHTS}


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_overall_is_weighted_average():
    assert overall_score(_sections(), matched_must=4, missing_must=1) == 80


def test_missing_most_must_haves_caps_score():
    assert overall_score(_sections(), matched_must=1, missing_must=3) == MUST_HAVE_CAP_SCORE


@pytest.mark.parametrize("score, label", [(90, "Strong fit"), (60, "Moderate fit"), (40, "Weak fit"), (10, "Poor fit")])
def test_verdict_labels(score, label):
    assert verdict_label(score) == label


# -------------------------------------------------------------------- trace
def test_trace_records_and_exports():
    seen = []
    trace = Trace(on_event=seen.append)
    trace.log("matcher", "message", "scored", {"score": 70})
    trace.handoff("matcher", "critic", "verify")
    assert [e.event for e in seen] == ["message", "handoff"]
    assert trace.to_dict()["event_count"] == 2
    assert "| matcher | handoff |" in trace.to_markdown()


def test_trace_rejects_unknown_event_type():
    with pytest.raises(ValueError):
        Trace().log("matcher", "guess", "not a real event type")
