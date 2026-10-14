"""Shared pieces for all agents: run context, pipeline state, logged tool calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from lib.config import Settings
from lib.llm import SupportsChatJSON
from lib.trace import Trace
from tools.errors import ToolError
from tools.text_tools import keyword_coverage, verify_evidence
from tools.web_tool import fetch_url, web_search

AGENTS = ("resume_parser", "job_analyst", "matcher", "critic", "career_coach")


class PipelineState(TypedDict, total=False):
    """The shared blackboard every agent reads from and writes to."""

    # input
    resume_filename: str
    resume_text: str
    resume_pages: int
    job_description: str
    job_url: str
    # orchestration
    plan: list[str]
    completed: list[str]
    next: str
    last: str
    failed: dict[str, str]           # {"agent": ..., "error": ...} from the last step
    agent_notes: dict[str, str]      # orchestrator's instructions to specific agents
    attempts: dict[str, int]
    status: str                      # running | completed | aborted
    message: str
    # agent outputs
    resume_profile: dict
    job_profile: dict
    keyword_coverage: dict
    match: dict
    match_history: list[dict]
    critique: dict
    critique_history: list[dict]
    critic_rounds: int
    critic_approved: bool
    arbitration: dict
    coaching: dict
    warnings: list[str]
    report: dict


@dataclass
class AgentContext:
    llm: SupportsChatJSON
    trace: Trace
    settings: Settings
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        defaults = {
            "fetch_url": fetch_url,
            "web_search": web_search,
            "verify_evidence": verify_evidence,
            "keyword_coverage": keyword_coverage,
        }
        self.tools = {**defaults, **self.tools}


def call_tool(ctx: AgentContext, agent: str, name: str, **kwargs: Any) -> Any:
    """Run a tool and record both the call and its outcome in the trace."""
    shown = {k: (v if len(str(v)) < 120 else str(v)[:117] + "...") for k, v in kwargs.items()}
    ctx.trace.log(agent, "tool_call", f"{name}({_fmt_args(shown)})", {"tool": name, "args": shown})
    try:
        result = ctx.tools[name](**kwargs)
    except ToolError as exc:
        ctx.trace.log(agent, "tool_result", f"{name} failed: {exc}", {"tool": name, "ok": False, "error": str(exc)})
        raise
    except Exception as exc:  # unexpected tool bug: surface as a ToolError so agents can recover
        ctx.trace.log(agent, "tool_result", f"{name} crashed: {exc}", {"tool": name, "ok": False, "error": repr(exc)})
        raise ToolError(f"{name} failed unexpectedly: {exc}") from exc
    ctx.trace.log(agent, "tool_result", f"{name} ok: {_preview(result)}", {"tool": name, "ok": True, "result": _clip(result)})
    return result


def to_prompt_json(value: Any, limit: int = 8000) -> str:
    text = json.dumps(value, indent=1, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items() if k not in {"resume_text", "text"})


def _preview(result: Any) -> str:
    if isinstance(result, str):
        return f"{len(result)} chars of text"
    if isinstance(result, list):
        return f"{len(result)} results"
    if isinstance(result, dict):
        if "coverage" in result:
            return f"coverage {result['coverage']:.0%} ({len(result.get('found', []))} found)"
        if "supported" in result:
            return "supported" if result["supported"] else "NOT supported"
    return str(result)[:80]


def _clip(result: Any) -> Any:
    if isinstance(result, str):
        return result[:1500] + ("..." if len(result) > 1500 else "")
    return result
