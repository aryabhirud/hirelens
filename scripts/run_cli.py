"""Run HireLens from the command line and save the report and the real execution trace.

Usage:
  python scripts/run_cli.py samples/inputs/sample_resume.pdf --job-file samples/inputs/job_backend_intern.txt
  python scripts/run_cli.py resume.pdf --job-url https://company.com/careers/123
  python scripts/run_cli.py resume.pdf --job-file samples/inputs/job_vague.txt --out samples/outputs/vague

Outputs (in --out, default samples/outputs/<run_id>/): report.json, trace.json, trace.md
These are the files to submit as the sample output and the agent execution trace.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentContext  # noqa: E402
from agents.graph import run_pipeline  # noqa: E402
from lib.config import load_settings  # noqa: E402
from lib.llm import LLMClient  # noqa: E402
from lib.schemas import HireLensInput  # noqa: E402
from lib.trace import Trace  # noqa: E402
from tools.pdf_tool import extract_pdf_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resume", type=Path)
    parser.add_argument("--job-file", type=Path)
    parser.add_argument("--job-url", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    resume_text, pages = extract_pdf_text(args.resume.read_bytes())
    job_text = args.job_file.read_text() if args.job_file else ""
    inp = HireLensInput(resume_filename=args.resume.name, resume_text=resume_text, resume_pages=pages,
                        job_description=job_text, job_url=args.job_url)

    settings = load_settings()
    trace = Trace(on_event=lambda e: print(f"[{e.elapsed_s:6.1f}s] {e.agent:<13} {e.event:<11} {e.summary}"))
    report = run_pipeline(inp, AgentContext(llm=LLMClient(settings), trace=trace, settings=settings))

    out = args.out or ROOT / "samples" / "outputs" / trace.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.json").write_text(inp.model_dump_json(indent=2))
    (out / "report.json").write_text(report.model_dump_json(indent=2))
    (out / "trace.json").write_text(trace.to_json())
    (out / "trace.md").write_text(trace.to_markdown())
    print(f"\n{report.status}: score {report.overall_score} ({report.verdict}) → saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
