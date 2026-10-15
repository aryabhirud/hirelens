# HireLens — Project Synopsis

**Team:** Arya Sanjay Bhirud (23070122050), Aryan Choudhary (23070122052), Arun Tati (23070122048)

## Problem
Students and early-career job seekers usually cannot tell how well they fit a role before applying. They either guess, or send the same resume to every posting and never learn why they were rejected. Generic "ATS score" tools count keywords but do not explain the score, do not check whether the match is real, and do not tell the candidate what to do next.

## Solution
HireLens is a resume-to-job matching assistant. The user uploads a resume (PDF) and pastes a job description, gives a job posting URL, or both. HireLens returns a 0–100 fit score with a section-wise breakdown (skills, experience, projects, education, keywords), the skill gaps, improved resume bullets, likely interview questions and concrete actions to close each gap.

The work is done by six cooperating AI agents built with LangGraph rather than one model answering everything. An **Orchestrator** plans each run and re-plans when something goes wrong. The **Resume Parser** turns the PDF into structured data. The **Job Analyst** separates must-have from nice-to-have requirements and, when the posting is vague or only a URL is given, decides for itself to fetch the page or search the web for the company's careers page. The **Matcher** scores each section and must quote the resume as evidence. The **Critic** checks every quote against the actual resume text and sends inflated or unsupported scores back to the Matcher for revision; if they still disagree after two revisions, the Orchestrator arbitrates. Only after the scores are verified does the **Career Coach** write targeted bullets, interview questions and gap actions. Every plan, decision, tool call and hand-off is recorded in a downloadable execution trace.

## Target user
Final-year students, fresh graduates and early-career professionals applying to internships and entry-level roles, who want an honest, explained first check of their fit for a specific job before applying. HireLens is a preparation aid, not a guarantee of selection.

## Scope relative to the approved CA2 description
The system matches the approved description. Implementation details added while building, none of which change the scope:
- The Orchestrator is itself an LLM agent that plans the run and re-plans on weak resume extraction, vague job data, agent failures, and unresolved Matcher–Critic disagreements.
- The Critic combines automatic evidence verification (tools) with LLM judgement, and a policy layer that will not approve scores whose quotes cannot be found in the resume.
- The overall score is computed by a fixed, published weighted formula from the agents' section scores, so it is transparent and repeatable.
- The backup project (campus timetable scheduler) was not pursued.
