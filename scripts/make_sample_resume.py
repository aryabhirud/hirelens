"""Generate the fictional sample resume used in tests and as the sample input.

Run:  python scripts/make_sample_resume.py
The person in it is invented; replace with a real (consenting) resume for demos.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "samples" / "inputs" / "sample_resume.pdf"

styles = getSampleStyleSheet()
H = ParagraphStyle("H", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.5, leading=12.5)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=10, bulletIndent=2)

CONTENT = [
    ("title", "Riya Deshmukh"),
    ("p", "B.Tech Computer Science (Final Year) · Pune, India · riya.deshmukh@example.com · github.com/riya-d-example"),
    ("h", "Summary"),
    ("p", "Final-year computer science student focused on backend development and applied machine learning. "
          "Comfortable building REST APIs in Python and deploying them with Docker."),
    ("h", "Education"),
    ("p", "B.Tech in Computer Science and Engineering, Example Institute of Technology, Pune — 2022 to 2026, CGPA 8.4/10"),
    ("p", "Higher Secondary (Science), Pune — 2022, 89%"),
    ("h", "Experience"),
    ("p", "<b>Software Development Intern</b>, Brightpath Labs, Pune — May 2025 to July 2025 (3 months)"),
    ("b", "Built REST APIs in Python with FastAPI for an internal inventory tool used by 40 staff."),
    ("b", "Wrote PostgreSQL queries and reduced the slowest report query from 9 seconds to 2 seconds by adding indexes."),
    ("b", "Containerised the service with Docker and set up a GitHub Actions pipeline that runs unit tests on every pull request."),
    ("p", "<b>Teaching Assistant, Data Structures</b>, Example Institute of Technology — Jan 2025 to Apr 2025"),
    ("b", "Held weekly lab sessions for 60 second-year students on trees, graphs and dynamic programming in C++."),
    ("h", "Projects"),
    ("p", "<b>CampusEats — food pre-ordering app</b> (Python, FastAPI, React, PostgreSQL)"),
    ("b", "Designed the database schema and order APIs; handled 300 orders during the college fest pilot."),
    ("p", "<b>Resume Screening Model</b> (Python, scikit-learn, pandas)"),
    ("b", "Trained a TF-IDF and logistic regression classifier to tag resumes by job category with 87% accuracy on a public dataset."),
    ("p", "<b>Traffic Signal Simulation</b> (Python, SUMO)"),
    ("b", "Simulated adaptive signal timing for a four-way junction and compared it with fixed timing."),
    ("h", "Skills"),
    ("p", "Languages: Python, C++, JavaScript, SQL · Backend: FastAPI, Flask, REST APIs · Data: pandas, NumPy, "
          "scikit-learn · Databases: PostgreSQL, MySQL · Tools: Git, Docker, GitHub Actions, Linux"),
    ("h", "Certifications"),
    ("p", "AWS Cloud Practitioner (2025)"),
]


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title="Riya Deshmukh - Resume")
    flow = []
    for kind, text in CONTENT:
        if kind == "title":
            flow.append(Paragraph(text, styles["Title"]))
        elif kind == "h":
            flow.append(Paragraph(text, H))
        elif kind == "b":
            flow.append(Paragraph(text, BUL, bulletText="•"))
        else:
            flow.append(Paragraph(text, B))
        flow.append(Spacer(1, 1.5))
    doc.build(flow)
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
