"""Resume PDF extraction and the job-posting web tools."""
from io import BytesIO

import pytest

from tools.errors import ToolError
from tools.pdf_tool import extract_pdf_text
from tools.web_tool import fetch_url, web_search


def test_pdf_extraction(resume_text):
    assert "Brightpath Labs" in resume_text
    assert len(resume_text) > 1000


@pytest.mark.parametrize("data, message", [
    (b"", "empty"),
    (b"hello world, not a pdf", "not a PDF"),
    (b"%PDF-1.4 garbage", "could not be read"),
    (b"%PDF" + b"0" * (5 * 1024 * 1024 + 1), "larger than 5 MB"),
], ids=["empty", "not-pdf", "corrupt-pdf", "oversized"])
def test_pdf_bad_inputs(data, message):
    with pytest.raises(ToolError, match=message):
        extract_pdf_text(data)


def test_pdf_with_no_text_is_rejected():
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf)
    c.rect(100, 100, 200, 200)  # a drawing, no text: behaves like a scanned resume
    c.save()
    with pytest.raises(ToolError, match="scanned"):
        extract_pdf_text(buf.getvalue())


@pytest.mark.parametrize("url", ["ftp://x.com/job", "not a url", "http://localhost/job", "http://192.168.1.4/job"])
def test_fetch_url_rejects_unsafe_urls(url):
    with pytest.raises(ToolError):
        fetch_url(url)


class _FakeResponse:
    def __init__(self, status=200, text="", content_type="text/html"):
        self.status_code, self.text, self.headers = status, text, {"content-type": content_type}


def test_fetch_url_extracts_readable_text(monkeypatch):
    html = "<html><script>ignore()</script><nav>menu</nav><h1>Backend Intern</h1>" + "<p>Python and SQL required.</p>" * 20
    monkeypatch.setattr("tools.web_tool.requests.get", lambda *a, **k: _FakeResponse(text=html))
    text = fetch_url("https://jobs.example.com/1")
    assert "Backend Intern" in text and "ignore()" not in text and "menu" not in text


@pytest.mark.parametrize("response, message", [
    (_FakeResponse(status=403), "HTTP 403"),
    (_FakeResponse(text="%PDF", content_type="application/pdf"), "not a web page"),
    (_FakeResponse(text="<div id=app></div>"), "JavaScript"),
])
def test_fetch_url_reports_unusable_pages(monkeypatch, response, message):
    monkeypatch.setattr("tools.web_tool.requests.get", lambda *a, **k: response)
    with pytest.raises(ToolError, match=message):
        fetch_url("https://jobs.example.com/1")


def test_web_search_rejects_empty_query():
    with pytest.raises(ToolError, match="Empty"):
        web_search("   ")
