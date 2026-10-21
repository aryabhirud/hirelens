"""Web tools used by the Job Analyst: fetch a job posting page, search the web."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

import requests

from tools.errors import ToolError

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 HireLens/1.0"
)
MAX_PAGE_CHARS = 12000


def _validate_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError(f"Not a valid web address: {url!r}")
    host = parsed.hostname or ""
    if host in {"localhost"} or host.endswith(".local"):
        raise ToolError("Local addresses are not allowed.")
    try:
        if ipaddress.ip_address(host).is_private:
            raise ToolError("Private network addresses are not allowed.")
    except ValueError:
        pass  # a hostname, not an IP literal
    return url


def fetch_url(url: str, timeout: int = 12) -> str:
    """Download a page and return its readable text."""
    from bs4 import BeautifulSoup

    url = _validate_url(url)
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    except requests.RequestException as exc:
        raise ToolError(f"Could not reach {url} ({type(exc).__name__}).") from exc
    if response.status_code >= 400:
        raise ToolError(f"{url} returned HTTP {response.status_code}.")
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise ToolError(f"{url} is not a web page ({content_type or 'unknown type'}).")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()
    text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n"))
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 150:
        raise ToolError(
            f"{url} returned almost no readable text (the page is probably rendered by JavaScript)."
        )
    return text[:MAX_PAGE_CHARS]


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web (DuckDuckGo, no API key needed)."""
    query = (query or "").strip()
    if not query:
        raise ToolError("Empty search query.")
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        raise ToolError(f"Web search failed ({type(exc).__name__}).") from exc
    cleaned = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results or []
    ]
    if not cleaned:
        raise ToolError(f"No search results for {query!r}.")
    return cleaned
