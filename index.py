"""Entrypoint for Vercel: it looks for a FastAPI instance named `app` in index.py.

Locally, run the same app with:  uvicorn server.main:app --reload
"""
from server.main import app  # noqa: F401
