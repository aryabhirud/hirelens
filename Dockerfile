# One container serves both the API and the web frontend.
# Works on Hugging Face Spaces (Docker), Render, Railway, or any Docker host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/hirelens

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health')"
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
