FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-nemotron-prep.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-nemotron-prep.txt

COPY . .

RUN adduser --disabled-password --gecos '' appuser
RUN mkdir -p app/static/nemotron/models && chown -R appuser:appuser app/static/nemotron/models
USER appuser

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["sh", "-c", "python -m app.nemotron_assets && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
