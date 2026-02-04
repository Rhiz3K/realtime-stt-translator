# Agent Notes (sr_live-translator)

This repository is a small FastAPI app that serves a password-protected UI with three STT engines:
- **Web Speech** – browser Web Speech API, sends recognized text to `/ws` for translation.
- **Deepgram** – streams audio to `/ws/deepgram` for Deepgram Nova-3 transcription + translation.
- **ElevenLabs** – streams audio to `/ws/elevenlabs`, server proxies to ElevenLabs Scribe v2 Realtime WS API.

Repo layout:
- `app/main.py`: FastAPI app, auth cookie, websocket handlers, Deepgram + ElevenLabs integration.
- `app/templates/*.html`: Jinja templates with inline CSS + inline JS.
- `tests/test_main.py`: pytest suite (FastAPI TestClient + websocket tests).

No Cursor rules found (`.cursor/rules/`, `.cursorrules` absent).
No Copilot rules found (`.github/copilot-instructions.md` absent).


## Quickstart

Create a venv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the app:
```bash
cp .env.example .env
# edit .env (APP_PASSWORD is required)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```


## Build / Lint / Test Commands

There is no separate build step (Python + templates). The core CI-like checks are:

Test suite:
```bash
pytest
```

Run a single test file:
```bash
pytest tests/test_main.py
```

Run a single test by node id (most precise):
```bash
pytest tests/test_main.py::test_ws_translates_text
```

Run tests matching a substring/expression:
```bash
pytest -k translate
```

Run tests with output (pytest.ini sets `-q` by default):
```bash
pytest -vv
```

Run tests with coverage (requires `pytest-cov` from `requirements-dev.txt`):
```bash
pytest --cov=app --cov-report=term-missing
```

Smoke-check importability (fast, catches syntax errors):
```bash
python -m compileall app tests
```

Lint/format:
- No linter/formatter is pinned in this repo.
- `.gitignore` mentions `.ruff_cache/` and `.mypy_cache/`, so you may see those tools locally.

If you have them installed locally, these are reasonable defaults:
```bash
ruff check .
ruff format .
mypy app
```


## Code Style Guidelines

### Python (general)

- Python version: code uses `X | None` unions, so target Python 3.10+.
- Imports: group in this order with blank lines: stdlib, third-party, local.
- Formatting: keep changes minimal and local; avoid repo-wide reformatting.
- Prefer f-strings; avoid `str.format` unless needed.
- Use `snake_case` for functions/vars, `PascalCase` for classes/types, `UPPER_SNAKE_CASE` for constants.
- Use leading underscore for internal helpers (e.g., `_translate`, `_sign`).
- Prefer explicit return types for non-trivial helpers; use `TypedDict` for JSON payload shapes.

### FastAPI / Starlette patterns

- Keep route handlers small; push logic into helpers.
- Websocket endpoints must:
  - Validate auth and origin before `accept()`.
  - Send structured JSON errors (`{"error": "..."}`) when possible.
  - Close with appropriate codes (1008 for policy/unauthorized, 1011 for server error).

### Async + blocking work

- Do not block the event loop.
  - If a library call is sync-only, offload with `asyncio.to_thread(...)`.
  - Apply timeouts with `asyncio.wait_for(...)`.
- When receiving events from other threads, use `loop.call_soon_threadsafe(...)`.
- Be careful cancelling tasks during shutdown; catch `asyncio.CancelledError`.

### Error handling + logging

- Prefer specific exceptions where practical; use broad `except Exception` only at outer boundaries
  (websocket loops, optional imports, translation calls).
- Log actionable context; do not log secrets (passwords, auth tokens, API keys).
- In websocket handlers, attempt to notify the client, then close; treat send/close failures as best-effort.

### Security / auth

- `APP_PASSWORD` is required; the app should refuse to operate without it.
- Auth tokens are signed with `AUTH_SECRET` (defaults to `APP_PASSWORD`).
- Use `secrets.compare_digest(...)` for secret comparisons.
- Keep redirects safe: only allow relative paths via `sanitize_next_path`.
- Origin checks:
  - If `ALLOWED_ORIGINS` is set, origin must be an exact match.
  - Otherwise, origin host must match the request Host header.

### Optional dependencies + version drift

- Deepgram SDK and googletrans have API drift.
  - Imports are wrapped in `try/except` and code uses duck-typing.
  - Preserve this pattern: keep Deepgram optional so `/` still works when SDK is missing.

### Templates / frontend

- Templates are Jinja (`app/templates/*.html`) and include inline CSS + inline JS.
- Keep accessibility attributes and semantics (aria labels, role=status, skip link).
- Avoid large formatting-only diffs in HTML/CSS/JS.
- When injecting content in JS, use `textContent` / `createTextNode` (avoid `innerHTML`) to prevent XSS.

### Tests

- Use `pytest`.
- Prefer `TestClient` for HTTP and `client.websocket_connect(...)` for websockets.
- Use `monkeypatch` to isolate external services (Translator, Deepgram client).
- When tests mutate module globals (e.g., `main.APP_PASSWORD`), do it inside fixtures.


## Repo-specific gotchas

- Don’t commit secrets:
  - `.env` is gitignored.
  - `credentials/*.json` is gitignored.
- Deepgram websocket logic uses threads + async queues; changes here can introduce races.
- Keep queue bounding behavior (drop old interim results when full) to avoid memory growth.
