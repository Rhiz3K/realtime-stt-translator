# Contributing

Thanks for improving the Czech live translator. Please keep this branch's
single-purpose architecture intact: one microphone, one Google STT pipeline,
and one atomic English/Russian result.

## Setup

Use Python 3.12 (the production and CI runtime), Node.js 24 for frontend
regressions, Git, and a microphone-capable browser. Other Python versions are
not part of the supported CI matrix.

```bash
git clone https://github.com/<your-username>/realtime-stt-translator.git
cd realtime-stt-translator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# Set GEMINI_API_KEY and APP_PASSWORD in .env.
pytest
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful checks:

```bash
pytest -vv
pytest tests/test_main.py
pytest -k interim
pytest --cov=app --cov-report=term-missing
python -m compileall app tests
node --test tests/frontend-runtime.test.cjs
```

No formatter is pinned. If available locally, use `ruff check .`,
`ruff format .`, and `mypy app`; avoid unrelated reformatting.

## Architecture rules

- `/ws/audio` accepts binary PCM16 mono audio plus only `{"type":"stop"}`.
- Audio, language, models, throttling, and destinations are fixed internal
  decisions—not environment variables or client configuration.
- Every translation result includes both `en` and `ru` in one JSON message.
- Keep at most one in-flight interim snapshot and one newest pending interim;
  committed finals pre-empt interim work, stay ordered, and are never dropped.
- Tests mock the Google SDK and must never use paid network calls.
- Keep API keys, auth cookies, passwords, and transcript text out of logs.

The backend is async: do not block the event loop, apply timeouts to external
calls, and explicitly cancel/drain tasks on every shutdown path. WebSockets
must validate authentication and Origin before `accept()` and use structured,
sanitized errors.

Frontend content must be inserted with `textContent` or `createTextNode`, never
`innerHTML`. Preserve semantic HTML, accessible labels, `role=status`, and the
English/Russian `lang` attributes.

## Workflow

Use a descriptive branch and focused commits. Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/), for example:

```text
feat(audio): improve fixed pcm resampling
fix(ws): preserve final after superseded interim
test(google): cover live session rollover
```

Before opening a pull request:

- run the full tests and compile check;
- document any new operational environment variable;
- include tests for protocol or scheduling changes;
- confirm no credentials or captured audio entered the diff;
- explain any change to the fixed model/cost choices.

Report security issues privately through GitHub's vulnerability reporting or a
draft security advisory rather than a public issue. See `SECURITY.md`.
