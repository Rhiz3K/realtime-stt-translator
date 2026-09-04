# CLAUDE.md

See `AGENTS.md` for the complete repository guidance.

This is a single-purpose FastAPI app. Browser microphone audio is streamed as
fixed PCM16/16 kHz frames to `/ws/audio`, proxied to Google Live Transcribe in
fixed Czech SMART mode, then translated by one structured Flash-Lite request
into an atomic English/Russian pair. There are no selectable engines,
providers, languages, model sizes, or chunk controls.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env  # set GEMINI_API_KEY and APP_PASSWORD
pytest
python -m compileall app tests
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Important invariants: validate auth and Origin before accepting the WebSocket;
never expose the Google key; coalesce/cancel interims but never drop finals;
emit EN and RU together; loop over SDK `receive()` because it ends after each
turn; use `textContent` in the UI; close all async tasks on Stop/disconnect.
