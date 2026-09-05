<!-- codebase-memory-mcp:start -->
# Codebase Knowledge Graph (codebase-memory-mcp)

This project uses codebase-memory-mcp to maintain a knowledge graph of the codebase.
ALWAYS prefer MCP graph tools over grep/glob/file-search for code discovery.

## Priority Order
1. `search_graph` — find functions, classes, routes, variables by pattern
2. `trace_path` — trace who calls a function or what it calls
3. `get_code_snippet` — read specific function/class source code
4. `query_graph` — run Cypher queries for complex patterns
5. `get_architecture` — high-level project summary

## When to fall back to grep/glob
- Searching for string literals, error messages, config values
- Searching non-code files (Dockerfiles, shell scripts, configs)
- When MCP tools return insufficient results
<!-- codebase-memory-mcp:end -->

# Agent Notes (realtime-stt-translator)

This branch is intentionally a single-purpose FastAPI application:

```text
browser microphone
  -> /ws/audio (raw PCM16, mono, 16 kHz)
  -> Google gemini-3.5-transcribe-live (fixed cs-CZ, SMART)
  -> one structured Flash-Lite request per selected transcript
  -> atomic {type, en, ru} interim/final message
```

There is no browser Web Speech path, local model, alternate STT engine,
translation provider, text input, language selector, or user-tunable audio
buffering. Do not add configuration messages to the WebSocket protocol.

## Layout

- `app/main.py`: auth, security headers, health routes, and the sole audio WS.
- `app/google_audio.py`: fixed Google models/configuration and cost-aware actor.
- `app/static/pcm-worklet.js`: fixed 100 ms PCM16 capture/resampling.
- `app/templates/index.html`: one Start/Stop action and EN/RU output.
- `tests/`: provider-free fakes; tests never call Google over the network.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
pytest
python -m compileall app tests
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Use Python 3.12, matching production and CI, and Node.js 24 for frontend tests.
Other Python versions are not covered by CI. No linter or formatter is pinned.
Keep edits local; if available, `ruff check .` and `ruff format .` are suitable.

## Invariants

- The Google API key stays server-side.
- Authenticate and validate WebSocket Origin before `accept()`.
- Results contain EN and RU in the same JSON object; never emit one language at
  a time.
- Keep at most one in-flight interim snapshot and one newest pending interim;
  coalesce any other hypotheses. Finals are FIFO, must not be dropped, and
  immediately cancel speculative interim work.
- Interim translation has no retry. Final translation retries once only for a
  timeout (including HTTP 408), 429, or 5xx response.
- Google SDK `AsyncSession.receive()` ends at each `turn_complete`; call it
  repeatedly for a continuous session.
- Stop must not treat the first final or turn completion as a stream flush.
  Keep receiving until upstream EOF or the bounded deadline; a timeout emits
  `transcription_incomplete`, never a successful `ended`.
- Keep exactly one browser reader alive during shutdown. Disconnect cancels
  paid requests before closing upstream transports. Bound stream-end sends.
- Bind worklet callbacks and Stop timers to a session generation/socket;
  close the port on cleanup and cap the browser's queued audio at 16 000 B.
- Failed finals leave permanent history warnings. Reset clears interim only.
- Live Transcribe sessions last at most ten minutes; the app finishes at 9:45
  with a recoverable `session_limit` event.
- Insert browser content with `textContent`, never `innerHTML`.
- Do not log transcripts, passwords, cookies, or API keys.

## Fixed model choices

- STT: `gemini-3.5-transcribe-live`, `cs-CZ`, `SMART`.
- Interim: `gemini-2.5-flash-lite`, thinking disabled.
- Final: `gemini-3.5-flash-lite`, minimal thinking.
- Audio: signed little-endian PCM16, mono, 16 kHz, 100 ms frames.

These constants deliberately live in code rather than environment variables.
Changing them is a product decision and requires tests plus README updates.

## Security and async style

- HMAC auth tokens use `AUTH_SECRET` (falling back to `APP_PASSWORD`) and
  `secrets.compare_digest`.
- Keep redirects relative via `sanitize_next_path`.
- `ALLOWED_ORIGINS`, when set, is an exact allowlist; otherwise WS Origin host
  must match Host.
- Never block the event loop. Bound external calls with timeouts and drain or
  cancel every spawned task during disconnect.
- Send sanitized JSON errors and use close codes 1008 (policy), 1009 (size),
  1011 (server), and 1013 (temporary overload).
