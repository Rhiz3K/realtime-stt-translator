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
  -> Web Speech (cs-CZ, continuous, interimResults)
  -> /ws/transcript (interim/final Czech text, never PCM)
  -> one structured Flash-Lite request per selected transcript
  -> atomic {type, en, ru} interim/final message

automatic fallback only when Web Speech is missing or fails:
  microphone -> /ws/audio -> Google gemini-3.5-transcribe-live -> same translator
```

Web Speech is the user-requested primary STT; Google audio is the automatic
backup. No engine selector, local model, alternate translation provider, typed
text input, language selector or user-tunable buffering. Do not add
configuration messages to either WebSocket protocol.

## Layout

- `app/main.py`: auth, security headers, health and protected audio/transcript WS.
- `app/google_audio.py`: fixed Google models/configuration and cost-aware actor.
- `app/static/pcm-worklet.js`: fixed 100 ms PCM16 capture/resampling.
- `app/templates/index.html`: page composed from `components/` Jinja macros.
- `app/static/design-tokens.css`, `translator.css`: tokens, atoms, molecules, layout.
- `app/static/translator-ui.js`: presentation, paired rendering, and telemetry.
- `app/static/interim-layout.js`: viewport-aware growth and shared EN/RU preview font fitting.
- `app/static/audio-session.js`: the sole Start/Stop and audio lifecycle.
- `app/static/web-speech.js`: Czech browser STT, results, restart and watchdog.
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
- Web Speech must never open Google Live STT or send PCM to the server. Local
  PCM is only for the microphone meter, captured duration and stalled-STT check.
- Start Google automatically if the browser API is absent. On a service failure,
  stop Web Speech, warn about a possible gap, drain accepted finals, then open
  the backup. Stop can cancel the switch. Never run both STT paths together.
  Do not switch on permission denial or translation/backend failure. Pass the
  already-authorized audio track to recognition.start(); capture errors within
  Web Speech may use the backup, but getUserMedia denial never may.
- Process Web Speech resultIndex/final indices once per recognition run;
  rebuild the entire replaceable interim suffix, including withdrawals.
  Normal empty onend may restart. Sustained audio without results triggers a
  fixed watchdog; silence alone does not. Timers/callbacks must be detached.
- Web Speech Stop calls recognition.stop(), accepts late finals through onend,
  then stops the text WS. A remaining draft/error/3-second timeout is explicitly
  unconfirmed, never promoted. Server Stop drains accepted finals while the
  sole browser reader keeps watching disconnect. Both routes share paid slots.
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
- Fixed manual endpointing (user-approved replacement of hybrid VAD) sends
  `activity_start` before the first PCM, and `activity_end` at acoustic pauses
  or a 12-second audio bound. Automatic Google VAD is disabled; keep
  NO_INTERRUPTION. Await `generation_complete` before starting the next
  activity, bounded to 1 second; on timeout stop visibly, never reopen silently.
  All PCM stays ordered and unchanged, including leading silence/prefixes.
  The generation acknowledgement gates the next activity only, never proves
  completion of the whole stream. Never promote an interim to a final.
  Keep thresholds in code, never browser settings or env vars.
- Google backup Stop must not treat the first final or turn completion as a stream flush.
  Keep receiving until upstream EOF or the bounded deadline; a timeout emits
  `transcription_incomplete`, never a successful `ended`.
  A genuinely empty session (no audio attempted and no transcript received)
  may end immediately without activity-end/EOF waiting. Never infer emptiness
  from rounded audio seconds, silence, or the absence of interim results.
- Keep exactly one browser reader alive during shutdown. Disconnect cancels
  paid requests before closing upstream transports. Bound activity sends.
- Bind worklet callbacks and Stop timers to a session generation/socket;
  close the port on cleanup and cap the browser's queued audio at 16 000 B.
- Failed finals leave permanent history warnings. Reset clears interim only.
  Unconfirmed completion is a compact warning once per session, not a failed
  final. Do not merge distinct failed final events or warnings across sessions.
  After final drain, preserve any remaining translated preview inside the
  warning as explicitly unconfirmed; never count it as a final.
- Interim failures are visible but do not stop capture or discard final work.
- Only presentation preferences (font, theme, compact toolbar, follow text)
  may be saved locally; never save transcripts or add provider settings.
- Interim grows into available viewport height before shrinking both languages
  together (14 px floor, then scrolling without truncation). Keep a small
  accessible history strip. Never mutate the saved font or final-history font
  when fitting; refit on viewport/toolbar changes and reset after final/clear.
- Telemetry is additive and transcript-free. Label the first STT delay honestly:
  it starts with the first audio frame and includes any leading silence;
  Web Speech measures in the browser, Google backup measures on the server.
- Live Transcribe sessions last at most ten minutes; the app finishes at 9:45
  with a recoverable `session_limit` event.
- Insert browser content with `textContent`, never `innerHTML`.
- Do not log transcripts, passwords, cookies, or API keys.

## Fixed model choices

- Primary STT: browser Web Speech, `cs-CZ`, continuous and interim enabled.
- Backup STT: `gemini-3.5-transcribe-live`, `cs-CZ`, `SMART`.
- Interim: `gemini-3.1-flash-lite`, minimal thinking.
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
