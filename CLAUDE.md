# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Real-time speech-to-text + translation web app: FastAPI backend, Jinja templates with inline JS/CSS, five switchable STT engines. `AGENTS.md` contains additional agent notes and code-style detail; `CONTRIBUTING.md` has the full workflow.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # set at least APP_PASSWORD

# Run dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Tests (pytest.ini sets -q, testpaths=tests, pythonpath=.)
pytest
pytest tests/test_main.py::test_ws_translates_text   # single test
pytest -k translate                                  # by substring
pytest --cov=app --cov-report=term-missing           # coverage

# Fast syntax check
python -m compileall app tests
```

No linter/formatter is pinned. If available locally: `ruff check .`, `ruff format .`, `mypy app` (expect noise from googletrans/deepgram typing). CI (`.github/workflows/ci.yml`) runs `pytest` on a self-hosted runner, Python 3.12; fork PRs are skipped. Code targets Python 3.10+ (`X | None` unions).

**Known issue:** three Deepgram tests (`test_ws_deepgram_happy_path_emits_interim_and_final`, `test_ws_deepgram_init_failure_sends_error`, `test_ws_deepgram_missing_sdk_returns_error`) can hang due to threading/SDK interactions — run them individually when touching Deepgram code and be ready to kill them.

Commits follow Conventional Commits (`feat(stt): ...`, `fix(ws): ...`). New env vars must be documented in `.env.example` (and README's table).

## Architecture

The entire backend is `app/main.py` (~1500 lines): auth, CSP middleware, HTTP routes, and three WebSocket endpoints. The entire frontend UI is `app/templates/index.html` (~3500 lines of inline JS/CSS) plus the local-Whisper engine in `app/static/whisper/`.

### Five STT engines, two data-flow shapes

1. **Text-only flow** — STT happens in the browser; the server only translates:
   - **Web Speech**: browser `SpeechRecognition` → text → `/ws`
   - **Whisper (local)**: mic → `pcm-worklet.js` (16 kHz int16 frames) → Silero VAD (`silero-vad.onnx`) segments speech → Transformers.js Whisper ONNX (WebGPU, or WASM/CPU fallback) in `whisper-engine.mjs` → text → `/ws`. Audio never leaves the client.
   - **Nemotron (local)**: mic → `pcm-worklet.js` → `mel.js` log-mel → fp16 cache-aware FastConformer encoder (onnxruntime-web, WebGPU with WASM fallback) + RNN-T greedy decode (`rnnt.js`) in `nemotron-engine.mjs` → text → `/ws`. Streaming, so it emits growing interims and commits a final at end-of-utterance (RMS VAD). Model assets are generated/gitignored (~1.2 GB) — see `app/static/nemotron/README.md`. Audio never leaves the client.
   - **ElevenLabs browser mode**: browser fetches a single-use token via `POST /api/elevenlabs/token`, connects directly to the ElevenLabs WS, sends recognized text to `/ws`.

2. **Audio-proxy flow** — browser streams raw PCM (16-bit, 16 kHz, mono) to the server, which proxies to a vendor STT API and translates results:
   - **`/ws/deepgram`**: Deepgram SDK live connection. The SDK callback runs in a separate listener thread; results cross into asyncio via `event_loop.call_soon_threadsafe` into a bounded `asyncio.Queue` (drops interim results when full, prefers keeping finals — preserve this bounding behavior). Shutdown drains the queue within a grace deadline.
   - **`/ws/elevenlabs`**: server opens its own WS to ElevenLabs Scribe (via `websockets`), then runs two concurrent tasks (`_forward_audio`, `_receive_transcripts`) until either stops.

### WebSocket protocol (`/ws` and audio endpoints)

Clients send typed JSON: `{"type": "config", ...}` (optional, first message), `{"type": "interim"|"final", "text": ..., "src": ..., "dests": [...]}`, `{"type": "ping"}`. Server replies with `{"type": ..., "original": ..., "translations": {...}}` or `{"error": ...}`. A legacy plain-text path (Czech → en/ru) is retained in `/ws` — don't break it. Interim messages carry a version counter server-side; stale interims are skipped before and after the (slow) translation call, while finals are always translated.

### Translation

`googletrans` (unofficial API, 1–3 s per call, can break). `_translate()` handles both sync and async variants of the library, offloads sync calls with `asyncio.to_thread`, and applies `TRANSLATE_TIMEOUT_SECONDS`. On failure the `Translator` is recreated (stale HTTP session) and a `translation_failed` error payload is sent.

### Auth & security

- HMAC-SHA256 signed tokens (`payload_b64.sig_b64`) in an httpOnly cookie; signed with `AUTH_SECRET` (falls back to `APP_PASSWORD`). Compare secrets only via `secrets.compare_digest`.
- WS endpoints validate auth **and** origin before `accept()` (`_require_ws_auth`); HTTP routes use `_require_http_auth`. Close codes: 1008 policy/unauthorized, 1011 server error.
- Origin rules: exact match against `ALLOWED_ORIGINS` if set, otherwise origin host must equal the Host header.
- `/login` is rate-limited in memory (10/60 s per IP) with Origin/Referer CSRF check; redirects sanitized by `sanitize_next_path`.
- `_CSPMiddleware` sets CSP, COOP/COEP (cross-origin isolation → SharedArrayBuffer → multi-threaded WASM for Whisper), and `no-cache` for `/static/whisper/*`.

### Version-drift defensiveness (intentional pattern — preserve it)

`deepgram-sdk` and `googletrans` have breaking API changes across versions. All their imports are wrapped in `try/except` with duck-typed fallbacks (`_looks_like_deepgram_results`, `_deepgram_send_finalize`, sync/async `_translate`). Deepgram must stay optional so the app still boots (Web Speech mode) without the SDK. Dependencies in `requirements.txt` are deliberately mostly unpinned.

## Cross-file sync constraints

These pairs must change together:

- **Transformers.js / ONNX Runtime versions** are pinned in *both* the CSP in `app/main.py` (`_CSPMiddleware`) and `app/static/whisper/whisper-engine.mjs` (import URL + `env.backends.onnx.wasm.wasmPaths`). The wasm must come from the same transformers.js build or session creation throws (`s._OrtGetInputName is not a function`); a CSP mismatch silently blocks the CDN fetch.
- **Whisper model list**: `WHISPER_MODELS` in `whisper-engine.mjs` ↔ the model `<select>` in `index.html` (marked with a "Keep in sync" comment).
- **Cache-buster**: `index.html` imports `/static/whisper/whisper-engine.mjs?v=N` — bump `N` when changing the engine file.
- **Nemotron engine**: the onnxruntime-web version is pinned in *both* the CSP in `app/main.py` (`_CSPMiddleware`) and `nemotron-engine.mjs` (import URL + `env.wasm.wasmPaths`); bump the `?v=N` on the `nemotron-engine.mjs` import in `index.html` when changing it. The fp16 model in `app/static/nemotron/models/` is gitignored and rebuilt by `scripts/prepare_nemotron_onnx.py` (`requirements-nemotron-prep.txt`). The encoder is uniformly fp16 (JS feeds fp16 via `f16.js`); `decoder_joint.onnx` stays fp32 on WASM. `mel.js` must stay numerically in sync with `scripts/nemotron_reference.py` (validated by `scripts/validate_mel.mjs`).
- **PCM framing**: `FRAME_SAMPLES = 512` in `pcm-worklet.js` matches the VAD timing constants (~32 ms/frame at 16 kHz) in `whisper-engine.mjs`.
- **Engine names**: `_ALL_ENGINES` / `ENABLED_ENGINES` in `main.py` ↔ engine dropdown and gating logic in `index.html` (template receives `enabled_engines`).

## Conventions that matter here

- Czech and English are mixed in comments, log messages, and UI labels (UI i18n is a roadmap item). Don't translate or "fix" them in unrelated PRs; avoid large formatting-only diffs in templates.
- Never block the event loop: sync library calls go through `asyncio.to_thread`, timeouts via `asyncio.wait_for`, cross-thread events via `loop.call_soon_threadsafe`.
- Broad `except Exception` is acceptable only at outer boundaries (WS loops, optional imports, translation calls); treat WS send/close failures as best-effort.
- Frontend JS: insert text via `textContent`/`createTextNode` (never `innerHTML`); wrap `JSON.parse` in try/catch in `onmessage` handlers; keep ARIA attributes.
- Tests use `TestClient` + `client.websocket_connect(...)`; the `client` fixture monkeypatches module globals (`main.APP_PASSWORD`, `main.ENABLED_ENGINES`, ...) — mutate globals only inside fixtures. External services (Translator, Deepgram) are faked via `monkeypatch`.
- The same AudioWorklet PCM processor is currently inlined in three places in `index.html` (Deepgram, ElevenLabs server, ElevenLabs browser) — a known wart; extraction is a roadmap item.
