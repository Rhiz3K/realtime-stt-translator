# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** this branch now has one fixed Czech microphone → Google Live
  Transcribe → atomic English/Russian Flash-Lite pipeline. Removed alternate
  engines, browser Web Speech, text translation, language/model selectors, and
  user-configurable audio buffering. See README for the supported env vars.
- Production/CI runtime documentation consistently targets Python 3.12;
  frontend regression tests use Node.js 24 without npm dependencies.

### Fixed

- Preserve multiple and delayed final transcripts after Stop; report an
  unconfirmed input tail explicitly instead of falsely declaring success.
- Bound stream-end sends and shutdown, release paid session slots on timeout,
  and cancel paid requests promptly when the browser disconnects during drain.
- Isolate worklet callbacks and fallback timers between sessions; stop visibly
  if the network cannot keep up with the fixed PCM stream.
- Retain missing-final warnings in history, clear stale interim text on reset,
  and omit upstream exception payloads and chained tracebacks from logs.

## [1.1.0] - 2026-06-09

### Added

- **Browser-local Whisper STT engine** (`whisper`) — a fourth STT engine that runs
  OpenAI Whisper as ONNX in the browser via
  [Transformers.js](https://github.com/huggingface/transformers.js). Audio stays on the
  device; only recognized text is sent to `/ws` for translation. Multilingual, Czech included.
  - Models selectable in Settings: `tiny`, `base`, `small`, `large-v3-turbo`
    (from [onnx-community](https://huggingface.co/onnx-community)).
  - Runs on **WebGPU** when available (including Android Chrome 121+) and falls back to
    **CPU/WASM**; the backend is selectable (Auto / WebGPU / CPU).
  - **Multi-threaded CPU** inference via cross-origin isolation (COOP/COEP +
    SharedArrayBuffer) — roughly 3–4× faster than single-threaded on multi-core devices.
  - **Silero VAD** (on-device ONNX, ~2 MB) for speech/non-speech segmentation, with an
    RMS energy gate as fallback if the VAD model can't load.
  - Anti-repetition decoding (`no_repeat_ngram_size`) to break Whisper's hallucination
    loops on silent or uncertain audio.
  - Bounded transcription backlog (drops the oldest segments) so latency can't grow
    without limit on slow devices.
- `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`
  response headers (required to enable multi-threaded WASM).
- GitHub Actions CI workflow — the test suite runs on pushes and pull requests to `main`.

### Changed

- `ENABLED_ENGINES` now accepts `whisper`.
- Content-Security-Policy allows jsDelivr and Hugging Face in `script-src` / `connect-src`
  and adds `'wasm-unsafe-eval'`, needed by Transformers.js and ONNX Runtime Web.
- The app now serves `/static` (the Whisper engine module, the PCM AudioWorklet, and the
  VAD model). `/static/whisper/*` is sent with `Cache-Control: no-cache` so the in-browser
  engine always revalidates.

### Fixed

- Mobile layout: Start / Stop / Settings could be clipped off-screen by a flexbox overflow
  trap on narrow viewports; the top-bar controls now wrap and stay reachable.
- Whisper startup failure (`s._OrtGetInputName is not a function`) caused by loading an
  ONNX Runtime wasm build that did not match the one Transformers.js was built against.

## [1.0.0] - 2026-02-04

Initial release.

- FastAPI app serving a password-protected, real-time STT + translation UI.
- Three STT engines: Web Speech API, Deepgram Nova-3, and ElevenLabs Scribe v2
  (server-side proxy and direct-browser modes).
- Real-time translation into two configurable target languages via googletrans.
- Cookie-based auth with HMAC-signed tokens, rate-limited login, CSRF/origin checks, and a
  Content-Security-Policy.
- Docker image with a health check; configuration via environment variables.
