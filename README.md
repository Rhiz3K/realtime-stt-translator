# Live Translator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](Dockerfile)

Real-time speech-to-text and translation web application. Speak into a microphone, see transcription appear instantly, and get live translations into two target languages simultaneously.

Built with [FastAPI](https://fastapi.tiangolo.com/), powered by seven interchangeable STT engines, and designed to run anywhere -- locally, in Docker, or behind a reverse proxy.

![UI screenshot](https://github.com/user-attachments/assets/4f4323e9-7cea-4cd3-a8c9-ad8b4d896147)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Local Development](#local-development)
  - [Docker](#docker)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Engine Selection](#engine-selection)
- [API Reference](#api-reference)
  - [HTTP Endpoints](#http-endpoints)
  - [WebSocket Endpoints](#websocket-endpoints)
  - [WebSocket Message Format](#websocket-message-format)
- [Testing](#testing)
- [Deployment](#deployment)
  - [Docker Compose](#docker-compose)
  - [Coolify](#coolify)
  - [Behind a Reverse Proxy](#behind-a-reverse-proxy)
- [Security](#security)
- [Roadmap](#roadmap)
  - [Optional Future Improvements](#optional-future-improvements)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

---

## Features

- **Seven STT engines** -- switchable in the UI at any time:

  | Engine | Runs on | API Key | Notes |
  |---|---|---|---|
  | **Web Speech API** | Browser API | None | Chrome/Edge recommended; may use the browser vendor's remote service unless experimental `processLocally` is enabled and available |
  | **Whisper (local)** | Browser | None | On-device ONNX via Transformers.js; in Auto device mode tiny/base/small fall back to CPU/WASM when WebGPU is unavailable (forcing WebGPU reports the failure instead), while large-v3-turbo requires WebGPU; selectable models are ~75 MB–800 MB and cached; inspired by [whisper_android](https://github.com/vilassn/whisper_android) |
  | **Nemotron (local)** | Browser | None | On-device streaming ONNX ([nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)) via onnxruntime-web; **WebGPU strongly recommended** (CPU/WASM works but isn't real-time); ~1.2 GB one-time download; model must be built first ([guide](app/static/nemotron/README.md)) |
  | **Parakeet v3 (local)** | Browser | None | On-device ONNX ([nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), TDT) via onnxruntime-web; multilingual incl. Czech (~11% FLEURS WER); **WebGPU recommended**; ~930 MB int8 one-time download; build assets first with `scripts/prepare_parakeet_onnx.py` |
  | **Deepgram Nova-3** | Server | Required | High accuracy, low latency |
  | **ElevenLabs Scribe v2** | Server or Browser | Required | Server-side proxy or direct browser connection |
  | **Azure AI Speech** | Browser → Azure | Required | Browser-direct Speech SDK connection using a short-lived token |

- **Real-time translation** into two configurable target languages -- [googletrans](https://github.com/ssut/py-googletrans) by default, optional [DeepL](https://www.deepl.com/pro-api) with automatic switch-over while Google rate-limits the server (or pick one provider in Settings)
- **Interim + final results** -- partial transcriptions shown live before the utterance is committed
- **Interim throttling** -- server-side message versioning skips stale translations to prevent queue buildup
- **Password-protected** -- cookie-based auth with HMAC-signed tokens (can be disabled for VPN/proxy setups)
- **Rate-limited login** -- 10 attempts per 60 seconds per IP
- **Engine access control** -- enable/disable engines per deployment via `ENABLED_ENGINES`
- **Security headers** -- CSP, X-Content-Type-Options, X-Frame-Options, CSRF mitigation
- **Dark mode** -- automatic (`prefers-color-scheme`) or manual toggle (light/dark/system)
- **Responsive UI** -- works on desktop, tablet, and mobile
- **Adjustable font size** -- slider for transcript readability (12--64 px, persisted)
- **Health check** -- `/health` endpoint for Docker `HEALTHCHECK` and load balancers

## Architecture

```
┌─────────────┐       ┌──────────────────────────────┐
│   Browser   │◄─────►│        FastAPI Server         │
│             │  WS   │                               │
│  Web Speech ├──────►│  /ws          (text → translate)│
│  Whisper    ├──────►│  /ws          (text → translate)│
│  Nemotron   ├──────►│  /ws          (text → translate)│
│  Parakeet   ├──────►│  /ws          (text → translate)│
│  Deepgram   ├──────►│  /ws/deepgram (audio → STT → tr.)│
│  ElevenLabs ├──────►│  /ws/elevenlabs (audio → STT → tr.)│
│             │       │                               │
│  ElevenLabs ├──────►│  ElevenLabs WS (browser mode) │
│  (browser)  │  WS   │  ↕ /ws for translation only   │
│  Azure      ├──────►│  Azure Speech (browser direct) │
│             │       │  ↕ /ws for translation only   │
└─────────────┘       └──────────┬───────────────────┘
                                 │
                    ┌────────────┼────────────┬────────────┐
                    ▼            ▼            ▼            ▼
              Deepgram API  ElevenLabs API  Azure Speech  googletrans
              (Nova-3 STT)  (Scribe v2)     (STT)         (translation)
```

**Web Speech** -- The browser's built-in `SpeechRecognition` API handles STT and sends recognized text to `/ws` for translation only. Recognition itself is not necessarily private/on-device: with the default `processLocally=false`, the browser may use a vendor service. Settings can require experimental on-device processing and explicitly install a language pack when the browser supports [`processLocally`](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition/processLocally), `available()`, and `install()`.

**Whisper (local)** -- The browser downloads a Whisper model ([onnx-community](https://huggingface.co/onnx-community) ONNX) and runs it on-device via Transformers.js, with [Silero VAD](https://github.com/snakers4/silero-vad) (also in-browser ONNX) segmenting speech from silence. It uses **WebGPU** when available (recommended, including Android Chrome 121+) and otherwise runs on **CPU/WASM** -- multi-threaded when the page is cross-origin isolated (the app sends the required COOP/COEP headers). The backend is selectable in Settings. Audio never leaves the client; only transcribed text is sent to `/ws` for translation. Similar in spirit to the native [whisper_android](https://github.com/vilassn/whisper_android), but runs entirely in the browser.

**Nemotron (local)** -- The browser runs NVIDIA's [Nemotron-3.5-ASR streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) model (a cache-aware FastConformer encoder + RNN-T decoder, 40 language-locales) fully on-device via onnxruntime-web. The encoder runs on **WebGPU** (strongly recommended — real-time) with a CPU/WASM fallback that works but isn't real-time for this 600 M model. Because it streams, it shows a growing live transcription and commits a final on each pause; audio never leaves the client, only text goes to `/ws`. The model assets (~1.2 GB fp16) are generated, not committed — build them once with `scripts/prepare_nemotron_onnx.py` (see [app/static/nemotron/README.md](app/static/nemotron/README.md)).

**Parakeet v3 (local)** -- The browser runs NVIDIA's [Parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer encoder + TDT decoder, 25 languages incl. Czech) fully on-device via onnxruntime-web, reusing the Nemotron log-mel front-end. Unlike Nemotron the int8 export is offline, so the engine re-encodes the growing utterance buffer for interims and commits a final on each pause. **WebGPU recommended**; audio never leaves the client, only text goes to `/ws`. The ~930 MB int8 assets are downloaded (not committed) by `scripts/prepare_parakeet_onnx.py` (see [app/static/parakeet/README.md](app/static/parakeet/README.md)).

**Deepgram** -- Raw PCM audio streams from the browser to `/ws/deepgram`. The server proxies it to the Deepgram SDK for transcription, then translates via googletrans.

**ElevenLabs (server mode)** -- Same pattern as Deepgram but using the ElevenLabs Scribe v2 Realtime WebSocket API at `/ws/elevenlabs`.

**ElevenLabs (browser mode)** -- The browser fetches a single-use token via `POST /api/elevenlabs/token`, connects directly to the ElevenLabs WS, and sends recognized text to `/ws` for translation (same flow as Web Speech).

**Azure AI Speech (browser-direct)** -- The browser fetches a short-lived token via `POST /api/azure/token`, then the Azure SpeechSDK streams mic audio directly to Azure and sends recognized text to `/ws` for translation (same flow as ElevenLabs browser mode). Good Czech support; free **F0** tier ~5 hrs/month.

## Quick Start

### Prerequisites

- Python 3.10+ (uses `X | None` union syntax)
- A microphone-capable browser (Chrome or Edge recommended for Web Speech)
- API keys for Deepgram, ElevenLabs, and/or Azure (optional -- Web Speech and local engines need none)

### Local Development

```bash
# Clone the repository
git clone https://github.com/Rhiz3K/realtime-stt-translator.git
cd realtime-stt-translator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — at minimum set APP_PASSWORD

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) and enter your password.

### Docker

```bash
docker build -t live-translator .
docker run -p 8000:8000 --env-file .env live-translator
```

Or run with inline environment variables:

```bash
docker run -p 8000:8000 \
  -e APP_PASSWORD=your-secret \
  -e ENABLED_ENGINES=webspeech,deepgram \
  -e DEEPGRAM_API_KEY=your-key \
  live-translator
```

## Configuration

### Environment Variables

Copy [`.env.example`](.env.example) and edit to taste. All variables have sensible defaults except `APP_PASSWORD`.

#### Authentication

| Variable | Required | Default | Description |
|---|---|---|---|
| `APP_PASSWORD` | Yes\* | -- | Login password. Required when `AUTH_ENABLED=true`. |
| `AUTH_ENABLED` | No | `true` | Set `false` to skip login (useful behind VPN or reverse proxy auth). |
| `AUTH_SECRET` | No | `APP_PASSWORD` | Separate HMAC signing secret for auth tokens. **Recommended for production.** |
| `AUTH_COOKIE_NAME` | No | `srlt_auth` | Name of the auth cookie. |
| `AUTH_TOKEN_TTL_SECONDS` | No | `43200` (12 h) | Auth token time-to-live. |
| `AUTH_COOKIE_SECURE` | No | Auto-detect | Force `Secure` flag on cookies. Set `true` behind HTTPS reverse proxy. |

#### Origins

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALLOWED_ORIGINS` | No | -- | Comma-separated allowed WebSocket origins. If empty, origin host must match request Host header. |

#### STT Engines

| Variable | Required | Default | Description |
|---|---|---|---|
| `ENABLED_ENGINES` | No | `webspeech` | Comma-separated list: `webspeech`, `whisper`, `nemotron`, `parakeet`, `deepgram`, `elevenlabs`, `azure`. Disabled engines appear grayed out in the UI. (`nemotron`/`parakeet` require building the model first — see [app/static/nemotron/README.md](app/static/nemotron/README.md) / [app/static/parakeet/README.md](app/static/parakeet/README.md).) |
| `NEMOTRON_AUTO_PREPARE` | No | `true` | When `nemotron` is enabled, create `app/static/nemotron/models` and build missing model assets in the background on container start. First run downloads ~2.6 GB and writes ~1.3 GB. Set `false` if you mount prebuilt assets and do not want automatic generation. |
| `NEMOTRON_PREPARE_TIMEOUT_SECONDS` | No | `1800` | Timeout for the Nemotron model preparation subprocess. Set `0` to disable, or increase it for slow disks/network. |
| `DEEPGRAM_API_KEY` | For Deepgram | -- | API key from [console.deepgram.com](https://console.deepgram.com/) |
| `DEEPGRAM_RESULT_QUEUE_SIZE` | No | `100` | Maximum queued Deepgram final transcripts. Pending interims are coalesced; an overloaded session is stopped instead of dropping a final. |
| `ELEVENLABS_API_KEY` | For ElevenLabs | -- | API key from [elevenlabs.io](https://elevenlabs.io/app/settings/api-keys) |
| `AZURE_SPEECH_KEY` | For Azure | -- | Speech resource key from the [Azure Portal](https://portal.azure.com/). Browser-direct mode; users may also supply their own key in the UI. |
| `AZURE_SPEECH_REGION` | For Azure | -- | Azure region of the Speech resource (e.g. `westeurope`). Free **F0** tier gives ~5 audio hours/month. |

#### Translation

| Variable | Required | Default | Description |
|---|---|---|---|
| `MAX_TEXT_LENGTH` | No | `5000` | Maximum accepted input text length per WebSocket message. |
| `TRANSLATE_TIMEOUT_SECONDS` | No | `10` | Timeout for a single translation call (seconds). |
| `DEEPL_API_KEY` | No | -- | Adds [DeepL](https://www.deepl.com/pro-api) as a translation provider (free keys end with `:fx`, 500k chars/month). |
| `DEEPL_API_URL` | No | derived from key | Override the DeepL translate endpoint (`api-free.deepl.com` for `:fx` keys, else `api.deepl.com`). |
| `TRANSLATE_PROVIDER` | No | `auto` | `auto` = googletrans first, switch to DeepL while Google answers HTTP 429; `google` / `deepl` pins one (never switches); `deepl,google` = explicit order. Users can override per session in Settings. |
| `TRANSLATE_FALLBACK_COOLDOWN_SECONDS` | No | `600` | How long `auto` skips a provider after a rate-limit answer. |
| `NEMOTRON_MODEL_DIR` | No | `app/static/nemotron/models` | Read by `scripts/prepare_nemotron_onnx.py` only — where to write the generated assets. |
| `PARAKEET_MODEL_DIR` | No | `app/static/parakeet/models` | Read by `scripts/prepare_parakeet_onnx.py` only — where to write the downloaded assets. |

### Engine Selection

Engines are enabled via the `ENABLED_ENGINES` environment variable:

```bash
# No vendor API keys (Web Speech + local Whisper)
ENABLED_ENGINES=webspeech,whisper

# Browser-only with the on-device Nemotron streaming model (build the model first)
ENABLED_ENGINES=webspeech,nemotron
NEMOTRON_AUTO_PREPARE=true
NEMOTRON_PREPARE_TIMEOUT_SECONDS=1800

# All engines
ENABLED_ENGINES=webspeech,whisper,nemotron,parakeet,deepgram,elevenlabs,azure

# Deepgram + ElevenLabs only
ENABLED_ENGINES=deepgram,elevenlabs
```

Disabled engines appear in the UI dropdown but are grayed out and cannot be selected.

## API Reference

### HTTP Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | Yes\* | Main UI. Renders login form if not authenticated. |
| `GET` | `/health` | No | Health check. Returns `{"status": "ok"}`. |
| `GET` | `/deepgram` | -- | Legacy redirect to `/`. |
| `POST` | `/login` | No | Form login (`password`, `next`). Sets auth cookie. Rate-limited. |
| `GET` | `/api/translate/languages` | Yes\* | Lists available translation languages. |
| `POST` | `/api/elevenlabs/token` | Yes\* | Creates single-use ElevenLabs Scribe token. Accepts optional `{"api_key": "..."}` body. |
| `POST` | `/api/azure/token` | Yes\* | Creates a short-lived Azure Speech token. Accepts optional `{"api_key": "...", "region": "..."}` body. |

\*Auth is required only when `AUTH_ENABLED=true` (default).

### WebSocket Endpoints

| Path | Input | Description |
|---|---|---|
| `/ws` | JSON text messages | Translates text (Web Speech, local Whisper/Nemotron/Parakeet, ElevenLabs browser mode, and Azure). |
| `/ws/deepgram` | Binary PCM audio | Streams audio to Deepgram for STT + translation. |
| `/ws/elevenlabs` | Binary PCM audio | Streams audio to ElevenLabs for STT + translation. |

All WebSocket endpoints require a matching origin header. A valid auth cookie is additionally required when `AUTH_ENABLED=true`.

### WebSocket Message Format

**Client -> Server** (`/ws`):

```jsonc
// Session config (optional, sent once at start)
{"type": "config", "translate": {"src": "cs", "dests": ["en", "ru"]}}

// Text messages
{"type": "interim", "text": "Ahoj svete", "src": "cs", "dests": ["en", "ru"]}
{"type": "final",   "text": "Ahoj svete", "src": "cs", "dests": ["en", "ru"]}

// Keepalive
{"type": "ping"}
```

**Server -> Client**:

```jsonc
// Translation result
{
  "type": "final",
  "original": "Ahoj svete",
  "dests": ["en", "ru"],
  "translations": {"en": "Hello world", "ru": "Privet mir"}
}

// Error
{"error": "translation_failed"}

// Keepalive response
{"type": "pong"}
```

**Client -> Server** (`/ws/deepgram`, `/ws/elevenlabs`):

The first message can optionally be a JSON config:

```jsonc
{
  "type": "config",
  "deepgram": {"language": "cs", "interim_results": true, "punctuate": true},
  "translate": {"src": "cs", "dests": ["en", "ru"]},
  "translate_interim": false
}
```

All subsequent messages are raw binary PCM audio (16-bit, 16 kHz, mono). In ElevenLabs manual-commit mode, the client may also send `{"type": "commit"}`.

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Verbose output
pytest -vv

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run a specific test
pytest tests/test_main.py::test_ws_translates_text

# Run tests matching a pattern
pytest -k translate

# Quick syntax check (no execution)
python -m compileall app tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow and known test issues.
Trusted pushes and same-repository pull requests also run the pytest suite through `.github/workflows/ci.yml`.

## Deployment

### Docker Compose

```yaml
services:
  live-translator:
    build: .
    ports:
      - "8000:8000"
    environment:
      APP_PASSWORD: ${APP_PASSWORD}
      AUTH_SECRET: ${AUTH_SECRET}
      ENABLED_ENGINES: webspeech,deepgram,elevenlabs
      DEEPGRAM_API_KEY: ${DEEPGRAM_API_KEY}
      ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY}
    restart: unless-stopped
```

### Coolify

1. Create a new service pointing to the GitHub repository.
2. Set environment variables in the Coolify dashboard (see [Configuration](#configuration)).
3. Deploy. The `Dockerfile` includes a `HEALTHCHECK` that Coolify uses automatically.

If `ENABLED_ENGINES` contains `nemotron`, the container starts Uvicorn immediately
and runs `python -m app.nemotron_assets` in the background. That helper creates
`app/static/nemotron/models` and, when any required model file is missing, runs
`scripts/prepare_nemotron_onnx.py`. The first run downloads ~2.6 GB, writes ~1.3
GB, and needs enough temporary disk and memory to convert the encoder. The prepare
subprocess times out after `NEMOTRON_PREPARE_TIMEOUT_SECONDS` (default 1800
seconds; `0` disables it). The rest of the app remains reachable while the assets
are being prepared; the Nemotron engine is usable once the model files are present.
For stable redeploys, mount `app/static/nemotron/models` as persistent storage, or
leave the generated files in the image/container storage if your Coolify setup
preserves them across rebuilds.

### Behind a Reverse Proxy

When running behind nginx, Caddy, or similar:

1. Set `AUTH_COOKIE_SECURE=true` if the proxy terminates TLS.
2. Set `ALLOWED_ORIGINS=https://your-domain.com` to restrict WebSocket origins.
3. Ensure the proxy forwards `Host`, `Origin`, and `X-Forwarded-For` headers.
4. Enable WebSocket proxying for `/ws`, `/ws/deepgram`, and `/ws/elevenlabs`.
5. For local Whisper: serve over HTTPS (microphone and WebGPU require a secure context) and pass the app's `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers through unmodified. If the proxy strips them, Whisper still works but loses the faster multi-threaded CPU path.

Example nginx location block:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location ~ ^/ws {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 86400;
}
```

## Security

- **Authentication** -- HMAC-SHA256 signed tokens in `httpOnly` cookies with configurable TTL.
- **Login rate limiting** -- 10 attempts per 60 seconds per IP (in-memory).
- **CSRF protection** -- Origin/Referer validation on login form submissions.
- **Content Security Policy** -- Restricts script sources, frame ancestors, and connect targets.
- **Permissions Policy** -- Limits microphone and on-device speech-recognition access to the same origin.
- **Cross-origin isolation** -- `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers (also enable multi-threaded WASM for local Whisper).
- **WebSocket origin check** -- Validates `Origin` header against `Host` or `ALLOWED_ORIGINS`.
- **Safe redirects** -- `sanitize_next_path` prevents open redirects after login.
- **No secrets in logs** -- Passwords and API keys are never logged.
- **Memory-only browser keys** -- API keys entered in Settings are not persisted to `localStorage`.

For vulnerability reporting, please see [SECURITY.md](SECURITY.md).

## Roadmap

The following improvements are planned or under consideration. Contributions welcome!

- [ ] **Extend CI** -- add linting and a Docker build to the existing pytest workflow
- [ ] **Internationalize the UI** -- currently Czech labels are hardcoded in templates
- [ ] **Session recording/export** -- save transcriptions and translations to a downloadable file

### Optional Future Improvements

These items would improve the project but are not blocking. They make great first contributions:

- [ ] **Expand test coverage** -- add tests for:
  - Browser-driven Web Speech lifecycle behavior across supported Chrome/Edge versions
  - Real vendor reconnect/error matrices for Deepgram, ElevenLabs, and Azure
- [ ] **Extract duplicated AudioWorklet PCM processor** code into a shared JavaScript constant -- the same processor is currently inlined in three places (Deepgram, ElevenLabs server mode, ElevenLabs browser mode)
- [x] **Second translation provider** -- DeepL is available via `DEEPL_API_KEY`; `TRANSLATE_PROVIDER=auto` switches to it while `googletrans` (unofficial API, per-IP HTTP 429) is rate limited, and the Settings dialog can pin either. Another provider (e.g. `google-cloud-translate`, Azure Translator) slots into `TranslationRouter` the same way

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Support

See [SUPPORT.md](SUPPORT.md).

## License

This project is licensed under the [MIT License](LICENSE).

---

Release history: [CHANGELOG.md](CHANGELOG.md).

Made with care by [Rhiz3K](https://github.com/Rhiz3K)
