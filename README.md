# Live Translator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](Dockerfile)

Real-time speech-to-text and translation web application. Speak into a microphone, see transcription appear instantly, and get live translations into two target languages simultaneously.

Built with [FastAPI](https://fastapi.tiangolo.com/), powered by three interchangeable STT engines, and designed to run anywhere -- locally, in Docker, or behind a reverse proxy.

![UI screenshot](https://github.com/user-attachments/assets/decc7103-55b0-47a3-8982-e19eaca54e90)

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

- **Three STT engines** -- switchable in the UI at any time:

  | Engine | Runs on | API Key | Notes |
  |---|---|---|---|
  | **Web Speech API** | Browser | None | Chrome/Edge recommended; no server cost |
  | **Deepgram Nova-3** | Server | Required | High accuracy, low latency |
  | **ElevenLabs Scribe v2** | Server or Browser | Required | Server-side proxy or direct browser connection |

- **Real-time translation** into two configurable target languages (powered by [googletrans](https://github.com/ssut/py-googletrans))
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
│  Deepgram   ├──────►│  /ws/deepgram (audio → STT → tr.)│
│  ElevenLabs ├──────►│  /ws/elevenlabs (audio → STT → tr.)│
│             │       │                               │
│  ElevenLabs ├──────►│  ElevenLabs WS (browser mode) │
│  (browser)  │  WS   │  ↕ /ws for translation only   │
└─────────────┘       └──────────┬───────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              Deepgram API  ElevenLabs API  googletrans
              (Nova-3 STT)  (Scribe v2)    (translation)
```

**Web Speech** -- The browser's built-in `SpeechRecognition` API handles STT locally; recognized text is sent to `/ws` for translation only.

**Deepgram** -- Raw PCM audio streams from the browser to `/ws/deepgram`. The server proxies it to the Deepgram SDK for transcription, then translates via googletrans.

**ElevenLabs (server mode)** -- Same pattern as Deepgram but using the ElevenLabs Scribe v2 Realtime WebSocket API at `/ws/elevenlabs`.

**ElevenLabs (browser mode)** -- The browser fetches a single-use token via `POST /api/elevenlabs/token`, connects directly to the ElevenLabs WS, and sends recognized text to `/ws` for translation (same flow as Web Speech).

## Quick Start

### Prerequisites

- Python 3.10+ (uses `X | None` union syntax)
- A microphone-capable browser (Chrome or Edge recommended for Web Speech)
- API keys for Deepgram and/or ElevenLabs (optional -- Web Speech works without any)

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
| `ENABLED_ENGINES` | No | `webspeech` | Comma-separated list: `webspeech`, `deepgram`, `elevenlabs`. Disabled engines appear grayed out in the UI. |
| `DEEPGRAM_API_KEY` | For Deepgram | -- | API key from [console.deepgram.com](https://console.deepgram.com/) |
| `DEEPGRAM_RESULT_QUEUE_SIZE` | No | `100` | Internal queue size for Deepgram transcription results. |
| `ELEVENLABS_API_KEY` | For ElevenLabs | -- | API key from [elevenlabs.io](https://elevenlabs.io/app/settings/api-keys) |

#### Translation

| Variable | Required | Default | Description |
|---|---|---|---|
| `MAX_TEXT_LENGTH` | No | `5000` | Maximum accepted input text length per WebSocket message. |
| `TRANSLATE_TIMEOUT_SECONDS` | No | `10` | Timeout for a single googletrans call (seconds). |

### Engine Selection

Engines are enabled via the `ENABLED_ENGINES` environment variable:

```bash
# Web Speech only (default — no API keys needed)
ENABLED_ENGINES=webspeech

# All engines
ENABLED_ENGINES=webspeech,deepgram,elevenlabs

# Deepgram + ElevenLabs (no Web Speech)
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

\*Auth is required only when `AUTH_ENABLED=true` (default).

### WebSocket Endpoints

| Path | Input | Description |
|---|---|---|
| `/ws` | JSON text messages | Translates text (Web Speech + ElevenLabs browser mode). |
| `/ws/deepgram` | Binary PCM audio | Streams audio to Deepgram for STT + translation. |
| `/ws/elevenlabs` | Binary PCM audio | Streams audio to ElevenLabs for STT + translation. |

All WebSocket endpoints require a valid auth cookie and matching origin header (when `AUTH_ENABLED=true`).

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

All subsequent messages are raw binary PCM audio (16-bit, 16 kHz, mono).

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

### Behind a Reverse Proxy

When running behind nginx, Caddy, or similar:

1. Set `AUTH_COOKIE_SECURE=true` if the proxy terminates TLS.
2. Set `ALLOWED_ORIGINS=https://your-domain.com` to restrict WebSocket origins.
3. Ensure the proxy forwards `Host`, `Origin`, and `X-Forwarded-For` headers.
4. Enable WebSocket proxying for `/ws`, `/ws/deepgram`, and `/ws/elevenlabs`.

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
- **WebSocket origin check** -- Validates `Origin` header against `Host` or `ALLOWED_ORIGINS`.
- **Safe redirects** -- `sanitize_next_path` prevents open redirects after login.
- **No secrets in logs** -- Passwords and API keys are never logged.

For vulnerability reporting, please see [SECURITY.md](SECURITY.md).

## Roadmap

The following improvements are planned or under consideration. Contributions welcome!

- [ ] **Add CI pipeline** -- GitHub Actions workflow for linting, testing, and Docker build
- [ ] **Internationalize the UI** -- currently Czech labels are hardcoded in templates
- [ ] **Session recording/export** -- save transcriptions and translations to a downloadable file

### Optional Future Improvements

These items would improve the project but are not blocking. They make great first contributions:

- [ ] **Pin dependency versions** in `requirements.txt` -- currently unpinned, which can cause breakage on fresh installs when upstream packages release breaking changes
- [ ] **Expand test coverage** -- add tests for:
  - ElevenLabs WebSocket happy-path
  - `/api/translate/languages` endpoint
  - `sanitize_next_path` edge cases
  - WebSocket config message handling
  - `MAX_TEXT_LENGTH` enforcement
  - Auth token expiry
- [ ] **Extract duplicated AudioWorklet PCM processor** code into a shared JavaScript constant -- the same processor is currently inlined in three places (Deepgram, ElevenLabs server mode, ElevenLabs browser mode)
- [ ] **Consider `google-cloud-translate` or `deepl` for production translation** -- the current `googletrans` library uses an unofficial API that can be slow (1--3 s per call) and occasionally breaks; a paid translation API would be more reliable for production deployments

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Support

See [SUPPORT.md](SUPPORT.md).

## License

This project is licensed under the [MIT License](LICENSE).

---

Made with care by [Rhiz3K](https://github.com/Rhiz3K)
