import asyncio
import base64
from collections import deque
import contextlib
from dataclasses import dataclass
import hashlib
import hmac
import inspect
import json
import logging
import os
import secrets
import threading
import time
from typing import TypedDict
from urllib.parse import urlencode, urlparse

import websockets as ws_lib
from anyio import EndOfStream
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from googletrans import Translator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.websockets import WebSocketDisconnect

logging.basicConfig(level=logging.INFO)

# deepgram-sdk has had breaking API changes across major versions. Treat it as an
# optional dependency so the app can still boot (at least for the Web Speech API
# mode) when Deepgram is not installed or an import path changes.
try:  # pragma: no cover - depends on installed deepgram-sdk version
    from deepgram import DeepgramClient  # type: ignore
except Exception:  # pragma: no cover
    DeepgramClient = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on installed deepgram-sdk version
    from deepgram.core.events import EventType  # type: ignore
except Exception:  # pragma: no cover
    class EventType:  # type: ignore[no-redef]
        MESSAGE = "message"
        ERROR = "error"
        CLOSE = "close"

try:  # pragma: no cover - deepgram-sdk v3 exported this, newer versions may not
    from deepgram.listen import ListenV1Results  # type: ignore
except Exception:  # pragma: no cover
    ListenV1Results = None  # type: ignore[assignment]

# Načteme .env proměnné (volitelně, pokud máte něco v .env)
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# --- Security middleware: Content-Security-Policy ---
class _CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: StarletteResponse = await call_next(request)
        # Inline scripts/styles are used throughout; connect-src must allow
        # ElevenLabs WS for browser mode.
        # jsDelivr is scoped to the two pinned packages we actually load
        # (Transformers.js and the ONNX Runtime build used by it and the VAD)
        # rather than the whole CDN. The bare-version entry covers the initial
        # import; the trailing-slash entries cover its /+esm and /dist/* sub-paths.
        jsdelivr = (
            "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0 "
            "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0/ "
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250306-ccf8fdd9ea/ "
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/"
        )
        # Azure AI Speech browser SDK (loaded only for the 'azure' engine). The
        # version here must match the one index.html loads (see AZURE_SDK_VERSION).
        azure_sdk = "https://cdn.jsdelivr.net/npm/microsoft-cognitiveservices-speech-sdk@1.50.0/"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' blob: {jsdelivr} {azure_sdk}; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            f"connect-src 'self' wss://api.elevenlabs.io {jsdelivr} "
            "https://huggingface.co https://cdn-lfs.huggingface.co "
            "https://cas-bridge.xethub.hf.co "
            "wss://*.stt.speech.microsoft.com https://*.api.cognitive.microsoft.com; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "microphone=(self), on-device-speech-recognition=(self)"
        )
        # Cross-origin isolation enables SharedArrayBuffer, which lets ONNX Runtime
        # Web run the local Whisper model multi-threaded on the CPU (much faster on
        # multi-core devices). COEP 'credentialless' still allows the cross-origin
        # CDN/Hugging Face fetches (transformers.js, ORT wasm, model weights) since
        # those send CORS headers.
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        # Always revalidate the local Whisper engine/worklet so a browser can't
        # pin a stale (and possibly broken) cached copy across reloads.
        if request.url.path.startswith("/static/whisper/"):
            response.headers["Cache-Control"] = "no-cache"
        # The Nemotron model weights (~1.2 GB) are immutable and must be cached
        # aggressively; the engine code is revalidated like Whisper's.
        elif request.url.path.startswith("/static/nemotron/models/"):
            if 200 <= response.status_code < 400:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/static/nemotron/"):
            response.headers["Cache-Control"] = "no-cache"
        # Parakeet mirrors Nemotron: immutable ~930 MB model weights, revalidated code.
        elif request.url.path.startswith("/static/parakeet/models/"):
            if 200 <= response.status_code < 400:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/static/parakeet/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(_CSPMiddleware)

# --- Helpers for safe env var parsing ---


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 0:
            logging.warning("Env %s=%s is negative, using default %d", name, raw, default)
            return default
        return value
    except ValueError:
        logging.warning("Env %s=%r is not a valid integer, using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
        if value <= 0:
            logging.warning("Env %s=%s is non-positive, using default %s", name, raw, default)
            return default
        return value
    except ValueError:
        logging.warning("Env %s=%r is not a valid number, using default %s", name, raw, default)
        return default


def _normalize_lang_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def _normalize_translate_dests(value: object) -> list[str] | None:
    """Return 1-2 normalized dest language codes, or None if invalid/empty."""

    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if len(out) >= 2:
            break
        if not isinstance(item, str):
            continue
        v = item.strip().lower()
        if v:
            out.append(v)
    if not out:
        return None
    if len(out) == 2 and out[0] == out[1]:
        out[1] = "ru" if out[0] != "ru" else "en"
    return out


# --- Konfigurace (z prostředí) ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
AUTH_SECRET = os.getenv("AUTH_SECRET") or APP_PASSWORD
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "srlt_auth")
AUTH_TOKEN_TTL_SECONDS = _env_int("AUTH_TOKEN_TTL_SECONDS", 43200)

if AUTH_ENABLED and AUTH_SECRET == APP_PASSWORD and APP_PASSWORD:
    logging.warning(
        "AUTH_SECRET is not set — falling back to APP_PASSWORD for token signing. "
        "Set a separate AUTH_SECRET for production."
    )

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_RESULT_QUEUE_SIZE = _env_int("DEEPGRAM_RESULT_QUEUE_SIZE", 100)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

# Azure AI Speech (browser-direct mode): the server only mints a short-lived
# auth token; the browser SpeechSDK does the streaming. Region must match the
# resource (e.g. "westeurope"). Free F0 tier gives ~5 audio hours/month.
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")

# Which STT engines are available to users.  Comma-separated list.
# Valid values: webspeech, whisper, nemotron, deepgram, elevenlabs, azure.  Default: webspeech only.
_ALL_ENGINES = {"webspeech", "whisper", "nemotron", "deepgram", "elevenlabs", "azure", "parakeet"}
_raw_engines = os.getenv("ENABLED_ENGINES", "webspeech").strip()
ENABLED_ENGINES: set[str] = {
    e.strip().lower() for e in _raw_engines.split(",") if e.strip().lower() in _ALL_ENGINES
} or {"webspeech"}

# Warn if an engine is enabled but its API key is missing.
if "deepgram" in ENABLED_ENGINES and not DEEPGRAM_API_KEY:
    logging.warning("Engine 'deepgram' is enabled but DEEPGRAM_API_KEY is not set.")
if "elevenlabs" in ENABLED_ENGINES and not ELEVENLABS_API_KEY:
    logging.warning(
        "Engine 'elevenlabs' is enabled but ELEVENLABS_API_KEY is not set. "
        "Server-side mode will fail; browser mode requires users to provide their own key."
    )
if "azure" in ENABLED_ENGINES and not (AZURE_SPEECH_KEY and AZURE_SPEECH_REGION):
    logging.warning(
        "Engine 'azure' is enabled but AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set. "
        "Token minting will fail until both are configured (or the client supplies its own)."
    )

MAX_TEXT_LENGTH = _env_int("MAX_TEXT_LENGTH", 5000)
TRANSLATE_TIMEOUT_SECONDS = _env_float("TRANSLATE_TIMEOUT_SECONDS", 10.0)
INTERIM_COALESCE_DELAY_SECONDS = 0.02

# --- Simple in-memory rate limiter for /login ---
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 60.0


class TranscriptResult(TypedDict):
    transcript: str
    is_final: bool


class LatestTranscriptQueue:
    """Preserve all committed transcripts and coalesce pending interim updates."""

    def __init__(self, *, max_finals: int = 100) -> None:
        self._finals: deque[TranscriptResult] = deque()
        self._interim: TranscriptResult | None = None
        self._wake = asyncio.Event()
        self._max_finals = max(1, max_finals)

    def put(self, result: TranscriptResult) -> bool:
        if result["is_final"]:
            self._interim = None
            if len(self._finals) >= self._max_finals:
                return False
            self._finals.append(result)
        else:
            self._interim = result
        self._wake.set()
        return True

    async def get(self) -> TranscriptResult:
        while True:
            if self._finals:
                return self._finals.popleft()
            if self._interim is not None:
                result = self._interim
                self._interim = None
                return result
            self._wake.clear()
            await self._wake.wait()

    def empty(self) -> bool:
        return not self._finals and self._interim is None


@dataclass(slots=True)
class TranslationWork:
    text: str
    msg_type: str
    src: str
    dests: list[str]
    typed: bool = True
    client_id: int | None = None
    client_sent_ms: float | None = None
    revision: int = 0


class LatestInterimQueue:
    """Keep every committed item but only the newest pending interim item."""

    def __init__(self, *, max_finals: int = 100) -> None:
        self._finals: deque[TranslationWork] = deque()
        self._interim: TranslationWork | None = None
        self._wake = asyncio.Event()
        self._closed = False
        self._max_finals = max(1, max_finals)
        self._revision = 0

    def put(self, work: TranslationWork) -> bool:
        if self._closed:
            return False
        self._revision += 1
        work.revision = self._revision
        if work.msg_type == "interim":
            self._interim = work
        else:
            # A final supersedes any still-pending hypothesis for the same stream.
            self._interim = None
            if len(self._finals) >= self._max_finals:
                return False
            self._finals.append(work)
        self._wake.set()
        return True

    @property
    def revision(self) -> int:
        return self._revision

    async def get(self) -> TranslationWork | None:
        while True:
            if self._finals:
                return self._finals.popleft()
            if self._interim is not None:
                work = self._interim
                self._interim = None
                return work
            if self._closed:
                return None
            self._wake.clear()
            await self._wake.wait()

    def close(self) -> None:
        self._closed = True
        self._wake.set()


try:
    from deepgram.extensions.types.sockets.listen_v1_control_message import (
        ListenV1ControlMessage,
    )
except Exception:  # pragma: no cover - optional dependency surface varies by deepgram-sdk version
    ListenV1ControlMessage = None  # type: ignore[assignment]


def _looks_like_deepgram_results(obj: object) -> bool:
    # deepgram-sdk has changed public result types across versions.
    # Use duck-typing so we don't hard-depend on a specific class import.
    return hasattr(obj, "channel") and hasattr(obj, "is_final")


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str) -> str:
    if not AUTH_SECRET:
        return ""
    mac = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(mac)


def create_auth_token() -> str:
    now = int(time.time())
    payload = {"iat": now, "exp": now + AUTH_TOKEN_TTL_SECONDS}
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    sig_b64 = _sign(payload_b64)
    return f"{payload_b64}.{sig_b64}"


def verify_auth_token(token: str | None) -> bool:
    if not token or not AUTH_SECRET:
        return False
    parts = token.split(".")
    if len(parts) != 2:
        return False
    payload_b64, sig_b64 = parts
    expected_sig = _sign(payload_b64)
    if not expected_sig or not secrets.compare_digest(expected_sig, sig_b64):
        return False
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return False
    exp = payload.get("exp")
    if not isinstance(exp, int):
        return False
    return exp >= int(time.time())


def sanitize_next_path(next_path: str | None) -> str:
    if not next_path or not next_path.startswith("/") or next_path.startswith("//"):
        return "/"
    return next_path


def is_origin_allowed(origin: str | None, host: str | None) -> bool:
    configured = os.getenv("ALLOWED_ORIGINS", "").strip()
    if configured:
        allowed = {o.strip() for o in configured.split(",") if o.strip()}
        return bool(origin) and origin in allowed

    if not origin or not host:
        return False
    try:
        parsed = urlparse(origin)
    except Exception:
        return False
    return parsed.netloc == host


def _cookie_secure_for_request(request: Request) -> bool:
    configured = os.getenv("AUTH_COOKIE_SECURE")
    if configured is None or configured == "":
        return request.url.scheme == "https"
    return configured.strip().lower() in {"1", "true", "yes", "on"}


def _render_login(request: Request, *, next_path: str, invalid_pwd: bool) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "password_prompt.html",
        {"invalid_pwd": invalid_pwd, "next_path": next_path},
    )


def _new_translator() -> Translator:
    # googletrans 4.x otherwise converts upstream HTTP failures into a dummy
    # "translation" containing the original text.
    try:
        return Translator(raise_exception=True)
    except TypeError:  # pragma: no cover - compatibility with older releases
        return Translator()


async def _close_translator(translator: Translator | None) -> None:
    if translator is None:
        return
    client = getattr(translator, "client", None)
    closer = getattr(client, "aclose", None)
    if callable(closer):
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logging.debug("Translator close failed: %s", exc)


def _validate_translation_result(result):
    # httpx.Response is falsy for 4xx/5xx statuses, so do not use ``a or b``
    # here: that would discard precisely the failed response we need to detect.
    response = getattr(result, "_response", None)
    if response is None:
        response = getattr(result, "response", None)
    status_code = getattr(response, "status_code", 200)
    if isinstance(status_code, int) and status_code >= 400:
        raise RuntimeError(f"Translation upstream returned HTTP {status_code}")
    return result


async def _translate(translator: Translator, text: str, *, src: str, dest: str):
    # googletrans has had both sync and async implementations across versions.
    # Run sync translate in a worker thread to avoid blocking the event loop.
    if inspect.iscoroutinefunction(translator.translate):
        result = await asyncio.wait_for(
            translator.translate(text, src=src, dest=dest),
            timeout=TRANSLATE_TIMEOUT_SECONDS,
        )
    else:
        result = await asyncio.wait_for(
            asyncio.to_thread(lambda: translator.translate(text, src=src, dest=dest)),
            timeout=TRANSLATE_TIMEOUT_SECONDS,
        )
    return _validate_translation_result(result)


def _translation_payload(
    work: TranslationWork,
    translations: dict[str, str],
    *,
    error: str | None = None,
    translate_ms: int = 0,
) -> dict:
    if not work.typed:
        payload: dict = {
            "original": work.text,
            "en": translations.get("en", ""),
            "ru": translations.get("ru", ""),
        }
        if error:
            payload["error"] = error
        return payload

    payload = {
        "type": work.msg_type if work.msg_type in {"interim", "final"} else "final",
        "original": work.text,
        "dests": work.dests,
        "translations": translations,
        "timing": {"translate_ms": translate_ms},
    }
    if work.client_id is not None:
        payload["client_id"] = work.client_id
    if work.client_sent_ms is not None:
        payload["client_sent_ms"] = work.client_sent_ms
    if error:
        payload["error"] = error
    return payload


@contextlib.asynccontextmanager
async def _threaded_exit_stack():
    """Run potentially blocking synchronous context-manager shutdown off-loop."""
    stack = contextlib.ExitStack()
    try:
        yield stack
    finally:
        await asyncio.to_thread(stack.close)


def _deepgram_send_finalize(dg_socket) -> None:
    # deepgram-sdk v3 had send_finalize/send_close_stream with dedicated types.
    # deepgram-sdk v5 uses send_control(ListenV1ControlMessage(type=...)).
    if hasattr(dg_socket, "send_finalize"):
        try:
            from deepgram.listen.v1.types.listen_v1finalize import ListenV1Finalize  # type: ignore

            dg_socket.send_finalize(ListenV1Finalize(type="Finalize"))
            return
        except Exception:
            pass

    if hasattr(dg_socket, "send_control") and ListenV1ControlMessage is not None:
        try:
            dg_socket.send_control(ListenV1ControlMessage(type="Finalize"))
        except Exception:
            pass


def _deepgram_send_close_stream(dg_socket) -> None:
    if hasattr(dg_socket, "send_close_stream"):
        try:
            from deepgram.listen.v1.types.listen_v1close_stream import (  # type: ignore
                ListenV1CloseStream,
            )

            dg_socket.send_close_stream(ListenV1CloseStream(type="CloseStream"))
            return
        except Exception:
            pass

    if hasattr(dg_socket, "send_control") and ListenV1ControlMessage is not None:
        try:
            dg_socket.send_control(ListenV1ControlMessage(type="CloseStream"))
        except Exception:
            pass


@app.get("/health")
async def health():
    """Health check endpoint for Docker HEALTHCHECK and load balancers."""
    return {"status": "ok"}


def _is_same_origin(request: Request) -> bool:
    """Check that Origin or Referer header matches the request host (CSRF mitigation)."""
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlparse(origin)
            return parsed.netloc == request.headers.get("host", "")
        except Exception:
            return False
    referer = request.headers.get("referer")
    if referer:
        try:
            parsed = urlparse(referer)
            return parsed.netloc == request.headers.get("host", "")
        except Exception:
            return False
    # No Origin/Referer — allow (same-site navigation from address bar).
    return True


def _check_login_rate_limit(client_ip: str) -> bool:
    """Return True if the IP is within the allowed rate limit, False if blocked."""
    now = time.time()
    attempts = _LOGIN_ATTEMPTS.get(client_ip, [])
    # Prune old entries.
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS]
    _LOGIN_ATTEMPTS[client_ip] = attempts
    return len(attempts) < _LOGIN_MAX_ATTEMPTS


def _record_login_attempt(client_ip: str) -> None:
    _LOGIN_ATTEMPTS.setdefault(client_ip, []).append(time.time())


def _index_context() -> dict:
    """Template context shared by all routes that render index.html."""
    return {
        "enabled_engines": sorted(ENABLED_ENGINES),
    }


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """
    Vrátí index.html z app/templates (případně heslo).
    """
    if AUTH_ENABLED:
        if not APP_PASSWORD:
            return HTMLResponse("APP_PASSWORD not configured", status_code=500)

        if not verify_auth_token(request.cookies.get(AUTH_COOKIE_NAME)):
            return _render_login(request, next_path=request.url.path, invalid_pwd=False)

    return templates.TemplateResponse(request, "index.html", _index_context())


@app.get("/deepgram", response_class=HTMLResponse)
async def get_deepgram_index(request: Request):
    """Legacy endpoint — redirects to unified UI. Auth is enforced at /."""
    return RedirectResponse(url="/", status_code=303)


def _require_http_auth(request: Request) -> None:
    if not AUTH_ENABLED:
        return
    if not APP_PASSWORD:
        raise HTTPException(status_code=500, detail="server_not_configured")
    if not verify_auth_token(request.cookies.get(AUTH_COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="unauthorized")


def _require_engine_enabled(engine: str) -> None:
    if engine not in ENABLED_ENGINES:
        raise HTTPException(status_code=404, detail="engine_not_enabled")


@app.get("/api/translate/languages")
async def api_translate_languages(request: Request):
    """Return available translation languages (googletrans)."""
    _require_http_auth(request)
    try:
        from googletrans import LANGUAGES  # type: ignore

        languages = [{"code": code, "name": name} for code, name in LANGUAGES.items()]
        languages.sort(key=lambda x: (x["name"], x["code"]))
        return {"languages": languages}
    except Exception:
        return {"languages": []}


ELEVENLABS_TOKEN_URL = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"


@app.post("/api/elevenlabs/token")
async def api_elevenlabs_token(request: Request):
    """Create a single-use ElevenLabs token for browser-side Scribe connections.

    The client may supply its own API key in the JSON body (``api_key``).
    If omitted, the server-side ``ELEVENLABS_API_KEY`` env var is used.
    """
    _require_http_auth(request)
    _require_engine_enabled("elevenlabs")

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    api_key = ""
    if isinstance(body, dict) and isinstance(body.get("api_key"), str):
        api_key = body["api_key"].strip()
    if not api_key:
        api_key = ELEVENLABS_API_KEY

    if not api_key:
        raise HTTPException(status_code=400, detail="No ElevenLabs API key provided")

    import httpx  # lightweight async HTTP client (ships with FastAPI/Starlette)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                ELEVENLABS_TOKEN_URL,
                headers={"xi-api-key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            return {"token": data.get("token", "")}
    except httpx.HTTPStatusError as e:
        detail = f"ElevenLabs API error: {e.response.status_code}"
        try:
            detail = e.response.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create token: {e}")


def _valid_azure_region(region: str) -> bool:
    # Region is interpolated into the token URL host — restrict to the Azure
    # region charset to prevent SSRF/host injection.
    return bool(region) and len(region) <= 40 and all(c.isalnum() or c == "-" for c in region)


@app.post("/api/azure/token")
async def api_azure_token(request: Request):
    """Mint a short-lived Azure AI Speech auth token for browser-side recognition.

    The client may supply its own ``api_key``/``region`` in the JSON body;
    otherwise the server-side ``AZURE_SPEECH_KEY`` / ``AZURE_SPEECH_REGION`` are
    used. Azure's issueToken returns the token as plain text, valid ~10 minutes;
    the browser SpeechSDK consumes it via ``fromAuthorizationToken``.
    """
    _require_http_auth(request)
    _require_engine_enabled("azure")

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    api_key = ""
    region = ""
    if isinstance(body, dict):
        if isinstance(body.get("api_key"), str):
            api_key = body["api_key"].strip()
        if isinstance(body.get("region"), str):
            region = body["region"].strip()
    api_key = api_key or AZURE_SPEECH_KEY
    region = region or AZURE_SPEECH_REGION

    if not api_key or not region:
        raise HTTPException(status_code=400, detail="Azure Speech key/region not configured")
    if not _valid_azure_region(region):
        raise HTTPException(status_code=400, detail="Invalid Azure region")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
                headers={"Ocp-Apim-Subscription-Key": api_key, "Content-Length": "0"},
            )
            resp.raise_for_status()
            return {"token": resp.text, "region": region}
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Azure token error: {e.response.status_code}",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create Azure token: {e}")


@app.post("/login")
async def login(
    request: Request,
    password: str = Form(...),
    next_path: str = Form("/", alias="next"),
):
    if not AUTH_ENABLED:
        return RedirectResponse(url="/", status_code=303)
    if not APP_PASSWORD:
        return HTMLResponse("APP_PASSWORD not configured", status_code=500)

    # CSRF mitigation: verify that the request Origin/Referer matches our host.
    if not _is_same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-origin login not allowed")

    # Rate limiting.
    client_ip = request.client.host if request.client else "0.0.0.0"
    if not _check_login_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {int(_LOGIN_WINDOW_SECONDS)}s.",
        )

    next_path = sanitize_next_path(next_path)
    if not secrets.compare_digest(password, APP_PASSWORD):
        _record_login_attempt(client_ip)
        return _render_login(request, next_path=next_path, invalid_pwd=True)

    resp = RedirectResponse(url=next_path, status_code=303)
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        create_auth_token(),
        max_age=AUTH_TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure_for_request(request),
        path="/",
    )
    return resp


async def _require_ws_auth(websocket: WebSocket) -> bool:
    """Check WS auth. Returns True if allowed, False if closed with error."""
    if AUTH_ENABLED and not APP_PASSWORD:
        await websocket.close(code=1011, reason="Server not configured")
        return False
    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return False
    if AUTH_ENABLED and not verify_auth_token(websocket.cookies.get(AUTH_COOKIE_NAME)):
        await websocket.close(code=1008, reason="Unauthorized")
        return False
    return True


async def _require_ws_engine(websocket: WebSocket, engine: str) -> bool:
    if engine in ENABLED_ENGINES:
        return True
    await websocket.close(code=1008, reason="Engine not enabled")
    return False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Translate browser STT text while coalescing superseded interim hypotheses."""
    if not await _require_ws_auth(websocket):
        return

    await websocket.accept()
    logging.info("WebSocket /ws připojen")

    translator = _new_translator()
    session_src_lang = "cs"
    session_dest_langs: list[str] = ["en", "ru"]
    inbox = LatestInterimQueue()
    send_lock = asyncio.Lock()
    current_interim_task: asyncio.Future | None = None
    superseded_interim_tasks: set[asyncio.Future] = set()

    async def _send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def _reader() -> None:
        nonlocal session_src_lang, session_dest_langs, current_interim_task
        try:
            while True:
                raw = await websocket.receive_text()
                if not raw:
                    continue

                typed = False
                msg_type = "final"
                src_lang = session_src_lang
                dest_langs = list(session_dest_langs)
                text = raw
                client_id: int | None = None
                client_sent_ms: float | None = None

                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None

                if isinstance(parsed, dict):
                    if parsed.get("type") == "config":
                        tr_cfg = parsed.get("translate")
                        if isinstance(tr_cfg, dict):
                            src_norm = _normalize_lang_code(tr_cfg.get("src"))
                            if src_norm:
                                session_src_lang = src_norm
                            dests_norm = _normalize_translate_dests(tr_cfg.get("dests"))
                            if dests_norm:
                                session_dest_langs = dests_norm
                        continue
                    if parsed.get("type") == "ping":
                        await _send({"type": "pong"})
                        continue
                    if isinstance(parsed.get("text"), str):
                        typed = True
                        msg_type = parsed.get("type") if parsed.get("type") in {"interim", "final"} else "final"
                        text = parsed["text"]
                        src_lang = _normalize_lang_code(parsed.get("src")) or src_lang
                        dest_langs = _normalize_translate_dests(parsed.get("dests")) or dest_langs
                        cid = parsed.get("client_id")
                        if isinstance(cid, int) and cid >= 0:
                            client_id = cid
                        sent_ms = parsed.get("client_sent_ms")
                        if isinstance(sent_ms, (int, float)):
                            client_sent_ms = float(sent_ms)

                text = text.strip()
                work = TranslationWork(
                    text=text,
                    msg_type=msg_type,
                    src=src_lang,
                    dests=dest_langs,
                    typed=typed,
                    client_id=client_id,
                    client_sent_ms=client_sent_ms,
                )

                if not text or len(text) > MAX_TEXT_LENGTH:
                    empty_work = TranslationWork(
                        text="",
                        msg_type=msg_type,
                        src=src_lang,
                        dests=dest_langs,
                        typed=typed,
                        client_id=client_id,
                        client_sent_ms=client_sent_ms,
                    )
                    await _send(
                        _translation_payload(
                            empty_work,
                            {dest: "" for dest in dest_langs},
                            error="text_too_long" if len(text) > MAX_TEXT_LENGTH else None,
                        )
                    )
                    continue

                # A newer hypothesis or a committed final makes in-flight interim
                # work useless. Cancellation also lets a final bypass a slow request.
                if current_interim_task is not None and not current_interim_task.done():
                    superseded_interim_tasks.add(current_interim_task)
                    current_interim_task.cancel()
                if not inbox.put(work):
                    await _send({"error": "translation_queue_full"})
                    await websocket.close(code=1013, reason="Translation queue full")
                    return
        except (WebSocketDisconnect, EndOfStream):
            logging.info("WebSocket odpojen klientem.")
        finally:
            inbox.close()

    async def _worker() -> None:
        nonlocal translator, current_interim_task
        while True:
            work = await inbox.get()
            if work is None:
                return
            if work.msg_type == "interim":
                # Give the reader one event-loop turn to consume a burst's newer
                # hypothesis/final before starting network work for this interim.
                await asyncio.sleep(INTERIM_COALESCE_DELAY_SECONDS)
                if work.revision != inbox.revision:
                    continue
            started = time.perf_counter()
            task = asyncio.gather(
                *[_translate(translator, work.text, src=work.src, dest=dest) for dest in work.dests]
            )
            if work.msg_type == "interim":
                current_interim_task = task
            try:
                # Shield lets us distinguish an explicitly superseded child
                # translation from cancellation of this worker itself. The
                # reader marks the former before cancelling it.
                results = await asyncio.shield(task)
            except asyncio.CancelledError:
                if task in superseded_interim_tasks:
                    continue
                task.cancel()
                raise
            except Exception as exc:
                logging.error("Překlad selhal: %s", exc)
                old_translator = translator
                translator = _new_translator()
                await _close_translator(old_translator)
                response = _translation_payload(
                    work,
                    {dest: "" for dest in work.dests},
                    error="translation_failed",
                )
            else:
                translate_ms = int((time.perf_counter() - started) * 1000)
                response = _translation_payload(
                    work,
                    {dest: (result.text if result else "") for dest, result in zip(work.dests, results)},
                    translate_ms=translate_ms,
                )
            finally:
                superseded_interim_tasks.discard(task)
                if current_interim_task is task:
                    current_interim_task = None
            if work.msg_type == "interim" and work.revision != inbox.revision:
                continue
            await _send(response)

    reader_task = asyncio.create_task(_reader())
    worker_task = asyncio.create_task(_worker())
    try:
        done, pending = await asyncio.wait(
            {reader_task, worker_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        failure: Exception | None = None
        for task in done:
            try:
                task.result()
            except (WebSocketDisconnect, EndOfStream):
                pass
            except Exception as exc:
                failure = exc
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if failure is not None:
            raise failure
    except Exception as exc:
        logging.error("Nastala chyba v /ws: %s", exc)
        try:
            await _send({"error": "server_error"})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        inbox.close()
        if current_interim_task is not None:
            current_interim_task.cancel()
        await _close_translator(translator)


@app.websocket("/ws/deepgram")
async def deepgram_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pro Deepgram Nova-3 RSTT.
    Přijímá audio data z prohlížeče, posílá je do Deepgram, 
    vrací přepis a překlad.
    """
    if not await _require_ws_auth(websocket):
        return
    if not await _require_ws_engine(websocket, "deepgram"):
        return

    await websocket.accept()
    logging.info("WebSocket /ws/deepgram připojen")

    browser_send_lock = asyncio.Lock()

    async def _send_browser(payload: dict) -> None:
        async with browser_send_lock:
            await websocket.send_json(payload)
    
    if not DEEPGRAM_API_KEY:
        logging.error("DEEPGRAM_API_KEY není nastaven")
        await _send_browser({"error": "DEEPGRAM_API_KEY not configured"})
        await websocket.close()
        return

    if DeepgramClient is None:
        logging.error("deepgram-sdk není nainstalovaný nebo nejde importovat")
        await _send_browser({"error": "deepgram-sdk not installed"})
        await websocket.close()
        return
    
    stop_event = threading.Event()
    async_stop_event = asyncio.Event()
    process_task = None
    listen_thread: threading.Thread | None = None
    
    # Capture event loop for use in callbacks from other threads
    event_loop = asyncio.get_running_loop()

    # Defaults for Deepgram connect.
    # Model is fixed to match legacy behavior.
    dg_language = "cs"
    dg_interim_results = True
    dg_punctuate = True

    # Defaults for translation.
    translate_src = "cs"
    translate_dests: list[str] = ["en", "ru"]
    translate_interim = False

    first_audio: bytes | None = None

    # Optional session config as the first websocket text message.
    # If the client sends audio first, we keep it and proceed with defaults.
    try:
        first = await websocket.receive()
        if first.get("type") == "websocket.receive":
            if first.get("text"):
                try:
                    cfg = json.loads(first["text"])
                except Exception:
                    cfg = None

                if isinstance(cfg, dict) and cfg.get("type") == "config":
                    dg_cfg = cfg.get("deepgram")
                    if isinstance(dg_cfg, dict):
                        language = dg_cfg.get("language")
                        if isinstance(language, str) and language.strip():
                            dg_language = language.strip()
                        if isinstance(dg_cfg.get("interim_results"), bool):
                            dg_interim_results = dg_cfg["interim_results"]
                        if isinstance(dg_cfg.get("punctuate"), bool):
                            dg_punctuate = dg_cfg["punctuate"]

                    tr_cfg = cfg.get("translate")
                    if isinstance(tr_cfg, dict):
                        src_norm = _normalize_lang_code(tr_cfg.get("src"))
                        if src_norm:
                            translate_src = src_norm

                        dests_norm = _normalize_translate_dests(tr_cfg.get("dests"))
                        if dests_norm:
                            translate_dests = dests_norm

                    if isinstance(cfg.get("translate_interim"), bool):
                        translate_interim = cfg["translate_interim"]
            elif first.get("bytes"):
                first_audio = first["bytes"]
        elif first.get("type") == "websocket.disconnect":
            return
    except (WebSocketDisconnect, EndOfStream):
        return

    if len(translate_dests) == 2 and translate_dests[0] == translate_dests[1]:
        translate_dests[1] = "ru" if translate_dests[0] != "ru" else "en"
    translator = _new_translator()

    def _dg_payload(
        *,
        msg_type: str,
        original: str,
        translations: dict[str, str],
        error: str | None = None,
        timing: dict[str, int] | None = None,
    ) -> dict:
        payload: dict = {
            "type": msg_type,
            "original": original,
            "dests": translate_dests,
            "translations": translations,
        }
        # Backwards-compatible top-level fields.
        if "en" in translations:
            payload["en"] = translations["en"]
        if "ru" in translations:
            payload["ru"] = translations["ru"]
        if timing:
            payload["timing"] = timing
        if error:
            payload["error"] = error
        return payload

    try:
        # Inicializace Deepgram klienta
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        async with _threaded_exit_stack() as stack:
            # Vytvoření živého připojení s Nova-3 modelem
            connect_kwargs: dict[str, str] = {
                "model": "nova-3",
                "language": dg_language,
                "encoding": "linear16",
                "sample_rate": "16000",
                "channels": "1",
                "interim_results": "true" if dg_interim_results else "false",
                "punctuate": "true" if dg_punctuate else "false",
            }
            connect_obj = deepgram.listen.v1.connect(**connect_kwargs)
            # deepgram-sdk v5 returns a context manager, v3 returned an iterator.
            if hasattr(connect_obj, "__enter__"):
                dg_socket = await asyncio.to_thread(stack.enter_context, connect_obj)
            else:
                dg_socket_iterator = connect_obj
                dg_socket = await asyncio.to_thread(next, dg_socket_iterator)
                stack.callback(getattr(dg_socket_iterator, "close", lambda: None))

            logging.info("Deepgram Nova-3 připojení úspěšně spuštěno")
            
            # Queue pro předávání výsledků mezi vlákny
            result_queue = LatestTranscriptQueue(max_finals=DEEPGRAM_RESULT_QUEUE_SIZE)
            queue_overflowed = False

            # Grace window to drain final results after shutdown.
            shutdown_deadline: float | None = None
            
            # Callback pro příjem transkripce z Deepgram
            def on_message(result):
                try:
                    if _looks_like_deepgram_results(result):
                        # Check if alternatives exist and are non-empty
                        if (result.channel and 
                            result.channel.alternatives and 
                            len(result.channel.alternatives) > 0):
                            transcript = result.channel.alternatives[0].transcript
                            is_final = result.is_final
                            
                            if transcript and transcript.strip():
                                logging.debug(
                                    "Deepgram transcript received (chars=%d, final=%s)",
                                    len(transcript),
                                    is_final,
                                )
                                payload: TranscriptResult = {
                                    "transcript": transcript,
                                    "is_final": bool(is_final),
                                }

                                def _enqueue() -> None:
                                    nonlocal queue_overflowed
                                    # During shutdown we still want to enqueue final results produced
                                    # by Deepgram finalize/close, but we can drop interim updates.
                                    if stop_event.is_set() and not payload["is_final"]:
                                        return
                                    if queue_overflowed:
                                        return
                                    if result_queue.put(payload):
                                        return
                                    # Never evict a committed transcript. Stop the
                                    # overloaded session explicitly instead of
                                    # growing without bound or silently losing text.
                                    queue_overflowed = True
                                    logging.error("Deepgram final transcript queue is full")
                                    stop_event.set()
                                    async_stop_event.set()
                                    asyncio.create_task(
                                        _send_browser({"error": "transcript_queue_full"})
                                    )

                                event_loop.call_soon_threadsafe(_enqueue)
                except Exception as e:
                    logging.error(f"Chyba při zpracování transkripce: {str(e)}")
            
            def on_error(error):
                logging.error(f"Deepgram error: {error}")
                stop_event.set()

                def _notify() -> None:
                    async_stop_event.set()
                    async def _send() -> None:
                        try:
                            await _send_browser({"error": str(error)})
                        except Exception as send_err:
                            logging.debug(f"Nelze poslat Deepgram error: {send_err}")

                    asyncio.create_task(_send())

                event_loop.call_soon_threadsafe(_notify)
            
            def on_close(close):
                logging.info("Deepgram připojení uzavřeno")
                stop_event.set()
                event_loop.call_soon_threadsafe(async_stop_event.set)
            
            # Registrace callbacků
            dg_socket.on(EventType.MESSAGE, on_message)
            dg_socket.on(EventType.ERROR, on_error)
            dg_socket.on(EventType.CLOSE, on_close)
            
            # Spustit poslouchání v samostatném vlákně
            def listen_thread_func():
                try:
                    dg_socket.start_listening()
                except Exception as e:
                    logging.error(f"Listen thread error: {e}")
                    stop_event.set()
                    event_loop.call_soon_threadsafe(async_stop_event.set)
            
            listen_thread = threading.Thread(target=listen_thread_func, daemon=True)
            listen_thread.start()

            async def refresh_translator() -> None:
                nonlocal translator
                old_translator = translator
                translator = _new_translator()
                await _close_translator(old_translator)
            
            # Coroutine pro zpracování výsledků
            async def process_results():
                while True:
                    if stop_event.is_set() and result_queue.empty():
                        # Prefer to drain results until the listen thread ends, but don't hang forever.
                        if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                            break
                        if listen_thread is None or not listen_thread.is_alive():
                            break
                    try:
                        result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                        transcript = result["transcript"]
                        is_final = result["is_final"]
                        
                        if is_final:
                            try:
                                start_t = time.perf_counter()
                                results = await asyncio.gather(
                                    *[
                                        _translate(
                                            translator,
                                            transcript,
                                            src=translate_src,
                                            dest=dest,
                                        )
                                        for dest in translate_dests
                                    ]
                                )
                                translate_ms = int((time.perf_counter() - start_t) * 1000)
                                translations = {
                                    dest: (res.text if res else "")
                                    for dest, res in zip(translate_dests, results)
                                }
                                response = _dg_payload(
                                    msg_type="final",
                                    original=transcript,
                                    translations=translations,
                                    timing={"translate_ms": translate_ms},
                                )
                            except Exception as translate_err:
                                logging.error(f"Chyba při překladu: {translate_err}")
                                await refresh_translator()
                                response = _dg_payload(
                                    msg_type="final",
                                    original=transcript,
                                    translations={d: "" for d in translate_dests},
                                    error="translation_failed",
                                    timing={"translate_ms": 0},
                                )
                        else:
                            if translate_interim:
                                try:
                                    start_t = time.perf_counter()
                                    results = await asyncio.gather(
                                        *[
                                            _translate(
                                                translator,
                                                transcript,
                                                src=translate_src,
                                                dest=dest,
                                            )
                                            for dest in translate_dests
                                        ]
                                    )
                                    translate_ms = int((time.perf_counter() - start_t) * 1000)
                                    translations = {
                                        dest: (res.text if res else "")
                                        for dest, res in zip(translate_dests, results)
                                    }
                                    response = _dg_payload(
                                        msg_type="interim",
                                        original=transcript,
                                        translations=translations,
                                        timing={"translate_ms": translate_ms},
                                    )
                                except Exception as translate_err:
                                    logging.error(f"Chyba při překladu interim: {translate_err}")
                                    await refresh_translator()
                                    response = _dg_payload(
                                        msg_type="interim",
                                        original=transcript,
                                        translations={d: "" for d in translate_dests},
                                        error="translation_failed",
                                        timing={"translate_ms": 0},
                                    )
                            else:
                                response = _dg_payload(
                                    msg_type="interim",
                                    original=transcript,
                                    translations={d: "" for d in translate_dests},
                                    timing={"translate_ms": 0},
                                )
                         
                        await _send_browser(response)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        if not stop_event.is_set():
                            logging.error(f"Chyba při zpracování výsledku: {str(e)}")
            
            # Spustit task pro zpracování výsledků
            process_task = asyncio.create_task(process_results())
            
            # Přijímání audio dat z prohlížeče
            try:
                if first_audio:
                    await asyncio.to_thread(dg_socket.send_media, first_audio)
                while not stop_event.is_set():
                    receive_task = asyncio.create_task(websocket.receive())
                    upstream_stop_task = asyncio.create_task(async_stop_event.wait())
                    done, pending = await asyncio.wait(
                        {receive_task, upstream_stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for pending_task in pending:
                        pending_task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if upstream_stop_task in done:
                        break
                    msg = receive_task.result()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("type") != "websocket.receive":
                        continue
                    data = msg.get("bytes")
                    if data:
                        await asyncio.to_thread(dg_socket.send_media, data)
            except (WebSocketDisconnect, EndOfStream):
                logging.info("Klient odpojen")
            finally:
                stop_event.set()
                async_stop_event.set()
                try:
                    await asyncio.to_thread(_deepgram_send_finalize, dg_socket)
                    await asyncio.to_thread(_deepgram_send_close_stream, dg_socket)
                except Exception as e:
                    logging.warning(f"Deepgram close selhal: {e}")
                if listen_thread is not None:
                    await asyncio.to_thread(listen_thread.join, 1.0)

                # Let the processor drain queued results after finalize.
                shutdown_deadline = time.monotonic() + 1.5
                if process_task:
                    try:
                        await asyncio.wait_for(process_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        process_task.cancel()
                        try:
                            await process_task
                        except asyncio.CancelledError:
                            pass
    
    except (WebSocketDisconnect, EndOfStream):
        logging.info("Deepgram WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"Deepgram chyba: {str(e)}")
        try:
            await _send_browser({"error": str(e)})
        except Exception as send_err:
            logging.debug(f"Nelze poslat Deepgram error: {send_err}")
    finally:
        await _close_translator(translator)


@app.websocket("/ws/elevenlabs")
async def elevenlabs_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ElevenLabs Scribe v2 Realtime STT.
    Receives PCM audio from the browser, proxies it to the ElevenLabs
    realtime WS, translates transcripts and sends them back.
    """
    if not await _require_ws_auth(websocket):
        return
    if not await _require_ws_engine(websocket, "elevenlabs"):
        return

    await websocket.accept()
    logging.info("WebSocket /ws/elevenlabs připojen")

    if not ELEVENLABS_API_KEY:
        logging.error("ELEVENLABS_API_KEY není nastaven")
        await websocket.send_json({"error": "ELEVENLABS_API_KEY not configured"})
        await websocket.close()
        return

    # Session defaults.
    translate_src = "cs"
    translate_dests: list[str] = ["en", "ru"]
    translate_interim = True
    el_language_code = ""
    el_commit_strategy = "vad"

    # Read optional config message (first message may be JSON config or audio).
    first_audio: bytes | None = None
    try:
        first = await websocket.receive()
        if first.get("type") == "websocket.receive":
            if first.get("text"):
                try:
                    cfg = json.loads(first["text"])
                except Exception:
                    cfg = None

                if isinstance(cfg, dict) and cfg.get("type") == "config":
                    el_cfg = cfg.get("elevenlabs")
                    if isinstance(el_cfg, dict):
                        lang = el_cfg.get("language_code")
                        if isinstance(lang, str) and lang.strip():
                            el_language_code = lang.strip()
                        strategy = el_cfg.get("commit_strategy")
                        if isinstance(strategy, str) and strategy in {"vad", "manual"}:
                            el_commit_strategy = strategy

                    tr_cfg = cfg.get("translate")
                    if isinstance(tr_cfg, dict):
                        src_norm = _normalize_lang_code(tr_cfg.get("src"))
                        if src_norm:
                            translate_src = src_norm

                        dests_norm = _normalize_translate_dests(tr_cfg.get("dests"))
                        if dests_norm:
                            translate_dests = dests_norm

                    if isinstance(cfg.get("translate_interim"), bool):
                        translate_interim = cfg["translate_interim"]
            elif first.get("bytes"):
                first_audio = first["bytes"]
        elif first.get("type") == "websocket.disconnect":
            return
    except (WebSocketDisconnect, EndOfStream):
        return

    if len(translate_dests) == 2 and translate_dests[0] == translate_dests[1]:
        translate_dests[1] = "ru" if translate_dests[0] != "ru" else "en"
    translator = _new_translator()

    # Build ElevenLabs WS URL with encoded query parameters.
    el_params = {
        "model_id": "scribe_v2_realtime",
        "audio_format": "pcm_16000",
        "sample_rate": "16000",
        "commit_strategy": el_commit_strategy,
    }
    if el_language_code:
        el_params["language_code"] = el_language_code
    if el_commit_strategy == "vad":
        el_params["vad_silence_threshold_secs"] = "1.5"
    el_ws_url = f"{ELEVENLABS_WS_URL}?{urlencode(el_params)}"

    def _el_payload(
        *,
        msg_type: str,
        original: str,
        translations: dict[str, str],
        error: str | None = None,
        timing: dict[str, int] | None = None,
    ) -> dict:
        payload: dict = {
            "type": msg_type,
            "original": original,
            "dests": translate_dests,
            "translations": translations,
        }
        if timing:
            payload["timing"] = timing
        if error:
            payload["error"] = error
        return payload

    el_ws = None
    stop_event = asyncio.Event()
    transcript_inbox = LatestInterimQueue()
    browser_send_lock = asyncio.Lock()
    current_el_interim_task: asyncio.Future | None = None
    superseded_el_interim_tasks: set[asyncio.Future] = set()

    async def _send_browser(payload: dict) -> None:
        async with browser_send_lock:
            await websocket.send_json(payload)

    try:
        el_ws = await ws_lib.connect(
            el_ws_url,
            additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
        )
        logging.info("ElevenLabs Scribe WS připojeno")

        # Wait for session_started before forwarding audio.
        session_msg_raw = await asyncio.wait_for(el_ws.recv(), timeout=10)
        session_msg = json.loads(session_msg_raw)
        logging.info(f"ElevenLabs session started: {session_msg.get('session_id', '')}")

        async def _forward_audio():
            """Read PCM audio from browser WS and forward to ElevenLabs as base64."""
            try:
                if first_audio:
                    audio_b64 = base64.b64encode(first_audio).decode("ascii")
                    await el_ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": audio_b64,
                        "commit": False,
                        "sample_rate": 16000,
                    }))

                while not stop_event.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("type") != "websocket.receive":
                        continue

                    data = msg.get("bytes")
                    if data:
                        audio_b64 = base64.b64encode(data).decode("ascii")
                        await el_ws.send(json.dumps({
                            "message_type": "input_audio_chunk",
                            "audio_base_64": audio_b64,
                            "commit": False,
                            "sample_rate": 16000,
                        }))
                    elif msg.get("text"):
                        # Client may send JSON commands (e.g. commit).
                        try:
                            cmd = json.loads(msg["text"])
                            if isinstance(cmd, dict) and cmd.get("type") == "commit":
                                await el_ws.send(json.dumps({
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": "",
                                    "commit": True,
                                    "sample_rate": 16000,
                                }))
                            elif isinstance(cmd, dict) and cmd.get("type") == "ping":
                                await _send_browser({"type": "pong"})
                        except Exception:
                            pass
            except (WebSocketDisconnect, EndOfStream):
                logging.info("ElevenLabs: klient odpojen")
            except Exception as e:
                logging.error(f"ElevenLabs forward_audio error: {e}")
            finally:
                stop_event.set()

        async def _receive_transcripts():
            """Read upstream continuously and enqueue translation outside this loop."""
            nonlocal current_el_interim_task
            try:
                async for raw in el_ws:
                    if stop_event.is_set():
                        break
                    try:
                        ev = json.loads(raw)
                    except Exception:
                        continue

                    msg_type = ev.get("message_type", "")

                    if msg_type == "partial_transcript":
                        text = ev.get("text", "").strip()
                        if not text:
                            continue
                        if translate_interim:
                            if current_el_interim_task is not None and not current_el_interim_task.done():
                                superseded_el_interim_tasks.add(current_el_interim_task)
                                current_el_interim_task.cancel()
                            transcript_inbox.put(
                                TranslationWork(
                                    text=text,
                                    msg_type="interim",
                                    src=translate_src,
                                    dests=list(translate_dests),
                                )
                            )
                        else:
                            await _send_browser(
                                _el_payload(
                                    msg_type="interim",
                                    original=text,
                                    translations={d: "" for d in translate_dests},
                                    timing={"translate_ms": 0},
                                )
                            )

                    elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                        text = ev.get("text", "").strip()
                        if not text:
                            continue
                        if current_el_interim_task is not None and not current_el_interim_task.done():
                            superseded_el_interim_tasks.add(current_el_interim_task)
                            current_el_interim_task.cancel()
                        accepted = transcript_inbox.put(
                            TranslationWork(
                                text=text,
                                msg_type="final",
                                src=translate_src,
                                dests=list(translate_dests),
                            )
                        )
                        if not accepted:
                            await _send_browser({"error": "translation_queue_full"})
                            break

                    elif msg_type in (
                        "input_error", "error", "auth_error", "transcriber_error",
                        "quota_exceeded", "throttled", "rate_limited", "queue_overflow",
                        "resource_exhausted", "session_time_limit_exceeded",
                        "chunk_size_exceeded", "insufficient_audio_activity", "unaccepted_terms",
                    ):
                        error_msg = ev.get("error", ev.get("message", str(ev)))
                        logging.error(f"ElevenLabs error: {error_msg}")
                        await _send_browser({"error": f"ElevenLabs: {error_msg}"})
                        break

            except Exception as e:
                if not stop_event.is_set():
                    logging.error(f"ElevenLabs receive_transcripts error: {e}")
            finally:
                transcript_inbox.close()
                stop_event.set()

        async def _translate_transcripts():
            nonlocal translator, current_el_interim_task
            while True:
                work = await transcript_inbox.get()
                if work is None:
                    return
                if work.msg_type == "interim":
                    await asyncio.sleep(INTERIM_COALESCE_DELAY_SECONDS)
                    if work.revision != transcript_inbox.revision:
                        continue
                started = time.perf_counter()
                task = asyncio.gather(
                    *[_translate(translator, work.text, src=work.src, dest=dest) for dest in work.dests]
                )
                if work.msg_type == "interim":
                    current_el_interim_task = task
                try:
                    results = await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task in superseded_el_interim_tasks:
                        continue
                    task.cancel()
                    raise
                except Exception as exc:
                    logging.error("ElevenLabs translation error: %s", exc)
                    old_translator = translator
                    translator = _new_translator()
                    await _close_translator(old_translator)
                    response = _translation_payload(
                        work,
                        {dest: "" for dest in work.dests},
                        error="translation_failed",
                    )
                else:
                    response = _translation_payload(
                        work,
                        {dest: (result.text if result else "") for dest, result in zip(work.dests, results)},
                        translate_ms=int((time.perf_counter() - started) * 1000),
                    )
                finally:
                    superseded_el_interim_tasks.discard(task)
                    if current_el_interim_task is task:
                        current_el_interim_task = None
                if work.msg_type == "interim" and work.revision != transcript_inbox.revision:
                    continue
                await _send_browser(response)

        # Keep reading upstream while translation runs independently. If upstream
        # closes, drain already-queued finals before tearing down the browser socket.
        forward_task = asyncio.create_task(_forward_audio())
        receive_task = asyncio.create_task(_receive_transcripts())
        translate_task = asyncio.create_task(_translate_transcripts())

        done, pending_io = await asyncio.wait(
            {forward_task, receive_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        upstream_finished = receive_task in done and forward_task not in done
        stop_event.set()
        for task in pending_io:
            task.cancel()
        await asyncio.gather(*pending_io, return_exceptions=True)

        if upstream_finished:
            try:
                await asyncio.wait_for(
                    translate_task,
                    timeout=TRANSLATE_TIMEOUT_SECONDS + 1,
                )
            except asyncio.TimeoutError:
                translate_task.cancel()
        else:
            translate_task.cancel()
        await asyncio.gather(translate_task, return_exceptions=True)

    except (WebSocketDisconnect, EndOfStream):
        logging.info("ElevenLabs WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"ElevenLabs chyba: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception as send_err:
            logging.debug(f"Nelze poslat ElevenLabs error: {send_err}")
    finally:
        transcript_inbox.close()
        if current_el_interim_task is not None:
            current_el_interim_task.cancel()
        await _close_translator(translator)
        if el_ws:
            try:
                await el_ws.close()
            except Exception:
                pass
