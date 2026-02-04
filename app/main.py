import asyncio
import contextlib
import inspect
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import TypedDict
from urllib.parse import urlparse

import websockets as ws_lib

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from googletrans import Translator
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


# --- Security middleware: Content-Security-Policy ---
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class _CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: StarletteResponse = await call_next(request)
        # Inline scripts/styles are used throughout; connect-src must allow
        # ElevenLabs WS for browser mode.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' blob:; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' wss://api.elevenlabs.io; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
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

# Which STT engines are available to users.  Comma-separated list.
# Valid values: webspeech, deepgram, elevenlabs.  Default: webspeech only.
_ALL_ENGINES = {"webspeech", "deepgram", "elevenlabs"}
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

MAX_TEXT_LENGTH = _env_int("MAX_TEXT_LENGTH", 5000)
TRANSLATE_TIMEOUT_SECONDS = _env_float("TRANSLATE_TIMEOUT_SECONDS", 10.0)

# --- Simple in-memory rate limiter for /login ---
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 60.0


class TranscriptResult(TypedDict):
    transcript: str
    is_final: bool


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


async def _translate(translator: Translator, text: str, *, src: str, dest: str):
    # googletrans has had both sync and async implementations across versions.
    # Run sync translate in a worker thread to avoid blocking the event loop.
    if inspect.iscoroutinefunction(translator.translate):
        return await asyncio.wait_for(
            translator.translate(text, src=src, dest=dest),
            timeout=TRANSLATE_TIMEOUT_SECONDS,
        )
    return await asyncio.wait_for(
        asyncio.to_thread(lambda: translator.translate(text, src=src, dest=dest)),
        timeout=TRANSLATE_TIMEOUT_SECONDS,
    )


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

        # Legacy podpora: ?pwd=... (DEPRECATED — password is exposed in URL/logs)
        legacy_pwd = request.query_params.get("pwd")
        if legacy_pwd is not None:
            logging.warning(
                "Deprecated ?pwd= query auth used from %s — migrate to the login form",
                request.client.host if request.client else "unknown",
            )
            if secrets.compare_digest(legacy_pwd, APP_PASSWORD):
                resp = RedirectResponse(url=request.url.path, status_code=303)
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
            return _render_login(request, next_path=request.url.path, invalid_pwd=True)

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
    if not AUTH_ENABLED:
        return True
    if not APP_PASSWORD:
        await websocket.close(code=1011, reason="Server not configured")
        return False
    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return False
    if not verify_auth_token(websocket.cookies.get(AUTH_COOKIE_NAME)):
        await websocket.close(code=1008, reason="Unauthorized")
        return False
    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Přijímá text (z prohlížeče) -> překládá do EN a RU pomocí googletrans -> posílá zpět JSON.
    """
    if not await _require_ws_auth(websocket):
        return

    await websocket.accept()
    logging.info("WebSocket /ws připojen")

    translator = Translator()

    session_src_lang = "cs"
    session_dest_langs: list[str] = ["en", "ru"]

    # --- Interim dedup: version counter so we can skip stale interims ---
    # Each incoming message increments the counter. Before starting a slow
    # translation for an interim message, we check whether a newer message
    # has already arrived — if so, the interim is stale and we skip it.
    _msg_version = 0

    try:
        while True:
            # Čekáme na text z frontendu
            raw = await websocket.receive_text()
            if not raw:
                continue

            _msg_version += 1
            my_version = _msg_version

            wants_typed_response = False
            msg_type: str | None = None
            src_lang = session_src_lang
            dest_langs: list[str] = list(session_dest_langs)
            text = raw
            client_id: int | None = None
            client_sent_ms: float | None = None
            try:
                parsed = json.loads(raw)
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
                        await websocket.send_json({"type": "pong"})
                        continue

                    if isinstance(parsed.get("text"), str):
                        wants_typed_response = True
                        msg_type = parsed.get("type")
                        text = parsed["text"]

                        cid = parsed.get("client_id")
                        if isinstance(cid, int) and cid >= 0:
                            client_id = cid
                        cts = parsed.get("client_sent_ms")
                        if isinstance(cts, (int, float)):
                            client_sent_ms = float(cts)

                        src_val = parsed.get("src")
                        src_norm = _normalize_lang_code(src_val)
                        if src_norm:
                            src_lang = src_norm

                        dests_norm = _normalize_translate_dests(parsed.get("dests"))
                        if dests_norm:
                            dest_langs = dests_norm
            except Exception:
                # Legacy klient posílá prostý text.
                pass

            text = text.strip()

            if len(dest_langs) == 2 and dest_langs[0] == dest_langs[1]:
                dest_langs[1] = "ru" if dest_langs[0] != "ru" else "en"

            def _legacy_payload(*, original: str, en: str, ru: str, error: str | None = None) -> dict:
                payload: dict = {"original": original, "en": en, "ru": ru}
                if error:
                    payload["error"] = error
                return payload

            def _typed_payload(
                *,
                original: str,
                translations: dict[str, str],
                error: str | None = None,
                timing: dict[str, int] | None = None,
            ) -> dict:
                normalized_type = msg_type if msg_type in {"interim", "final"} else "final"
                payload: dict = {
                    "type": normalized_type,
                    "original": original,
                    "dests": dest_langs,
                    "translations": translations,
                }
                if client_id is not None:
                    payload["client_id"] = client_id
                if client_sent_ms is not None:
                    payload["client_sent_ms"] = client_sent_ms
                if timing:
                    payload["timing"] = timing
                if error:
                    payload["error"] = error
                return payload

            if not text:
                if wants_typed_response:
                    await websocket.send_json(
                        _typed_payload(
                            original="",
                            translations={d: "" for d in dest_langs},
                            timing={"translate_ms": 0},
                        )
                    )
                else:
                    await websocket.send_json(_legacy_payload(original="", en="", ru=""))
                continue
            if len(text) > MAX_TEXT_LENGTH:
                if wants_typed_response:
                    await websocket.send_json(
                        _typed_payload(
                            original="",
                            translations={d: "" for d in dest_langs},
                            error="text_too_long",
                            timing={"translate_ms": 0},
                        )
                    )
                else:
                    await websocket.send_json(
                        _legacy_payload(original="", en="", ru="", error="text_too_long")
                    )
                continue

            # Skip stale interim messages: if a newer message has already
            # arrived while we were processing, this interim is outdated.
            # Final messages are always translated (they represent committed text).
            is_interim = msg_type == "interim"
            if is_interim and my_version != _msg_version:
                logging.debug("Skipping stale interim v%d (current v%d)", my_version, _msg_version)
                continue

            try:
                if wants_typed_response:
                    start_t = time.perf_counter()
                    results = await asyncio.gather(
                        *[
                            _translate(translator, text, src=src_lang, dest=dest)
                            for dest in dest_langs
                        ]
                    )
                    translate_ms = int((time.perf_counter() - start_t) * 1000)
                    # Check again after translation — if a new message arrived
                    # during the (slow) translation, discard this stale interim result.
                    if is_interim and my_version != _msg_version:
                        logging.debug("Discarding stale interim translation v%d", my_version)
                        continue
                    response = _typed_payload(
                        original=text,
                        translations={
                            dest: (res.text if res else "")
                            for dest, res in zip(dest_langs, results)
                        },
                        timing={"translate_ms": translate_ms},
                    )
                else:
                    translation_en, translation_ru = await asyncio.gather(
                        _translate(translator, text, src="cs", dest="en"),
                        _translate(translator, text, src="cs", dest="ru"),
                    )
                    response = _legacy_payload(
                        original=text,
                        en=translation_en.text,
                        ru=translation_ru.text,
                    )
            except Exception as e:
                logging.error(f"Překlad selhal: {str(e)}")
                # Recreate translator – the googletrans HTTP session may have gone stale.
                translator = Translator()
                if wants_typed_response:
                    response = _typed_payload(
                        original=text,
                        translations={d: "" for d in dest_langs},
                        error="translation_failed",
                        timing={"translate_ms": 0},
                    )
                else:
                    response = _legacy_payload(
                        original=text,
                        en="",
                        ru="",
                        error="translation_failed",
                    )

            # Odešleme JSON s překladem
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logging.info("WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"Nastala chyba: {str(e)}")
        try:
            await websocket.send_json({"error": "server_error"})
        except Exception as send_err:
            logging.debug(f"Nelze poslat server_error: {send_err}")
        try:
            await websocket.close(code=1011)
        except Exception as close_err:
            logging.debug(f"Nelze zavřít websocket: {close_err}")


@app.websocket("/ws/deepgram")
async def deepgram_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pro Deepgram Nova-3 RSTT.
    Přijímá audio data z prohlížeče, posílá je do Deepgram, 
    vrací přepis a překlad.
    """
    if not await _require_ws_auth(websocket):
        return

    await websocket.accept()
    logging.info("WebSocket /ws/deepgram připojen")
    
    if not DEEPGRAM_API_KEY:
        logging.error("DEEPGRAM_API_KEY není nastaven")
        await websocket.send_json({"error": "DEEPGRAM_API_KEY not configured"})
        await websocket.close()
        return

    if DeepgramClient is None:
        logging.error("deepgram-sdk není nainstalovaný nebo nejde importovat")
        await websocket.send_json({"error": "deepgram-sdk not installed"})
        await websocket.close()
        return
    
    translator = Translator()
    stop_event = threading.Event()
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
    except WebSocketDisconnect:
        return

    if len(translate_dests) == 2 and translate_dests[0] == translate_dests[1]:
        translate_dests[1] = "ru" if translate_dests[0] != "ru" else "en"

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

        with contextlib.ExitStack() as stack:
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
                dg_socket = stack.enter_context(connect_obj)
            else:
                dg_socket_iterator = connect_obj
                dg_socket = next(dg_socket_iterator)
                stack.callback(getattr(dg_socket_iterator, "close", lambda: None))

            logging.info("Deepgram Nova-3 připojení úspěšně spuštěno")
            
            # Queue pro předávání výsledků mezi vlákny
            result_queue: asyncio.Queue[TranscriptResult] = asyncio.Queue(
                maxsize=DEEPGRAM_RESULT_QUEUE_SIZE
            )

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
                                logging.info(f"Deepgram transkripce: {transcript} (final: {is_final})")
                                payload: TranscriptResult = {
                                    "transcript": transcript,
                                    "is_final": bool(is_final),
                                }

                                def _enqueue() -> None:
                                    # During shutdown we still want to enqueue final results produced
                                    # by Deepgram finalize/close, but we can drop interim updates.
                                    if stop_event.is_set() and not payload["is_final"]:
                                        return
                                    # Udržet frontu omezenou. Preferujeme dropovat interim, ne final.
                                    if result_queue.full():
                                        if payload["is_final"]:
                                            drained: list[TranscriptResult] = []
                                            try:
                                                while True:
                                                    drained.append(result_queue.get_nowait())
                                            except asyncio.QueueEmpty:
                                                pass

                                            dropped = False
                                            kept: list[TranscriptResult] = []
                                            for item in drained:
                                                if not dropped and not item.get("is_final"):
                                                    dropped = True
                                                    continue
                                                kept.append(item)
                                            # If the queue had only final items, drop the oldest one.
                                            if not dropped and kept:
                                                kept = kept[1:]

                                            for item in kept:
                                                try:
                                                    result_queue.put_nowait(item)
                                                except asyncio.QueueFull:
                                                    break
                                        else:
                                            try:
                                                result_queue.get_nowait()
                                            except asyncio.QueueEmpty:
                                                # Fronta je v tomto okamžiku prázdná – není co odstranit.
                                                pass
                                    try:
                                        result_queue.put_nowait(payload)
                                    except asyncio.QueueFull:
                                        # Pokud je fronta stále plná, tento výsledek přeskočíme.
                                        pass

                                event_loop.call_soon_threadsafe(_enqueue)
                except Exception as e:
                    logging.error(f"Chyba při zpracování transkripce: {str(e)}")
            
            def on_error(error):
                logging.error(f"Deepgram error: {error}")
                stop_event.set()

                def _notify() -> None:
                    async def _send() -> None:
                        try:
                            await websocket.send_json({"error": str(error)})
                        except Exception as send_err:
                            logging.debug(f"Nelze poslat Deepgram error: {send_err}")

                    asyncio.create_task(_send())

                event_loop.call_soon_threadsafe(_notify)
            
            def on_close(close):
                logging.info("Deepgram připojení uzavřeno")
                stop_event.set()
            
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
            
            listen_thread = threading.Thread(target=listen_thread_func, daemon=True)
            listen_thread.start()
            
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
                         
                        await websocket.send_json(response)
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
                    dg_socket.send_media(first_audio)
                while not stop_event.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("type") != "websocket.receive":
                        continue
                    data = msg.get("bytes")
                    if data:
                        dg_socket.send_media(data)
            except WebSocketDisconnect:
                logging.info("Klient odpojen")
            finally:
                stop_event.set()
                try:
                    _deepgram_send_finalize(dg_socket)
                    _deepgram_send_close_stream(dg_socket)
                except Exception as e:
                    logging.warning(f"Deepgram close selhal: {e}")
                if listen_thread is not None:
                    listen_thread.join(timeout=1.0)

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
    
    except WebSocketDisconnect:
        logging.info("Deepgram WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"Deepgram chyba: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception as send_err:
            logging.debug(f"Nelze poslat Deepgram error: {send_err}")


@app.websocket("/ws/elevenlabs")
async def elevenlabs_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ElevenLabs Scribe v2 Realtime STT.
    Receives PCM audio from the browser, proxies it to the ElevenLabs
    realtime WS, translates transcripts and sends them back.
    """
    if not await _require_ws_auth(websocket):
        return

    await websocket.accept()
    logging.info("WebSocket /ws/elevenlabs připojen")

    if not ELEVENLABS_API_KEY:
        logging.error("ELEVENLABS_API_KEY není nastaven")
        await websocket.send_json({"error": "ELEVENLABS_API_KEY not configured"})
        await websocket.close()
        return

    translator = Translator()

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
    except WebSocketDisconnect:
        return

    if len(translate_dests) == 2 and translate_dests[0] == translate_dests[1]:
        translate_dests[1] = "ru" if translate_dests[0] != "ru" else "en"

    # Build ElevenLabs WS URL with query parameters.
    el_params = (
        f"model_id=scribe_v2_realtime"
        f"&audio_format=pcm_16000"
        f"&sample_rate=16000"
        f"&commit_strategy={el_commit_strategy}"
    )
    if el_language_code:
        el_params += f"&language_code={el_language_code}"
    if el_commit_strategy == "vad":
        el_params += "&vad_silence_threshold_secs=1.5"
    el_ws_url = f"{ELEVENLABS_WS_URL}?{el_params}"

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
                                await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass
            except WebSocketDisconnect:
                logging.info("ElevenLabs: klient odpojen")
            except Exception as e:
                logging.error(f"ElevenLabs forward_audio error: {e}")
            finally:
                stop_event.set()

        async def _receive_transcripts():
            """Read transcripts from ElevenLabs and send translated results to browser."""
            nonlocal translator
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
                            try:
                                start_t = time.perf_counter()
                                results = await asyncio.gather(
                                    *[
                                        _translate(translator, text, src=translate_src, dest=dest)
                                        for dest in translate_dests
                                    ]
                                )
                                translate_ms = int((time.perf_counter() - start_t) * 1000)
                                translations = {
                                    dest: (res.text if res else "")
                                    for dest, res in zip(translate_dests, results)
                                }
                                response = _el_payload(
                                    msg_type="interim",
                                    original=text,
                                    translations=translations,
                                    timing={"translate_ms": translate_ms},
                                )
                            except Exception as te:
                                logging.error(f"ElevenLabs interim translation error: {te}")
                                translator = Translator()
                                response = _el_payload(
                                    msg_type="interim",
                                    original=text,
                                    translations={d: "" for d in translate_dests},
                                    error="translation_failed",
                                    timing={"translate_ms": 0},
                                )
                        else:
                            response = _el_payload(
                                msg_type="interim",
                                original=text,
                                translations={d: "" for d in translate_dests},
                                timing={"translate_ms": 0},
                            )
                        await websocket.send_json(response)

                    elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                        text = ev.get("text", "").strip()
                        if not text:
                            continue
                        try:
                            start_t = time.perf_counter()
                            results = await asyncio.gather(
                                *[
                                    _translate(translator, text, src=translate_src, dest=dest)
                                    for dest in translate_dests
                                ]
                            )
                            translate_ms = int((time.perf_counter() - start_t) * 1000)
                            translations = {
                                dest: (res.text if res else "")
                                for dest, res in zip(translate_dests, results)
                            }
                            response = _el_payload(
                                msg_type="final",
                                original=text,
                                translations=translations,
                                timing={"translate_ms": translate_ms},
                            )
                        except Exception as te:
                            logging.error(f"ElevenLabs translation error: {te}")
                            translator = Translator()
                            response = _el_payload(
                                msg_type="final",
                                original=text,
                                translations={d: "" for d in translate_dests},
                                error="translation_failed",
                                timing={"translate_ms": 0},
                            )
                        await websocket.send_json(response)

                    elif msg_type in ("input_error", "error", "auth_error",
                                      "transcriber_error", "quota_exceeded"):
                        error_msg = ev.get("error", ev.get("message", str(ev)))
                        logging.error(f"ElevenLabs error: {error_msg}")
                        await websocket.send_json({"error": f"ElevenLabs: {error_msg}"})

            except Exception as e:
                if not stop_event.is_set():
                    logging.error(f"ElevenLabs receive_transcripts error: {e}")
            finally:
                stop_event.set()

        # Run both tasks concurrently; when one stops the other is cancelled.
        forward_task = asyncio.create_task(_forward_audio())
        receive_task = asyncio.create_task(_receive_transcripts())

        done, pending = await asyncio.wait(
            {forward_task, receive_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop_event.set()
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logging.info("ElevenLabs WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"ElevenLabs chyba: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception as send_err:
            logging.debug(f"Nelze poslat ElevenLabs error: {send_err}")
    finally:
        if el_ws:
            try:
                await el_ws.close()
            except Exception:
                pass
