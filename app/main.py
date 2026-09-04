from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import quote, urlparse

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import errors, types
from starlette.middleware.base import BaseHTTPMiddleware

from app.google_audio import (
    AUDIO_MIME_TYPE,
    FINAL_TIMEOUT_SECONDS,
    MAX_AUDIO_CHUNK_BYTES,
    MAX_FINAL_BACKLOG,
    TRANSCRIBE_MODEL,
    GeminiPairTranslator,
    TranslationBacklogFull,
    TranslationCoordinator,
    live_transcription_config,
    transcript_from_message,
)


load_dotenv()

logger = logging.getLogger("realtime_translator")
BASE_DIR = Path(__file__).resolve().parent


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> tuple[bool, bool]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default, True
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True, True
    if normalized in _FALSE_VALUES:
        return False, True
    # Authentication configuration must fail closed on a typo.
    return True, False


def _env_int(name: str, default: int) -> tuple[int, bool]:
    try:
        return int(os.getenv(name, str(default))), True
    except ValueError:
        return default, False


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
AUTH_ENABLED, _AUTH_ENABLED_VALID = _env_bool("AUTH_ENABLED", True)
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
AUTH_SECRET = os.getenv("AUTH_SECRET", "").strip() or APP_PASSWORD
AUTH_COOKIE_NAME = (
    os.getenv("AUTH_COOKIE_NAME", "translator_auth").strip() or "translator_auth"
)
AUTH_TOKEN_TTL_SECONDS, _AUTH_TOKEN_TTL_VALID = _env_int(
    "AUTH_TOKEN_TTL_SECONDS", 12 * 60 * 60
)
AUTH_COOKIE_SECURE_RAW = os.getenv("AUTH_COOKIE_SECURE")

_PLACEHOLDERS = {
    "change-me",
    "changeme",
    "replace-me",
    "replace-with-a-long-random-value",
    "your-api-key",
    "your-password",
}
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 60.0
_LOGIN_TRACKED_IP_CAP = 10_000
_login_attempts: dict[str, deque[float]] = {}
_login_attempts_lock = threading.Lock()

# Google currently limits one live-transcription connection to ten minutes.
# End slightly early so the browser receives a controlled, restartable result.
_SESSION_MAX_SECONDS = 9 * 60 + 45
_GOOGLE_SETUP_TIMEOUT_SECONDS = 15.0
_GOOGLE_CLOSE_TIMEOUT_SECONDS = 2.0
_FINAL_TRANSCRIPT_GRACE_SECONDS = 2.5
_FINAL_DRAIN_SECONDS = (MAX_FINAL_BACKLOG + 1) * (
    (2 * FINAL_TIMEOUT_SECONDS) + 0.25
) + 1
_AUDIO_BYTES_PER_SECOND = 16_000 * 2
_AUDIO_BURST_BYTES = _AUDIO_BYTES_PER_SECOND * 2
_MAX_CONCURRENT_AUDIO_SESSIONS = 4
_audio_session_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_AUDIO_SESSIONS)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "; ".join(
                (
                    "default-src 'self'",
                    "script-src 'self' 'unsafe-inline'",
                    "style-src 'self' 'unsafe-inline'",
                    "img-src 'self' data:",
                    "connect-src 'self'",
                    "worker-src 'self'",
                    "object-src 'none'",
                    "base-uri 'none'",
                    "frame-ancestors 'none'",
                    "form-action 'self'",
                )
            ),
        )
        response.headers.setdefault("Permissions-Policy", "microphone=(self)")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cache-Control", "no-store")
        if _request_is_https(request):
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


app = FastAPI(title="Czech live translation", docs_url=None, redoc_url=None)
app.add_middleware(SecurityHeadersMiddleware)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def configuration_errors() -> list[str]:
    failures: list[str] = []
    if not GEMINI_API_KEY or GEMINI_API_KEY.lower() in _PLACEHOLDERS:
        failures.append("gemini_api_key_missing")
    if AUTH_ENABLED:
        if not _AUTH_ENABLED_VALID:
            failures.append("auth_enabled_invalid")
        if not APP_PASSWORD or APP_PASSWORD.lower() in _PLACEHOLDERS:
            failures.append("app_password_missing")
        if not AUTH_SECRET or AUTH_SECRET.lower() in _PLACEHOLDERS:
            failures.append("auth_secret_missing")
        if not _AUTH_TOKEN_TTL_VALID or AUTH_TOKEN_TTL_SECONDS <= 0:
            failures.append("auth_token_ttl_invalid")
        if AUTH_COOKIE_SECURE_RAW and not _valid_optional_bool(AUTH_COOKIE_SECURE_RAW):
            failures.append("auth_cookie_secure_invalid")
    return failures


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64url") from exc


def _sign(payload_b64: str) -> str:
    if not AUTH_SECRET:
        return ""
    digest = hmac.new(
        AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
    ).digest()
    return _b64url_encode(digest)


def create_auth_token() -> str:
    now = int(time.time())
    payload = {"iat": now, "exp": now + AUTH_TOKEN_TTL_SECONDS}
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_auth_token(token: str | None) -> bool:
    if not token or not AUTH_SECRET:
        return False
    parts = token.split(".")
    if len(parts) != 2:
        return False
    payload_b64, signature = parts
    expected = _sign(payload_b64)
    if not expected or not secrets.compare_digest(
        expected.encode("ascii"), signature.encode("utf-8")
    ):
        return False
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return False
    issued_at = payload.get("iat")
    expires_at = payload.get("exp")
    now = int(time.time())
    return (
        isinstance(issued_at, int)
        and isinstance(expires_at, int)
        and issued_at <= now + 60
        and expires_at >= now
    )


def sanitize_next_path(next_path: str | None) -> str:
    if not next_path or not next_path.startswith("/"):
        return "/"
    if next_path.startswith("//") or "\\" in next_path:
        return "/"
    parsed = urlparse(next_path)
    if parsed.scheme or parsed.netloc:
        return "/"
    return next_path


def is_origin_allowed(origin: str | None, host: str | None) -> bool:
    configured = os.getenv("ALLOWED_ORIGINS", "").strip()
    if configured:
        allowed = {value.strip() for value in configured.split(",") if value.strip()}
        return bool(origin) and origin in allowed
    if not origin or not host:
        return False
    try:
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc == host
    except Exception:  # pragma: no cover - urlparse accepts arbitrary strings
        return False


def _request_is_https(request: Request) -> bool:
    # Uvicorn rewrites ASGI scope["scheme"] only for trusted proxy addresses;
    # reading X-Forwarded-Proto here directly would bypass that trust boundary.
    return request.url.scheme == "https"


def _valid_optional_bool(raw: str) -> bool:
    return raw.strip().lower() in (_TRUE_VALUES | _FALSE_VALUES)


def _cookie_secure_for_request(request: Request) -> bool:
    if AUTH_COOKIE_SECURE_RAW is not None and AUTH_COOKIE_SECURE_RAW.strip():
        normalized = AUTH_COOKIE_SECURE_RAW.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        return True
    return _request_is_https(request)


def _is_same_origin(request: Request) -> bool:
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return True
    try:
        return urlparse(source).netloc == request.headers.get("host", "")
    except Exception:  # pragma: no cover - urlparse accepts arbitrary strings
        return False


def _prune_login_attempts(client_ip: str, now: float) -> deque[float] | None:
    attempts = _login_attempts.get(client_ip)
    if attempts is None:
        return None
    cutoff = now - _LOGIN_WINDOW_SECONDS
    while attempts and attempts[0] <= cutoff:
        attempts.popleft()
    if not attempts:
        _login_attempts.pop(client_ip, None)
        return None
    return attempts


def _check_login_rate_limit(client_ip: str) -> bool:
    with _login_attempts_lock:
        attempts = _prune_login_attempts(client_ip, time.monotonic())
        return attempts is None or len(attempts) < _LOGIN_MAX_ATTEMPTS


def _record_login_attempt(client_ip: str) -> None:
    with _login_attempts_lock:
        now = time.monotonic()
        attempts = _prune_login_attempts(client_ip, now)
        if attempts is None:
            if len(_login_attempts) >= _LOGIN_TRACKED_IP_CAP:
                for tracked_ip in list(_login_attempts):
                    _prune_login_attempts(tracked_ip, now)
            if len(_login_attempts) >= _LOGIN_TRACKED_IP_CAP:
                _login_attempts.pop(next(iter(_login_attempts)))
            attempts = deque()
            _login_attempts[client_ip] = attempts
        attempts.append(now)


def _clear_login_attempts(client_ip: str) -> None:
    with _login_attempts_lock:
        _login_attempts.pop(client_ip, None)


def _render_login(
    request: Request, *, next_path: str = "/", invalid_pwd: bool = False
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="password_prompt.html",
        context={
            "next_path": sanitize_next_path(next_path),
            "invalid_pwd": invalid_pwd,
        },
        status_code=401 if invalid_pwd else 200,
    )


@app.get("/health")
@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    failures = configuration_errors()
    if failures:
        return JSONResponse(
            {"status": "not_ready", "checks": failures}, status_code=503
        )
    return {"status": "ready"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if AUTH_ENABLED and not verify_auth_token(request.cookies.get(AUTH_COOKIE_NAME)):
        return RedirectResponse(
            url=f"/login?next={quote('/', safe='')}", status_code=303
        )
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/"):  # noqa: A002
    if not AUTH_ENABLED:
        return RedirectResponse(url="/", status_code=303)
    return _render_login(request, next_path=next)


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
    if not _is_same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-origin login not allowed")

    client_ip = request.client.host if request.client else "0.0.0.0"
    if not _check_login_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts")

    next_path = sanitize_next_path(next_path)
    if not secrets.compare_digest(
        password.encode("utf-8"), APP_PASSWORD.encode("utf-8")
    ):
        _record_login_attempt(client_ip)
        return _render_login(request, next_path=next_path, invalid_pwd=True)

    _clear_login_attempts(client_ip)
    response = RedirectResponse(url=next_path, status_code=303)
    response.set_cookie(
        AUTH_COOKIE_NAME,
        create_auth_token(),
        max_age=AUTH_TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure_for_request(request),
        path="/",
    )
    return response


async def _require_ws_auth(websocket: WebSocket) -> bool:
    if AUTH_ENABLED and not APP_PASSWORD:
        await websocket.close(code=1011, reason="Server not configured")
        return False
    if not is_origin_allowed(
        websocket.headers.get("origin"), websocket.headers.get("host")
    ):
        await websocket.close(code=1008, reason="Origin not allowed")
        return False
    if AUTH_ENABLED and not verify_auth_token(websocket.cookies.get(AUTH_COOKIE_NAME)):
        await websocket.close(code=1008, reason="Unauthorized")
        return False
    return True


class _JsonSender:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def send(self, payload: dict[str, object]) -> None:
        async with self._lock:
            await self._websocket.send_json(payload)


class _ClientProtocolError(RuntimeError):
    def __init__(self, code: int, reason: str, error_code: str) -> None:
        super().__init__(reason)
        self.close_code = code
        self.reason = reason
        self.error_code = error_code


class _ReportedSessionError(RuntimeError):
    """Terminal error already sent to the browser before a bounded drain."""

    def __init__(self, close_code: int, reason: str) -> None:
        super().__init__(reason)
        self.close_code = close_code
        self.reason = reason


class _AudioRateLimiter:
    """Bound paid audio bytes to realtime with a two-second jitter allowance."""

    def __init__(self) -> None:
        self._tokens = float(_AUDIO_BURST_BYTES)
        self._updated_at = time.monotonic()

    def consume(self, size: int) -> bool:
        now = time.monotonic()
        elapsed = max(0.0, now - self._updated_at)
        self._updated_at = now
        self._tokens = min(
            float(_AUDIO_BURST_BYTES),
            self._tokens + (elapsed * _AUDIO_BYTES_PER_SECOND),
        )
        if size > self._tokens:
            return False
        self._tokens -= size
        return True


def _create_google_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


@asynccontextmanager
async def _connect_google_live(async_client: object):
    """Bound setup, then leave the established Live session untimed."""

    connection = async_client.live.connect(
        model=TRANSCRIBE_MODEL, config=live_transcription_config()
    )
    google_session = await asyncio.wait_for(
        connection.__aenter__(), timeout=_GOOGLE_SETUP_TIMEOUT_SECONDS
    )
    try:
        yield google_session
    except BaseException as exc:
        suppressed = await asyncio.wait_for(
            connection.__aexit__(type(exc), exc, exc.__traceback__),
            timeout=_GOOGLE_CLOSE_TIMEOUT_SECONDS,
        )
        if not suppressed:
            raise
    else:
        await asyncio.wait_for(
            connection.__aexit__(None, None, None),
            timeout=_GOOGLE_CLOSE_TIMEOUT_SECONDS,
        )


async def _receive_browser_audio(websocket: WebSocket, google_session: object) -> bool:
    """Forward fixed-format audio. Return True for a deliberate Stop."""

    limiter = _AudioRateLimiter()
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return False

        audio = message.get("bytes")
        if audio is not None:
            if not audio or len(audio) % 2:
                raise _ClientProtocolError(1003, "Invalid PCM frame", "invalid_audio")
            if len(audio) > MAX_AUDIO_CHUNK_BYTES:
                raise _ClientProtocolError(
                    1009, "Audio frame too large", "audio_too_large"
                )
            if not limiter.consume(len(audio)):
                raise _ClientProtocolError(
                    1008, "Audio sent faster than realtime", "audio_rate_exceeded"
                )
            await google_session.send_realtime_input(
                audio=types.Blob(data=audio, mime_type=AUDIO_MIME_TYPE)
            )
            continue

        raw = message.get("text")
        try:
            control = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            control = None
        if control == {"type": "stop"}:
            return True
        raise _ClientProtocolError(
            1003, "Unsupported client message", "invalid_message"
        )


async def _receive_google_transcripts(
    google_session: object,
    coordinator: TranslationCoordinator,
    stop_requested: asyncio.Event,
    final_after_stop: asyncio.Event,
) -> None:
    # AsyncSession.receive() ends after every model turn, so reconnect the
    # iterator while keeping the same Live session open.
    while True:
        received = False
        async for message in google_session.receive():
            received = True
            event = transcript_from_message(message)
            if event is None:
                continue
            kind, text = event
            coordinator.submit(kind, text)
            if kind == "final" and stop_requested.is_set():
                final_after_stop.set()
        if stop_requested.is_set() and final_after_stop.is_set():
            return
        if not received:
            raise RuntimeError("Google Live session ended without a response")


async def _cancel_tasks(*tasks: asyncio.Task[object] | None) -> None:
    selected = [task for task in tasks if task is not None]
    active = [task for task in selected if not task.done()]
    for task in active:
        task.cancel()
    if selected:
        await asyncio.gather(*selected, return_exceptions=True)


async def _close_google_session(google_session: object) -> None:
    close = getattr(google_session, "close", None)
    if callable(close):
        with suppress(Exception):
            await asyncio.wait_for(close(), timeout=_GOOGLE_CLOSE_TIMEOUT_SECONDS)


async def _drain_translations(
    coordinator: TranslationCoordinator, translation_task: asyncio.Task[None]
) -> None:
    # close() discards only the speculative interim. Accepted finals remain in
    # the bounded FIFO and therefore have a finite worst-case drain time.
    coordinator.close()
    await asyncio.wait_for(translation_task, timeout=_FINAL_DRAIN_SECONDS)


async def _serve_audio_session(websocket: WebSocket, sender: _JsonSender) -> None:
    client = _create_google_client()
    async_client = client.aio
    coordinator: TranslationCoordinator | None = None
    browser_task: asyncio.Task[bool] | None = None
    google_task: asyncio.Task[None] | None = None
    translation_task: asyncio.Task[None] | None = None
    limit_task: asyncio.Task[None] | None = None

    try:
        async with _connect_google_live(async_client) as google_session:
            stop_requested = asyncio.Event()
            final_after_stop = asyncio.Event()
            coordinator = TranslationCoordinator(
                GeminiPairTranslator(async_client), sender.send
            )
            translation_task = asyncio.create_task(coordinator.run())
            google_task = asyncio.create_task(
                _receive_google_transcripts(
                    google_session, coordinator, stop_requested, final_after_stop
                )
            )
            browser_task = asyncio.create_task(
                _receive_browser_audio(websocket, google_session)
            )
            limit_task = asyncio.create_task(asyncio.sleep(_SESSION_MAX_SECONDS))

            await sender.send({"type": "ready"})
            done, _ = await asyncio.wait(
                {browser_task, google_task, translation_task, limit_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # On an upstream/queue failure, stop buying audio first, then drain
            # every final already accepted before surfacing the terminal error.
            if google_task in done:
                try:
                    google_task.result()
                except BaseException as exc:
                    terminal_error = exc
                else:
                    terminal_error = RuntimeError(
                        "Google Live session ended unexpectedly"
                    )
                google_task = None
                await _cancel_tasks(browser_task, limit_task)
                browser_task = None
                limit_task = None
                reported_error = await _report_terminal_error(sender, terminal_error)
                await _close_google_session(google_session)
                await _drain_translations(coordinator, translation_task)
                translation_task = None
                raise reported_error from terminal_error
            if translation_task in done:
                translation_task.result()
                raise RuntimeError("translation actor ended unexpectedly")

            deliberate_stop = False
            if browser_task in done:
                try:
                    deliberate_stop = browser_task.result()
                except BaseException as exc:
                    await _cancel_tasks(google_task, limit_task)
                    google_task = None
                    limit_task = None
                    reported_error = await _report_terminal_error(sender, exc)
                    await _close_google_session(google_session)
                    await _drain_translations(coordinator, translation_task)
                    translation_task = None
                    raise reported_error from exc
                if not deliberate_stop:
                    return
            elif limit_task in done:
                await sender.send(
                    {"type": "error", "code": "session_limit", "recoverable": True}
                )
                # No audio may follow audio_stream_end.  The user-triggered
                # Stop path has already completed this task; the timed path has
                # to stop it explicitly before finalizing Google.
                await _cancel_tasks(browser_task)
                browser_task = None
                deliberate_stop = True

            if deliberate_stop:
                stop_requested.set()
                upstream_error: BaseException | None = None
                try:
                    await google_session.send_realtime_input(audio_stream_end=True)
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(
                            final_after_stop.wait(),
                            timeout=_FINAL_TRANSCRIPT_GRACE_SECONDS,
                        )
                    # A final and turn-complete can arrive in separate frames.
                    await asyncio.sleep(0.10)
                except Exception as exc:
                    upstream_error = exc

                if google_task is not None:
                    if not google_task.done():
                        google_task.cancel()
                    result = (
                        await asyncio.gather(google_task, return_exceptions=True)
                    )[0]
                    if (
                        isinstance(result, BaseException)
                        and not isinstance(result, asyncio.CancelledError)
                        and upstream_error is None
                    ):
                        upstream_error = result
                google_task = None

                await _close_google_session(google_session)
                await _drain_translations(coordinator, translation_task)
                translation_task = None
                if upstream_error is not None:
                    raise upstream_error
                await sender.send({"type": "ended"})
                await websocket.close(code=1000, reason="Finished")
    finally:
        if coordinator is not None:
            coordinator.close()
        await _cancel_tasks(browser_task, google_task, translation_task, limit_task)
        with suppress(Exception):
            await asyncio.wait_for(
                async_client.aclose(), timeout=_GOOGLE_CLOSE_TIMEOUT_SECONDS
            )


def _status_code_from_google_error(exc: BaseException) -> int | None:
    value = getattr(exc, "code", None)
    if isinstance(value, int):
        return value
    value = getattr(exc, "status_code", None)
    return value if isinstance(value, int) else None


async def _best_effort_error(
    sender: _JsonSender, code: str, *, recoverable: bool = False
) -> None:
    with suppress(Exception):
        await sender.send({"type": "error", "code": code, "recoverable": recoverable})


async def _report_terminal_error(
    sender: _JsonSender, exc: BaseException
) -> _ReportedSessionError:
    """Tell the UI to release its microphone before final translations drain."""

    if isinstance(exc, asyncio.CancelledError):
        raise exc
    if isinstance(exc, _ClientProtocolError):
        code = exc.error_code
        recoverable = False
        close_code = exc.close_code
        reason = exc.reason
    elif isinstance(exc, TranslationBacklogFull):
        code = "translation_backlog_full"
        recoverable = False
        close_code = 1013
        reason = "Try again later"
    elif isinstance(exc, errors.APIError):
        status = _status_code_from_google_error(exc)
        if status == 429:
            code = "google_rate_limited"
            close_code = 1013
            reason = "Google rate limited"
        else:
            logger.warning("Google API session failed (status=%s)", status)
            code = "google_unavailable"
            close_code = 1011
            reason = "Google unavailable"
        recoverable = True
    elif isinstance(exc, asyncio.TimeoutError):
        code = "google_timeout"
        recoverable = True
        close_code = 1011
        reason = "Google timeout"
    else:
        logger.error(
            "Audio translation session failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        code = "session_failed"
        recoverable = True
        close_code = 1011
        reason = "Session failed"

    await _best_effort_error(sender, code, recoverable=recoverable)
    return _ReportedSessionError(close_code, reason)


async def _best_effort_close(websocket: WebSocket, code: int, reason: str) -> None:
    with suppress(Exception):
        await websocket.close(code=code, reason=reason)


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    if not await _require_ws_auth(websocket):
        return
    if configuration_errors():
        await websocket.close(code=1011, reason="Server not configured")
        return

    slot_acquired = _audio_session_slots.acquire(blocking=False)
    if not slot_acquired:
        await websocket.accept()
        sender = _JsonSender(websocket)
        await _best_effort_error(sender, "server_busy", recoverable=True)
        await _best_effort_close(websocket, 1013, "Server busy")
        return
    try:
        # accept() is inside the slot's finally scope: an aborted handshake must
        # not permanently consume one of the fixed paid-session slots.
        await websocket.accept()
        sender = _JsonSender(websocket)
        try:
            await _serve_audio_session(websocket, sender)
        except WebSocketDisconnect:
            return
        except _ReportedSessionError as exc:
            # Its JSON error was sent before the possibly long final drain.
            await _best_effort_close(websocket, exc.close_code, exc.reason)
        except _ClientProtocolError as exc:
            await _best_effort_error(sender, exc.error_code)
            await _best_effort_close(websocket, exc.close_code, exc.reason)
        except TranslationBacklogFull:
            await _best_effort_error(sender, "translation_backlog_full")
            await _best_effort_close(websocket, 1013, "Try again later")
        except errors.APIError as exc:
            status = _status_code_from_google_error(exc)
            if status == 429:
                await _best_effort_error(
                    sender, "google_rate_limited", recoverable=True
                )
                await _best_effort_close(websocket, 1013, "Google rate limited")
            else:
                logger.warning("Google API session failed (status=%s)", status)
                await _best_effort_error(sender, "google_unavailable", recoverable=True)
                await _best_effort_close(websocket, 1011, "Google unavailable")
        except asyncio.TimeoutError:
            await _best_effort_error(sender, "google_timeout", recoverable=True)
            await _best_effort_close(websocket, 1011, "Google timeout")
        except Exception:
            logger.exception("Audio translation session failed")
            await _best_effort_error(sender, "session_failed", recoverable=True)
            await _best_effort_close(websocket, 1011, "Session failed")
    finally:
        _audio_session_slots.release()
