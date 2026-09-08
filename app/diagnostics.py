"""Content-free JSON diagnostics, correlated across one WebSocket's tasks.

Use Uvicorn's configured INFO logger so these records reach container stderr
without enabling verbose HTTP/SDK logging (which may expose credentials).
Only allowlisted enums and bounded numbers may enter a record, never payloads,
exception messages, tracebacks, headers, URLs or client-supplied identifiers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from secrets import token_hex


logger = logging.getLogger("uvicorn.error")
_STRINGS = {
    "stt": {"web_speech", "google_audio"},
    "kind": {"interim", "final"},
    "outcome": {
        "handler_returned",
        "completed",
        "client_disconnect",
        "server_cancelled",
        "unhandled_error",
        "auth_rejected",
        "not_configured",
        "server_busy",
        "terminal_error",
    },
    "error_kind": {"timeout", "http", "invalid_response", "internal"},
    "cancel_reason": {
        "session_shutdown",
        "final_preempted",
        "interim_withdrawn",
        "actor_cancelled",
    },
    "error_code": {
        "invalid_audio",
        "audio_too_large",
        "audio_rate_exceeded",
        "invalid_message",
        "transcript_too_large",
        "transcript_rate_exceeded",
        "server_busy",
        "translation_backlog_full",
        "transcription_incomplete",
        "session_limit",
        "google_rate_limited",
        "google_unavailable",
        "google_timeout",
        "session_failed",
        "translation_failed",
        "interim_translation_failed",
        "missing_json",
        "invalid_json",
        "unexpected_fields",
        "empty_language",
        "invalid_response",
    },
    "model": {"gemini-3.1-flash-lite", "gemini-3.5-flash-lite"},
    "finish_reason": {
        "STOP",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "IMAGE_SAFETY",
        "UNEXPECTED_TOOL_CALL",
        "NO_IMAGE",
        "FINISH_REASON_UNSPECIFIED",
    },
}
_COUNTERS = {
    "interim_received",
    "final_received",
    "interim_started",
    "final_started",
    "interim_succeeded",
    "final_succeeded",
    "interim_failed",
    "final_failed",
    "interim_cancelled",
    "final_cancelled",
    "interim_withdrawn",
    "audio_bytes",
    "retries",
}
_NUMBERS = _COUNTERS | {
    "elapsed_ms",
    "input_chars",
    "queue_ms",
    "translation_ms",
    "attempt_ms",
    "attempt",
    "work_id",
    "http_status",
    "close_code",
    "first_stt_ms",
    "first_transcript_ms_since_session",
    "pending_finals",
    "timeout_ms",
}
_EVENTS = {
    "session_open",
    "session_ready",
    "session_stop_requested",
    "session_error",
    "session_close",
    "session_disconnect",
    "session_close_requested",
    "stt_progress",
    "translation_started",
    "translation_succeeded",
    "translation_failed",
    "translation_cancelled",
    "translation_attempt_failed",
    "translation_retry",
    "upstream_error",
    "session_exception",
}
_current: ContextVar[SessionDiagnostics | None] = ContextVar(
    "diagnostics", default=None
)
_work: ContextVar[int | None] = ContextVar("translation_work", default=None)


class SessionDiagnostics:
    def __init__(self, stt: str) -> None:
        self.id = token_hex(8)
        self.stt = stt
        self.started = time.monotonic()
        self.counts = dict.fromkeys(_COUNTERS, 0)
        self.outcome = "handler_returned"
        self.stop_requested = False
        self.last_progress = float("-inf")
        self.first_transcript_ms: int | None = None


def event(name: str, **fields: object) -> None:
    state = _current.get()
    if state is None or name not in _EVENTS:
        return
    record = {
        "event": name,
        "session_id": state.id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "elapsed_ms": max(0, round((time.monotonic() - state.started) * 1000)),
        "stt": state.stt if state.stt in _STRINGS["stt"] else "unknown",
    }
    if _work.get() is not None:
        record["work_id"] = _work.get()
    for key, value in fields.items():
        if key in _STRINGS and type(value) is str and value in _STRINGS[key]:
            record[key] = value
        elif key in _NUMBERS and type(value) is int and 0 <= value <= 10**12:
            record[key] = value
    if name == "session_close":
        record["stop_requested"] = state.stop_requested
    level = (
        logging.WARNING
        if name.endswith("failed")
        or name in {"session_error", "upstream_error", "session_exception"}
        else logging.INFO
    )
    logger.log(level, "translator %s", json.dumps(record, sort_keys=True))


def count(name: str, amount: int = 1) -> None:
    state = _current.get()
    if state is not None and name in _COUNTERS:
        state.counts[name] += amount


def transcript(kind: str, input_chars: int, first_stt_ms: int | None) -> None:
    state = _current.get()
    if state is None:
        return
    count(f"{kind}_received")
    now = time.monotonic()
    if state.first_transcript_ms is None:
        state.first_transcript_ms = max(0, round((now - state.started) * 1000))
    # Do not write every Web Speech hypothesis or every 100 ms audio frame.
    # Finals and the first result are visible; interim progress is bounded.
    if kind == "final" or now - state.last_progress >= 5:
        state.last_progress = now
        event(
            "stt_progress",
            kind=kind,
            input_chars=input_chars,
            first_stt_ms=first_stt_ms,
            first_transcript_ms_since_session=state.first_transcript_ms,
            **state.counts,
        )


def stop_requested() -> None:
    state = _current.get()
    if state is not None and not state.stop_requested:
        state.stop_requested = True
        event("session_stop_requested")


def outcome(value: str) -> None:
    state = _current.get()
    if state is not None and value in _STRINGS["outcome"]:
        state.outcome = value


def disconnect(code: object) -> None:
    state = _current.get()
    if state is not None and state.outcome == "handler_returned":
        outcome("client_disconnect")
    event("session_disconnect", close_code=code)


def error_fields(exc: BaseException) -> dict[str, object]:
    # Never stringify exc, inspect its response/body, or include its traceback.
    if isinstance(exc, TimeoutError):
        return {"error_kind": "timeout"}
    response_code = getattr(exc, "diagnostic_code", None)
    if type(response_code) is str and response_code in {
        "missing_json",
        "invalid_json",
        "unexpected_fields",
        "empty_language",
        "invalid_response",
    }:
        return {"error_kind": "invalid_response", "error_code": response_code}
    status = getattr(exc, "code", None)
    if type(status) is not int:
        status = getattr(exc, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        return {"error_kind": "http", "http_status": status}
    return {"error_kind": "internal"}


@contextmanager
def work(work_id: int):
    token = _work.set(work_id)
    try:
        yield
    finally:
        _work.reset(token)


@contextmanager
def session(stt: str):
    state = SessionDiagnostics(stt)
    token = _current.set(state)
    event("session_open")
    try:
        yield
    except asyncio.CancelledError:
        outcome("server_cancelled")
        raise
    except BaseException:
        outcome("unhandled_error")
        raise
    finally:
        try:
            event("session_close", outcome=state.outcome, **state.counts)
        finally:
            _current.reset(token)
