import asyncio
import json
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app import diagnostics, google_audio


PRIVATE = "PRIVATE_CANARY transcript password cookie API-key raw-provider-response"


def test_default_uvicorn_configuration_emits_info_diagnostics_to_stderr():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from logging.config import dictConfig
from uvicorn.config import LOGGING_CONFIG
from app import diagnostics
dictConfig(LOGGING_CONFIG)
with diagnostics.session('web_speech'):
    diagnostics.event('session_ready')
""",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert '"event": "session_open"' in result.stderr
    assert '"event": "session_ready"' in result.stderr
    assert '"event": "session_close"' in result.stderr
    assert result.stdout == ""


def records(caplog):
    return [
        json.loads(record.getMessage().removeprefix("translator "))
        for record in caplog.records
        if record.getMessage().startswith("translator ")
    ]


def test_schema_drops_payloads_secrets_free_text_and_tracebacks(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with diagnostics.session("web_speech"):
        diagnostics.event(
            "translation_attempt_failed",
            kind="interim",
            http_status=429,
            attempt_ms=25,
            input_chars=100,
            model=PRIVATE,
            error_code=PRIVATE,
            finish_reason=PRIVATE,
            queue_ms=PRIVATE,
            translation_ms=float("inf"),
            headers={"Cookie": PRIVATE},
            text=PRIVATE,
            en=PRIVATE,
            ru=PRIVATE,
            api_key=PRIVATE,
            exception=ValueError(PRIVATE),
            exc_info=True,
            session_id=PRIVATE,
        )
        diagnostics.event(PRIVATE, text=PRIVATE)
        diagnostics.event(
            "upstream_error", **diagnostics.error_fields(ValueError(PRIVATE))
        )
    data = records(caplog)
    failure = data[1]
    assert failure["http_status"] == 429
    assert failure["attempt_ms"] == 25
    assert failure["input_chars"] == 100
    assert set(failure) == {
        "event",
        "session_id",
        "timestamp",
        "elapsed_ms",
        "stt",
        "kind",
        "http_status",
        "attempt_ms",
        "input_chars",
    }
    assert PRIVATE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert diagnostics._current.get() is None


@pytest.mark.parametrize(
    "kind,status", [("interim", 429), ("final", 429), ("final", 503)]
)
def test_http_failures_log_attempts_status_and_retry_without_response_body(
    caplog, monkeypatch, kind, status
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 0)

    class ProviderFailure(RuntimeError):
        code = status

    async def generate_content(**_kwargs):
        raise ProviderFailure(PRIVATE)

    async def scenario():
        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        emitted = []
        finished = asyncio.Event()

        async def emit(payload):
            emitted.append(payload)
            finished.set()

        with diagnostics.session("web_speech"):
            actor = google_audio.TranslationCoordinator(
                google_audio.GeminiPairTranslator(client), emit
            )
            task = asyncio.create_task(actor.run())
            actor.submit(kind, PRIVATE)
            await asyncio.wait_for(finished.wait(), 2)
            actor.close()
            await task
        assert emitted[0]["code"] == (
            "translation_failed" if kind == "final" else "interim_translation_failed"
        )

    asyncio.run(scenario())
    data = records(caplog)
    attempts = [r for r in data if r["event"] == "translation_attempt_failed"]
    assert len(attempts) == (2 if kind == "final" else 1)
    assert all(
        r["http_status"] == status and r["error_kind"] == "http" for r in attempts
    )
    assert {r["work_id"] for r in attempts} == {1}
    assert len({r["session_id"] for r in data}) == 1
    assert data[-1][f"{kind}_failed"] == 1
    assert data[-1]["retries"] == (1 if kind == "final" else 0)
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "empty_language"])
def test_timeout_and_invalid_pair_are_distinguishable_without_logging_content(
    caplog, monkeypatch, failure
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    monkeypatch.setattr(google_audio, "INTERIM_TIMEOUT_SECONDS", 0.005)

    async def generate_content(**_kwargs):
        if failure == "timeout":
            await asyncio.Future()
        if failure == "invalid_json":
            return SimpleNamespace(
                text=PRIVATE, candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")]
            )
        return SimpleNamespace(parsed={"en": PRIVATE, "ru": ""})

    async def scenario():
        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        with diagnostics.session("web_speech"), diagnostics.work(4):
            with pytest.raises((TimeoutError, google_audio.TranslationFailure)):
                await google_audio.GeminiPairTranslator(client).translate(
                    PRIVATE, "interim"
                )

    asyncio.run(scenario())
    event = next(
        r for r in records(caplog) if r["event"] == "translation_attempt_failed"
    )
    assert event["work_id"] == 4
    assert event["error_kind"] == (
        "timeout" if failure == "timeout" else "invalid_response"
    )
    if failure != "timeout":
        assert event["error_code"] == failure
    if failure == "invalid_json":
        assert event["finish_reason"] == "MAX_TOKENS"
    assert PRIVATE not in caplog.text


def test_interim_input_logging_is_coalesced_but_summary_counts_every_result(
    caplog, monkeypatch
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    now = [10.0]
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: now[0])
    with diagnostics.session("web_speech"):
        for _ in range(500):
            diagnostics.transcript("interim", 5, None)
        now[0] += 5
        diagnostics.transcript("interim", 10, None)
        diagnostics.transcript("final", 10, None)
    data = records(caplog)
    assert len([r for r in data if r["event"] == "stt_progress"]) == 3
    assert data[-1]["interim_received"] == 501
    assert data[-1]["final_received"] == 1
    assert data[-1]["audio_bytes"] == 0
    assert "first_stt_ms" not in data[1]  # Browser-only timing must not be invented.


def test_parallel_session_ids_counts_and_work_context_are_isolated(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    async def scenario():
        async def run(stt, count):
            with diagnostics.session(stt):
                for _ in range(count):
                    diagnostics.transcript("final", 5, None)
                    await asyncio.sleep(0)
                with diagnostics.work(count):
                    task = asyncio.create_task(log_from_child())
                await task

        async def log_from_child():
            await asyncio.sleep(0)
            diagnostics.event("translation_started", kind="final")

        await asyncio.gather(run("web_speech", 2), run("google_audio", 3))
        assert diagnostics._current.get() is None
        assert diagnostics._work.get() is None

    asyncio.run(scenario())
    summaries = [r for r in records(caplog) if r["event"] == "session_close"]
    assert len({r["session_id"] for r in summaries}) == 2
    for summary in summaries:
        expected = 2 if summary["stt"] == "web_speech" else 3
        assert summary["final_received"] == expected
        child = next(
            r
            for r in records(caplog)
            if r["event"] == "translation_started"
            and r["session_id"] == summary["session_id"]
        )
        assert child["work_id"] == expected
        assert "work_id" not in summary


def test_cancelled_request_is_not_logged_as_provider_failure(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 0)

    async def scenario():
        started = asyncio.Event()

        class Translator:
            async def translate(self, _text, _kind):
                started.set()
                await asyncio.Future()

        async def emit(_payload):
            pass

        with diagnostics.session("web_speech"):
            actor = google_audio.TranslationCoordinator(Translator(), emit)
            task = asyncio.create_task(actor.run())
            actor.submit("interim", PRIVATE)
            await asyncio.wait_for(started.wait(), 1)
            actor.close()
            await task

    asyncio.run(scenario())
    data = records(caplog)
    event = next(r for r in data if r["event"] == "translation_cancelled")
    assert event["cancel_reason"] == "session_shutdown"
    assert data[-1]["interim_cancelled"] == 1
    assert data[-1]["interim_failed"] == 0
    assert not any(r["event"] == "translation_attempt_failed" for r in data)
    assert PRIVATE not in caplog.text


def test_session_summary_and_context_cleanup_on_server_cancellation(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with pytest.raises(asyncio.CancelledError):
        with diagnostics.session("web_speech"):
            raise asyncio.CancelledError(PRIVATE)
    assert records(caplog)[-1]["outcome"] == "server_cancelled"
    assert diagnostics._current.get() is None
    assert PRIVATE not in caplog.text
