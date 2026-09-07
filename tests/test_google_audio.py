import asyncio
import struct
from types import SimpleNamespace

import pytest

from app import google_audio


def _message(*, interim: str | None = None, final: str | None = None):
    return SimpleNamespace(
        server_content=SimpleNamespace(
            interim_input_transcription=(
                SimpleNamespace(text=interim) if interim is not None else None
            ),
            input_transcription=(
                SimpleNamespace(text=final) if final is not None else None
            ),
        )
    )


def test_live_config_is_fixed_clean_czech_transcription():
    config = google_audio.live_transcription_config()

    assert [value.value for value in config.response_modalities] == ["TEXT"]
    assert config.input_audio_transcription.language_codes == ["cs-CZ"]
    assert config.input_audio_transcription.mode.value == "SMART"
    activity = config.realtime_input_config.automatic_activity_detection
    assert config.realtime_input_config.activity_handling.value == "NO_INTERRUPTION"
    assert activity.disabled is True
    assert activity.model_dump(exclude_none=True) == {"disabled": True}
    assert google_audio.TRANSCRIBE_MODEL == "gemini-3.5-transcribe-live"
    assert google_audio.AUDIO_MIME_TYPE == "audio/pcm;rate=16000"


def _pcm(value=8000, samples=1600):
    return struct.pack("<h", value) * samples


def test_endpointing_does_not_turn_initial_or_continued_silence_into_speech():
    segmenter = google_audio.AudioSegmenter()
    assert not any(segmenter.boundary_after(_pcm(0)) for _ in range(150))
    assert not any(segmenter.boundary_after(_pcm()) for _ in range(25))
    assert [segmenter.boundary_after(_pcm(0)) for _ in range(4)] == [
        False,
        False,
        False,
        True,
    ]
    assert not any(segmenter.boundary_after(_pcm(0)) for _ in range(150))


def test_endpointing_prefers_short_pause_after_target_and_bounds_continuous_speech():
    segmenter = google_audio.AudioSegmenter()
    assert not any(segmenter.boundary_after(_pcm()) for _ in range(65))
    assert not segmenter.boundary_after(_pcm(0))
    assert segmenter.boundary_after(_pcm(0))
    boundaries = [segmenter.boundary_after(_pcm()) for _ in range(240)]
    assert [index + 1 for index, boundary in enumerate(boundaries) if boundary] == [
        120,
        240,
    ]


def test_endpointing_counts_partial_frames_and_little_endian_audio():
    segmenter = google_audio.AudioSegmenter()
    # Same 12 seconds split into 10 ms frames, including negative PCM samples.
    boundaries = [segmenter.boundary_after(_pcm(-10000, 160)) for _ in range(1200)]
    assert sum(boundaries) == 1
    assert boundaries[-1]


class _ManualUpstream:
    def __init__(self):
        self.sent = []
        self.messages = asyncio.Queue()
        self.end_sent = asyncio.Event()
        self.closed = False

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)
        if "activity_end" in kwargs:
            self.end_sent.set()

    async def receive(self):
        yield await self.messages.get()

    async def close(self):
        self.closed = True


def test_manual_activity_waits_for_generation_before_reopening_without_losing_pcm(
    monkeypatch,
):
    monkeypatch.setattr(google_audio, "SEGMENT_MAX_SECONDS", 0.2)

    async def scenario():
        upstream = _ManualUpstream()
        session = google_audio.GoogleAudioSession(upstream)
        received = []

        async def read():
            while True:
                async for message in session.receive():
                    received.append(message)

        reader = asyncio.create_task(read())
        frames = [_pcm(0, 1), _pcm(1000), _pcm(-2000), _pcm(3000, 5)]
        try:
            # Silence and a partial prefix must follow activity_start too.
            await session.send_audio(frames[0])
            await session.send_audio(frames[1])
            boundary = asyncio.create_task(session.send_audio(frames[2]))
            await upstream.end_sent.wait()
            assert [next(iter(item)) for item in upstream.sent] == [
                "activity_start",
                "audio",
                "audio",
                "audio",
                "activity_end",
            ]
            final = _message(final="hotovo")
            final.server_content.turn_complete = True
            await upstream.messages.put(final)
            # A final/turn_complete alone is not this activity's generation ack.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not boundary.done()
            completion = SimpleNamespace(
                server_content=SimpleNamespace(generation_complete=True)
            )
            await upstream.messages.put(completion)
            await asyncio.wait_for(boundary, 0.1)
            await session.send_audio(frames[3])
            await session.end_input()
            await session.end_input()
            assert [
                item["audio"].data for item in upstream.sent if "audio" in item
            ] == frames
            assert [next(iter(item)) for item in upstream.sent][-4:] == [
                "activity_end",
                "activity_start",
                "audio",
                "activity_end",
            ]
            assert received == [final, completion]
            assert not any("audio_stream_end" in item for item in upstream.sent)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await session.close()
        assert upstream.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_manual_boundary_cannot_reopen_after_timeout_or_cancel_or_use_stale_ack(
    monkeypatch, cancel
):
    monkeypatch.setattr(google_audio, "SEGMENT_MAX_SECONDS", 0.1)
    monkeypatch.setattr(google_audio, "TURN_FINALIZATION_TIMEOUT_SECONDS", 0.02)

    async def scenario():
        upstream = _ManualUpstream()
        session = google_audio.GoogleAudioSession(upstream)
        # Previous completion must be cleared before issuing a new end.
        await upstream.messages.put(
            SimpleNamespace(server_content=SimpleNamespace(generation_complete=True))
        )
        async for _ in session.receive():
            pass
        task = asyncio.create_task(session.send_audio(_pcm()))
        await upstream.end_sent.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        with pytest.raises(TimeoutError):
            await session.send_audio(_pcm())
        await session.end_input()
        assert [next(iter(item)) for item in upstream.sent] == [
            "activity_start",
            "audio",
            "activity_end",
        ]

    asyncio.run(scenario())


def test_empty_manual_session_does_not_send_an_activity_end():
    async def scenario():
        upstream = _ManualUpstream()
        session = google_audio.GoogleAudioSession(upstream)
        await session.end_input()
        assert upstream.sent == []

    asyncio.run(scenario())


def test_final_transcript_wins_when_message_contains_both_fields():
    assert google_audio.transcript_from_message(
        _message(interim="rozpracované", final="hotovo")
    ) == ("final", "hotovo")


def test_empty_or_unrelated_google_message_is_ignored():
    assert google_audio.transcript_from_message(SimpleNamespace()) is None
    assert google_audio.transcript_from_message(_message(interim="  ")) is None


class _Models:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _translator(responses):
    models = _Models(responses)
    return google_audio.GeminiPairTranslator(SimpleNamespace(models=models)), models


def test_interim_translation_uses_cheapest_model_and_one_pair_request():
    translator, models = _translator(
        [SimpleNamespace(parsed={"en": "Hello", "ru": "Привет"})]
    )

    pair = asyncio.run(translator.translate("Ahoj", "interim"))

    assert pair == {"en": "Hello", "ru": "Привет"}
    assert len(models.calls) == 1
    call = models.calls[0]
    assert call["model"] == "gemini-3.1-flash-lite"
    assert call["contents"] == "Ahoj"
    assert call["config"].thinking_config.thinking_level.value == "MINIMAL"
    assert call["config"].temperature is None
    assert call["config"].max_output_tokens == 2048
    assert call["config"].response_json_schema == google_audio.TRANSLATION_SCHEMA


def test_final_translation_uses_quality_model_and_bounded_retry(monkeypatch):
    class TransientError(RuntimeError):
        code = 503

    translator, models = _translator(
        [
            TransientError("temporary"),
            SimpleNamespace(text='{"en":"Done","ru":"Готово"}'),
        ]
    )
    monkeypatch.setattr(google_audio.asyncio, "sleep", lambda _: _no_wait())

    pair = asyncio.run(translator.translate("Hotovo", "final"))

    assert pair == {"en": "Done", "ru": "Готово"}
    assert [call["model"] for call in models.calls] == [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
    ]
    assert models.calls[0]["config"].thinking_config.thinking_level.value == "MINIMAL"
    assert models.calls[0]["config"].temperature is None
    assert models.calls[0]["config"].max_output_tokens == 8192


@pytest.mark.parametrize("status", [408, 429, 500, 501, 507, 599])
def test_retry_classification_covers_all_transient_http_statuses(status):
    assert google_audio._is_transient_error(SimpleNamespace(code=status))


def test_retry_classification_rejects_permanent_client_error():
    assert not google_audio._is_transient_error(SimpleNamespace(code=400))
    assert not google_audio._is_transient_error(ConnectionError("not an SDK status"))


async def _no_wait():
    return None


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(text="not json"),
        SimpleNamespace(parsed={"en": "Only one"}),
        SimpleNamespace(parsed={"en": "", "ru": "Пусто"}),
        SimpleNamespace(parsed={"en": "Hi", "ru": 42}),
        SimpleNamespace(parsed={"en": "Hi", "ru": "Привет", "extra": "no"}),
    ],
)
def test_pair_validation_rejects_malformed_or_partial_json(response):
    translator, _ = _translator([response])

    with pytest.raises(Exception):
        asyncio.run(translator.translate("Ahoj", "interim"))


def test_rapid_interims_are_coalesced_to_latest(monkeypatch):
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 0.015)
    monkeypatch.setattr(google_audio, "INTERIM_MIN_INTERVAL_SECONDS", 0.0)

    class Translator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, kind):
            self.calls.append((text, kind))
            return {"en": f"EN:{text}", "ru": f"RU:{text}"}

    async def scenario():
        emitted = []
        translator = Translator()

        async def emit(value):
            emitted.append(value)

        coordinator = google_audio.TranslationCoordinator(translator, emit)
        worker = asyncio.create_task(coordinator.run())
        coordinator.submit("interim", "a")
        await asyncio.sleep(0.002)
        coordinator.submit("interim", "ah")
        await asyncio.sleep(0.002)
        coordinator.submit("interim", "ahoj")
        await asyncio.sleep(0.030)
        coordinator.close()
        await worker
        return translator.calls, emitted

    calls, emitted = asyncio.run(scenario())
    assert calls == [("ahoj", "interim")]
    assert emitted == [{"type": "interim", "en": "EN:ahoj", "ru": "RU:ahoj"}]


def test_final_preempts_interim_and_is_never_split(monkeypatch):
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 1.0)

    class Translator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, kind):
            self.calls.append((text, kind))
            return {"en": "Final", "ru": "Финал"}

    async def scenario():
        emitted = []
        translator = Translator()

        async def emit(value):
            emitted.append(value)

        coordinator = google_audio.TranslationCoordinator(translator, emit)
        worker = asyncio.create_task(coordinator.run())
        coordinator.submit("interim", "roz")
        await asyncio.sleep(0)
        coordinator.submit("final", "rozhodnuto")
        await asyncio.sleep(0.01)
        coordinator.close()
        await worker
        return translator.calls, emitted

    calls, emitted = asyncio.run(scenario())
    assert calls == [("rozhodnuto", "final")]
    assert emitted == [{"type": "final", "en": "Final", "ru": "Финал"}]


def test_continuous_interims_cannot_starve_periodic_preview(monkeypatch):
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 0.010)
    monkeypatch.setattr(google_audio, "INTERIM_MIN_INTERVAL_SECONDS", 0.025)

    class Translator:
        async def translate(self, text, kind):
            # Slower than the hypothesis cadence: an implementation that
            # cancels every in-flight request will never emit while speech
            # remains continuous.
            await asyncio.sleep(0.020)
            return {"en": f"EN:{text}", "ru": f"RU:{text}"}

    async def scenario():
        emitted = []

        async def emit(value):
            emitted.append(value)

        coordinator = google_audio.TranslationCoordinator(Translator(), emit)
        worker = asyncio.create_task(coordinator.run())
        for index in range(24):
            coordinator.submit("interim", f"word-{index}")
            await asyncio.sleep(0.005)
        emitted_during_stream = len(emitted)
        await asyncio.sleep(0.040)
        coordinator.close()
        await worker
        return emitted_during_stream, emitted

    emitted_during_stream, emitted = asyncio.run(scenario())
    assert emitted_during_stream >= 3
    assert len(emitted) >= 3
    assert len(emitted) < 12
    assert emitted[-1] == {
        "type": "interim",
        "en": "EN:word-23",
        "ru": "RU:word-23",
    }


def test_newer_interim_keeps_inflight_snapshot_but_final_preempts_it(monkeypatch):
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 0.0)
    monkeypatch.setattr(google_audio, "INTERIM_MIN_INTERVAL_SECONDS", 0.0)

    class Translator:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = []

        async def translate(self, text, kind):
            if kind == "interim":
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.append(text)
                    raise
            return {"en": f"EN:{text}", "ru": f"RU:{text}"}

    async def newer_interim_scenario():
        emitted = []
        translator = Translator()

        async def emit(value):
            emitted.append(value)

        coordinator = google_audio.TranslationCoordinator(translator, emit)
        worker = asyncio.create_task(coordinator.run())
        coordinator.submit("interim", "first")
        await translator.started.wait()
        coordinator.submit("interim", "newest")
        await asyncio.sleep(0)
        assert translator.cancelled == []
        translator.release.set()
        await asyncio.sleep(0.010)
        coordinator.close()
        await worker
        return emitted

    async def final_scenario():
        emitted = []
        translator = Translator()

        async def emit(value):
            emitted.append(value)

        coordinator = google_audio.TranslationCoordinator(translator, emit)
        worker = asyncio.create_task(coordinator.run())
        coordinator.submit("interim", "unfinished")
        await translator.started.wait()
        coordinator.submit("final", "committed")
        await asyncio.sleep(0.010)
        coordinator.close()
        await worker
        return translator.cancelled, emitted

    assert asyncio.run(newer_interim_scenario()) == [
        {"type": "interim", "en": "EN:first", "ru": "RU:first"},
        {"type": "interim", "en": "EN:newest", "ru": "RU:newest"},
    ]
    cancelled, emitted = asyncio.run(final_scenario())
    assert cancelled == ["unfinished"]
    assert emitted == [{"type": "final", "en": "EN:committed", "ru": "RU:committed"}]


def test_failed_final_emits_sanitized_recoverable_error():
    class Translator:
        async def translate(self, text, kind):
            raise RuntimeError("secret upstream detail")

    async def scenario():
        emitted = []

        async def emit(value):
            emitted.append(value)

        coordinator = google_audio.TranslationCoordinator(Translator(), emit)
        worker = asyncio.create_task(coordinator.run())
        coordinator.submit("final", "text")
        await asyncio.sleep(0)
        coordinator.close()
        await worker
        return emitted

    assert asyncio.run(scenario()) == [
        {"type": "error", "code": "translation_failed", "recoverable": True}
    ]


def test_close_drains_every_committed_final_in_fifo_order():
    class Translator:
        async def translate(self, text, kind):
            await asyncio.sleep(0)
            return {"en": f"EN:{text}", "ru": f"RU:{text}"}

    async def scenario():
        emitted = []

        async def emit(value):
            emitted.append(value)

        coordinator = google_audio.TranslationCoordinator(Translator(), emit)
        worker = asyncio.create_task(coordinator.run())
        coordinator.submit("final", "one")
        coordinator.submit("final", "two")
        coordinator.submit("final", "three")
        coordinator.close()
        await worker
        return emitted

    assert asyncio.run(scenario()) == [
        {"type": "final", "en": "EN:one", "ru": "RU:one"},
        {"type": "final", "en": "EN:two", "ru": "RU:two"},
        {"type": "final", "en": "EN:three", "ru": "RU:three"},
    ]


def test_interim_failure_is_reported_without_retry_and_next_final_survives(monkeypatch):
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 0.0)

    class MissingModel(RuntimeError):
        code = 404

    translator, models = _translator(
        [
            MissingModel("private provider detail"),
            SimpleNamespace(parsed={"en": "Final", "ru": "Финал"}),
        ]
    )

    async def scenario():
        emitted = []
        failed = asyncio.Event()

        async def emit(value):
            emitted.append(value)
            if value["type"] == "error":
                failed.set()

        coordinator = google_audio.TranslationCoordinator(translator, emit)
        worker = asyncio.create_task(coordinator.run())
        coordinator.submit("interim", "private transcript")
        await asyncio.wait_for(failed.wait(), timeout=1)
        coordinator.submit("final", "hotovo")
        coordinator.close()
        await worker
        return emitted

    assert asyncio.run(scenario()) == [
        {"type": "error", "code": "interim_translation_failed", "recoverable": True},
        {"type": "final", "en": "Final", "ru": "Финал"},
    ]
    assert len(models.calls) == 2


def test_telemetry_measures_provider_time_not_browser_send_time(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(google_audio.time, "monotonic", lambda: clock[0])

    class Translator:
        async def translate(self, text, kind):
            clock[0] += 0.4
            return {"en": "Done", "ru": "Готово"}

    async def scenario():
        reports = []

        async def emit(value):
            clock[0] += 1.0  # A slow browser must not inflate provider latency.

        async def report(value):
            reports.append(value)

        coordinator = google_audio.TranslationCoordinator(Translator(), emit, report)
        coordinator.note_audio()
        clock[0] += 0.25
        coordinator.submit("final", "private transcript")
        clock[0] += 0.05
        coordinator.close()
        await coordinator.run()
        return reports

    reports = asyncio.run(scenario())
    assert [report["stage"] for report in reports] == ["translating", "translated"]
    assert reports[-1] == {
        "type": "telemetry",
        "kind": "final",
        "stage": "translated",
        "hypotheses": 0,
        "finals_received": 1,
        "first_stt_ms": 250,
        "queue_ms": 50,
        "translation_ms": 400,
    }
    assert "private transcript" not in str(reports)


def test_missing_audio_timestamp_does_not_fabricate_stt_latency():
    async def scenario():
        translator, _ = _translator(
            [SimpleNamespace(parsed={"en": "Hi", "ru": "Привет"})]
        )
        reports = []

        async def emit(value):
            pass

        async def report(value):
            reports.append(value)

        coordinator = google_audio.TranslationCoordinator(translator, emit, report)
        coordinator.submit("final", "Ahoj")
        coordinator.close()
        await coordinator.run()
        return reports

    assert all(report["first_stt_ms"] is None for report in asyncio.run(scenario()))
