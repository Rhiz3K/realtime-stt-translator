import asyncio
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
    assert google_audio.TRANSCRIBE_MODEL == "gemini-3.5-transcribe-live"
    assert google_audio.AUDIO_MIME_TYPE == "audio/pcm;rate=16000"


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
    assert call["model"] == "gemini-2.5-flash-lite"
    assert call["contents"] == "Ahoj"
    assert call["config"].thinking_config.thinking_budget == 0
    assert call["config"].temperature == 0
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
