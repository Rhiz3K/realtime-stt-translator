"""Fixed Google audio-to-transcript-to-translation pipeline.

The public application deliberately exposes no model, language, or buffering
settings.  This module is the one place where those cost/quality decisions live.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, TypedDict

from google.genai import types


TRANSCRIBE_MODEL = "gemini-3.5-transcribe-live"
INTERIM_TRANSLATION_MODEL = "gemini-3.1-flash-lite"
FINAL_TRANSLATION_MODEL = "gemini-3.5-flash-lite"

SOURCE_LANGUAGE = "cs-CZ"
AUDIO_MIME_TYPE = "audio/pcm;rate=16000"

# The browser emits exactly 100 ms frames (3,200 bytes), with one smaller tail
# frame on Stop.  Accepting larger frames would let a modified client purchase
# Google audio time faster than wall-clock time.
MAX_AUDIO_CHUNK_BYTES = 3_200
MAX_TRANSCRIPT_CHARS = 12_000
MAX_FINAL_BACKLOG = 32

# Fixed endpointing: prefer a pause, but don't repeatedly translate an
# ever-growing paragraph when a speaker (or background music) never goes quiet.
SEGMENT_MIN_SECONDS = 2.0
SEGMENT_TARGET_SECONDS = 6.0
SEGMENT_MAX_SECONDS = 12.0
# Do not reopen a manual activity while Google is still finalizing it. This
# acknowledgement gates the next activity only, never confirms a stream flush.
TURN_FINALIZATION_TIMEOUT_SECONDS = 1.0

# Translating every rapidly-changing ASR hypothesis wastes tokens and makes the
# UI flicker.  The first useful preview arrives quickly, then updates at most
# roughly once per second.  Finals always pre-empt this delay.
INTERIM_DEBOUNCE_SECONDS = 0.35
INTERIM_MIN_INTERVAL_SECONDS = 0.90
INTERIM_TIMEOUT_SECONDS = 6.0
FINAL_TIMEOUT_SECONDS = 15.0


class TranslationPair(TypedDict):
    en: str
    ru: str


TranscriptKind = Literal["interim", "final"]
EmitJson = Callable[[dict[str, object]], Awaitable[None]]


TRANSLATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "en": {"type": "string"},
        "ru": {"type": "string"},
    },
    "required": ["en", "ru"],
    "additionalProperties": False,
}

TRANSLATION_INSTRUCTION = """\
Translate the supplied Czech speech transcript faithfully and naturally into
English and Russian. Return JSON with exactly two string fields: en and ru.
Preserve names, numbers, intent, register, and formatting. Do not explain,
summarize, censor, or add facts. The text may be an unfinished live interim;
translate only the words present and never complete the speaker's thought.
"""


class TranslationFailure(RuntimeError):
    """The model did not return one valid English/Russian pair."""


class TranslationBacklogFull(RuntimeError):
    """Final results cannot be dropped without corrupting the transcript."""


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    status = getattr(exc, "code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    return status in {408, 429} or (isinstance(status, int) and 500 <= status < 600)


def live_transcription_config() -> types.LiveConnectConfig:
    """Return the fixed, clean Czech live-transcription configuration."""

    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        # Explicit, serialized activity boundaries keep interim recognition
        # alive without relying on repeated hybrid audio_stream_end resets.
        realtime_input_config=types.RealtimeInputConfig(
            activity_handling="NO_INTERRUPTION",
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,
            ),
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=[SOURCE_LANGUAGE],
            mode="SMART",
        ),
    )


class AudioSegmenter:
    """Request upstream finalization without dropping/replaying any PCM.

    This is a cheap endpoint hint, not an STT engine or a proof of silence.
    Only Google's final text is committed. Activity starts before the first
    PCM frame, including leading silence, so no prefix audio is clipped.
    Count samples rather than wall time so jitter cannot create extra turns.
    """

    def __init__(self) -> None:
        self._elapsed = 0.0
        self._quiet = 0.0
        self._envelope = 0.0
        self._active = False

    def boundary_after(self, audio: bytes) -> bool:
        duration = len(audio) / 32_000
        samples = len(audio) // 2
        if not samples:
            return False
        energy = sum(value * value for (value,) in struct.iter_unpack("<h", audio))
        rms = math.sqrt(energy / samples) / 32_768
        self._envelope = max(rms, self._envelope * math.exp(-duration / 3))
        threshold = max(0.003, min(0.04, self._envelope * 0.22))
        quiet = rms < threshold
        if not quiet:
            self._active = True
        if not self._active:
            return False
        self._elapsed += duration
        self._quiet = self._quiet + duration if quiet else 0.0
        pause = 0.2 if self._elapsed >= SEGMENT_TARGET_SECONDS else 0.4
        if self._elapsed + 1e-9 >= SEGMENT_MAX_SECONDS or (
            self._elapsed >= SEGMENT_MIN_SECONDS and self._quiet + 1e-9 >= pause
        ):
            self._elapsed = self._quiet = 0.0
            self._active = False
            return True
        return False


class GoogleAudioSession:
    """Ordered manual activities over one Google connection.

    The caller serializes send_audio/end_input and bounds transport sends.
    One concurrent receive iterator must stay alive to observe generation
    completion. Never reopen after a missing acknowledgement: fail visibly
    instead of silently accumulating one long turn or losing live previews.
    """

    def __init__(self, session: object) -> None:
        self._session = session
        self._segmenter = AudioSegmenter()
        self._activity_open = False
        self._turn_done = asyncio.Event()
        self._boundary_failed = False

    async def send_audio(self, audio: bytes) -> None:
        if self._boundary_failed:
            raise TimeoutError("Previous audio activity did not finish")
        if not self._activity_open:
            await self._session.send_realtime_input(
                activity_start=types.ActivityStart()
            )
            self._activity_open = True
        await self._session.send_realtime_input(
            audio=types.Blob(data=audio, mime_type=AUDIO_MIME_TYPE)
        )
        if self._segmenter.boundary_after(audio):
            # Leave the guard set on *any* failure/cancellation, including a
            # blocked end send whose delivery cannot safely be retried.
            self._boundary_failed = True
            await self.end_input()
            await asyncio.wait_for(
                self._turn_done.wait(), timeout=TURN_FINALIZATION_TIMEOUT_SECONDS
            )
            self._boundary_failed = False

    async def end_input(self) -> None:
        # Stop sends at most one end for the current activity. It does NOT
        # await this per-turn signal as evidence that all input was processed.
        if self._activity_open:
            self._turn_done.clear()
            await self._session.send_realtime_input(activity_end=types.ActivityEnd())
            self._activity_open = False

    async def receive(self):
        async for message in self._session.receive():
            content = getattr(message, "server_content", None)
            if getattr(content, "generation_complete", False):
                self._turn_done.set()
            yield message

    async def close(self) -> None:
        await self._session.close()


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def transcript_from_message(message: object) -> tuple[TranscriptKind, str] | None:
    """Extract one authoritative transcript event from a Live API message.

    A final wins if Google happens to put both fields in the same message.  This
    avoids paying for an interim translation that will be immediately replaced.
    """

    server_content = getattr(message, "server_content", None)
    if server_content is None:
        return None

    final = getattr(server_content, "input_transcription", None)
    final_text = _clean_text(getattr(final, "text", None))
    if final_text:
        return "final", final_text

    interim = getattr(server_content, "interim_input_transcription", None)
    interim_text = _clean_text(getattr(interim, "text", None))
    if interim_text:
        return "interim", interim_text
    return None


class GeminiPairTranslator:
    """Translate a Czech transcript into EN and RU in one structured call."""

    def __init__(self, async_client: object) -> None:
        self._client = async_client

    async def translate(self, text: str, kind: TranscriptKind) -> TranslationPair:
        model = (
            FINAL_TRANSLATION_MODEL if kind == "final" else INTERIM_TRANSLATION_MODEL
        )
        thinking = types.ThinkingConfig(thinking_level="MINIMAL")
        config = types.GenerateContentConfig(
            system_instruction=TRANSLATION_INSTRUCTION,
            # Both fixed models are Gemini 3.x: retain default sampling.
            # This is a ceiling, not reserved/billed output.  A generous final
            # limit prevents long utterances from becoming truncated JSON.
            max_output_tokens=8_192 if kind == "final" else 2_048,
            response_mime_type="application/json",
            response_json_schema=TRANSLATION_SCHEMA,
            thinking_config=thinking,
        )

        attempts = 2 if kind == "final" else 1
        for attempt in range(attempts):
            try:
                response = await asyncio.wait_for(
                    self._client.models.generate_content(
                        model=model,
                        contents=text,
                        config=config,
                    ),
                    timeout=(
                        FINAL_TIMEOUT_SECONDS
                        if kind == "final"
                        else INTERIM_TIMEOUT_SECONDS
                    ),
                )
                return self._parse_response(response)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt + 1 >= attempts or not _is_transient_error(exc):
                    raise
                # Retry only a committed final.  Retrying speculative interims
                # costs money and usually returns after a newer hypothesis.
                await asyncio.sleep(0.20)

        raise TranslationFailure("translation attempts exhausted")  # pragma: no cover

    @staticmethod
    def _parse_response(response: object) -> TranslationPair:
        value = getattr(response, "parsed", None)
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if not isinstance(value, dict):
            raw = getattr(response, "text", None)
            if not isinstance(raw, str):
                raise TranslationFailure("translation response has no JSON text")
            try:
                value = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise TranslationFailure(
                    "translation response is invalid JSON"
                ) from exc

        if set(value) != {"en", "ru"}:
            raise TranslationFailure("translation response has unexpected fields")
        en = _clean_text(value.get("en"))
        ru = _clean_text(value.get("ru"))
        if not en or not ru:
            raise TranslationFailure("translation response contains an empty language")
        return {"en": en, "ru": ru}


@dataclass(frozen=True, slots=True)
class _TranslationWork:
    kind: TranscriptKind
    text: str
    revision: int
    eligible_at: float | None
    received_at: float


class _LatestTranslationQueue:
    """Keep only the newest interim while never dropping committed finals."""

    def __init__(self) -> None:
        self._finals: deque[_TranslationWork] = deque()
        self._interim: _TranslationWork | None = None
        self._event = asyncio.Event()
        self._closed = False
        self.revision = 0

    def put(
        self, kind: TranscriptKind, text: str, eligible_at: float | None = None
    ) -> _TranslationWork:
        if self._closed:
            raise RuntimeError("translation queue is closed")
        self.revision += 1
        work = _TranslationWork(
            kind=kind,
            text=text,
            revision=self.revision,
            eligible_at=eligible_at,
            received_at=time.monotonic(),
        )
        if kind == "final":
            if len(self._finals) >= MAX_FINAL_BACKLOG:
                raise TranslationBacklogFull("final translation backlog is full")
            self._interim = None
            self._finals.append(work)
        else:
            self._interim = work
        self._event.set()
        return work

    async def get(self) -> _TranslationWork | None:
        while True:
            if self._finals:
                return self._finals.popleft()
            if self._interim is not None:
                work = self._interim
                self._interim = None
                return work
            if self._closed:
                return None
            self._event.clear()
            await self._event.wait()

    def clear_interim(self) -> None:
        self.revision += 1
        self._interim = None

    def close(self) -> None:
        self._closed = True
        self._interim = None
        self._event.set()


class TranslationCoordinator:
    """Cost-aware latest-interim/final translation actor for one audio session."""

    def __init__(
        self,
        translator: GeminiPairTranslator,
        emit: EmitJson,
        report: EmitJson | None = None,
    ) -> None:
        self._translator = translator
        self._emit = emit
        self._report = report
        self._hypotheses = 0
        self._finals_received = 0
        self._audio_started_at: float | None = None
        self._first_stt_ms: int | None = None
        self._queue = _LatestTranslationQueue()
        self._active_interim: asyncio.Task[None] | None = None
        self._active_interim_in_flight = False
        self._last_interim_text = ""
        self._last_interim_started_at = float("-inf")
        self._interim_deadline: float | None = None
        self._latest_final_revision = 0
        self._closing = False

    def note_audio(self) -> None:
        if self._audio_started_at is None:
            self._audio_started_at = time.monotonic()

    @property
    def has_input(self) -> bool:
        """Whether any audio was attempted or a transcript was received.

        Check actual input, not rounded UI seconds or detected speech: even a
        short/silent frame may still have an unconfirmed upstream result.
        """
        return (
            self._audio_started_at is not None
            or self._hypotheses > 0
            or self._finals_received > 0
        )

    async def _telemetry(
        self, kind: TranscriptKind, stage: str, **timings: int
    ) -> None:
        if self._report is not None:
            await self._report(
                {
                    "type": "telemetry",
                    "kind": kind,
                    "stage": stage,
                    "hypotheses": self._hypotheses,
                    "finals_received": self._finals_received,
                    "first_stt_ms": self._first_stt_ms,
                    **timings,
                }
            )

    def submit(self, kind: TranscriptKind, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if self._first_stt_ms is None and self._audio_started_at is not None:
            self._first_stt_ms = max(
                0, round((time.monotonic() - self._audio_started_at) * 1000)
            )
        if kind == "interim":
            self._hypotheses += 1
        else:
            self._finals_received += 1
        if kind == "interim" and len(text) > MAX_TRANSCRIPT_CHARS:
            text = text[:MAX_TRANSCRIPT_CHARS]
        if kind == "interim" and text == self._last_interim_text:
            return False

        if kind == "interim":
            self._last_interim_text = text
            # Preserve the first pending hypothesis' deadline across rapid
            # replacements. This is a throttle with latest-value coalescing,
            # not a trailing debounce that can starve continuous speech.
            if self._interim_deadline is None:
                now = time.monotonic()
                self._interim_deadline = max(
                    now + INTERIM_DEBOUNCE_SECONDS,
                    self._last_interim_started_at + INTERIM_MIN_INTERVAL_SECONDS,
                )
            eligible_at = self._interim_deadline
        else:
            self._last_interim_text = ""
            self._interim_deadline = None
            eligible_at = None

        work = self._queue.put(kind, text, eligible_at)
        if kind == "final":
            self._latest_final_revision = work.revision
        if self._active_interim and not self._active_interim.done():
            # Replacing a sleeping debounce is free. Once a paid request is in
            # flight, keep it as the periodic snapshot and retain just the
            # newest pending interim. A committed final always pre-empts it.
            if kind == "final" or not self._active_interim_in_flight:
                self._active_interim.cancel()
        return True

    def clear_interim(self) -> None:
        """Withdraw a browser hypothesis without discarding committed finals."""
        self._queue.clear_interim()
        self._last_interim_text = ""
        self._interim_deadline = None
        if self._active_interim and not self._active_interim.done():
            self._active_interim.cancel()

    def close(self) -> None:
        self._closing = True
        self._interim_deadline = None
        if self._active_interim and not self._active_interim.done():
            self._active_interim.cancel()
        self._queue.close()

    async def run(self) -> None:
        while (work := await self._queue.get()) is not None:
            task = asyncio.create_task(self._process(work))
            if work.kind == "interim":
                self._active_interim = task
            try:
                await task
            except asyncio.CancelledError:
                # A replacement may cancel a sleeping debounce; a final also
                # cancels an in-flight interim. Propagate actor shutdown.
                if asyncio.current_task() and asyncio.current_task().cancelling():
                    task.cancel()
                    raise
            finally:
                if self._active_interim is task:
                    self._active_interim = None

    async def _process(self, work: _TranslationWork) -> None:
        if work.kind == "interim":
            deadline = work.eligible_at or time.monotonic()
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            if work.revision != self._queue.revision or self._closing:
                return
            self._last_interim_started_at = time.monotonic()
            if self._interim_deadline == work.eligible_at:
                self._interim_deadline = None

        if work.kind == "interim":
            self._active_interim_in_flight = True
        try:
            queue_ms = max(0, round((time.monotonic() - work.received_at) * 1000))
            await self._telemetry(work.kind, "translating", queue_ms=queue_ms)
            started_at = time.monotonic()
            try:
                pair = await self._translator.translate(work.text, work.kind)
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._emit(
                    {
                        "type": "error",
                        "code": "translation_failed"
                        if work.kind == "final"
                        else "interim_translation_failed",
                        "recoverable": True,
                    }
                )
                await self._telemetry(work.kind, "failed", queue_ms=queue_ms)
                return

            if work.kind == "interim" and (
                self._closing or work.revision < self._latest_final_revision
            ):
                return
            translation_ms = max(0, round((time.monotonic() - started_at) * 1000))
            await self._emit({"type": work.kind, "en": pair["en"], "ru": pair["ru"]})
            await self._telemetry(
                work.kind,
                "translated",
                queue_ms=queue_ms,
                translation_ms=translation_ms,
            )
        finally:
            if work.kind == "interim":
                self._active_interim_in_flight = False
