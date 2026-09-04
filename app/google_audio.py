"""Fixed Google audio-to-transcript-to-translation pipeline.

The public application deliberately exposes no model, language, or buffering
settings.  This module is the one place where those cost/quality decisions live.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, TypedDict

from google.genai import types


TRANSCRIBE_MODEL = "gemini-3.5-transcribe-live"
INTERIM_TRANSLATION_MODEL = "gemini-2.5-flash-lite"
FINAL_TRANSLATION_MODEL = "gemini-3.5-flash-lite"

SOURCE_LANGUAGE = "cs-CZ"
AUDIO_MIME_TYPE = "audio/pcm;rate=16000"

# The browser emits exactly 100 ms frames (3,200 bytes), with one smaller tail
# frame on Stop.  Accepting larger frames would let a modified client purchase
# Google audio time faster than wall-clock time.
MAX_AUDIO_CHUNK_BYTES = 3_200
MAX_TRANSCRIPT_CHARS = 12_000
MAX_FINAL_BACKLOG = 32

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
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=[SOURCE_LANGUAGE],
            mode="SMART",
        ),
    )


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
        thinking = (
            types.ThinkingConfig(thinking_level="MINIMAL")
            if kind == "final"
            else types.ThinkingConfig(thinking_budget=0)
        )
        config = types.GenerateContentConfig(
            system_instruction=TRANSLATION_INSTRUCTION,
            # Gemini 3.x quality is tuned for its default sampling.  Keep the
            # deterministic override only on the older, disposable interim.
            temperature=0 if kind == "interim" else None,
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

    def close(self) -> None:
        self._closed = True
        self._interim = None
        self._event.set()


class TranslationCoordinator:
    """Cost-aware latest-interim/final translation actor for one audio session."""

    def __init__(self, translator: GeminiPairTranslator, emit: EmitJson) -> None:
        self._translator = translator
        self._emit = emit
        self._queue = _LatestTranslationQueue()
        self._active_interim: asyncio.Task[None] | None = None
        self._active_interim_in_flight = False
        self._last_interim_text = ""
        self._last_interim_started_at = float("-inf")
        self._interim_deadline: float | None = None
        self._latest_final_revision = 0
        self._closing = False

    def submit(self, kind: TranscriptKind, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
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
            try:
                pair = await self._translator.translate(work.text, work.kind)
            except asyncio.CancelledError:
                raise
            except Exception:
                if work.kind == "final":
                    await self._emit(
                        {
                            "type": "error",
                            "code": "translation_failed",
                            "recoverable": True,
                        }
                    )
                return

            if work.kind == "interim" and (
                self._closing or work.revision < self._latest_final_revision
            ):
                return
            await self._emit({"type": work.kind, "en": pair["en"], "ru": pair["ru"]})
        finally:
            if work.kind == "interim":
                self._active_interim_in_flight = False
