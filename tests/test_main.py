import types
import base64
import json
import asyncio
import gc
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.main as main


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech", "whisper", "nemotron", "deepgram", "elevenlabs", "azure"})
    # Translation-provider state is process-wide (rate-limit cooldowns); give every
    # test a clean, googletrans-only default so switch-over tests opt in explicitly.
    monkeypatch.setattr(main, "DEEPL_API_KEY", "")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "auto")
    monkeypatch.setattr(main, "_PROVIDER_COOLDOWN_UNTIL", {})
    return TestClient(main.app)


def _assert_login_h1(html: str) -> None:
    assert re.search(r"<h1[^>]*>Sign in</h1>", html)


def test_get_index_requires_password(client):
    resp = client.get("/")
    assert resp.status_code == 200
    _assert_login_h1(resp.text)
    assert "Incorrect password" not in resp.text


def test_query_password_does_not_attempt_login(client):
    resp = client.get("/?pwd=wrong")
    assert resp.status_code == 200
    _assert_login_h1(resp.text)
    assert "Incorrect password" not in resp.text


def test_query_password_cannot_authenticate(client):
    resp = client.get("/?pwd=test-password", follow_redirects=False)
    assert resp.status_code == 200
    _assert_login_h1(resp.text)


def test_post_password_serves_index_html(client):
    resp = client.post(
        "/login",
        data={"password": "test-password", "next": "/"},
        follow_redirects=False,
    )
    assert resp.status_code == 303

    resp = client.get("/")
    assert resp.status_code == 200
    assert "<title>Live Translator</title>" in resp.text


def test_get_deepgram_always_redirects_to_index(client):
    """The /deepgram legacy endpoint now always redirects to /."""
    resp = client.get("/deepgram", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers.get("location") == "/"


class _FakeTranslation:
    def __init__(self, text: str):
        self.text = text


def test_latest_interim_queue_coalesces_interims_and_prioritizes_finals():
    async def scenario():
        queue = main.LatestInterimQueue()
        queue.put(main.TranslationWork("a", "interim", "cs", ["en"]))
        queue.put(main.TranslationWork("ab", "interim", "cs", ["en"]))
        assert (await queue.get()).text == "ab"

        queue.put(main.TranslationWork("abc?", "interim", "cs", ["en"]))
        queue.put(main.TranslationWork("abc", "final", "cs", ["en"]))
        queue.put(main.TranslationWork("def", "final", "cs", ["en"]))
        assert (await queue.get()).text == "abc"
        assert (await queue.get()).text == "def"

    asyncio.run(scenario())


def test_deepgram_result_queue_never_evicts_finals_for_interims():
    async def scenario():
        queue = main.LatestTranscriptQueue()
        queue.put({"transcript": "final-1", "is_final": True})
        queue.put({"transcript": "interim-1", "is_final": False})
        queue.put({"transcript": "interim-2", "is_final": False})
        queue.put({"transcript": "final-2", "is_final": True})
        assert (await queue.get())["transcript"] == "final-1"
        assert (await queue.get())["transcript"] == "final-2"

    asyncio.run(scenario())


def test_deepgram_result_queue_rejects_overflow_without_evicting_final():
    async def scenario():
        queue = main.LatestTranscriptQueue(max_finals=1)
        assert queue.put({"transcript": "keep", "is_final": True}) is True
        assert queue.put({"transcript": "reject", "is_final": True}) is False
        assert (await queue.get())["transcript"] == "keep"

    asyncio.run(scenario())


def test_translation_http_error_is_not_mistaken_for_a_valid_result():
    class FailedResponse:
        status_code = 503

        def __bool__(self):
            # httpx.Response uses status-dependent truthiness.
            return False

    result = _FakeTranslation("source text echoed by upstream")
    result._response = FailedResponse()

    with pytest.raises(RuntimeError, match="HTTP 503"):
        main._validate_translation_result(result)


def test_ws_translates_text(client, monkeypatch):
    class FakeAsyncTranslator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, src, dest):
            self.calls.append((text, src, dest))
            return _FakeTranslation(f"{dest}:{text}")

    fake_translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: fake_translator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_text("Ahoj")
        data = ws.receive_json()

    assert data == {"original": "Ahoj", "en": "en:Ahoj", "ru": "ru:Ahoj"}
    assert fake_translator.calls == [("Ahoj", "cs", "en"), ("Ahoj", "cs", "ru")]


def test_ws_typed_translates_single_dest(client, monkeypatch):
    class FakeAsyncTranslator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, src, dest):
            self.calls.append((text, src, dest))
            return _FakeTranslation(f"{dest}:{text}")

    fake_translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: fake_translator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"]}})
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["original"] == "Ahoj"
    assert data["dests"] == ["en"]
    assert data["translations"] == {"en": "en:Ahoj"}
    assert isinstance(data.get("timing", {}), dict)
    assert fake_translator.calls == [("Ahoj", "cs", "en")]


def test_ws_empty_text_does_not_call_translator(client, monkeypatch):
    class FakeAsyncTranslator:
        async def translate(self, *_args, **_kwargs):
            raise AssertionError("translate should not be called for empty input")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_text("   ")
        data = ws.receive_json()

    assert data == {"original": "", "en": "", "ru": ""}


def test_ws_ping_pong(client, monkeypatch):
    """Server responds to keepalive ping with pong."""

    class FakeAsyncTranslator:
        async def translate(self, *_args, **_kwargs):
            raise AssertionError("translate should not be called for ping")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "ping"})
        data = ws.receive_json()

    assert data == {"type": "pong"}


def test_ws_cancels_superseded_interim_before_final(client, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class SlowTranslator:
        async def translate(self, text, src, dest):
            if text == "a":
                started.set()
                await asyncio.to_thread(release.wait)
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", SlowTranslator)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "interim", "text": "a", "dests": ["en"]})
        assert started.wait(timeout=1)
        ws.send_json({"type": "interim", "text": "ab", "dests": ["en"]})
        ws.send_json({"type": "final", "text": "abc", "dests": ["en"]})
        release.set()
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["original"] == "abc"
    assert data["translations"] == {"en": "en:abc"}


def test_ws_rejects_oversized_text_without_translation(client, monkeypatch):
    class FakeAsyncTranslator:
        async def translate(self, *_args, **_kwargs):
            raise AssertionError("oversized text must not be translated")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)
    monkeypatch.setattr(main, "MAX_TEXT_LENGTH", 3)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "final", "text": "four", "dests": ["en"]})
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["original"] == ""
    assert data["translations"] == {"en": ""}
    assert data["error"] == "text_too_long"


def test_ws_elevenlabs_missing_api_key_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "")

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        payload = ws.receive_json()

    assert payload == {"error": "ELEVENLABS_API_KEY not configured"}


def test_ws_deepgram_missing_api_key_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "")

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        payload = ws.receive_json()

    assert payload == {"error": "DEEPGRAM_API_KEY not configured"}


def test_ws_deepgram_init_failure_sends_error(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")

    class BoomDeepgramClient:
        def __init__(self, api_key):
            raise RuntimeError("boom")

    monkeypatch.setattr(main, "DeepgramClient", BoomDeepgramClient)

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        # The endpoint blocks on the first client message (config or audio) before
        # it constructs DeepgramClient, so send a config to trigger the failing init.
        # Without this the server never reaches the boom and both sides deadlock.
        ws.send_json({"type": "config", "deepgram": {"language": "cs"}})
        payload = ws.receive_json()

    # Generic code out; the exception detail stays in the server log.
    assert payload == {"error": "server_error"}


def test_ws_deepgram_missing_sdk_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setattr(main, "DeepgramClient", None)

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        payload = ws.receive_json()

    assert payload == {"error": "deepgram-sdk not installed"}


def test_ws_deepgram_happy_path_delivers_translated_final(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")

    class FakeAlt:
        def __init__(self, transcript: str):
            self.transcript = transcript

    class FakeChannel:
        def __init__(self, transcript: str):
            self.alternatives = [FakeAlt(transcript)]

    class FakeListenV1Results:
        def __init__(self, transcript: str, is_final: bool):
            self.channel = FakeChannel(transcript)
            self.is_final = is_final

    class FakeAsyncTranslator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, src, dest):
            self.calls.append((text, src, dest))
            return _FakeTranslation(f"{dest}:{text}")

    translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: translator)

    class FakeDgSocket:
        def __init__(self):
            self._handlers = {}
            self.sent_media = []
            self.finalized = False
            self.closed = False

        def on(self, event_type, callback):
            self._handlers[event_type] = callback

        def start_listening(self):
            msg_cb = self._handlers.get(main.EventType.MESSAGE)
            if msg_cb:
                msg_cb(FakeListenV1Results("prubezne", False))
                msg_cb(FakeListenV1Results("finalni", True))

        def send_media(self, data):
            self.sent_media.append(data)

        def send_finalize(self, _message=None):
            self.finalized = True

        def send_close_stream(self, _message=None):
            self.closed = True

    class _FakeSocketIterator:
        def __init__(self, socket):
            self._socket = socket
            self._sent = False
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._sent:
                raise StopIteration
            self._sent = True
            return self._socket

        def close(self):
            self.closed = True

    fake_socket = FakeDgSocket()
    fake_iter = _FakeSocketIterator(fake_socket)

    class FakeDeepgramClient:
        def __init__(self, api_key):
            self.api_key = api_key

            class _V1:
                def connect(self, **_kwargs):
                    return fake_iter

            class _Listen:
                v1 = _V1()

            self.listen = _Listen()

    monkeypatch.setattr(main, "DeepgramClient", FakeDeepgramClient)

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        ws.send_bytes(b"\x00\x01")
        # The listener thread hands the interim and the final to the loop as two
        # separate callbacks, so whether the worker drains the interim before the
        # final coalesces it away is pure scheduling — accept either ordering and
        # read until the final. Coalescing itself is pinned deterministically by
        # the LatestTranscriptQueue unit tests above.
        final = None
        for _ in range(2):
            message = ws.receive_json()
            if message["type"] == "final":
                final = message
                break
            assert message["type"] == "interim"
            assert message["original"] == "prubezne"

    assert final is not None, "no final transcript arrived"
    assert final["type"] == "final"
    assert final["original"] == "finalni"
    assert final["dests"] == ["en", "ru"]
    assert final["translations"] == {"en": "en:finalni", "ru": "ru:finalni"}
    assert final["en"] == "en:finalni"
    assert final["ru"] == "ru:finalni"
    assert isinstance(final.get("timing", {}), dict)
    assert translator.calls == [("finalni", "cs", "en"), ("finalni", "cs", "ru")]
    assert fake_socket.sent_media == [b"\x00\x01"]


def test_ws_elevenlabs_happy_path_translates_final_without_blocking_receiver(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "test-key")

    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    class FakeElevenSocket:
        def __init__(self):
            self.events = iter(
                [
                    '{"message_type":"partial_transcript","text":"ah"}',
                    '{"message_type":"committed_transcript","text":"ahoj"}',
                ]
            )
            self.closed = False

        async def recv(self):
            return '{"message_type":"session_started","session_id":"session-1"}'

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

        async def send(self, _payload):
            pass

        async def close(self):
            self.closed = True

    upstream = FakeElevenSocket()
    connected = {}

    async def fake_connect(url, additional_headers):
        connected["url"] = url
        connected["headers"] = additional_headers
        return upstream

    monkeypatch.setattr(main.ws_lib, "connect", fake_connect)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        ws.send_json(
            {
                "type": "config",
                "elevenlabs": {"language_code": "cs-CZ", "commit_strategy": "manual"},
                "translate": {"src": "cs", "dests": ["en"]},
                "translate_interim": True,
            }
        )
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["original"] == "ahoj"
    assert data["translations"] == {"en": "en:ahoj"}
    assert "language_code=cs-CZ" in connected["url"]
    assert connected["headers"] == {"xi-api-key": "test-key"}
    assert upstream.closed is True


def _elevenlabs_socket_emitting(events):
    """Fake Scribe upstream that replays `events` (raw JSON strings) once."""

    class FakeElevenSocket:
        def __init__(self):
            self.events = iter(events)
            self.closed = False

        async def recv(self):
            return '{"message_type":"session_started","session_id":"session-1"}'

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

        async def send(self, _payload):
            pass

        async def close(self):
            self.closed = True

    return FakeElevenSocket()


def _connect_fake_elevenlabs(monkeypatch, upstream):
    async def fake_connect(_url, additional_headers):
        return upstream

    monkeypatch.setattr(main.ws_lib, "connect", fake_connect)


def test_ws_elevenlabs_transient_error_warns_without_ending_session(client, monkeypatch):
    """A commit during silence must not tear down the whole STT session."""
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "test-key")

    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    upstream = _elevenlabs_socket_emitting(
        [
            '{"message_type":"insufficient_audio_activity","error":"no speech detected"}',
            '{"message_type":"committed_transcript","text":"ahoj"}',
        ]
    )
    _connect_fake_elevenlabs(monkeypatch, upstream)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"]}})
        warning = ws.receive_json()
        final = ws.receive_json()

    # Typed payload -> the browser shows it but keeps recording.
    assert warning["type"] == "warning"
    assert warning["error"] == "ElevenLabs: no speech detected"
    # The session survived: the transcript after the warning still arrives.
    assert final["type"] == "final"
    assert final["original"] == "ahoj"
    assert final["translations"] == {"en": "en:ahoj"}


def test_ws_elevenlabs_fatal_error_ends_session(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "test-key")

    upstream = _elevenlabs_socket_emitting(
        [
            '{"message_type":"quota_exceeded","error":"out of credits"}',
            '{"message_type":"committed_transcript","text":"nikdy"}',
        ]
    )
    _connect_fake_elevenlabs(monkeypatch, upstream)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"]}})
        error = ws.receive_json()

    # Typeless error -> terminal by convention: the browser stops the session on it.
    assert "type" not in error
    assert error["error"] == "ElevenLabs: out of credits"
    # The server stopped reading upstream, so the transcript queued behind the error
    # was never processed. (The socket itself stays open until the client leaves —
    # teardown is client-driven, which is what the typeless-error convention buys.)
    assert next(upstream.events, None) is not None
    assert upstream.closed is True


# --- /api/elevenlabs/token tests ---


def test_elevenlabs_token_requires_auth(client):
    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 401


def test_elevenlabs_token_missing_api_key(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "")

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 400
    assert "No ElevenLabs API key" in resp.json()["detail"]


def test_elevenlabs_token_uses_env_key(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "xi-env-key")

    import httpx

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"token": "tok_abc123"}

        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            assert kwargs.get("headers", {}).get("xi-api-key") == "xi-env-key"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeAsyncClient())

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 200
    assert resp.json() == {"token": "tok_abc123"}


def test_elevenlabs_token_uses_client_key(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "xi-env-key")

    import httpx

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"token": "tok_client"}

        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            # Should use the client-provided key, not env key.
            assert kwargs.get("headers", {}).get("xi-api-key") == "xi-my-key"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeAsyncClient())

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={"api_key": "xi-my-key"})
    assert resp.status_code == 200
    assert resp.json() == {"token": "tok_client"}


# --- AUTH_ENABLED tests ---


def test_auth_disabled_skips_login(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "APP_PASSWORD", "")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech"})

    c = TestClient(main.app)
    resp = c.get("/")
    assert resp.status_code == 200
    assert "<title>Live Translator</title>" in resp.text


def test_auth_disabled_ws_no_cookie_needed(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "APP_PASSWORD", "")

    c = TestClient(main.app)
    with c.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "ping"})
        data = ws.receive_json()
        assert data == {"type": "pong"}


def test_auth_disabled_still_rejects_cross_origin_ws(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "APP_PASSWORD", "")

    c = TestClient(main.app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with c.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
            pass
    assert exc_info.value.code == 1008


def test_disabled_vendor_engines_are_rejected_server_side(client, monkeypatch):
    client.post("/login", data={"password": "test-password", "next": "/"})
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech"})
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "server-key")
    monkeypatch.setattr(main, "AZURE_SPEECH_KEY", "azure-key")
    monkeypatch.setattr(main, "AZURE_SPEECH_REGION", "westeurope")

    assert client.post("/api/elevenlabs/token", json={}).status_code == 404
    assert client.post("/api/azure/token", json={}).status_code == 404

    for path in ("/ws/deepgram", "/ws/elevenlabs"):
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(path, headers={"origin": "http://testserver"}):
                pass
        assert exc_info.value.code == 1008


# --- ENABLED_ENGINES tests ---


def test_enabled_engines_passed_to_template(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech", "deepgram"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    # webspeech and deepgram should NOT have disabled attribute
    assert 'value="webspeech" ' in resp.text  # not disabled
    assert 'value="deepgram" ' in resp.text    # not disabled
    # elevenlabs should be disabled
    assert 'value="elevenlabs" disabled' in resp.text


def test_enabled_engines_includes_whisper_in_template(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"whisper"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    assert 'value="whisper" ' in resp.text
    # Engine module is loaded on demand via dynamic import(); ensure there is no
    # eager <script src="..."> tag pulling it during initial page load.
    assert '<script src="/static/whisper/whisper-engine.mjs"' not in resp.text


def test_static_whisper_worklet_served(client):
    resp = client.get("/static/whisper/pcm-worklet.js")
    assert resp.status_code == 200
    assert "Int16PCMProcessor" in resp.text
    # /static/whisper/* must always revalidate so browsers can't pin a stale
    # (possibly ABI-incompatible) engine copy across deploys.
    assert resp.headers.get("cache-control") == "no-cache"


def test_static_whisper_engine_served(client):
    resp = client.get("/static/whisper/whisper-engine.mjs")
    assert resp.status_code == 200
    assert "WhisperLocalEngine" in resp.text
    assert resp.headers.get("cache-control") == "no-cache"


def test_csp_allows_whisper_runtime_sources(client):
    # The local Whisper engine needs WASM execution plus Transformers.js/ONNX from
    # jsdelivr and the model weights from Hugging Face. The CSP must permit these.
    resp = client.get("/health")
    csp = resp.headers.get("Content-Security-Policy", "")
    assert "'wasm-unsafe-eval'" in csp
    assert "https://cdn.jsdelivr.net" in csp
    assert "https://huggingface.co" in csp


def test_cross_origin_isolation_headers(client):
    # COOP+COEP make the page cross-origin isolated, which enables SharedArrayBuffer
    # and lets ONNX Runtime Web run the Whisper WASM/CPU path multi-threaded.
    resp = client.get("/health")
    assert resp.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert resp.headers.get("Cross-Origin-Embedder-Policy") == "credentialless"


def test_permissions_policy_allows_same_origin_microphone_and_local_speech(client):
    policy = client.get("/health").headers.get("Permissions-Policy", "")
    assert "microphone=(self)" in policy
    assert "on-device-speech-recognition=(self)" in policy


def test_enabled_engines_includes_nemotron_in_template(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"nemotron"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    assert 'value="nemotron" ' in resp.text


def test_static_nemotron_engine_served(client):
    resp = client.get("/static/nemotron/nemotron-engine.mjs")
    assert resp.status_code == 200
    assert "NemotronLocalEngine" in resp.text
    # Engine code revalidates like Whisper's so a stale copy can't be pinned.
    assert resp.headers.get("cache-control") == "no-cache"


def test_missing_nemotron_model_assets_are_not_cached_immutably(client):
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)
    # A user can hit Nemotron before the background prepare job finishes. Never
    # let a browser pin that transient 404 as an immutable model response.
    resp = client.get("/static/nemotron/models/__missing_model_asset__.json")
    assert resp.status_code == 404
    assert resp.headers.get("cache-control") == "no-store"


def test_existing_nemotron_model_assets_cached_immutably(client):
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)
    path = Path("app/static/nemotron/models/__cache_test__.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok")
    try:
        resp = client.get("/static/nemotron/models/__cache_test__.txt")
    finally:
        path.unlink(missing_ok=True)

    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "public, max-age=31536000, immutable"


def test_csp_allows_nemotron_onnxruntime(client):
    # The local Nemotron engine loads onnxruntime-web 1.20.1 (encoder on WebGPU,
    # decoder on WASM); the CSP must permit that pinned build.
    resp = client.get("/health")
    csp = resp.headers.get("Content-Security-Policy", "")
    assert "onnxruntime-web@1.20.1" in csp


def test_enabled_engines_default_webspeech_only(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    assert 'value="deepgram" disabled' in resp.text
    assert 'value="elevenlabs" disabled' in resp.text


def test_translate_languages_requires_auth_and_returns_sorted_choices(client):
    assert client.get("/api/translate/languages").status_code == 401
    client.post("/login", data={"password": "test-password", "next": "/"})

    response = client.get("/api/translate/languages")
    assert response.status_code == 200
    languages = response.json()["languages"]
    assert any(item["code"] == "en" for item in languages)
    assert languages == sorted(languages, key=lambda item: (item["name"], item["code"]))


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (None, "/"),
        ("", "/"),
        ("https://evil.example", "/"),
        ("//evil.example", "/"),
        # Browsers normalize "\" to "/" in Location, so these are protocol-relative too.
        ("/\\evil.example", "/"),
        ("/\\\\evil.example", "/"),
        ("\\/evil.example", "/"),
        # A newline would let the value split the redirect header.
        ("/safe\r\nX-Injected: 1", "/"),
        ("/safe/path?x=1", "/safe/path?x=1"),
    ],
)
def test_sanitize_next_path(candidate, expected):
    assert main.sanitize_next_path(candidate) == expected


def test_auth_token_expires(monkeypatch):
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "AUTH_TOKEN_TTL_SECONDS", 10)
    monkeypatch.setattr(main.time, "time", lambda: 100)
    token = main.create_auth_token()
    assert main.verify_auth_token(token) is True

    monkeypatch.setattr(main.time, "time", lambda: 111)
    assert main.verify_auth_token(token) is False


# --- /health endpoint ---


def test_health_endpoint_no_auth_needed(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- Rate limiting on /login ---


def test_login_rate_limiting(client, monkeypatch):
    # Reset rate limiter state.
    monkeypatch.setattr(main, "_LOGIN_ATTEMPTS", {})
    monkeypatch.setattr(main, "_LOGIN_MAX_ATTEMPTS", 3)

    for _ in range(3):
        resp = client.post(
            "/login",
            data={"password": "wrong", "next": "/"},
            follow_redirects=False,
        )
        assert resp.status_code == 200  # renders login form

    resp = client.post(
        "/login",
        data={"password": "wrong", "next": "/"},
        follow_redirects=False,
    )
    assert resp.status_code == 429


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_rendered_template_inline_javascript_parses(client, tmp_path):
    """index.html is ~5k lines of inline JS that nothing else syntax-checks.

    Parsing the *rendered* output (not the raw template) is what makes this
    possible — Jinja placeholders would otherwise break the parse.
    """
    client.post("/login", data={"password": "test-password", "next": "/"})
    html = client.get("/").text

    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    assert blocks, "no inline <script> blocks found in the rendered page"

    checked = 0
    for index, block in enumerate(blocks):
        if not block.strip():
            continue
        path = tmp_path / f"block_{index}.js"
        path.write_text(block)
        proc = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"inline script block {index} is not valid JS:\n{proc.stderr}"
        checked += 1
    assert checked, "no non-empty inline script blocks were checked"


# --- Wave A: env parsing / normalization helpers ---


@pytest.mark.parametrize(
    "raw,expected",
    [("5", 5), ("0", 0), ("-1", 7), ("abc", 7), ("", 7), ("1.5", 7)],
)
def test_env_int_falls_back_to_default_on_bad_input(monkeypatch, raw, expected):
    monkeypatch.setenv("SRLT_TEST_INT", raw)
    assert main._env_int("SRLT_TEST_INT", 7) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("2.5", 2.5), ("0", 7.0), ("-3", 7.0), ("abc", 7.0), ("", 7.0)],
)
def test_env_float_falls_back_to_default_on_bad_input(monkeypatch, raw, expected):
    monkeypatch.setenv("SRLT_TEST_FLOAT", raw)
    assert main._env_float("SRLT_TEST_FLOAT", 7.0) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("en", None),                      # not a list
        (["en", "ru", "de"], ["en", "ru"]),  # capped at two
        ([1, "en"], ["en"]),               # non-strings skipped
        ([""], None),                      # blanks are not dests
        (["  EN  "], ["en"]),              # trimmed + lowercased
        (["en", "en"], ["en", "ru"]),      # duplicate pair rewritten
        (["ru", "ru"], ["ru", "en"]),
    ],
)
def test_normalize_translate_dests(value, expected):
    assert main._normalize_translate_dests(value) == expected


# --- Wave A: auth token internals ---


def test_verify_auth_token_rejects_malformed_tokens(monkeypatch):
    monkeypatch.setattr(main, "AUTH_SECRET", "unit-test-secret")

    valid = main.create_auth_token()
    payload_b64, sig_b64 = valid.split(".")
    assert main.verify_auth_token(valid) is True

    assert main.verify_auth_token(None) is False
    assert main.verify_auth_token("") is False
    assert main.verify_auth_token("no-dot") is False
    assert main.verify_auth_token("a.b.c") is False
    assert main.verify_auth_token(f"{payload_b64}.{sig_b64}x") is False  # bad signature

    # Correctly signed, but the payload is not decodable JSON.
    junk = main._b64url_encode(b"not-json")
    assert main.verify_auth_token(f"{junk}.{main._sign(junk)}") is False

    # Correctly signed JSON, but `exp` is not an int.
    bad_exp = main._b64url_encode(b'{"exp":"soon"}')
    assert main.verify_auth_token(f"{bad_exp}.{main._sign(bad_exp)}") is False


def test_signing_without_a_secret_never_validates(monkeypatch):
    monkeypatch.setattr(main, "AUTH_SECRET", "")
    assert main._sign("payload") == ""
    assert main.verify_auth_token("payload.signature") is False


def test_is_origin_allowed_uses_allowlist_when_configured(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
    assert main.is_origin_allowed("https://a.example", "anything") is True
    assert main.is_origin_allowed("https://b.example", "anything") is True
    assert main.is_origin_allowed("https://evil.example", "anything") is False

    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    # Without an allowlist the origin host must match the Host header.
    assert main.is_origin_allowed("https://host.example", "host.example") is True
    assert main.is_origin_allowed("https://other.example", "host.example") is False
    assert main.is_origin_allowed(None, "host.example") is False
    assert main.is_origin_allowed("https://host.example", None) is False


# --- Wave A: /login and unconfigured-server paths ---


def test_login_rejects_cross_origin_post(client):
    resp = client.post(
        "/login",
        data={"password": "test-password", "next": "/"},
        headers={"origin": "https://evil.example"},
        follow_redirects=False,
    )
    assert resp.status_code == 403


def test_login_rejects_cross_origin_referer(client):
    resp = client.post(
        "/login",
        data={"password": "test-password", "next": "/"},
        headers={"referer": "https://evil.example/page"},
        follow_redirects=False,
    )
    assert resp.status_code == 403


def test_login_without_app_password_is_a_server_error(client, monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "")
    resp = client.post("/login", data={"password": "x"}, follow_redirects=False)
    assert resp.status_code == 500


def test_index_without_app_password_is_a_server_error(client, monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "")
    resp = client.get("/")
    assert resp.status_code == 500


def test_translate_languages_without_app_password_is_a_server_error(client, monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "")
    resp = client.get("/api/translate/languages")
    assert resp.status_code == 500


def test_translate_languages_falls_back_when_googletrans_lacks_languages(client, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "googletrans", types.SimpleNamespace())
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.get("/api/translate/languages")
    assert resp.status_code == 200
    assert resp.json() == {"languages": []}


# --- Wave A: parakeet static cache headers (mirrors the nemotron pair) ---


def test_missing_parakeet_model_assets_are_not_cached_immutably(client):
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)
    resp = client.get("/static/parakeet/models/__missing_model_asset__.json")
    assert resp.status_code == 404
    assert resp.headers.get("cache-control") == "no-store"


def test_existing_parakeet_model_assets_cached_immutably(client):
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)
    path = Path("app/static/parakeet/models/__cache_test__.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok")
    try:
        resp = client.get("/static/parakeet/models/__cache_test__.txt")
    finally:
        path.unlink(missing_ok=True)

    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "public, max-age=31536000, immutable"


def test_parakeet_engine_module_is_revalidated(client):
    resp = client.get("/static/parakeet/parakeet-engine.mjs")
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "no-cache"


# --- Wave A: queue micro-branches ---


def test_latest_transcript_queue_serves_and_clears_a_lone_interim():
    async def scenario():
        queue = main.LatestTranscriptQueue()
        assert queue.empty() is True
        queue.put({"transcript": "interim", "is_final": False})
        assert queue.empty() is False
        assert (await queue.get())["transcript"] == "interim"
        assert queue.empty() is True

    asyncio.run(scenario())


def test_latest_interim_queue_rejects_puts_after_close():
    async def scenario():
        queue = main.LatestInterimQueue()
        queue.close()
        assert queue.put(main.TranslationWork("a", "final", "cs", ["en"])) is False
        assert await queue.get() is None

    asyncio.run(scenario())


def test_latest_interim_queue_rejects_finals_past_the_cap():
    async def scenario():
        queue = main.LatestInterimQueue(max_finals=1)
        assert queue.put(main.TranslationWork("keep", "final", "cs", ["en"])) is True
        assert queue.put(main.TranslationWork("reject", "final", "cs", ["en"])) is False
        assert (await queue.get()).text == "keep"

    asyncio.run(scenario())


# --- Wave A: POST /api/azure/token ---


def _fake_httpx_post(monkeypatch, handler):
    """Route httpx.AsyncClient(...).post through `handler(url, **kwargs)`."""
    import httpx

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, url, **kwargs):
            return handler(url, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: FakeAsyncClient())


class _FakeTextResponse:
    def __init__(self, text="", status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        pass


def test_azure_token_requires_auth(client):
    assert client.post("/api/azure/token", json={}).status_code == 401


def test_azure_token_returns_404_when_engine_disabled(client, monkeypatch):
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech"})
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)
    assert client.post("/api/azure/token", json={}).status_code == 404


def test_azure_token_uses_env_key_and_region(client, monkeypatch):
    monkeypatch.setattr(main, "AZURE_SPEECH_KEY", "env-key")
    monkeypatch.setattr(main, "AZURE_SPEECH_REGION", "westeurope")
    seen = {}

    def handler(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers", {})
        return _FakeTextResponse("tok-from-azure")

    _fake_httpx_post(monkeypatch, handler)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/azure/token", json={})
    assert resp.status_code == 200
    assert resp.json() == {"token": "tok-from-azure", "region": "westeurope"}
    assert seen["url"] == "https://westeurope.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "env-key"


def test_azure_token_prefers_client_supplied_key_and_region(client, monkeypatch):
    monkeypatch.setattr(main, "AZURE_SPEECH_KEY", "env-key")
    monkeypatch.setattr(main, "AZURE_SPEECH_REGION", "westeurope")
    seen = {}

    def handler(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers", {})
        return _FakeTextResponse("tok-client")

    _fake_httpx_post(monkeypatch, handler)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/azure/token", json={"api_key": "my-key", "region": "northeurope"})
    assert resp.status_code == 200
    assert resp.json() == {"token": "tok-client", "region": "northeurope"}
    assert seen["url"].startswith("https://northeurope.")
    assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "my-key"


def test_azure_token_without_configuration_is_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "AZURE_SPEECH_KEY", "")
    monkeypatch.setattr(main, "AZURE_SPEECH_REGION", "")
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/azure/token", json={})
    assert resp.status_code == 400
    assert "not configured" in resp.json()["detail"]


@pytest.mark.parametrize("region", ["west europe", "west_europe", "we!", "a" * 41, "../evil"])
def test_azure_token_rejects_regions_outside_the_charset(client, monkeypatch, region):
    """The region is interpolated into the upstream URL — it must not inject a host."""
    monkeypatch.setattr(main, "AZURE_SPEECH_KEY", "env-key")

    def handler(url, **_kwargs):
        raise AssertionError(f"upstream must not be called for region {region!r} (url={url})")

    _fake_httpx_post(monkeypatch, handler)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/azure/token", json={"region": region})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid Azure region"


def test_azure_token_passes_through_upstream_status(client, monkeypatch):
    import httpx

    monkeypatch.setattr(main, "AZURE_SPEECH_KEY", "bad-key")
    monkeypatch.setattr(main, "AZURE_SPEECH_REGION", "westeurope")

    def handler(url, **_kwargs):
        response = httpx.Response(401, request=httpx.Request("POST", url))
        raise httpx.HTTPStatusError("unauthorized", request=response.request, response=response)

    _fake_httpx_post(monkeypatch, handler)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/azure/token", json={})
    assert resp.status_code == 401
    assert "401" in resp.json()["detail"]


def test_azure_token_maps_network_failure_to_502(client, monkeypatch):
    import httpx

    monkeypatch.setattr(main, "AZURE_SPEECH_KEY", "env-key")
    monkeypatch.setattr(main, "AZURE_SPEECH_REGION", "westeurope")

    def handler(_url, **_kwargs):
        raise httpx.ConnectError("dns failure")

    _fake_httpx_post(monkeypatch, handler)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/azure/token", json={})
    assert resp.status_code == 502
    assert "Failed to create Azure token" in resp.json()["detail"]


# --- Wave A: /ws translation failure + ElevenLabs token error paths ---


def test_ws_translation_failure_reports_error_and_recreates_the_translator(client, monkeypatch):
    """A failed translation must not silently drop the final or reuse a stale session."""
    created = []

    class BrokenTranslator:
        def __init__(self):
            created.append(self)

        async def translate(self, *_args, **_kwargs):
            raise RuntimeError("upstream refused")

    monkeypatch.setattr(main, "Translator", BrokenTranslator)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["error"] == "translation_failed"
    assert data["original"] == "Ahoj"
    assert data["translations"] == {"en": ""}
    # The stale HTTP session is dropped and a fresh Translator built for the retry.
    assert len(created) >= 2


def test_elevenlabs_token_rejects_a_non_json_upstream_body(client, monkeypatch):
    """A 200 whose body isn't JSON must not surface as a token-shaped success."""
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "xi-env-key")

    class FakeResponse:
        status_code = 200
        text = "definitely not json"

        def json(self):
            raise ValueError("not json")

        def raise_for_status(self):
            pass

    _fake_httpx_post(monkeypatch, lambda _url, **_kw: FakeResponse())
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 502
    assert "Failed to create token" in resp.json()["detail"]


def test_elevenlabs_token_passes_through_upstream_status(client, monkeypatch):
    import httpx

    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "xi-env-key")

    def handler(url, **_kwargs):
        response = httpx.Response(429, json={"detail": "quota"}, request=httpx.Request("POST", url))
        raise httpx.HTTPStatusError("too many", request=response.request, response=response)

    _fake_httpx_post(monkeypatch, handler)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 429
    # The upstream's own detail is preferred over the generic message.
    assert resp.json()["detail"] == "quota"


def test_elevenlabs_token_maps_network_failure_to_502(client, monkeypatch):
    import httpx

    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "xi-env-key")

    def handler(_url, **_kwargs):
        raise httpx.ConnectError("dns failure")

    _fake_httpx_post(monkeypatch, handler)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 502


# --- Wave B: audio forwarding, interim translation, worker failure ---


def test_ws_elevenlabs_forwards_audio_commit_and_ping(client, monkeypatch):
    """_forward_audio: browser PCM -> base64 chunks, plus the commit/ping commands."""
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "test-key")

    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    upstream_sent = []

    class GatedElevenSocket:
        def __init__(self):
            self.committed = asyncio.Event()
            self.closed = False
            self._done = False

        async def recv(self):
            return '{"message_type":"session_started","session_id":"session-1"}'

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            # Keep the upstream open until the client commits.
            await self.committed.wait()
            self._done = True
            return '{"message_type":"committed_transcript","text":"ahoj"}'

        async def send(self, payload):
            message = json.loads(payload)
            upstream_sent.append(message)
            if message.get("commit"):
                self.committed.set()

        async def close(self):
            self.closed = True

    upstream = GatedElevenSocket()
    _connect_fake_elevenlabs(monkeypatch, upstream)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"]}})
        ws.send_bytes(b"\x00\x01\x02\x03")
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        ws.send_json({"type": "commit"})
        final = ws.receive_json()

    assert pong == {"type": "pong"}
    assert final["type"] == "final"
    assert final["translations"] == {"en": "en:ahoj"}

    audio = [m for m in upstream_sent if m.get("audio_base_64")]
    assert audio, f"no audio forwarded: {upstream_sent}"
    assert base64.b64decode(audio[0]["audio_base_64"]) == b"\x00\x01\x02\x03"
    assert audio[0]["commit"] is False
    assert audio[0]["sample_rate"] == 16000

    commits = [m for m in upstream_sent if m.get("commit")]
    assert commits and commits[-1]["audio_base_64"] == ""


def test_ws_deepgram_translates_interim_when_requested(client, monkeypatch):
    """translate_interim=true routes interims through the translator too."""
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")

    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    class FakeAlt:
        def __init__(self, transcript):
            self.transcript = transcript

    class FakeChannel:
        def __init__(self, transcript):
            self.alternatives = [FakeAlt(transcript)]

    class FakeResults:
        def __init__(self, transcript, is_final):
            self.channel = FakeChannel(transcript)
            self.is_final = is_final

    interim_delivered = threading.Event()

    class FakeDgSocket:
        def __init__(self):
            self._handlers = {}

        def on(self, event_type, callback):
            self._handlers[event_type] = callback

        def start_listening(self):
            msg_cb = self._handlers.get(main.EventType.MESSAGE)
            msg_cb(FakeResults("prubezne", False))
            # Hold the final back so the interim cannot be coalesced away — the
            # point of this test is the interim translation branch.
            interim_delivered.wait(timeout=10)
            msg_cb(FakeResults("finalni", True))

        def send_media(self, data):
            pass

        def send_finalize(self, _message=None):
            pass

        def send_close_stream(self, _message=None):
            pass

    fake_socket = FakeDgSocket()

    class _FakeSocketIterator:
        def __init__(self):
            self._sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._sent:
                raise StopIteration
            self._sent = True
            return fake_socket

        def close(self):
            pass

    fake_iter = _FakeSocketIterator()

    class FakeDeepgramClient:
        def __init__(self, api_key):
            class _V1:
                def connect(self, **_kwargs):
                    return fake_iter

            class _Listen:
                v1 = _V1()

            self.listen = _Listen()

    monkeypatch.setattr(main, "DeepgramClient", FakeDeepgramClient)
    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    try:
        with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
            ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"]}, "translate_interim": True})
            ws.send_bytes(b"\x00\x01")
            interim = ws.receive_json()
            interim_delivered.set()
            final = ws.receive_json()
    finally:
        interim_delivered.set()

    assert interim["type"] == "interim"
    assert interim["original"] == "prubezne"
    assert interim["translations"] == {"en": "en:prubezne"}
    assert final["type"] == "final"
    assert final["translations"] == {"en": "en:finalni"}


def test_ws_unexpected_worker_error_reports_server_error_and_closes_1011(client, monkeypatch):
    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    def boom(*_args, **_kwargs):
        raise RuntimeError("payload construction failed")

    monkeypatch.setattr(main, "_translation_payload", boom)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()
        # Typeless error -> terminal, and the server closes with 1011.
        assert data == {"error": "server_error"}
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()

    assert excinfo.value.code == 1011


def _deepgram_client_for(fake_socket):
    """Wrap a fake socket in the v3-style iterator/client the endpoint expects."""

    class _FakeSocketIterator:
        def __init__(self):
            self._sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._sent:
                raise StopIteration
            self._sent = True
            return fake_socket

        def close(self):
            pass

    iterator = _FakeSocketIterator()

    class FakeDeepgramClient:
        def __init__(self, api_key):
            class _V1:
                def connect(self, **_kwargs):
                    return iterator

            class _Listen:
                v1 = _V1()

            self.listen = _Listen()

    return FakeDeepgramClient


def test_ws_deepgram_upstream_error_is_forwarded_and_stops_the_session(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")

    class FakeDgSocket:
        def __init__(self):
            self._handlers = {}
            self.closed = False

        def on(self, event_type, callback):
            self._handlers[event_type] = callback

        def start_listening(self):
            self._handlers[main.EventType.ERROR]("upstream exploded")

        def send_media(self, data):
            pass

        def send_finalize(self, _message=None):
            pass

        def send_close_stream(self, _message=None):
            self.closed = True

    monkeypatch.setattr(main, "DeepgramClient", _deepgram_client_for(FakeDgSocket()))
    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        ws.send_bytes(b"\x00\x01")
        data = ws.receive_json()

    # Typeless error -> the browser tears the session down.
    assert "type" not in data
    assert data["error"] == "upstream exploded"


def test_ws_elevenlabs_untranslated_interim_still_reaches_the_browser(client, monkeypatch):
    """translate_interim defaults off: interims are forwarded with empty translations."""
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "test-key")

    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    upstream = _elevenlabs_socket_emitting(
        [
            '{"message_type":"partial_transcript","text":"ah"}',
            '{"message_type":"committed_transcript","text":"ahoj"}',
        ]
    )
    _connect_fake_elevenlabs(monkeypatch, upstream)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        ws.send_json(
            {
                "type": "config",
                "translate": {"src": "cs", "dests": ["en"]},
                "translate_interim": False,
            }
        )
        messages = []
        for _ in range(2):
            message = ws.receive_json()
            messages.append(message)
            if message["type"] == "final":
                break

    final = messages[-1]
    assert final["type"] == "final"
    assert final["translations"] == {"en": "en:ahoj"}
    for interim in messages[:-1]:
        assert interim["type"] == "interim"
        # Untranslated: the text is forwarded, the translation slot stays empty.
        assert interim["original"] == "ah"
        assert interim["translations"] == {"en": ""}


# --- Wave C: small helpers that are cheaper to test than to exempt ---


def test_close_translator_tolerates_missing_and_failing_clients():
    async def scenario():
        await main._close_translator(None)  # no translator at all

        class Closed:
            def __init__(self):
                self.calls = 0

            async def aclose(self):
                self.calls += 1

        client = Closed()
        await main._close_translator(types.SimpleNamespace(client=client))
        assert client.calls == 1

        class Broken:
            async def aclose(self):
                raise RuntimeError("already closed")

        # A failing close must not propagate — it runs on teardown paths.
        await main._close_translator(types.SimpleNamespace(client=Broken()))
        await main._close_translator(types.SimpleNamespace(client=object()))

    asyncio.run(scenario())


def test_cookie_secure_follows_scheme_unless_overridden(monkeypatch):
    class FakeUrl:
        def __init__(self, scheme):
            self.scheme = scheme

    class FakeRequest:
        def __init__(self, scheme):
            self.url = FakeUrl(scheme)

    monkeypatch.delenv("AUTH_COOKIE_SECURE", raising=False)
    assert main._cookie_secure_for_request(FakeRequest("https")) is True
    assert main._cookie_secure_for_request(FakeRequest("http")) is False

    monkeypatch.setenv("AUTH_COOKIE_SECURE", "true")
    assert main._cookie_secure_for_request(FakeRequest("http")) is True

    monkeypatch.setenv("AUTH_COOKIE_SECURE", "no")
    assert main._cookie_secure_for_request(FakeRequest("https")) is False


def test_ws_echoes_client_correlation_fields(client, monkeypatch):
    """client_id/client_sent_ms round-trip so the browser can measure latency."""

    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json(
            {
                "type": "final",
                "text": "Ahoj",
                "src": "cs",
                "dests": ["en"],
                "client_id": 42,
                "client_sent_ms": 123.5,
            }
        )
        data = ws.receive_json()

    assert data["client_id"] == 42
    assert data["client_sent_ms"] == 123.5


def test_ws_supports_a_synchronous_translator(client, monkeypatch):
    """googletrans ships both sync and async variants; the sync one is offloaded."""
    calls = []

    class SyncTranslator:
        def translate(self, text, src, dest):
            calls.append((text, src, dest, threading.current_thread().name))
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", SyncTranslator)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()

    assert data["translations"] == {"en": "en:Ahoj"}
    assert calls and calls[0][:3] == ("Ahoj", "cs", "en")


def test_ws_deepgram_listener_crash_is_reported_to_the_browser(client, monkeypatch):
    """A dead listener thread must not leave the UI recording into a void."""
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")

    class ExplodingDgSocket:
        def on(self, event_type, callback):
            pass

        def start_listening(self):
            raise RuntimeError("listener died")

        def send_media(self, data):
            pass

        def send_finalize(self, _message=None):
            pass

        def send_close_stream(self, _message=None):
            pass

    monkeypatch.setattr(main, "DeepgramClient", _deepgram_client_for(ExplodingDgSocket()))
    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        ws.send_bytes(b"\x00\x01")
        data = ws.receive_json()

    # Typeless error -> the browser stops the session instead of hanging.
    assert "type" not in data
    assert data["error"] == "deepgram_listener_failed"


def test_latest_interim_queue_exposes_closed_state():
    queue = main.LatestInterimQueue()
    assert queue.closed is False
    queue.close()
    assert queue.closed is True


def test_worker_drains_queued_final_after_supersede_when_inbox_closed(monkeypatch):
    """Upstream (ElevenLabs) can end — closing the inbox — right after a committed
    final is queued behind a superseded interim. The worker must translate and
    send that final, not raise its own CancelledError and drop it. `inbox.closed`
    is not the teardown signal; only the worker task being cancelled is."""
    sent = []

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowTranslator:
            async def translate(self, text, src, dest):
                if text == "interim":
                    started.set()
                    await release.wait()
                return _FakeTranslation(f"{dest}:{text}")

        monkeypatch.setattr(main, "Translator", SlowTranslator)

        queue = main.LatestInterimQueue()

        async def send(payload):
            sent.append(payload)

        session = main.TranslationSession(queue, send, log_label="test")
        worker = asyncio.create_task(session.run())
        try:
            queue.put(main.TranslationWork("interim", "interim", "cs", ["en"]))
            await asyncio.wait_for(started.wait(), timeout=1)

            # Final supersedes the interim; the upstream then ends and closes the
            # inbox, all before the interim's cancellation lands on the worker.
            session.supersede_interim()
            assert queue.put(main.TranslationWork("final", "final", "cs", ["en"]))
            queue.close()
            release.set()

            await asyncio.wait_for(worker, timeout=2)
        finally:
            if not worker.done():
                worker.cancel()
            await session.aclose()

    asyncio.run(scenario())

    finals = [p for p in sent if p.get("type") == "final"]
    assert len(finals) == 1, f"final was dropped: {sent}"
    assert finals[0]["original"] == "final"
    assert finals[0]["translations"] == {"en": "en:final"}


def test_translation_session_retrieves_cancelled_provider_gather(monkeypatch):
    """Disconnecting during a translation must not leave asyncio's gather future
    unobserved. That was the noisy CancelledError recorded in the Coolify logs."""

    async def scenario():
        started = asyncio.Event()
        never_release = asyncio.Event()
        loop_errors = []

        class SlowTranslator:
            async def translate(self, text, src, dest):
                started.set()
                await never_release.wait()
                return _FakeTranslation(f"{dest}:{text}")

        monkeypatch.setattr(main, "Translator", SlowTranslator)
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        queue = main.LatestInterimQueue()

        async def send(_payload):
            pass

        session = main.TranslationSession(queue, send, log_label="test")
        worker = asyncio.create_task(session.run())
        try:
            assert queue.put(main.TranslationWork("final", "final", "cs", ["en"]))
            await asyncio.wait_for(started.wait(), timeout=1)
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        finally:
            if not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await session.aclose()

        # _GatheringFuture and its callbacks form a short-lived reference cycle.
        # Release the finished worker frame and collect it while our exception
        # handler is still installed.
        worker = None
        session = None
        queue = None
        gc.collect()
        await asyncio.sleep(0)
        loop.set_exception_handler(previous_handler)

        return loop_errors

    errors = asyncio.run(scenario())
    assert not [
        context
        for context in errors
        if "exception was never retrieved" in str(context.get("message", "")).lower()
    ]


def test_ws_translation_failure_cancels_siblings_before_closing_the_translator(client, monkeypatch):
    """gather() leaves siblings running after the first failure. They must be
    cancelled *before* the error path closes the translator they are still using —
    asserting only that they end up cancelled would pass either way, since endpoint
    teardown cancels them anyway."""
    events = []
    real_close = main._close_translator

    async def recording_close(translator):
        events.append("translator-closed")
        return await real_close(translator)

    class PartiallyFailingTranslator:
        async def translate(self, text, src, dest):
            if dest == "en":
                await asyncio.sleep(0.05)  # let the sibling get going first
                raise RuntimeError("en upstream refused")
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                events.append("sibling-cancelled")
                raise
            events.append("sibling-completed")
            return _FakeTranslation("late")

    monkeypatch.setattr(main, "Translator", PartiallyFailingTranslator)
    monkeypatch.setattr(main, "_close_translator", recording_close)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en", "ru"]})
        data = ws.receive_json()

    assert data["error"] == "translation_failed"
    assert "sibling-completed" not in events
    assert "sibling-cancelled" in events, f"sibling was never cancelled: {events}"
    assert events.index("sibling-cancelled") < events.index("translator-closed"), (
        f"translator closed while a sibling was still using it: {events}"
    )


@pytest.mark.parametrize(
    "path",
    ["/static/nemotron/models/encoder_fp16.onnx", "/static/parakeet/models/encoder-int8.onnx"],
)
def test_model_weights_are_not_downloadable_without_auth(client, path):
    """Hundreds of MB per file, served by StaticFiles, which has no auth of its own."""
    resp = client.get(path)
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "path",
    [
        # httpx keeps "//" and "%2e%2e" verbatim (it collapses only literal "."/
        # ".."), so these reach the server non-canonically; StaticFiles then
        # normalizes them onto a real model file. A raw startswith gate on the
        # request path would 401 the canonical spelling but serve these.
        "/static/nemotron//models/encoder_fp16.onnx",
        "/static/nemotron/%2e%2e/nemotron/models/encoder_fp16.onnx",
        "/static/whisper/%2e%2e/nemotron/models/encoder_fp16.onnx",
        "/static/parakeet//models/encoder-int8.onnx",
        "/static/parakeet/%2e%2e/parakeet/models/encoder-int8.onnx",
    ],
)
def test_model_weights_gate_is_not_bypassed_by_noncanonical_paths(client, path):
    """The auth middleware must normalize the path the same way StaticFiles does,
    or a scraper pulls ~1.2 GB per request with no session via "//"/"%2e%2e"."""
    resp = client.get(path)
    assert resp.status_code == 401


def test_model_weights_are_served_without_auth_when_auth_is_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"nemotron"})
    path = Path("app/static/nemotron/models/__auth_test__.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok")
    try:
        resp = TestClient(main.app).get("/static/nemotron/models/__auth_test__.txt")
    finally:
        path.unlink(missing_ok=True)

    assert resp.status_code == 200
    assert resp.text == "ok"


def test_ws_elevenlabs_translation_failure_reports_and_recycles(client, monkeypatch):
    """ElevenLabs runs the same TranslationSession as /ws, so it inherits the
    failure handling that used to live only in /ws."""
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "test-key")
    created = []

    class BrokenTranslator:
        def __init__(self):
            created.append(self)

        async def translate(self, *_args, **_kwargs):
            raise RuntimeError("upstream refused")

    monkeypatch.setattr(main, "Translator", BrokenTranslator)

    upstream = _elevenlabs_socket_emitting(
        ['{"message_type":"committed_transcript","text":"ahoj"}']
    )
    _connect_fake_elevenlabs(monkeypatch, upstream)
    client.post("/login", data={"password": "test-password", "next": "/"})

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"]}})
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["error"] == "translation_failed"
    assert data["original"] == "ahoj"
    assert data["translations"] == {"en": ""}
    assert len(created) >= 2  # stale HTTP session dropped, fresh Translator built


def test_translation_session_is_used_by_both_websocket_endpoints():
    """Guards the extraction: a future edit must not fork the logic again."""
    source = Path("app/main.py").read_text()
    assert source.count("TranslationSession(") == 2  # constructed once per endpoint
    assert 'log_label="Překlad selhal"' in source
    assert 'log_label="ElevenLabs translation error"' in source
    # The forked copies are gone, not merely unused.
    assert "_translate_transcripts" not in source
    assert "superseded_el_interim_tasks" not in source


# --- Translation providers: preference parsing, router, DeepL, config plumbing ---


class _FakeProvider:
    def __init__(self, name, behaviour):
        self.name = name
        self.behaviour = behaviour  # callable(text, src, dest) -> str, or raises
        self.calls = []
        self.recycled = 0
        self.closed = 0

    async def translate(self, text, *, src, dest):
        self.calls.append((text, src, dest))
        return self.behaviour(text, src, dest)

    async def recycle(self):
        self.recycled += 1

    async def aclose(self):
        self.closed += 1


def _rate_limited(*_args):
    raise main.TranslationRateLimited("google: HTTP 429")


class _FakeDeepL:
    """Stands in for DeepLTranslateProvider inside TranslationRouter (per session)."""

    name = "deepl"
    instances: list = []

    def __init__(self, api_key, *, api_url=""):
        self.api_key = api_key
        self.api_url = api_url
        self.calls = []
        _FakeDeepL.instances.append(self)

    async def translate(self, text, *, src, dest):
        self.calls.append((text, src, dest))
        return f"deepl:{dest}:{text}"

    async def recycle(self):
        pass

    async def aclose(self):
        pass


@pytest.mark.parametrize(
    "raw, deepl_key, expected",
    [
        ("auto", "k:fx", ["google", "deepl"]),
        ("", "k:fx", ["google", "deepl"]),
        ("auto", "", ["google"]),
        ("deepl,google", "k:fx", ["deepl", "google"]),
        ("deepl", "k:fx", ["deepl"]),
        ("google", "k:fx", ["google"]),
        ("deepl", "", ["google"]),  # pinned but unconfigured -> the only usable one
        ("bogus,deepl", "k:fx", ["deepl"]),  # unknown names dropped
        ("GOOGLE, deepl", "k:fx", ["google", "deepl"]),
    ],
)
def test_translate_provider_order_parsing(monkeypatch, raw, deepl_key, expected):
    monkeypatch.setattr(main, "DEEPL_API_KEY", deepl_key)
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", raw)
    assert main._translate_provider_order() == expected


def test_translate_provider_order_honours_a_session_pin(monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "k:fx")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "auto")
    assert main._translate_provider_order("deepl") == ["deepl"]
    assert main._translate_provider_order("google") == ["google"]
    assert main._translate_provider_order("auto") == ["google", "deepl"]
    # A pin on an unconfigured provider falls back to the server order.
    monkeypatch.setattr(main, "DEEPL_API_KEY", "")
    assert main._translate_provider_order("deepl") == ["google"]


@pytest.mark.parametrize(
    "raw, deepl_key, expected",
    [
        ("auto", "k:fx", "auto"),
        ("deepl", "k:fx", "deepl"),
        ("google,deepl", "k:fx", "auto"),
        ("deepl", "", "google"),  # pinned to something unconfigured -> the one that works
    ],
)
def test_translate_provider_ui_default(monkeypatch, raw, deepl_key, expected):
    monkeypatch.setattr(main, "DEEPL_API_KEY", deepl_key)
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", raw)
    assert main._translate_provider_ui_default() == expected


def test_normalize_translate_provider(monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "k:fx")
    assert main._normalize_translate_provider("deepl") == "deepl"
    assert main._normalize_translate_provider(" Google ") == "google"
    assert main._normalize_translate_provider("auto") == "auto"
    assert main._normalize_translate_provider("bing") is None
    assert main._normalize_translate_provider(42) is None
    monkeypatch.setattr(main, "DEEPL_API_KEY", "")
    assert main._normalize_translate_provider("deepl") is None


@pytest.mark.parametrize(
    "exc, expected",
    [
        (Exception("Unexpected status code \"429\" from ['translate.googleapis.com']"), True),
        (RuntimeError("Translation upstream returned HTTP 429"), True),
        (RuntimeError("Translation upstream returned HTTP 456"), True),
        (RuntimeError("Translation upstream returned HTTP 500"), False),
        (RuntimeError("timed out after 429 ms"), False),
    ],
)
def test_looks_rate_limited_message_forms(exc, expected):
    assert main._looks_rate_limited(exc) is expected


def test_looks_rate_limited_reads_response_status():
    class WithResponse(Exception):
        def __init__(self, status):
            super().__init__("boom")
            self.response = types.SimpleNamespace(status_code=status)

    assert main._looks_rate_limited(WithResponse(429)) is True
    assert main._looks_rate_limited(WithResponse(503)) is False


def test_translation_router_switches_to_fallback_on_rate_limit(monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "k:fx")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "auto")
    monkeypatch.setattr(main, "TRANSLATE_FALLBACK_COOLDOWN_SECONDS", 600.0)
    google = _FakeProvider("google", _rate_limited)
    deepl = _FakeProvider("deepl", lambda text, src, dest: f"deepl:{dest}:{text}")
    cooldown: dict = {}
    router = main.TranslationRouter({"google": google, "deepl": deepl}, cooldown=cooldown)

    async def scenario():
        first = await router.translate("auto", "ahoj", src="cs", dest="en")
        second = await router.translate("auto", "svete", src="cs", dest="en")
        return first, second

    first, second = asyncio.run(scenario())

    assert first == ("deepl:en:ahoj", "deepl")
    assert second == ("deepl:en:svete", "deepl")
    # Google was tried once, then skipped for the cooldown; DeepL served both.
    assert len(google.calls) == 1
    assert len(deepl.calls) == 2
    assert cooldown["google"] > time.monotonic()


def test_translation_router_retries_primary_once_its_cooldown_expired(monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "k:fx")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "auto")
    google = _FakeProvider("google", lambda text, src, dest: f"google:{text}")
    deepl = _FakeProvider("deepl", lambda text, src, dest: f"deepl:{text}")
    cooldown = {"google": time.monotonic() - 1}  # block already lifted
    router = main.TranslationRouter({"google": google, "deepl": deepl}, cooldown=cooldown)

    assert asyncio.run(router.translate("auto", "x", src="cs", dest="en")) == ("google:x", "google")
    assert deepl.calls == []


def test_translation_router_tries_cooling_providers_last_oldest_block_first(monkeypatch):
    """Everything limited: still try, oldest block first — the limit may have lifted."""
    monkeypatch.setattr(main, "DEEPL_API_KEY", "k:fx")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "auto")
    now = time.monotonic()
    google = _FakeProvider("google", lambda *_: "g")
    deepl = _FakeProvider("deepl", lambda *_: "d")
    router = main.TranslationRouter(
        {"google": google, "deepl": deepl},
        cooldown={"google": now + 100, "deepl": now + 50},
    )
    assert [p.name for p in router.candidates("auto")] == ["deepl", "google"]


def test_translation_router_pinned_provider_never_switches(monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "k:fx")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "auto")
    google = _FakeProvider("google", _rate_limited)
    deepl = _FakeProvider("deepl", lambda *_: "d")
    router = main.TranslationRouter({"google": google, "deepl": deepl}, cooldown={})

    with pytest.raises(RuntimeError, match="rate limited"):
        asyncio.run(router.translate("google", "x", src="cs", dest="en"))
    assert deepl.calls == []


def test_translation_router_recycles_and_closes_every_provider():
    google = _FakeProvider("google", lambda *_: "g")
    deepl = _FakeProvider("deepl", lambda *_: "d")
    router = main.TranslationRouter({"google": google, "deepl": deepl}, cooldown={})

    async def scenario():
        await router.recycle()
        await router.aclose()

    asyncio.run(scenario())
    assert (google.recycled, deepl.recycled, google.closed, deepl.closed) == (1, 1, 1, 1)


def _deepl_provider(handler, key="key:fx", **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return main.DeepLTranslateProvider(key, client=client, **kwargs)


def test_deepl_provider_translates_and_maps_language_codes():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"translations": [{"detected_source_language": "CS", "text": "Hello world"}]}
        )

    provider = _deepl_provider(handler)
    text = asyncio.run(provider.translate("Ahoj světe", src="cs", dest="en"))

    assert text == "Hello world"
    assert seen["url"] == "https://api-free.deepl.com/v2/translate"  # ":fx" key -> free host
    assert seen["auth"] == "DeepL-Auth-Key key:fx"
    assert seen["body"] == {"text": ["Ahoj světe"], "target_lang": "EN-US", "source_lang": "CS"}
    asyncio.run(provider.aclose())


def test_deepl_provider_picks_host_from_key_and_honours_override():
    idle = httpx.MockTransport(lambda request: httpx.Response(500))
    pro = main.DeepLTranslateProvider("abc", client=httpx.AsyncClient(transport=idle))
    custom = main.DeepLTranslateProvider(
        "abc:fx", api_url="http://localhost:9/v2/translate", client=httpx.AsyncClient(transport=idle)
    )
    assert pro.api_url == "https://api.deepl.com/v2/translate"
    assert custom.api_url == "http://localhost:9/v2/translate"

    async def close():
        await pro.aclose()
        await custom.aclose()

    asyncio.run(close())


def test_deepl_provider_omits_source_lang_it_cannot_name():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"translations": [{"text": "ok"}]})

    provider = _deepl_provider(handler)
    asyncio.run(provider.translate("x", src="sw", dest="de"))  # Swahili: not a DeepL source
    assert "source_lang" not in seen["body"]
    assert seen["body"]["target_lang"] == "DE"
    asyncio.run(provider.aclose())


@pytest.mark.parametrize("code", [429, 456])
def test_deepl_provider_reports_rate_limits(code):
    provider = _deepl_provider(lambda request: httpx.Response(code, json={"message": "limit"}))
    with pytest.raises(main.TranslationRateLimited):
        asyncio.run(provider.translate("x", src="cs", dest="en"))
    asyncio.run(provider.aclose())


def test_deepl_provider_other_errors_are_plain_failures():
    bad_lang = _deepl_provider(
        lambda request: httpx.Response(400, json={"message": "Value for 'target_lang' not supported."})
    )
    with pytest.raises(RuntimeError, match="HTTP 400"):
        asyncio.run(bad_lang.translate("x", src="cs", dest="xx"))

    odd_body = _deepl_provider(lambda request: httpx.Response(200, json={"nope": []}))
    with pytest.raises(RuntimeError, match="unexpected response"):
        asyncio.run(odd_body.translate("x", src="cs", dest="en"))

    def explode(request):
        raise httpx.ConnectError("refused", request=request)

    down = _deepl_provider(explode)
    with pytest.raises(RuntimeError, match="request failed"):
        asyncio.run(down.translate("x", src="cs", dest="en"))

    async def close():
        for provider in (bad_lang, odd_body, down):
            await provider.aclose()

    asyncio.run(close())


@pytest.mark.parametrize(
    "code, expected",
    [("en", "EN-US"), ("pt", "PT-PT"), ("zh-cn", "ZH-HANS"), ("zh-tw", "ZH-HANT"), ("de", "DE"), ("no", "NB"), ("iw", "HE")],
)
def test_deepl_target_lang_mapping(code, expected):
    assert main._deepl_target_lang(code) == expected


@pytest.mark.parametrize(
    "code, expected",
    [("cs", "CS"), ("zh-cn", "ZH"), ("en-US", "EN"), ("iw", "HE"), ("no", "NB"), ("sw", None)],
)
def test_deepl_source_lang_mapping(code, expected):
    assert main._deepl_source_lang(code) == expected


def test_ws_pins_translation_provider_from_config(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "key:fx")
    monkeypatch.setattr(main, "DeepLTranslateProvider", _FakeDeepL)

    class GoogleMustNotRun:
        async def translate(self, *_args, **_kwargs):
            raise AssertionError("google must not be used when deepl is pinned")

    monkeypatch.setattr(main, "Translator", GoogleMustNotRun)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"], "provider": "deepl"}})
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()

    assert data["translations"] == {"en": "deepl:en:Ahoj"}
    assert data["provider"] == "deepl"


def test_ws_auto_switches_to_deepl_when_google_is_rate_limited(client, monkeypatch):
    """The production failure mode: Google answers 429 for this server's IP. Auto
    mode must deliver the translation from DeepL instead of `translation_failed`,
    and remember the block so the next request skips Google outright."""
    monkeypatch.setattr(main, "DEEPL_API_KEY", "key:fx")
    monkeypatch.setattr(main, "DeepLTranslateProvider", _FakeDeepL)
    google_calls = []

    class ThrottledGoogle:
        async def translate(self, text, src, dest):
            google_calls.append(text)
            raise Exception("Unexpected status code \"429\" from ['translate.googleapis.com']")

    monkeypatch.setattr(main, "Translator", ThrottledGoogle)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        first = ws.receive_json()
        ws.send_json({"type": "final", "text": "Světe", "src": "cs", "dests": ["en"]})
        second = ws.receive_json()

    assert "error" not in first and "error" not in second
    assert first["translations"] == {"en": "deepl:en:Ahoj"}
    assert second["translations"] == {"en": "deepl:en:Světe"}
    assert first["provider"] == second["provider"] == "deepl"
    assert google_calls == ["Ahoj"]  # skipped for the cooldown on the second message
    assert main._PROVIDER_COOLDOWN_UNTIL["google"] > time.monotonic()


def test_ws_pinned_google_reports_rate_limit_instead_of_switching(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "key:fx")
    monkeypatch.setattr(main, "DeepLTranslateProvider", _FakeDeepL)
    _FakeDeepL.instances.clear()

    class ThrottledGoogle:
        async def translate(self, *_args, **_kwargs):
            raise Exception("Unexpected status code \"429\" from ['translate.googleapis.com']")

    monkeypatch.setattr(main, "Translator", ThrottledGoogle)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"], "provider": "google"}})
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()

    assert data["error"] == "translation_failed"
    assert data["translations"] == {"en": ""}
    assert all(instance.calls == [] for instance in _FakeDeepL.instances)


def test_ws_ignores_an_unknown_translation_provider(client, monkeypatch):
    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"{dest}:{text}")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"], "provider": "bing"}})
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()

    assert data["translations"] == {"en": "en:Ahoj"}
    assert data["provider"] == "google"


def test_index_context_exposes_translate_providers(monkeypatch):
    monkeypatch.setattr(main, "DEEPL_API_KEY", "")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "auto")
    context = main._index_context()
    assert context["translate_providers"] == ["google"]
    assert context["translate_provider_default"] == "auto"

    monkeypatch.setattr(main, "DEEPL_API_KEY", "k:fx")
    monkeypatch.setattr(main, "TRANSLATE_PROVIDER", "deepl")
    context = main._index_context()
    assert context["translate_providers"] == ["google", "deepl"]
    assert context["translate_provider_default"] == "deepl"


def test_template_exposes_translation_provider_setting():
    html = Path("app/templates/index.html").read_text()

    assert 'id="translateProvider"' in html
    assert "{{ translate_providers | tojson }}" in html
    assert "{{ translate_provider_default | tojson }}" in html
    assert "function populateTranslateProviderSelect" in html
    assert 'id="statProvider"' in html
    # Every WS session tells the server which provider it wants:
    # /ws (1) + Deepgram (dbg echo + send = 2) + ElevenLabs server mode (1).
    assert html.count("provider: translateProvider") == 4
    assert "translateProvider," in html.split("function persistSettings")[1].split("}")[0]


# --- Auth: non-ASCII secrets ---


def test_login_handles_non_ascii_password_and_cookie(client, monkeypatch):
    """secrets.compare_digest() raises TypeError on non-ASCII str. A Czech
    APP_PASSWORD with diacritics must simply work, and a cookie carrying a stray
    non-ASCII byte (anyone can send one) must land on the login page, not a 500."""
    monkeypatch.setattr(main, "APP_PASSWORD", "Křemílek42")

    assert main.verify_auth_token("abc.ÿ") is False
    resp = client.get("/", headers={b"cookie": "srlt_auth=abc.ÿ".encode("latin-1")})
    assert resp.status_code == 200
    _assert_login_h1(resp.text)

    wrong = client.post(
        "/login",
        data={"password": "Křemílek", "next": "/"},
        follow_redirects=False,
    )
    assert wrong.status_code == 200
    assert "Incorrect password" in wrong.text

    right = client.post(
        "/login",
        data={"password": "Křemílek42", "next": "/"},
        follow_redirects=False,
    )
    assert right.status_code == 303
    assert client.get("/").status_code == 200
    assert "<title>Live Translator</title>" in client.get("/").text
