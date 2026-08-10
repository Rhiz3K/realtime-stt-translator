import asyncio
from pathlib import Path
import re
import threading

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

    assert payload == {"error": "boom"}


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
    # A user can hit Nemotron before the background prepare job finishes. Never
    # let a browser pin that transient 404 as an immutable model response.
    resp = client.get("/static/nemotron/models/__missing_model_asset__.json")
    assert resp.status_code == 404
    assert resp.headers.get("cache-control") == "no-store"


def test_existing_nemotron_model_assets_cached_immutably(client):
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
