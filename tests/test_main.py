import re

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech", "deepgram", "elevenlabs"})
    return TestClient(main.app)


def _assert_login_h1(html: str) -> None:
    assert re.search(r"<h1[^>]*>Sign in</h1>", html)


def test_get_index_requires_password(client):
    resp = client.get("/")
    assert resp.status_code == 200
    _assert_login_h1(resp.text)
    assert "Incorrect password" not in resp.text


def test_get_index_wrong_password_shows_error(client):
    resp = client.get("/?pwd=wrong")
    assert resp.status_code == 200
    _assert_login_h1(resp.text)
    assert "Incorrect password" in resp.text


def test_get_index_correct_password_serves_index_html(client):
    resp = client.get("/?pwd=test-password", follow_redirects=False)
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
        payload = ws.receive_json()

    assert payload == {"error": "boom"}


def test_ws_deepgram_missing_sdk_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setattr(main, "DeepgramClient", None)

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        payload = ws.receive_json()

    assert payload == {"error": "deepgram-sdk not installed"}


def test_ws_deepgram_happy_path_emits_interim_and_final(client, monkeypatch):
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
        interim = ws.receive_json()
        final = ws.receive_json()

    assert interim["type"] == "interim"
    assert interim["original"] == "prubezne"
    assert interim["dests"] == ["en", "ru"]
    assert interim["translations"] == {"en": "", "ru": ""}
    assert interim["en"] == ""
    assert interim["ru"] == ""
    assert isinstance(interim.get("timing", {}), dict)

    assert final["type"] == "final"
    assert final["original"] == "finalni"
    assert final["dests"] == ["en", "ru"]
    assert final["translations"] == {"en": "en:finalni", "ru": "ru:finalni"}
    assert final["en"] == "en:finalni"
    assert final["ru"] == "ru:finalni"
    assert isinstance(final.get("timing", {}), dict)
    assert translator.calls == [("finalni", "cs", "en"), ("finalni", "cs", "ru")]
    assert fake_socket.sent_media == [b"\x00\x01"]


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
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        data = ws.receive_json()
        assert data == {"type": "pong"}


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
