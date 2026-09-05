import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from app import google_audio, main


def _event(*, interim=None, final=None, turn_complete=False):
    return SimpleNamespace(
        server_content=SimpleNamespace(
            interim_input_transcription=(
                SimpleNamespace(text=interim) if interim is not None else None
            ),
            input_transcription=(
                SimpleNamespace(text=final) if final is not None else None
            ),
            turn_complete=turn_complete,
        )
    )


class FakeGoogleSession:
    def __init__(self):
        self.sent = []
        self.turns = asyncio.Queue()
        self.closed = False
        self.eof_on_stop = True

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)
        if kwargs.get("audio") is not None:
            await self.turns.put([_event(interim="ahoj")])
        if kwargs.get("audio_stream_end"):
            await self.turns.put([_event(final="ahoj", turn_complete=True)])
            if self.eof_on_stop:
                await self.turns.put(main.errors.APIError(1000, "Normal closure"))

    def receive(self):
        async def one_turn():
            turn = await self.turns.get()
            if isinstance(turn, BaseException):
                raise turn
            for item in turn:
                yield item

        return one_turn()

    async def close(self):
        self.closed = True


class FakeConnect:
    def __init__(self, owner):
        self.owner = owner

    async def __aenter__(self):
        if self.owner.connect_hangs:
            await asyncio.Event().wait()
        if self.owner.connect_error:
            raise self.owner.connect_error
        return self.owner.session

    async def __aexit__(self, *_):
        if self.owner.exit_hangs:
            await asyncio.Event().wait()
        self.owner.session.closed = True


class FakeLive:
    def __init__(self, owner):
        self.owner = owner

    def connect(self, **kwargs):
        self.owner.connect_calls.append(kwargs)
        return FakeConnect(self.owner)


class FakeModels:
    def __init__(self, owner):
        self.owner = owner

    async def generate_content(self, **kwargs):
        self.owner.translation_calls.append(kwargs)
        if self.owner.translation_release is not None:
            self.owner.translation_started.set()
            try:
                while not self.owner.translation_release.is_set():
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                self.owner.translation_cancelled.set()
                raise
        text = kwargs["contents"]
        return SimpleNamespace(parsed={"en": f"EN:{text}", "ru": f"RU:{text}"})


class FakeAsyncClient:
    def __init__(self, owner):
        self.live = FakeLive(owner)
        self.models = FakeModels(owner)
        self.owner = owner

    async def aclose(self):
        self.owner.client_closed = True
        self.owner.closed_event.set()


class FakeGoogleClient:
    def __init__(self, *, connect_error=None, connect_hangs=False, exit_hangs=False):
        self.session = FakeGoogleSession()
        self.connect_calls = []
        self.translation_calls = []
        self.client_closed = False
        self.closed_event = threading.Event()
        self.connect_error = connect_error
        self.connect_hangs = connect_hangs
        self.exit_hangs = exit_hangs
        self.translation_started = threading.Event()
        self.translation_cancelled = threading.Event()
        self.translation_release = None
        self.aio = FakeAsyncClient(self)


@pytest.fixture(autouse=True)
def clean_login_limiter():
    main._login_attempts.clear()
    yield
    main._login_attempts.clear()


@pytest.fixture
def open_client(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-google-key")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def google_client(monkeypatch):
    fake = FakeGoogleClient()
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-google-key")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "_create_google_client", lambda: fake)
    monkeypatch.setattr(google_audio, "INTERIM_DEBOUNCE_SECONDS", 0.001)
    monkeypatch.setattr(google_audio, "INTERIM_MIN_INTERVAL_SECONDS", 0.0)
    with TestClient(main.app) as client:
        yield client, fake


def test_liveness_does_not_depend_on_google_configuration(monkeypatch):
    monkeypatch.setattr(main, "GEMINI_API_KEY", "")
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/health/live").json() == {"status": "ok"}


def test_readiness_requires_google_key(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "GEMINI_API_KEY", "")
    with TestClient(main.app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": ["gemini_api_key_missing"],
    }


def test_readiness_accepts_fixed_google_pipeline(open_client):
    response = open_client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_auth_boolean_typo_fails_closed_and_is_not_ready(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "treu")
    assert main._env_bool("AUTH_ENABLED", True) == (True, False)

    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "_AUTH_ENABLED_VALID", False)
    monkeypatch.setattr(main, "APP_PASSWORD", "real-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "real-secret")
    monkeypatch.setattr(main, "GEMINI_API_KEY", "real-key")
    assert "auth_enabled_invalid" in main.configuration_errors()


def test_invalid_cookie_secure_value_fails_secure_and_readiness(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "real-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "real-secret")
    monkeypatch.setattr(main, "GEMINI_API_KEY", "real-key")
    monkeypatch.setattr(main, "AUTH_COOKIE_SECURE_RAW", "flase")
    assert "auth_cookie_secure_invalid" in main.configuration_errors()
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )
    assert main._cookie_secure_for_request(request) is True


def test_root_redirects_to_login_when_auth_is_enabled(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "correct horse")
    monkeypatch.setattr(main, "AUTH_SECRET", "separate-secret")
    with TestClient(main.app) as client:
        response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


def test_login_sets_http_only_cookie_and_unlocks_app(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "tajné heslo")
    monkeypatch.setattr(main, "AUTH_SECRET", "signing-secret")
    monkeypatch.setattr(main, "AUTH_COOKIE_SECURE_RAW", "false")
    with TestClient(main.app) as client:
        response = client.post(
            "/login",
            data={"password": "tajné heslo", "next": "/"},
            headers={"origin": "http://testserver"},
            follow_redirects=False,
        )
        page = client.get("/")
    assert response.status_code == 303
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    assert page.status_code == 200


def test_login_rejects_cross_origin_and_bad_password(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "heslo")
    monkeypatch.setattr(main, "AUTH_SECRET", "secret")
    with TestClient(main.app) as client:
        cross_origin = client.post(
            "/login",
            data={"password": "heslo"},
            headers={"origin": "https://attacker.example"},
        )
        wrong = client.post(
            "/login",
            data={"password": "špatně"},
            headers={"origin": "http://testserver"},
        )
    assert cross_origin.status_code == 403
    assert wrong.status_code == 401
    assert "Nesprávné heslo" in wrong.text


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (None, "/"),
        ("https://evil.example", "/"),
        ("//evil.example", "/"),
        ("/\\evil", "/"),
        ("/safe?x=1", "/safe?x=1"),
    ],
)
def test_sanitize_next_path(candidate, expected):
    assert main.sanitize_next_path(candidate) == expected


def test_signed_token_rejects_tampering_and_non_ascii_signature(monkeypatch):
    monkeypatch.setattr(main, "AUTH_SECRET", "secret")
    token = main.create_auth_token()
    assert main.verify_auth_token(token)
    payload, signature = token.split(".")
    assert not main.verify_auth_token(f"{payload}.{signature[:-1]}x")
    assert not main.verify_auth_token(f"{payload}.ž")
    assert not main.verify_auth_token(f"ž.{signature}")


def test_origin_defaults_to_exact_request_host(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    assert main.is_origin_allowed("https://translator.example", "translator.example")
    assert not main.is_origin_allowed("https://attacker.example", "translator.example")
    assert not main.is_origin_allowed("ftp://translator.example", "translator.example")
    assert not main.is_origin_allowed(None, "translator.example")


def test_security_headers_only_allow_same_origin_microphone(open_client):
    response = open_client.get("/")
    assert response.headers["permissions-policy"] == "microphone=(self)"
    csp = response.headers["content-security-policy"]
    assert "connect-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "cdn" not in csp.lower()


def test_single_purpose_page_has_no_settings_or_legacy_engines(open_client):
    html = open_client.get("/").text.lower()
    assert "<select" not in html
    assert 'type="range"' not in html
    assert "webspeech" not in html
    assert "deepgram" not in html
    assert "elevenlabs" not in html
    assert "nemotron" not in html
    assert "parakeet" not in html
    assert "chunk" not in html
    assert "/ws/audio" in html
    assert 'lang="en"' in html
    assert 'lang="ru"' in html


def test_legacy_http_routes_are_gone(open_client):
    for path in (
        "/deepgram",
        "/api/translate/languages",
        "/api/azure/token",
        "/api/elevenlabs/token",
    ):
        assert open_client.get(path).status_code == 404


def test_ws_forwards_pcm_and_emits_atomic_interim_and_final(google_client):
    client, fake = google_client
    pcm = b"\x01\x00" * 1600

    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(pcm)
        interim = websocket.receive_json()
        websocket.send_text(json.dumps({"type": "stop"}))
        final = websocket.receive_json()
        ended = websocket.receive_json()

    assert interim == {"type": "interim", "en": "EN:ahoj", "ru": "RU:ahoj"}
    assert final == {"type": "final", "en": "EN:ahoj", "ru": "RU:ahoj"}
    assert ended == {"type": "ended"}
    assert len(fake.connect_calls) == 1
    connect = fake.connect_calls[0]
    assert connect["model"] == "gemini-3.5-transcribe-live"
    assert connect["config"].input_audio_transcription.language_codes == ["cs-CZ"]
    assert connect["config"].input_audio_transcription.mode.value == "SMART"
    assert fake.session.sent[0]["audio"].data == pcm
    assert fake.session.sent[0]["audio"].mime_type == "audio/pcm;rate=16000"
    assert fake.session.sent[-1] == {"audio_stream_end": True}
    assert [call["model"] for call in fake.translation_calls] == [
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash-lite",
    ]
    assert all(
        set(call["config"].response_json_schema["required"]) == {"en", "ru"}
        for call in fake.translation_calls
    )
    assert fake.session.closed
    assert fake.client_closed


@pytest.mark.parametrize(
    ("payload", "error_code", "close_code"),
    [
        (
            {"text": json.dumps({"type": "config", "language": "de"})},
            "invalid_message",
            1003,
        ),
        ({"bytes": b"x"}, "invalid_audio", 1003),
        (
            {"bytes": b"\0" * (google_audio.MAX_AUDIO_CHUNK_BYTES + 2)},
            "audio_too_large",
            1009,
        ),
    ],
)
def test_ws_rejects_configuration_and_invalid_audio(
    google_client, payload, error_code, close_code
):
    client, fake = google_client
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        if "text" in payload:
            websocket.send_text(payload["text"])
        else:
            websocket.send_bytes(payload["bytes"])
        assert websocket.receive_json() == {
            "type": "error",
            "code": error_code,
            "recoverable": False,
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
    assert closed.value.code == close_code
    assert not fake.translation_calls


def test_ws_checks_origin_before_accepting(google_client):
    client, fake = google_client
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect(
            "/ws/audio", headers={"origin": "https://attacker.example"}
        ):
            pass
    assert closed.value.code == 1008
    assert not fake.connect_calls


def test_ws_checks_auth_before_accepting(monkeypatch):
    fake = FakeGoogleClient()
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "password")
    monkeypatch.setattr(main, "AUTH_SECRET", "secret")
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-google-key")
    monkeypatch.setattr(main, "_create_google_client", lambda: fake)
    with TestClient(main.app) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(
                "/ws/audio", headers={"origin": "http://testserver"}
            ):
                pass
    assert closed.value.code == 1008
    assert not fake.connect_calls


def test_ws_refuses_missing_google_key_before_accept(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "GEMINI_API_KEY", "")
    with TestClient(main.app) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(
                "/ws/audio", headers={"origin": "http://testserver"}
            ):
                pass
    assert closed.value.code == 1011


def test_google_live_setup_timeout_reports_error_and_releases_slot(monkeypatch):
    fake = FakeGoogleClient(connect_hangs=True)
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-google-key")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "_create_google_client", lambda: fake)
    monkeypatch.setattr(main, "_audio_session_slots", slots)
    monkeypatch.setattr(main, "_GOOGLE_SETUP_TIMEOUT_SECONDS", 0.01)

    with TestClient(main.app) as client:
        with client.websocket_connect(
            "/ws/audio", headers={"origin": "http://testserver"}
        ) as websocket:
            assert websocket.receive_json() == {
                "type": "error",
                "code": "google_timeout",
                "recoverable": True,
            }
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()

    assert closed.value.code == 1011
    assert fake.closed_event.wait(timeout=1.0)
    assert slots.acquire(blocking=False)
    slots.release()


def test_google_live_context_exit_is_bounded(monkeypatch):
    fake = FakeGoogleClient(exit_hangs=True)
    monkeypatch.setattr(main, "_GOOGLE_CLOSE_TIMEOUT_SECONDS", 0.01)

    async def scenario():
        with pytest.raises(asyncio.TimeoutError):
            async with main._connect_google_live(fake.aio):
                pass

    asyncio.run(scenario())


def test_terminal_error_precedes_drain_without_dropping_accepted_final(monkeypatch):
    class FailingAfterFinalSession(FakeGoogleSession):
        def __init__(self):
            super().__init__()
            self.receive_calls = 0

        def receive(self):
            async def turn():
                self.receive_calls += 1
                if self.receive_calls == 1:
                    yield _event(final="přijatý finální text", turn_complete=True)
                    return
                raise RuntimeError("upstream failed after accepted final")

            return turn()

    fake = FakeGoogleClient()
    fake.session = FailingAfterFinalSession()
    fake.translation_release = threading.Event()
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-google-key")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "_create_google_client", lambda: fake)

    with TestClient(main.app) as client:
        with client.websocket_connect(
            "/ws/audio", headers={"origin": "http://testserver"}
        ) as websocket:
            assert websocket.receive_json() == {"type": "ready"}
            assert websocket.receive_json() == {
                "type": "error",
                "code": "session_failed",
                "recoverable": True,
            }
            assert fake.translation_started.wait(timeout=1.0)
            fake.translation_release.set()
            assert websocket.receive_json() == {
                "type": "final",
                "en": "EN:přijatý finální text",
                "ru": "RU:přijatý finální text",
            }
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()

    assert closed.value.code == 1011
    assert fake.client_closed


def test_audio_rate_limiter_bounds_paid_bytes_to_realtime_burst(monkeypatch):
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)
    limiter = main._AudioRateLimiter()
    assert google_audio.MAX_AUDIO_CHUNK_BYTES == 3200
    assert all(limiter.consume(3200) for _ in range(20))
    assert not limiter.consume(2)


def test_ws_rejects_new_session_when_fixed_global_cap_is_full(
    google_client, monkeypatch
):
    client, fake = google_client

    class FullSlots:
        def acquire(self, *, blocking):
            assert blocking is False
            return False

    monkeypatch.setattr(main, "_audio_session_slots", FullSlots())
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {
            "type": "error",
            "code": "server_busy",
            "recoverable": True,
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
    assert closed.value.code == 1013
    assert not fake.connect_calls


def test_receive_loop_reopens_sdk_iterator_for_multiple_turns():
    class Session:
        def __init__(self):
            self.index = 0

        def receive(self):
            async def turn():
                self.index += 1
                if self.index == 1:
                    yield _event(final="první", turn_complete=True)
                elif self.index == 2:
                    yield _event(final="druhá", turn_complete=True)
                else:
                    raise RuntimeError("test complete")

            return turn()

    class Coordinator:
        def __init__(self):
            self.values = []

        def submit(self, kind, text):
            self.values.append((kind, text))

    async def scenario():
        coordinator = Coordinator()
        with pytest.raises(RuntimeError, match="test complete"):
            await main._receive_google_transcripts(
                Session(), coordinator, asyncio.Event()
            )
        return coordinator.values

    assert asyncio.run(scenario()) == [("final", "první"), ("final", "druhá")]


def test_browser_disconnect_closes_google_and_workers(google_client):
    client, fake = google_client
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        websocket.close()
        assert fake.closed_event.wait(timeout=1.0)
    assert fake.session.closed
    assert fake.client_closed


def test_session_limit_stops_input_before_stream_end_and_drains_final(
    google_client, monkeypatch
):
    client, fake = google_client
    monkeypatch.setattr(main, "_SESSION_MAX_SECONDS", 0.005)

    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        assert websocket.receive_json() == {
            "type": "error",
            "code": "session_limit",
            "recoverable": True,
        }
        assert websocket.receive_json() == {
            "type": "final",
            "en": "EN:ahoj",
            "ru": "RU:ahoj",
        }
        assert websocket.receive_json() == {"type": "ended"}

    assert fake.session.sent == [{"audio_stream_end": True}]


class StopSequenceSession(FakeGoogleSession):
    """Finals in distinct SDK turns, with controllable delays and stream EOF."""

    def __init__(self, results, *, eof=True):
        super().__init__()
        self.results = iter(results)
        self.eof = eof
        self.end_sent = asyncio.Event()

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)
        if kwargs.get("audio_stream_end"):
            self.end_sent.set()

    def receive(self):
        async def turn():
            await self.end_sent.wait()
            item = next(self.results, None)
            if item is None:
                if self.eof:
                    raise main.errors.APIError(1000, "Normal closure")
                await asyncio.Event().wait()
            delay, text = item
            await asyncio.sleep(delay)
            yield _event(final=text, turn_complete=True)

        return turn()


@pytest.mark.parametrize("delay", [0.01, 2.75])
def test_stop_receives_earlier_and_last_final_across_sdk_turns(google_client, delay):
    client, fake = google_client
    fake.session = StopSequenceSession(
        [(0, "předchozí věta"), (delay, "poslední věta")]
    )
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_json({"type": "stop"})
        assert websocket.receive_json() == {
            "type": "final",
            "en": "EN:předchozí věta",
            "ru": "RU:předchozí věta",
        }
        assert websocket.receive_json() == {
            "type": "final",
            "en": "EN:poslední věta",
            "ru": "RU:poslední věta",
        }
        assert websocket.receive_json() == {"type": "ended"}
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
    assert closed.value.code == 1000
    assert fake.closed_event.wait(1)


@pytest.mark.parametrize("results", [[], [(0, "dřívější final")]])
def test_stop_timeout_never_claims_complete_even_after_a_final(
    google_client, monkeypatch, results
):
    client, fake = google_client
    fake.session = StopSequenceSession(results, eof=False)
    monkeypatch.setattr(main, "_FINAL_TRANSCRIPT_GRACE_SECONDS", 0.05)
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_json({"type": "stop"})
        received = []
        with pytest.raises(WebSocketDisconnect) as closed:
            while True:
                received.append(websocket.receive_json())
    assert [r["type"] for r in received] == ["final"] * len(results) + ["error"]
    assert received[-1]["code"] == "transcription_incomplete"
    assert closed.value.code == 1011
    assert fake.closed_event.wait(1)


class ObservedSlots:
    def __init__(self):
        self.semaphore = threading.BoundedSemaphore(1)
        self.released = threading.Event()

    def acquire(self, *, blocking):
        return self.semaphore.acquire(blocking=blocking)

    def release(self):
        self.semaphore.release()
        self.released.set()


@pytest.mark.parametrize("disconnect", [False, True])
def test_blocked_stream_end_releases_handler_and_paid_slot(
    google_client, monkeypatch, disconnect
):
    client, fake = google_client
    entered = threading.Event()

    async def blocked_end(**kwargs):
        assert kwargs == {"audio_stream_end": True}
        entered.set()
        await asyncio.Event().wait()

    fake.session.send_realtime_input = blocked_end
    slots = ObservedSlots()
    monkeypatch.setattr(main, "_audio_session_slots", slots)
    monkeypatch.setattr(main, "_GOOGLE_SEND_TIMEOUT_SECONDS", 0.05)
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_json({"type": "stop"})
        assert entered.wait(1)
        if disconnect:
            websocket.close()
        else:
            assert websocket.receive_json()["code"] == "google_timeout"
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
            assert closed.value.code == 1011
        assert fake.closed_event.wait(1)
        assert slots.released.wait(1)
    assert fake.session.closed


@pytest.mark.parametrize("trigger", ["stop", "session_limit", "upstream"])
@pytest.mark.parametrize("slow_close", [False, True])
def test_disconnect_during_final_drain_cancels_paid_request_without_retry(
    google_client, monkeypatch, trigger, slow_close
):
    client, fake = google_client
    fake.translation_release = threading.Event()
    fake.exit_hangs = slow_close
    monkeypatch.setattr(main, "_GOOGLE_CLOSE_TIMEOUT_SECONDS", 0.3)
    slots = ObservedSlots()
    monkeypatch.setattr(main, "_audio_session_slots", slots)
    if trigger == "session_limit":
        monkeypatch.setattr(main, "_SESSION_MAX_SECONDS", 0.01)
    if trigger == "upstream":

        class FailedSession(FakeGoogleSession):
            def receive(self):
                async def turn():
                    yield _event(final="přijatý text", turn_complete=True)
                    raise ValueError("provider failure")

                return turn()

        fake.session = FailedSession()

    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        if trigger == "stop":
            websocket.send_json({"type": "stop"})
        else:
            assert websocket.receive_json()["type"] == "error"
        assert fake.translation_started.wait(1)
        websocket.close()
        # Cancellation must not wait for a stalled connection.__aexit__().
        assert fake.translation_cancelled.wait(0.15)
        assert fake.closed_event.wait(1)
        assert slots.released.wait(1)
    assert len(fake.translation_calls) == 1


def test_shutdown_deadline_cancels_final_drain_and_releases_slot(
    google_client, monkeypatch
):
    client, fake = google_client
    fake.translation_release = threading.Event()
    slots = ObservedSlots()
    monkeypatch.setattr(main, "_audio_session_slots", slots)
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT_SECONDS", 0.05)
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_json({"type": "stop"})
        assert fake.translation_started.wait(1)
        assert websocket.receive_json()["code"] == "google_timeout"
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()
        assert fake.translation_cancelled.wait(1)
        assert slots.released.wait(1)


@pytest.mark.parametrize("phase", ["setup", "receive"])
def test_provider_exception_payload_and_chain_never_enter_logs(
    google_client, caplog, phase
):
    client, fake = google_client
    marker = "PRIVATE_TRANSCRIPT_NEVER_LOG"
    failure = ValueError(f"Failed to parse response: {marker}")
    failure.__cause__ = RuntimeError(f"raw payload: {marker}")
    if phase == "setup":
        fake.connect_error = failure
    else:

        class FailedSession(FakeGoogleSession):
            def receive(self):
                async def turn():
                    raise failure
                    yield  # pragma: no cover

                return turn()

        fake.session = FailedSession()
    with caplog.at_level("ERROR", logger="realtime_translator"):
        with client.websocket_connect(
            "/ws/audio", headers={"origin": "http://testserver"}
        ) as websocket:
            if phase == "receive":
                assert websocket.receive_json() == {"type": "ready"}
            assert websocket.receive_json()["code"] == "session_failed"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_json()
    assert "ValueError" in caplog.text
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)
