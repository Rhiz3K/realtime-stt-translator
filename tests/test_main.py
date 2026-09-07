import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from app import google_audio, main


def _receive_app_message(websocket):
    """Existing lifecycle assertions ignore additive, separately tested telemetry."""
    while (message := websocket.receive_json()).get("type") == "telemetry":
        pass
    return message


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
            generation_complete=turn_complete,
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
        if kwargs.get("activity_end") is not None:
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


def test_web_speech_translates_both_languages_without_opening_google_stt(google_client):
    client, fake = google_client
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        ws.send_json({"type": "interim", "text": "dobrý"})
        assert _receive_app_message(ws) == {
            "type": "interim",
            "en": "EN:dobrý",
            "ru": "RU:dobrý",
        }
        ws.send_json({"type": "final", "text": "Dobrý den."})
        ws.send_json({"type": "final", "text": "Dobrý den."})
        ws.send_json({"type": "stop"})
        for _ in range(2):
            assert _receive_app_message(ws) == {
                "type": "final",
                "en": "EN:Dobrý den.",
                "ru": "RU:Dobrý den.",
            }
        assert _receive_app_message(ws) == {"type": "ended"}
    assert fake.connect_calls == []
    assert fake.client_closed
    assert [call["model"] for call in fake.translation_calls] == [
        google_audio.INTERIM_TRANSLATION_MODEL,
        google_audio.FINAL_TRANSLATION_MODEL,
        google_audio.FINAL_TRANSLATION_MODEL,
    ]


def test_empty_web_speech_stop_does_not_call_any_google_api(google_client):
    client, fake = google_client
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        ws.send_json({"type": "stop"})
        assert _receive_app_message(ws) == {"type": "ended"}
    assert fake.connect_calls == fake.translation_calls == []


@pytest.mark.parametrize(
    "payload,code",
    [
        (None, "invalid_message"),
        ([], "invalid_message"),
        ({"type": "config", "lang": "en-US"}, "invalid_message"),
        ({"type": "final", "text": ""}, "invalid_message"),
        ({"type": "final", "text": 5}, "invalid_message"),
        ({"type": "interim", "text": "x", "model": "other"}, "invalid_message"),
        (
            {"type": "final", "text": "x" * (google_audio.MAX_TRANSCRIPT_CHARS + 1)},
            "transcript_too_large",
        ),
    ],
)
def test_web_speech_validates_protocol_before_paid_translation(
    google_client, payload, code
):
    client, fake = google_client
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        ws.send_json(payload)
        assert _receive_app_message(ws)["code"] == code
        with pytest.raises(WebSocketDisconnect):
            _receive_app_message(ws)
    assert fake.connect_calls == fake.translation_calls == []


def test_web_speech_rejects_binary_audio(google_client):
    client, fake = google_client
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        ws.send_bytes(b"\0\0")
        assert _receive_app_message(ws)["code"] == "invalid_message"
    assert fake.connect_calls == fake.translation_calls == []


@pytest.mark.parametrize(
    "origin,authenticated",
    [
        ("https://evil.example", True),
        (None, True),
        ("http://testserver", False),
    ],
)
def test_web_speech_checks_auth_and_origin_before_accept(
    google_client, monkeypatch, origin, authenticated
):
    client, fake = google_client
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "valid-password")
    if authenticated:
        client.cookies.set(main.AUTH_COOKIE_NAME, main.create_auth_token())
    with pytest.raises(WebSocketDisconnect) as caught:
        with client.websocket_connect(
            "/ws/transcript", headers={"origin": origin} if origin else {}
        ):
            pass
    assert caught.value.code == 1008
    assert fake.connect_calls == fake.translation_calls == []


def test_web_speech_uses_shared_paid_session_limit(google_client, monkeypatch):
    client, fake = google_client
    monkeypatch.setattr(main, "_audio_session_slots", threading.BoundedSemaphore(0))
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws)["code"] == "server_busy"
    assert fake.connect_calls == fake.translation_calls == []


def test_web_speech_disconnect_cancels_paid_final_during_stop_drain(google_client):
    client, fake = google_client
    fake.translation_release = threading.Event()
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        ws.send_json({"type": "final", "text": "ahoj"})
        assert fake.translation_started.wait(2)
        ws.send_json({"type": "stop"})
        ws.close()
    assert fake.translation_cancelled.wait(2)
    assert fake.closed_event.wait(2)
    assert fake.connect_calls == []


def test_withdrawn_browser_interim_cancels_paid_preview_but_keeps_finals(google_client):
    client, fake = google_client
    fake.translation_release = threading.Event()
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        ws.send_json({"type": "interim", "text": "rozpracované"})
        assert fake.translation_started.wait(2)
        ws.send_json({"type": "interim", "text": ""})
        assert _receive_app_message(ws) == {"type": "interim_reset"}
        assert fake.translation_cancelled.wait(2)
        fake.translation_release.set()
        ws.send_json({"type": "final", "text": "hotovo"})
        ws.send_json({"type": "stop"})
        assert _receive_app_message(ws) == {
            "type": "final",
            "en": "EN:hotovo",
            "ru": "RU:hotovo",
        }
        assert _receive_app_message(ws) == {"type": "ended"}


def test_web_speech_message_rate_is_bounded(google_client):
    client, _ = google_client
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        for _ in range(100):
            ws.send_json({"type": "interim", "text": ""})
        while (result := _receive_app_message(ws))["type"] != "error":
            assert result == {"type": "interim_reset"}
        assert result["code"] == "transcript_rate_exceeded"


def test_web_speech_protocol_error_reports_then_drains_accepted_final(google_client):
    client, fake = google_client
    fake.translation_release = threading.Event()
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        ws.send_json({"type": "final", "text": "uložit"})
        assert fake.translation_started.wait(2)
        ws.send_json({"type": "config"})
        assert _receive_app_message(ws)["code"] == "invalid_message"
        fake.translation_release.set()
        assert _receive_app_message(ws) == {
            "type": "final",
            "en": "EN:uložit",
            "ru": "RU:uložit",
        }
        with pytest.raises(WebSocketDisconnect) as caught:
            _receive_app_message(ws)
        assert caught.value.code == 1008


def test_web_speech_server_limit_finishes_without_opening_google_stt(
    google_client, monkeypatch
):
    client, fake = google_client
    monkeypatch.setattr(main, "_SESSION_MAX_SECONDS", 0.01)
    with client.websocket_connect(
        "/ws/transcript", headers={"origin": "http://testserver"}
    ) as ws:
        assert _receive_app_message(ws) == {"type": "ready"}
        assert _receive_app_message(ws)["code"] == "session_limit"
        assert _receive_app_message(ws) == {"type": "ended"}
    assert fake.connect_calls == []


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


def test_login_form_preserves_same_origin_for_native_post(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "heslo")
    monkeypatch.setattr(main, "AUTH_SECRET", "secret")
    with TestClient(main.app) as client:
        page = client.get("/login")
        wrong = client.post(
            "/login",
            data={"password": "wrong"},
            headers={"origin": "http://testserver"},
        )
    assert page.status_code == 200
    assert wrong.status_code == 401
    for response in (page, wrong):
        # no-referrer makes native form POSTs send Origin: null in browsers.
        assert response.headers["referrer-policy"] == "same-origin"
        assert '<form method="post" action="/login">' in response.text
        assert "form-action 'self'" in response.headers["content-security-policy"]


@pytest.mark.parametrize("origin", ["https://attacker.example", "null"])
def test_login_rejects_cross_origin_and_bad_password(monkeypatch, origin):
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "APP_PASSWORD", "heslo")
    monkeypatch.setattr(main, "AUTH_SECRET", "secret")
    with TestClient(main.app) as client:
        cross_origin = client.post(
            "/login",
            data={"password": "heslo"},
            headers={"origin": origin},
        )
        wrong = client.post(
            "/login",
            data={"password": "špatně"},
            headers={"origin": "http://testserver"},
        )
    assert cross_origin.status_code == 403
    assert cross_origin.json() == {"detail": "Cross-origin login not allowed"}
    assert "set-cookie" not in cross_origin.headers
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
    assert html.count('type="range"') == 1
    assert 'id="font-size"' in html
    assert "webspeech" not in html
    assert "deepgram" not in html
    assert "elevenlabs" not in html
    assert "nemotron" not in html
    assert "parakeet" not in html
    assert "chunk" not in html
    assert "/static/audio-session.js" in html
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
        assert _receive_app_message(websocket) == {"type": "ready"}
        websocket.send_bytes(pcm)
        interim = _receive_app_message(websocket)
        websocket.send_text(json.dumps({"type": "stop"}))
        final = _receive_app_message(websocket)
        ended = _receive_app_message(websocket)

    assert interim == {"type": "interim", "en": "EN:ahoj", "ru": "RU:ahoj"}
    assert final == {"type": "final", "en": "EN:ahoj", "ru": "RU:ahoj"}
    assert ended == {"type": "ended"}
    assert len(fake.connect_calls) == 1
    connect = fake.connect_calls[0]
    assert connect["model"] == "gemini-3.5-transcribe-live"
    assert connect["config"].input_audio_transcription.language_codes == ["cs-CZ"]
    assert connect["config"].input_audio_transcription.mode.value == "SMART"
    assert fake.session.sent[1]["audio"].data == pcm
    assert fake.session.sent[1]["audio"].mime_type == "audio/pcm;rate=16000"
    assert set(fake.session.sent[0]) == {"activity_start"}
    assert set(fake.session.sent[-1]) == {"activity_end"}
    assert [call["model"] for call in fake.translation_calls] == [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
    ]
    assert all(
        set(call["config"].response_json_schema["required"]) == {"en", "ru"}
        for call in fake.translation_calls
    )
    assert fake.session.closed
    assert fake.client_closed


@pytest.mark.parametrize("endpoint_hangs", [False, True])
def test_manual_boundaries_preserve_pcm_order_and_bound_marker_send(
    monkeypatch, endpoint_hangs
):
    monkeypatch.setattr(google_audio, "SEGMENT_MAX_SECONDS", 0.2)
    monkeypatch.setattr(main, "_GOOGLE_SEND_TIMEOUT_SECONDS", 0.01)
    frames = [b"\x40\x1f" * 1600 for _ in range(4)]

    async def scenario():
        sent = []
        messages = iter(
            [
                *[{"type": "websocket.receive", "bytes": frame} for frame in frames],
                {"type": "websocket.receive", "text": '{"type":"stop"}'},
                {"type": "websocket.receive", "bytes": frames[0]},
                {"type": "websocket.disconnect"},
            ]
        )

        class Browser:
            async def receive(self):
                return next(messages)

        class Google:
            def __init__(self):
                self.messages = asyncio.Queue()

            async def send_realtime_input(self, **kwargs):
                sent.append(kwargs)
                if "activity_end" in kwargs:
                    if endpoint_hangs:
                        await asyncio.Future()
                    await self.messages.put(_event(turn_complete=True))

            async def receive(self):
                yield await self.messages.get()

        google = main.GoogleAudioSession(Google())

        async def read():
            while True:
                async for _ in google.receive():
                    pass

        reader = asyncio.create_task(read())

        stop = asyncio.Event()
        lock = asyncio.Lock()
        if endpoint_hangs:
            with pytest.raises(asyncio.TimeoutError):
                await main._receive_browser_audio(Browser(), google, stop, lock)
            assert len(sent) == 4
        else:
            await main._receive_browser_audio(Browser(), google, stop, lock)
            assert stop.is_set()
            assert [item["audio"].data for item in sent if "audio" in item] == frames
            assert [
                index for index, item in enumerate(sent) if "activity_end" in item
            ] == [3, 7]
            assert not any("audio_stream_end" in item for item in sent)
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        assert not lock.locked()

    asyncio.run(scenario())


def test_missing_generation_ack_stops_visibly_and_keeps_accepted_final(
    google_client, monkeypatch
):
    client, fake = google_client
    monkeypatch.setattr(google_audio, "SEGMENT_MAX_SECONDS", 0.1)
    monkeypatch.setattr(google_audio, "TURN_FINALIZATION_TIMEOUT_SECONDS", 0.05)

    class NoGenerationAck(FakeGoogleSession):
        async def send_realtime_input(self, **kwargs):
            self.sent.append(kwargs)
            if "activity_end" in kwargs:
                # Final text does not satisfy the generation acknowledgement.
                await self.turns.put([_event(final="přijatá věta")])

    fake.session = NoGenerationAck()
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert _receive_app_message(websocket) == {"type": "ready"}
        websocket.send_bytes(b"\x40\x1f" * 1600)
        received = []
        with pytest.raises(WebSocketDisconnect) as closed:
            while True:
                received.append(_receive_app_message(websocket))
    assert closed.value.code == 1011
    assert {
        "type": "final",
        "en": "EN:přijatá věta",
        "ru": "RU:přijatá věta",
    } in received
    assert any(item.get("code") == "google_timeout" for item in received)
    assert not any(item["type"] == "ended" for item in received)
    assert [next(iter(item)) for item in fake.session.sent] == [
        "activity_start",
        "audio",
        "activity_end",
    ]
    assert fake.closed_event.wait(1)


def test_ws_sends_safe_progress_and_timings_without_changing_pair_contract(
    google_client,
):
    client, _ = google_client
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(b"\x01\x00" * 1600)
        progress = websocket.receive_json()
        pair = websocket.receive_json()
        timing = websocket.receive_json()
        assert progress["type"] == "telemetry"
        assert progress["stage"] == "translating"
        assert progress["kind"] == "interim"
        assert progress["hypotheses"] == 1
        assert progress["first_stt_ms"] >= 0
        assert pair == {"type": "interim", "en": "EN:ahoj", "ru": "RU:ahoj"}
        assert timing["stage"] == "translated"
        assert timing["translation_ms"] >= 0
        assert timing["queue_ms"] >= 0
        assert "ahoj" not in json.dumps([progress, timing])


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
        assert _receive_app_message(websocket) == {"type": "ready"}
        if "text" in payload:
            websocket.send_text(payload["text"])
        else:
            websocket.send_bytes(payload["bytes"])
        assert _receive_app_message(websocket) == {
            "type": "error",
            "code": error_code,
            "recoverable": False,
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_app_message(websocket)
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
            assert _receive_app_message(websocket) == {
                "type": "error",
                "code": "google_timeout",
                "recoverable": True,
            }
            with pytest.raises(WebSocketDisconnect) as closed:
                _receive_app_message(websocket)

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
            assert _receive_app_message(websocket) == {"type": "ready"}
            assert _receive_app_message(websocket) == {
                "type": "error",
                "code": "session_failed",
                "recoverable": True,
            }
            assert fake.translation_started.wait(timeout=1.0)
            fake.translation_release.set()
            assert _receive_app_message(websocket) == {
                "type": "final",
                "en": "EN:přijatý finální text",
                "ru": "RU:přijatý finální text",
            }
            with pytest.raises(WebSocketDisconnect) as closed:
                _receive_app_message(websocket)

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
        assert _receive_app_message(websocket) == {
            "type": "error",
            "code": "server_busy",
            "recoverable": True,
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_app_message(websocket)
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
        assert _receive_app_message(websocket) == {"type": "ready"}
        websocket.close()
        assert fake.closed_event.wait(timeout=1.0)
    assert fake.session.closed
    assert fake.client_closed


def test_session_limit_stops_input_before_stream_end_and_drains_final(
    google_client, monkeypatch
):
    client, fake = google_client
    fake.session = StopSequenceSession([(0, "ahoj")])
    monkeypatch.setattr(main, "_SESSION_MAX_SECONDS", 0.05)

    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert _receive_app_message(websocket) == {"type": "ready"}
        websocket.send_bytes(b"\0\0")
        assert _receive_app_message(websocket) == {
            "type": "error",
            "code": "session_limit",
            "recoverable": True,
        }
        assert _receive_app_message(websocket) == {
            "type": "final",
            "en": "EN:ahoj",
            "ru": "RU:ahoj",
        }
        assert _receive_app_message(websocket) == {"type": "ended"}

    assert set(fake.session.sent[0]) == {"activity_start"}
    assert fake.session.sent[1]["audio"].data == b"\0\0"
    assert set(fake.session.sent[-1]) == {"activity_end"}


class StopSequenceSession(FakeGoogleSession):
    """Finals in distinct SDK turns, with controllable delays and stream EOF."""

    def __init__(self, results, *, eof=True):
        super().__init__()
        self.results = iter(results)
        self.eof = eof
        self.end_sent = asyncio.Event()

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)
        if kwargs.get("activity_end") is not None:
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


@pytest.mark.parametrize("trigger", ["stop", "session_limit"])
def test_empty_session_finishes_without_stream_end_or_false_warning(
    google_client, monkeypatch, trigger
):
    client, fake = google_client
    fake.session = StopSequenceSession([], eof=False)
    # Waiting for the upstream in this test would exceed the whole shutdown
    # budget. No audio means there is genuinely no tail to confirm.
    monkeypatch.setattr(main, "_FINAL_TRANSCRIPT_GRACE_SECONDS", 60)
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT_SECONDS", 0.2)
    if trigger == "session_limit":
        monkeypatch.setattr(main, "_SESSION_MAX_SECONDS", 0.01)
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert _receive_app_message(websocket) == {"type": "ready"}
        if trigger == "stop":
            websocket.send_json({"type": "stop"})
            websocket.send_json({"type": "stop"})
        else:
            assert _receive_app_message(websocket)["code"] == "session_limit"
        assert _receive_app_message(websocket) == {"type": "ended"}
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_app_message(websocket)
    assert closed.value.code == 1000
    assert fake.closed_event.wait(1)
    assert fake.session.sent == []
    assert fake.translation_calls == []
    assert fake.session.closed


@pytest.mark.parametrize("delay", [0.01, 2.75])
def test_stop_receives_earlier_and_last_final_across_sdk_turns(google_client, delay):
    client, fake = google_client
    fake.session = StopSequenceSession(
        [(0, "předchozí věta"), (delay, "poslední věta")]
    )
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert _receive_app_message(websocket) == {"type": "ready"}
        websocket.send_bytes(b"\0\0")
        websocket.send_json({"type": "stop"})
        assert _receive_app_message(websocket) == {
            "type": "final",
            "en": "EN:předchozí věta",
            "ru": "RU:předchozí věta",
        }
        assert _receive_app_message(websocket) == {
            "type": "final",
            "en": "EN:poslední věta",
            "ru": "RU:poslední věta",
        }
        assert _receive_app_message(websocket) == {"type": "ended"}
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_app_message(websocket)
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
        assert _receive_app_message(websocket) == {"type": "ready"}
        # Even one silent sample counts: a rounded UI timer of 00:00 must not
        # be mistaken for an empty input stream.
        websocket.send_bytes(b"\0\0")
        websocket.send_json({"type": "stop"})
        received = []
        with pytest.raises(WebSocketDisconnect) as closed:
            while True:
                received.append(_receive_app_message(websocket))
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
        if "audio" in kwargs or "activity_start" in kwargs:
            return
        assert set(kwargs) == {"activity_end"}
        entered.set()
        await asyncio.Event().wait()

    fake.session.send_realtime_input = blocked_end
    slots = ObservedSlots()
    monkeypatch.setattr(main, "_audio_session_slots", slots)
    monkeypatch.setattr(main, "_GOOGLE_SEND_TIMEOUT_SECONDS", 0.05)
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert _receive_app_message(websocket) == {"type": "ready"}
        websocket.send_bytes(b"\0\0")
        websocket.send_json({"type": "stop"})
        assert entered.wait(1)
        if disconnect:
            websocket.close()
        else:
            assert _receive_app_message(websocket)["code"] == "google_timeout"
            with pytest.raises(WebSocketDisconnect) as closed:
                _receive_app_message(websocket)
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
    fake.session = StopSequenceSession([(0, "přijatý text")])
    monkeypatch.setattr(main, "_GOOGLE_CLOSE_TIMEOUT_SECONDS", 0.3)
    slots = ObservedSlots()
    monkeypatch.setattr(main, "_audio_session_slots", slots)
    if trigger == "session_limit":
        monkeypatch.setattr(main, "_SESSION_MAX_SECONDS", 0.05)
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
        assert _receive_app_message(websocket) == {"type": "ready"}
        if trigger != "upstream":
            websocket.send_bytes(b"\0\0")
        if trigger == "stop":
            websocket.send_json({"type": "stop"})
        else:
            assert _receive_app_message(websocket)["type"] == "error"
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
    fake.session = StopSequenceSession([(0, "přijatý text")])
    fake.translation_release = threading.Event()
    slots = ObservedSlots()
    monkeypatch.setattr(main, "_audio_session_slots", slots)
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT_SECONDS", 0.05)
    with client.websocket_connect(
        "/ws/audio", headers={"origin": "http://testserver"}
    ) as websocket:
        assert _receive_app_message(websocket) == {"type": "ready"}
        websocket.send_bytes(b"\0\0")
        websocket.send_json({"type": "stop"})
        assert fake.translation_started.wait(1)
        assert _receive_app_message(websocket)["code"] == "google_timeout"
        with pytest.raises(WebSocketDisconnect):
            _receive_app_message(websocket)
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
                assert _receive_app_message(websocket) == {"type": "ready"}
            assert _receive_app_message(websocket)["code"] == "session_failed"
            with pytest.raises(WebSocketDisconnect):
                _receive_app_message(websocket)
    assert "ValueError" in caplog.text
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)
