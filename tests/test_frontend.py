from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "app/templates/index.html").read_text(encoding="utf-8")
TOOLBAR = (ROOT / "app/templates/components/toolbar.html").read_text(encoding="utf-8")
UI = (ROOT / "app/static/translator-ui.js").read_text(encoding="utf-8")
SESSION = (ROOT / "app/static/audio-session.js").read_text(encoding="utf-8")
WORKLET = (ROOT / "app/static/pcm-worklet.js").read_text(encoding="utf-8")


def test_frontend_has_presentation_controls_but_no_pipeline_configuration():
    assert TOOLBAR.count('<button id="toggle"') == 1
    assert TOOLBAR.count('type="range"') == 1
    assert 'id="font-size"' in TOOLBAR
    for control in (
        "fullscreen-toggle",
        "theme-toggle",
        "compact-toggle",
        "follow-latest",
    ):
        assert control in TOOLBAR
    for source in (INDEX, TOOLBAR, UI, SESSION):
        assert "<select" not in source.lower()
        assert 'type="number"' not in source.lower()
        assert "WebSpeech" not in source
        assert "SpeechRecognition" not in source
        assert 'type: "config"' not in source
    assert 'JSON.stringify({type: "stop"})' in SESSION
    assert "localStorage" not in SESSION


def test_pair_rendering_is_atomic_and_uses_text_content():
    assert "function renderInterim(en, ru)" in UI
    assert "function renderFinal(en, ru)" in UI
    assert "liveEn.textContent = en" in UI
    assert "liveRu.textContent = ru" in UI
    assert '.querySelector(".result-en").textContent = en' in UI
    assert '.querySelector(".result-ru").textContent = ru' in UI
    assert "innerHTML" not in UI + SESSION
    assert "{{ translation_pair(live=true) }}" in INDEX
    assert "{{ translation_pair() }}" in INDEX


def test_stop_flushes_audio_before_sending_control_message():
    flush_at = SESSION.index('captureNode.port.postMessage({type: "flush"})')
    stop_function_at = SESSION.index("function sendStop(run, activeSocket)")
    assert stop_function_at < flush_at
    assert 'event.data.type === "flushed"' in SESSION
    assert SESSION.count('JSON.stringify({type: "stop"})') == 1


def test_worklet_emits_fixed_pcm16_100ms_frames():
    assert "this.targetRate = 16000" in WORKLET
    assert "new Int16Array(1600)" in WORKLET
    assert "32768" in WORKLET and "32767" in WORKLET
    assert 'registerProcessor("pcm16-capture"' in WORKLET
    assert "this.ratio = sampleRate / this.targetRate" in WORKLET


@pytest.mark.parametrize("script", ["translator-ui.js", "audio-session.js"])
def test_application_javascript_parses_when_node_is_available(script):
    node = shutil.which("node")
    if not node:
        return
    assert f'<script src="/static/{script}" defer></script>' in INDEX
    source = ROOT / "app/static" / script
    result = subprocess.run(
        [node, "--check", str(source)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_worklet_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if not node:
        return
    result = subprocess.run(
        [node, "--check", str(ROOT / "app/static/pcm-worklet.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_frontend_session_runtime_regressions():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for browser API runtime regression tests")
    result = subprocess.run(
        [node, "--test", str(ROOT / "tests/frontend-runtime.test.cjs")],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
