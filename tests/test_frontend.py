from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "app/templates/index.html").read_text(encoding="utf-8")
WORKLET = (ROOT / "app/static/pcm-worklet.js").read_text(encoding="utf-8")


def test_frontend_has_only_one_action_and_no_configuration_controls():
    assert INDEX.count('<button id="toggle"') == 1
    assert "<select" not in INDEX.lower()
    assert 'type="range"' not in INDEX.lower()
    assert 'type="number"' not in INDEX.lower()
    assert "localStorage" not in INDEX
    assert "WebSpeech" not in INDEX
    assert "SpeechRecognition" not in INDEX
    assert 'JSON.stringify({type: "stop"})' in INDEX
    assert 'type: "config"' not in INDEX


def test_pair_rendering_is_atomic_and_uses_text_content():
    assert "function renderInterim(en, ru)" in INDEX
    assert "function renderFinal(en, ru)" in INDEX
    assert "liveEn.textContent = en" in INDEX
    assert "liveRu.textContent = ru" in INDEX
    assert '.querySelector(".result-en").textContent = en' in INDEX
    assert '.querySelector(".result-ru").textContent = ru' in INDEX
    assert "innerHTML" not in INDEX


def test_stop_flushes_audio_before_sending_control_message():
    flush_at = INDEX.index('captureNode.port.postMessage({type: "flush"})')
    stop_function_at = INDEX.index("function sendStop(run, activeSocket)")
    assert stop_function_at < flush_at
    assert 'event.data.type === "flushed"' in INDEX
    assert INDEX.count('JSON.stringify({type: "stop"})') == 1


def test_worklet_emits_fixed_pcm16_100ms_frames():
    assert "this.targetRate = 16000" in WORKLET
    assert "new Int16Array(1600)" in WORKLET
    assert "32768" in WORKLET and "32767" in WORKLET
    assert 'registerProcessor("pcm16-capture"' in WORKLET
    assert "this.ratio = sampleRate / this.targetRate" in WORKLET


def test_inline_javascript_parses_when_node_is_available(tmp_path):
    node = shutil.which("node")
    if not node:
        return
    scripts = re.findall(r"<script>(.*?)</script>", INDEX, flags=re.DOTALL)
    assert len(scripts) == 1
    source = tmp_path / "index-inline.js"
    source.write_text(scripts[0], encoding="utf-8")
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
