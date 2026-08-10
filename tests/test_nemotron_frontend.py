import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_node(script):
    """Run an ES-module snippet against the real engine sources.

    Surfaces the JS error text on failure — check=True alone would only report
    the exit status and hide which assertion threw.
    """
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n--- stderr ---\n{proc.stderr}\n--- stdout ---\n{proc.stdout}")


@requires_node
def test_nemotron_webgpu_limit_helper_selects_encoder_variant_by_storage_buffer_limit():
    script = """
        import {
          checkNemotronWebGpuLimits,
          nemotronModelAssetUrl,
          selectNemotronEncoderVariantForLimit,
        } from './app/static/nemotron/nemotron-engine.mjs';

        const standard = selectNemotronEncoderVariantForLimit(25);
        if (standard.file !== 'encoder_fp16.onnx' || standard.minStorageBuffers !== 25) {
          throw new Error(`unexpected standard variant: ${JSON.stringify(standard)}`);
        }

        const limit16 = checkNemotronWebGpuLimits({ maxStorageBuffersPerShaderStage: 16 });
        if (limit16.ok !== true || limit16.variant.file !== 'encoder_fp16_concat16.onnx') {
          throw new Error(`unexpected limit16 result: ${JSON.stringify(limit16)}`);
        }

        const limit8 = checkNemotronWebGpuLimits({ maxStorageBuffersPerShaderStage: 8 });
        if (limit8.ok !== true || limit8.variant.file !== 'encoder_fp16_concat8.onnx') {
          throw new Error(`unexpected limit8 result: ${JSON.stringify(limit8)}`);
        }

        const tooLow = checkNemotronWebGpuLimits({ maxStorageBuffersPerShaderStage: 7 });
        if (tooLow.ok !== false || !tooLow.reason.includes('7') || !tooLow.reason.includes('8')) {
          throw new Error(`unexpected too-low result: ${JSON.stringify(tooLow)}`);
        }

        const url = nemotronModelAssetUrl('/static/nemotron/models', 'vocab.json');
        if (url !== '/static/nemotron/models/vocab.json?v=2') {
          throw new Error(`unexpected model asset URL: ${url}`);
        }
    """
    run_node(script)


def test_nemotron_template_exposes_local_engine_progress_status():
    html = (ROOT / "app/templates/index.html").read_text()

    assert 'id="localEngineProgress"' in html
    assert 'function showLocalEngineProgress' in html
    assert "handleLocalEngineStatus(status)" in html
    assert "nemotron-engine.mjs?v=4" in html


def test_nemotron_template_exposes_webgpu_debug_panel():
    html = (ROOT / "app/templates/index.html").read_text()

    assert 'id="nemotronDevice"' in html
    assert 'id="checkNemotronWebGpu"' in html
    assert 'id="nemotronWebGpuDebugOutput"' in html
    assert "function checkNemotronWebGpuDebug" in html
    assert "maxStorageBuffersPerShaderStage" in html
    assert "chrome://flags/#enable-unsafe-webgpu" in html


@requires_node
def test_parakeet_final_inference_does_not_erase_new_capture_audio():
    script = """
        import { ParakeetLocalEngine } from './app/static/parakeet/parakeet-engine.mjs';

        let release;
        const gate = new Promise((resolve) => { release = resolve; });
        const events = [];
        const engine = new ParakeetLocalEngine({ onTranscript: (event) => events.push(event) });
        engine._model = {
          transcribe: async (audio) => {
            if (audio.length !== 512) throw new Error(`unexpected final snapshot: ${audio.length}`);
            await gate;
            return { text: 'first' };
          },
        };
        engine._frames = [new Float32Array(512)];
        engine._inSpeech = true;
        engine._schedule(true);
        await new Promise((resolve) => setTimeout(resolve, 0));

        // Audio for the next utterance arrives while the prior final is in flight.
        engine._frames.push(new Float32Array(512).fill(0.5));
        release();
        await new Promise((resolve) => setTimeout(resolve, 0));

        if (engine._frames.length !== 1) {
          throw new Error(`new capture was erased: ${engine._frames.length}`);
        }
        if (events.length !== 1 || events[0].text !== 'first' || events[0].isFinal !== true) {
          throw new Error(`unexpected events: ${JSON.stringify(events)}`);
        }
    """
    run_node(script)


@requires_node
def test_nemotron_failed_final_resets_only_decoder_state():
    script = """
        import { NemotronLocalEngine } from './app/static/nemotron/nemotron-engine.mjs';

        const engine = new NemotronLocalEngine({ onStatus: () => {} });
        engine._finalJobs = [new Float32Array(512)];
        engine._busy = true;
        engine._frames = [new Float32Array(256).fill(0.5)];
        let resets = 0;
        engine._resetDecode = () => { resets += 1; };
        engine._process = async () => { throw new Error('inference failed'); };

        await engine._drain();

        if (resets !== 1) throw new Error(`decoder reset count: ${resets}`);
        if (engine._frames.length !== 1) throw new Error('live capture was erased');
        if (engine._busy !== false) throw new Error('drain remained busy');
    """
    run_node(script)


@requires_node
def test_nemotron_discarded_blip_resets_decoder_state_after_inflight_job():
    """A partially decoded noise blip must not leak its stream state forward.

    The blip's interim advances _consumed/_uttTokens and the fp16 caches; if the
    discard only cleared capture state, the next utterance would be decoded at a
    stale mel offset with the blip's tokens prepended.
    """
    script = """
        import { NemotronLocalEngine } from './app/static/nemotron/nemotron-engine.mjs';

        let release;
        const gate = new Promise((resolve) => { release = resolve; });
        let streamResets = 0;
        const engine = new NemotronLocalEngine({ onStatus: () => {} });
        engine._model = { resetStream: () => { streamResets += 1; } };
        // Stand in for an interim decode: it advances decoder state after an await.
        engine._process = async () => {
          await gate;
          engine._consumed = 56;
          engine._uttTokens.push(7);
        };

        engine._frames = [new Float32Array(512)];
        engine._inSpeech = true;
        engine._schedule(false);
        await new Promise((resolve) => setTimeout(resolve, 0));

        // Blip falls below MIN_SPEECH_FRAMES while its interim is still running.
        engine._discardBlip();
        if (engine._finalJobs.length !== 1) {
          throw new Error(`decode reset was not queued: ${engine._finalJobs.length}`);
        }
        // A second blip must not queue a redundant reset behind the first.
        engine._discardBlip();
        if (engine._finalJobs.length !== 1) {
          throw new Error(`redundant reset queued: ${engine._finalJobs.length}`);
        }

        release();
        await new Promise((resolve) => setTimeout(resolve, 0));
        await new Promise((resolve) => setTimeout(resolve, 0));

        if (engine._consumed !== 0) throw new Error(`stale mel offset: ${engine._consumed}`);
        if (engine._uttTokens.length !== 0) {
          throw new Error(`stale tokens: ${JSON.stringify(engine._uttTokens)}`);
        }
        if (streamResets !== 1) throw new Error(`encoder cache resets: ${streamResets}`);
        if (engine._busy !== false) throw new Error('drain remained busy');
    """
    run_node(script)


def test_template_uses_per_session_webspeech_recognizer_identity():
    html = (ROOT / "app/templates/index.html").read_text()

    assert "function _createWebSpeechRecognition(sessionGen)" in html
    assert "recognition === recognizer" in html
    assert "_detachWebSpeechRecognition(recognizer)" in html
    assert "recognition = _createWebSpeechRecognition(sessionGen)" in html


def test_template_hardens_manual_commit_backpressure_and_log_growth():
    html = (ROOT / "app/templates/index.html").read_text()

    assert 'id="commitButton"' in html
    assert "function requestStopRecording()" in html
    assert "const _EL_MANUAL_STOP_TIMEOUT_MS = 12000" in html
    assert "function _canSendAudio(ws)" in html
    assert "ws.bufferedAmount <= _AUDIO_WS_HIGH_WATER_BYTES" in html
    assert html.count("socket === sessionSocket") >= 2
    assert "_elBrowserWs === elSocket" in html
    assert "startElevenLabsServerRecording(true)" in html
    assert "const _MAX_RENDERED_SEGMENTS = 500" in html
    assert 'id="recognizedContainer" tabindex="0" role="log"' in html


def test_whisper_large_model_cannot_fall_through_to_wasm():
    source = (ROOT / "app/static/whisper/whisper-engine.mjs").read_text()

    assert "this.modelKey === 'large-v3-turbo' && !wantsWebGpu" in source
    assert "large-v3-turbo requires WebGPU" in source
