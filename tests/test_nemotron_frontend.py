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
    assert "nemotron-engine.mjs?v=8" in html


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


def test_template_defers_single_shot_stop_until_final_translation():
    html = (ROOT / "app/templates/index.html").read_text()

    assert "let _pendingFinalClientId = null" in html
    assert "if (msgType === 'final') _pendingFinalClientId = clientId;" in html
    assert "function _finishSingleShotStop()" in html
    assert "_singleShotStopTimer = setTimeout(_finishSingleShotStop, _SINGLE_SHOT_STOP_TIMEOUT_MS)" in html
    # Single-shot onend must not stop while that final is still unanswered.
    assert "webspeech:end (single-shot mode, waiting for final translation)" in html
    assert "_pendingFinalClientId == null && _pendingFinalTexts.length === 0" in html


def test_template_preserves_trailing_webspeech_interim_during_end_and_stop():
    html = (ROOT / "app/templates/index.html").read_text()

    assert "let webSpeechLastInterimText = '';" in html
    assert "function _commitWebSpeechTrailingInterim(reason)" in html
    assert "_commitWebSpeechTrailingInterim('onend')" in html
    # A user Stop must ask Web Speech to flush its last result, then leave /ws
    # alive until the corresponding final translation has reached the browser.
    assert "function _requestWebSpeechStop()" in html
    assert "recognizer.stop();" in html
    assert "_webSpeechStopRecognitionEnded = true;" in html
    assert "_WEB_SPEECH_STOP_TIMEOUT_MS" in html


@requires_node
def test_local_engines_release_onnx_sessions_on_dispose():
    """Every stop drops the engine; without release() the ORT memory is stranded."""
    script = """
        import { NemotronLocalEngine, NemotronModel } from './app/static/nemotron/nemotron-engine.mjs';
        import { ParakeetLocalEngine, ParakeetModel } from './app/static/parakeet/parakeet-engine.mjs';

        for (const [name, Model] of [['nemotron', NemotronModel], ['parakeet', ParakeetModel]]) {
          // Object.create skips the constructor (it needs real ORT sessions + config).
          const model = Object.create(Model.prototype);
          const released = [];
          model.enc = { release: async () => { released.push('encoder'); } };
          model.decoder = { session: { release: async () => { released.push('decoder'); } } };

          await model.dispose();

          if (released.length !== 2) {
            throw new Error(`${name}: released ${JSON.stringify(released)}, expected both sessions`);
          }
          if (model.enc !== null || model.decoder.session !== null) {
            throw new Error(`${name}: dispose kept a session reference`);
          }
        }

        for (const [name, Engine] of [['nemotron', NemotronLocalEngine], ['parakeet', ParakeetLocalEngine]]) {
          const engine = new Engine({ onStatus: () => {} });
          let disposed = 0;
          engine._model = { dispose: async () => { disposed += 1; } };

          await engine.dispose();

          if (disposed !== 1) throw new Error(`${name}: model dispose count ${disposed}`);
          if (engine._model !== null) throw new Error(`${name}: engine kept the model`);
        }
    """
    run_node(script)


@requires_node
def test_local_engines_await_inflight_decode_before_releasing_sessions():
    """dispose() must not release the ORT session while a decode is still
    suspended inside _drain(): that is a wasm use-after-free, and the ORT JSEP
    module-global session state stays set until the stale run resolves, so the
    next instance's first run() throws 'Session already started'."""
    script = """
        import { NemotronLocalEngine } from './app/static/nemotron/nemotron-engine.mjs';
        import { ParakeetLocalEngine } from './app/static/parakeet/parakeet-engine.mjs';

        for (const [name, Engine] of [['nemotron', NemotronLocalEngine], ['parakeet', ParakeetLocalEngine]]) {
          const order = [];
          let release;
          const gate = new Promise((r) => { release = r; });
          const engine = new Engine({ onStatus: () => {} });
          engine._model = {
            dispose: async () => { order.push('release'); },
            resetStream: () => {},
          };
          engine._process = async () => { await gate; order.push('decode-done'); };

          // Launch a drain with one queued final, let it reach the decode await,
          // then stop + dispose while the decode is still suspended.
          engine._frames = [new Float32Array(512)];
          engine._schedule(true);
          await new Promise((r) => setTimeout(r, 0));
          engine.stop();

          let settled = false;
          const disposed = engine.dispose().then(() => { settled = true; });
          await new Promise((r) => setTimeout(r, 0));
          if (settled) throw new Error(`${name}: dispose resolved before the decode settled`);

          release();
          await disposed;
          if (order[0] !== 'decode-done' || order[1] !== 'release') {
            throw new Error(`${name}: released session before decode finished: ${JSON.stringify(order)}`);
          }
          if (engine._model !== null) throw new Error(`${name}: dispose kept the model`);
        }
    """
    run_node(script)


def test_whisper_awaits_inflight_transcription_before_dispose():
    """Same hazard for Whisper: the Transformers.js pipeline (and its ORT session)
    must not be disposed while a transcription is still running in _processQueue().
    whisper-engine.mjs imports transformers.js from a URL at module top level, so
    it can't be driven under node like the ONNX engines — assert the ordering in
    source: _processQueue() must publish its promise and dispose() must await it
    before releasing the transcriber."""
    source = (ROOT / "app/static/whisper/whisper-engine.mjs").read_text()

    assert "this._processingPromise = (async () => {" in source, (
        "_processQueue() no longer publishes the in-flight promise dispose() waits on"
    )

    dispose = source.split("async dispose()", 1)[1].split("_resetVad", 1)[0]
    assert "await this._processingPromise" in dispose, "dispose() no longer awaits the in-flight transcription"
    assert dispose.index("await this._processingPromise") < dispose.index(".dispose()"), (
        "dispose() releases the transcriber before awaiting the in-flight transcription"
    )


def test_template_releases_engine_sessions_on_every_stop():
    html = (ROOT / "app/templates/index.html").read_text()

    stop_helper = html.split("function stopLocalEnginePipeline")[1].split("function stopWhisperPipeline")[0]
    assert "engine.stop()" in stop_helper
    assert "engine.dispose()" in stop_helper


@requires_node
def test_nemotron_aborts_encoder_work_when_the_engine_is_stopped():
    """A stop during a long final must not keep dispatching encoder sub-chunks."""
    script = """
        import { NemotronModel } from './app/static/nemotron/nemotron-engine.mjs';

        const model = Object.create(NemotronModel.prototype);
        let encoded = 0;
        model._assembleChunk = () => ({ chunk: new Float32Array(1), P: 1 });
        model._encodeOne = async () => { encoded += 1; return { encoded: new Float32Array(1), Tout: 1 }; };
        model.decoder = { decode: async () => {} };

        // 10 sub-chunks worth of frames, aborted after the second.
        const consumed = await model.pushFrames(
          new Float32Array(1), 1, 0, 56 * 10, 0, () => {}, () => encoded >= 2,
        );

        if (encoded !== 2) throw new Error(`kept encoding after abort: ${encoded}`);
        if (consumed !== 56 * 2) throw new Error(`reported wrong consumed count: ${consumed}`);

        // Without the abort callback it runs to completion and reports every frame.
        encoded = 0;
        const all = await model.pushFrames(new Float32Array(1), 1, 0, 56 * 3, 0, () => {});
        if (encoded !== 3 || all !== 56 * 3) {
          throw new Error(`unaborted run: encoded=${encoded} consumed=${all}`);
        }
    """
    run_node(script)


@requires_node
def test_local_engines_abort_asset_fetches_on_stop():
    """The weights download inside InferenceSession.create() cannot be cancelled,
    but the small config/vocab fetches can — and must be, or a stop during load
    leaves them running."""
    script = """
        import { NemotronLocalEngine } from './app/static/nemotron/nemotron-engine.mjs';
        import { ParakeetLocalEngine } from './app/static/parakeet/parakeet-engine.mjs';

        for (const [name, Engine] of [['nemotron', NemotronLocalEngine], ['parakeet', ParakeetLocalEngine]]) {
          const engine = new Engine({ onStatus: () => {} });
          if (engine._abort.signal.aborted) throw new Error(`${name}: aborted before start`);
          engine.stop();
          if (!engine._abort.signal.aborted) throw new Error(`${name}: stop did not abort the fetches`);
        }
    """
    run_node(script)
