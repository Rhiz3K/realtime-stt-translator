// Browser-side Nemotron-3.5-ASR streaming engine (onnxruntime-web + WebGPU).
//
// Sibling of the local Whisper engine: mic -> PCM worklet -> ONNX in the browser
// -> text -> /ws (translation only). Unlike Whisper this model is cache-aware
// streaming, so it emits a *growing* hypothesis (real interims) and commits a
// final at end-of-utterance.
//
// Pipeline (ported 1:1 from scripts/nemotron_reference.py — the ground truth):
//   audio -> mel.js -> [encoder_fp16.onnx, WebGPU] -> [decoder_joint.onnx, WASM,
//   RNNT greedy] -> tokenizer.js -> text.
//
// NemotronModel is the model core (load + streaming encode + decode), testable
// headless (scripts/nemotron_poc.html drives it on a WAV). NemotronLocalEngine
// wraps it with mic capture + VAD for index.html.

import { computeLogMel, N_MELS, HOP, SR } from "./mel.js";
import { RnntDecoder } from "./rnnt.js";
import { Tokenizer } from "./tokenizer.js";
import { f32ToF16, f16ToF32, zerosF16 } from "./f16.js";

const CHUNK_FRAMES = 56;   // mel frames consumed per encoder step (8960 samples / 160)
const PRE_ENCODE = 9;      // mel frames of left context prepended from the prior chunk

const prod = (a) => a.reduce((x, y) => x * y, 1);

export class NemotronModel {
  constructor(ort, { encSession, decSession, tokenizer, config, normalize }) {
    this.ort = ort;
    this.enc = encSession;
    this.tokenizer = tokenizer;
    this.config = config;
    this.normalize = normalize; // "none" | "per_feature"
    this.blankId = config.blank_id;
    this.cacheShapes = config.cache_shapes;
    this.decoder = new RnntDecoder(ort, decSession, {
      blankId: config.blank_id,
      hiddenDim: config.hidden_dim || 1024,
    });
    this.resetStream();
  }

  static async load(ort, { dir = "/static/nemotron/models", device = "auto", normalize = "none", onStatus = () => {} } = {}) {
    const [config, tokenizer] = await Promise.all([
      fetch(`${dir}/config.json`).then((r) => r.json()),
      Tokenizer.load(`${dir}/vocab.json`),
    ]);

    // Encoder: WebGPU (with WASM fallback). External fp16 weights via externalData.
    const encUrl = `${dir}/encoder_fp16.onnx`;
    const encOpts = {
      externalData: [{ path: "encoder_fp16.onnx.data", data: `${dir}/encoder_fp16.onnx.data` }],
      graphOptimizationLevel: "all",
    };
    let encSession;
    const wantGpu = device === "auto" || device === "webgpu";
    if (wantGpu) {
      try {
        onStatus("Loading Nemotron encoder on GPU (WebGPU)…");
        encSession = await ort.InferenceSession.create(encUrl, { ...encOpts, executionProviders: ["webgpu"] });
        onStatus("Nemotron encoder ready (WebGPU)");
      } catch (e) {
        if (device === "webgpu") throw e;
        onStatus("WebGPU unavailable — loading Nemotron on CPU (WASM, slower)…");
      }
    }
    if (!encSession) {
      encSession = await ort.InferenceSession.create(encUrl, { ...encOpts, executionProviders: ["wasm"] });
      onStatus("Nemotron encoder ready (CPU/WASM)");
    }

    // Decoder/joint: tiny, queried per token — WASM avoids per-token GPU dispatch.
    const decSession = await ort.InferenceSession.create(`${dir}/decoder_joint.onnx`, {
      executionProviders: ["wasm"],
    });

    return new NemotronModel(ort, { encSession, decSession, tokenizer, config, normalize });
  }

  promptIndex(lang) {
    const d = this.config.prompt_dictionary || {};
    if (lang && lang in d) return d[lang];
    const base = (lang || "").split("-")[0];
    if (base in d) return d[base];
    return d.auto ?? 0;
  }

  /** Reset cache + decoder state. Call at the start of each utterance. */
  resetStream() {
    const cs = this.cacheShapes;
    this.cChan = zerosF16(prod(cs.cache_last_channel)); // fp16 (Uint16Array) caches
    this.cTime = zerosF16(prod(cs.cache_last_time));
    this.cLen = BigInt64Array.of(0n);
    this.decoder.reset();
  }

  /** Build [128, PRE_ENCODE+count] chunk: left context (zero-padded at start) + main. */
  _assembleChunk(melData, T, fromFrame, count) {
    const P = PRE_ENCODE + count;
    const chunk = new Float32Array(N_MELS * P);
    for (let m = 0; m < N_MELS; m++) {
      const row = m * T;
      const dst = m * P;
      for (let p = 0; p < PRE_ENCODE; p++) {
        const src = fromFrame - PRE_ENCODE + p;
        chunk[dst + p] = src >= 0 ? melData[row + src] : 0;
      }
      for (let c = 0; c < count; c++) chunk[dst + PRE_ENCODE + c] = melData[row + fromFrame + c];
    }
    return { chunk, P };
  }

  async _encodeOne(chunk, P, promptIndex) {
    const ort = this.ort;
    const cs = this.cacheShapes;
    const out = await this.enc.run({
      processed_signal: new ort.Tensor("float16", f32ToF16(chunk), [1, N_MELS, P]),
      processed_signal_length: new ort.Tensor("int64", BigInt64Array.of(BigInt(P)), [1]),
      cache_last_channel: new ort.Tensor("float16", this.cChan, cs.cache_last_channel),
      cache_last_time: new ort.Tensor("float16", this.cTime, cs.cache_last_time),
      cache_last_channel_len: new ort.Tensor("int64", this.cLen, [1]),
      prompt_index: new ort.Tensor("int64", BigInt64Array.of(BigInt(promptIndex)), [1]),
    });
    // caches stay fp16 (Uint16Array) across calls; encoded -> fp32 for the decoder
    this.cChan = out.cache_last_channel_next.data;
    this.cTime = out.cache_last_time_next.data;
    this.cLen = out.cache_last_channel_len_next.data;
    return { encoded: f16ToF32(out.encoded.data), Tout: out.encoded.dims[2] };
  }

  /**
   * Encode + greedily decode mel frames [fromFrame, fromFrame+count), sub-chunked
   * into CHUNK_FRAMES steps so each encoder call matches the export's streaming
   * chunk. Caches + decoder state persist across calls (continuous stream).
   * @param {(id:number)=>void} emit per emitted token id
   */
  async pushFrames(melData, T, fromFrame, count, promptIndex, emit) {
    let off = 0;
    while (off < count) {
      const n = Math.min(CHUNK_FRAMES, count - off);
      const { chunk, P } = this._assembleChunk(melData, T, fromFrame + off, n);
      const { encoded, Tout } = await this._encodeOne(chunk, P, promptIndex);
      await this.decoder.decode(encoded, Tout, emit);
      off += n;
    }
  }

  /** Convenience for offline/whole-buffer use (PoC): mel a full clip, decode it. */
  async transcribe(audioF32, lang = "en") {
    this.resetStream();
    const { data, nFrames } = computeLogMel(audioF32, { normalize: this.normalize });
    const ids = [];
    await this.pushFrames(data, nFrames, 0, nFrames, this.promptIndex(lang), (id) => ids.push(id));
    return { ids, text: this.tokenizer.decode(ids) };
  }
}

// --- mic-driven streaming wrapper (the index.html-facing engine) ---------------
const FRAME_SAMPLES = 512;          // matches /static/whisper/pcm-worklet.js
const CHUNK_SAMPLES = CHUNK_FRAMES * HOP; // 8960 — one encoder step worth of audio
const RMS_SPEECH = 0.012;           // normalised-RMS "loud" gate (same as Whisper's fallback)
const SILENCE_FRAMES_END = 28;      // ~900 ms of silence ends an utterance
const MIN_SPEECH_FRAMES = 8;        // ignore sub-256 ms blips
const PRE_ROLL_FRAMES = 8;          // lead-in kept from just before speech onset
const MAX_UTT_SAMPLES = 20 * SR;    // hard cap (~20 s) -> force a final
const EDGE_MARGIN_FRAMES = 2;       // hold back reflect-edge mel frames from interims

function rmsNorm(int16) {
  let s = 0;
  for (let i = 0; i < int16.length; i++) { const v = int16[i] / 32768; s += v * v; }
  return Math.sqrt(s / int16.length);
}

/**
 * Mic -> 16 kHz PCM worklet -> streaming Nemotron -> growing interim text, with a
 * committed final at end-of-utterance (RMS silence). Mirrors WhisperLocalEngine's
 * public surface: new NemotronLocalEngine({...}); await load(); await start(stream);
 * stop(). onTranscript({text, isFinal}) fires interims (growing) and finals.
 */
export class NemotronLocalEngine {
  constructor({ language = "cs", device = "auto", normalize = "none", onStatus = () => {}, onTranscript = () => {} } = {}) {
    this.language = language;
    this.device = device;
    this.normalize = normalize;
    this.onStatus = onStatus;
    this.onTranscript = onTranscript;
    this._model = null;
    this._ctx = null;
    this._node = null;
    this._stream = null;
    this._resetUtterance();
  }

  _resetUtterance() {
    this._frames = [];      // Float32Array(512) chunks of the current utterance
    this._preRoll = [];     // recent frames captured before speech onset
    this._uttTokens = [];
    this._consumed = 0;     // mel frames already fed to the model
    this._inSpeech = false;
    this._silence = 0;
    this._speechFrames = 0;
    this._busy = false;
    this._needsProcess = false;
    this._finalPending = false;
    if (this._model) this._model.resetStream();
  }

  async load() {
    this.onStatus("Loading Nemotron (first run fetches ~1.2 GB)…");
    const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    // Multi-threaded WASM needs cross-origin isolation (SharedArrayBuffer); cap at
    // 8 threads and fall back to single-threaded when not isolated.
    const hardwareConcurrency = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
    ort.env.wasm.numThreads =
      typeof self !== "undefined" && self.crossOriginIsolated
        ? Math.max(1, Math.min(hardwareConcurrency, 8))
        : 1;
    this._model = await NemotronModel.load(ort, { device: this.device, normalize: this.normalize, onStatus: this.onStatus });
    this._promptIndex = this._model.promptIndex(this.language);
  }

  async start(stream) {
    this._stream = stream;
    this._ctx = new AudioContext({ sampleRate: SR });
    await this._ctx.audioWorklet.addModule("/static/whisper/pcm-worklet.js");
    const src = this._ctx.createMediaStreamSource(stream);
    this._node = new AudioWorkletNode(this._ctx, "int16-pcm-processor");
    this._node.port.onmessage = (e) => this._onFrame(new Int16Array(e.data));
    src.connect(this._node);
    this._node.connect(this._ctx.destination); // worklet emits silence; keeps process() pulled
    this.onStatus("Listening (local Nemotron)…");
  }

  stop() {
    try { if (this._node) this._node.disconnect(); } catch (_e) {}
    try { if (this._ctx) this._ctx.close(); } catch (_e) {}
    try { if (this._stream) this._stream.getTracks().forEach((t) => t.stop()); } catch (_e) {}
    this._node = this._ctx = this._stream = null;
    this._resetUtterance();
  }

  _onFrame(int16) {
    const f = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f[i] = int16[i] / 32768;
    const loud = rmsNorm(int16) >= RMS_SPEECH;

    if (!this._inSpeech) {
      this._preRoll.push(f);
      if (this._preRoll.length > PRE_ROLL_FRAMES) this._preRoll.shift();
      if (loud) {
        this._inSpeech = true;
        this._frames.push(...this._preRoll);
        this._preRoll = [];
        this._silence = 0;
        this._speechFrames = 0;
      }
      return;
    }

    this._frames.push(f);
    if (loud) { this._silence = 0; this._speechFrames++; } else this._silence++;

    const samples = this._frames.length * FRAME_SAMPLES;
    if (samples >= MAX_UTT_SAMPLES) { this._schedule(true); return; }
    if (samples - this._consumed * HOP >= CHUNK_SAMPLES) this._schedule(false);
    if (this._silence >= SILENCE_FRAMES_END) {
      if (this._speechFrames >= MIN_SPEECH_FRAMES) this._schedule(true);
      else this._resetUtterance(); // discard a noise blip
    }
  }

  _schedule(isFinal) {
    this._needsProcess = true;
    if (isFinal) this._finalPending = true;
    if (this._busy) return;
    this._busy = true;
    queueMicrotask(() => this._drain());
  }

  async _drain() {
    while (this._needsProcess) {
      this._needsProcess = false;
      const isFinal = this._finalPending;
      this._finalPending = false;
      try {
        await this._process(isFinal);
      } catch (e) {
        this.onStatus("Nemotron error: " + (e && e.message ? e.message : e));
      }
      if (isFinal) break;
    }
    this._busy = false;
  }

  _flatten() {
    let n = 0;
    for (const f of this._frames) n += f.length;
    const out = new Float32Array(n);
    let o = 0;
    for (const f of this._frames) { out.set(f, o); o += f.length; }
    return out;
  }

  async _process(isFinal) {
    const { data, nFrames } = computeLogMel(this._flatten(), { normalize: this.normalize });
    const avail = nFrames - this._consumed;
    const count = isFinal ? avail : Math.floor((avail - EDGE_MARGIN_FRAMES) / CHUNK_FRAMES) * CHUNK_FRAMES;
    if (count > 0) {
      await this._model.pushFrames(data, nFrames, this._consumed, count, this._promptIndex, (id) => this._uttTokens.push(id));
      this._consumed += count;
    } else if (!isFinal) {
      return;
    }
    const text = this._model.tokenizer.decode(this._uttTokens);
    if (isFinal) {
      this.onTranscript({ text, isFinal: true });
      this._resetUtterance();
    } else {
      this.onTranscript({ text, isFinal: false });
    }
  }
}

export { computeLogMel, N_MELS, HOP, SR };
