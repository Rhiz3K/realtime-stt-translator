// Browser-side Nemotron-3.5-ASR streaming engine (onnxruntime-web + WebGPU).
//
// Sibling of the local Whisper engine: mic -> PCM worklet -> ONNX in the browser
// -> text -> /ws (translation only). Unlike Whisper this model is cache-aware
// streaming, so it emits a *growing* hypothesis (real interims) and commits a
// final at end-of-utterance.
//
// Pipeline (ported 1:1 from scripts/nemotron_reference.py — the ground truth):
//   audio -> mel.js -> [encoder_fp16*.onnx, WebGPU/WASM] -> [decoder_joint.onnx,
//   WASM, RNNT greedy] -> tokenizer.js -> text.
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
const MAX_PENDING_FINALS = 8;
// Queued in _finalJobs instead of audio: "reset the decoder here", see _discardBlip().
const DISCARD_DECODE = Symbol("discard-decode");
const ENCODER_DATA_FILE = "encoder_fp16.onnx.data";
const NEMOTRON_MODEL_ASSET_VERSION = "2";
const NEMOTRON_ENCODER_VARIANTS = [
  { minStorageBuffers: 25, file: "encoder_fp16.onnx", label: "standard" },
  { minStorageBuffers: 16, file: "encoder_fp16_concat16.onnx", label: "storage-buffer 16" },
  { minStorageBuffers: 8, file: "encoder_fp16_concat8.onnx", label: "storage-buffer 8" },
];
const WEBGPU_MIN_STORAGE_BUFFERS = NEMOTRON_ENCODER_VARIANTS[NEMOTRON_ENCODER_VARIANTS.length - 1].minStorageBuffers;

const prod = (a) => a.reduce((x, y) => x * y, 1);

function emitStatus(onStatus, text, progress, indeterminate = false) {
  onStatus({ text, progress, indeterminate });
}

// Transient notice rather than load progress: same object shape, different field,
// so the UI can show it without disturbing the progress bar.
function emitNotice(onStatus, message) {
  onStatus({ message: String(message) });
}

async function fetchJson(url, label) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${label} fetch failed: ${res.status}. Nemotron model assets are not ready on the server yet.`);
  }
  return res.json();
}

export function nemotronModelAssetUrl(dir, file) {
  const sep = String(dir).includes("?") ? "&" : "?";
  return `${dir}/${file}${sep}v=${NEMOTRON_MODEL_ASSET_VERSION}`;
}

export function checkNemotronWebGpuLimits(adapterOrLimits) {
  const limits = adapterOrLimits && adapterOrLimits.limits ? adapterOrLimits.limits : adapterOrLimits;
  const limit = Number(limits && limits.maxStorageBuffersPerShaderStage);
  if (!Number.isFinite(limit) || limit <= 0) {
    return {
      ok: false,
      limit: 0,
      reason: "WebGPU storage-buffer limit is unavailable",
    };
  }
  const variant = selectNemotronEncoderVariantForLimit(limit);
  if (!variant) {
    return {
      ok: false,
      limit,
      reason: `WebGPU storage-buffer limit ${limit} is below Nemotron minimum ${WEBGPU_MIN_STORAGE_BUFFERS}`,
    };
  }
  return {
    ok: true,
    limit,
    variant,
    reason: `WebGPU storage-buffer limit ${limit} supports Nemotron ${variant.label} encoder`,
  };
}

export function selectNemotronEncoderVariantForLimit(limitValue) {
  const limit = Number(limitValue);
  if (!Number.isFinite(limit) || limit <= 0) return null;
  return NEMOTRON_ENCODER_VARIANTS.find((variant) => limit >= variant.minStorageBuffers) || null;
}

async function probeNemotronWebGpuSupport() {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    return { ok: false, limit: 0, reason: "WebGPU is not available in this browser" };
  }
  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return {
      ok: false,
      limit: 0,
      reason: `WebGPU adapter check failed: ${e && e.message ? e.message : String(e)}`,
    };
  }
  if (!adapter) {
    return { ok: false, limit: 0, reason: "WebGPU adapter is not available" };
  }
  return checkNemotronWebGpuLimits(adapter);
}

function encoderOptions(dir) {
  return {
    externalData: [{ path: ENCODER_DATA_FILE, data: nemotronModelAssetUrl(dir, ENCODER_DATA_FILE) }],
    graphOptimizationLevel: "all",
  };
}

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
    emitStatus(onStatus, "Checking Nemotron model assets...", 10, true);
    const [config, tokenizer] = await Promise.all([
      fetchJson(nemotronModelAssetUrl(dir, "config.json"), "config"),
      Tokenizer.load(nemotronModelAssetUrl(dir, "vocab.json")),
    ]);
    emitStatus(onStatus, "Nemotron model assets found", 25);

    // Encoder: WebGPU (with WASM fallback). Lower-limit WebGPU variants split
    // large Concat nodes but reuse the same external fp16 weights file.
    let encSession;
    const wantGpu = device === "auto" || device === "webgpu";
    if (wantGpu) {
      const gpuSupport = await probeNemotronWebGpuSupport();
      if (!gpuSupport.ok) {
        if (device === "webgpu") {
          throw new Error(`Nemotron WebGPU is not supported on this device: ${gpuSupport.reason}`);
        }
        emitStatus(onStatus, `${gpuSupport.reason}; loading Nemotron on CPU (WASM, slower)...`, 35, true);
      } else {
        try {
          const variant = gpuSupport.variant || NEMOTRON_ENCODER_VARIANTS[0];
          const encUrl = nemotronModelAssetUrl(dir, variant.file);
          emitStatus(onStatus, `Loading Nemotron encoder on GPU (WebGPU, ${variant.label})...`, 35, true);
          encSession = await ort.InferenceSession.create(encUrl, { ...encoderOptions(dir), executionProviders: ["webgpu"] });
          emitStatus(onStatus, `Nemotron encoder ready (WebGPU, ${variant.label})`, 70);
        } catch (e) {
          if (device === "webgpu") throw e;
          emitStatus(onStatus, "WebGPU failed; loading Nemotron on CPU (WASM, slower)...", 35, true);
        }
      }
    }
    if (!encSession) {
      if (!wantGpu) {
        emitStatus(onStatus, "Loading Nemotron encoder on CPU (WASM)...", 35, true);
      }
      const cpuVariant = NEMOTRON_ENCODER_VARIANTS[0];
      encSession = await ort.InferenceSession.create(nemotronModelAssetUrl(dir, cpuVariant.file), { ...encoderOptions(dir), executionProviders: ["wasm"] });
      emitStatus(onStatus, "Nemotron encoder ready (CPU/WASM)", 70);
    }

    // Decoder/joint: tiny, queried per token — WASM avoids per-token GPU dispatch.
    emitStatus(onStatus, "Loading Nemotron decoder...", 78, true);
    const decSession = await ort.InferenceSession.create(nemotronModelAssetUrl(dir, "decoder_joint.onnx"), {
      executionProviders: ["wasm"],
    });
    emitStatus(onStatus, "Nemotron model ready", 95);

    return new NemotronModel(ort, { encSession, decSession, tokenizer, config, normalize });
  }

  promptIndex(lang) {
    const d = this.config.prompt_dictionary || {};
    if (lang && lang in d) return d[lang];
    const base = (lang || "").split("-")[0];
    if (base in d) return d[base];
    return d.auto ?? 0;
  }

  /**
   * Release the ORT sessions. index.html drops the engine on every stop, but the
   * wasm/WebGPU allocations behind these sessions are not reclaimed by GC, so
   * without this every start/stop cycle strands another ~1.2 GB.
   */
  async dispose() {
    const sessions = [this.enc, this.decoder && this.decoder.session];
    this.enc = null;
    if (this.decoder) this.decoder.session = null;
    for (const session of sessions) {
      try { if (session && session.release) await session.release(); } catch (_e) { /* ignore */ }
    }
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
   * @param {() => boolean} [shouldAbort] checked between sub-chunks, so a stop
   *   during a long final does not keep dispatching encoder work at a dead engine
   * @returns {Promise<number>} mel frames actually consumed (< count if aborted)
   */
  async pushFrames(melData, T, fromFrame, count, promptIndex, emit, shouldAbort) {
    let off = 0;
    while (off < count) {
      if (shouldAbort && shouldAbort()) break;
      const n = Math.min(CHUNK_FRAMES, count - off);
      const { chunk, P } = this._assembleChunk(melData, T, fromFrame + off, n);
      const { encoded, Tout } = await this._encodeOne(chunk, P, promptIndex);
      await this.decoder.decode(encoded, Tout, emit);
      off += n;
    }
    return off;
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
    this._lifecycle = 0;
    this._resetUtterance();
  }

  _resetCapture() {
    this._frames = [];      // Float32Array(512) chunks of the current utterance
    this._preRoll = [];     // recent frames captured before speech onset
    this._inSpeech = false;
    this._silence = 0;
    this._speechFrames = 0;
  }

  _resetDecode() {
    this._uttTokens = [];
    this._consumed = 0;     // mel frames already fed to the model
    if (this._model) this._model.resetStream();
  }

  _resetUtterance() {
    this._resetCapture();
    this._resetDecode();
    this._busy = false;
    this._interimRequested = false;
    this._finalJobs = [];
  }

  async load() {
    emitStatus(this.onStatus, "Loading Nemotron runtime...", 5, true);
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
    emitStatus(this.onStatus, "Listening (local Nemotron)", 100);
  }

  stop() {
    this._lifecycle++;
    // Detach the handler before anything else: AudioContext.close() is async, so a
    // frame already in flight would otherwise mutate freshly reset capture state.
    try { if (this._node) this._node.port.onmessage = null; } catch (_e) {}
    try { if (this._node) this._node.disconnect(); } catch (_e) {}
    try { if (this._ctx) this._ctx.close(); } catch (_e) {}
    try { if (this._stream) this._stream.getTracks().forEach((t) => t.stop()); } catch (_e) {}
    this._node = this._ctx = this._stream = null;
    this._resetUtterance();
  }

  /** Release the model's ORT sessions. Call after stop(); not reusable afterwards. */
  async dispose() {
    const model = this._model;
    this._model = null;
    if (model) {
      try { await model.dispose(); } catch (_e) { /* ignore */ }
    }
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
      else this._discardBlip();
    }
  }

  /**
   * Drop a sub-MIN_SPEECH_FRAMES noise blip. Capture state can go immediately,
   * but a blip long enough to reach the silence cut-off (pre-roll + loud frames
   * + ~900 ms of silence) has usually already been decoded by an interim, so the
   * decoder reset must be sequenced behind any in-flight job — otherwise the next
   * utterance inherits the blip's tokens, mel offset and encoder caches.
   */
  _discardBlip() {
    this._resetCapture();
    this._interimRequested = false;
    // Nothing decoded and nothing running -> no stream state to throw away.
    if (!this._busy && this._consumed === 0 && !this._uttTokens.length) return;
    // A reset already waiting at the tail covers this blip too: jobs drain in
    // order, so no decoding can have happened since it was queued.
    if (this._finalJobs[this._finalJobs.length - 1] === DISCARD_DECODE) return;
    this._finalJobs.push(DISCARD_DECODE);
    if (this._busy) return;
    this._busy = true;
    queueMicrotask(() => this._drain());
  }

  _schedule(isFinal) {
    if (isFinal) {
      const audio = this._flatten();
      this._resetCapture();
      this._interimRequested = false;
      if (audio.length) {
        this._finalJobs.push(audio);
        if (this._finalJobs.length > MAX_PENDING_FINALS) {
          // Drop the oldest audio segment, never a queued decode reset: resets cost
          // nothing and the segments behind them would inherit stale stream state.
          // The push above guarantees at least one audio job, so the index is valid.
          this._finalJobs.splice(this._finalJobs.findIndex((job) => job !== DISCARD_DECODE), 1);
          emitNotice(this.onStatus, "Nemotron can't keep up — dropped the oldest buffered final segment");
        }
      }
    } else {
      this._interimRequested = true;
    }
    if (this._busy) return;
    this._busy = true;
    queueMicrotask(() => this._drain());
  }

  async _drain() {
    const lifecycle = this._lifecycle;
    while (this._finalJobs.length || this._interimRequested) {
      let audio;
      let isFinal;
      if (this._finalJobs.length) {
        const job = this._finalJobs.shift();
        if (job === DISCARD_DECODE) {
          this._resetDecode(); // queued by _discardBlip(): drop what the blip decoded
          continue;
        }
        audio = job;
        isFinal = true;
      } else {
        this._interimRequested = false;
        audio = this._flatten();
        isFinal = false;
      }
      if (!audio.length) continue;
      try {
        await this._process(audio, isFinal);
      } catch (e) {
        emitNotice(this.onStatus, "Nemotron error: " + (e && e.message ? e.message : e));
      } finally {
        // A failed final must not leak decoder/cache state into the next
        // utterance. Capture state is independent and may already contain new
        // microphone frames, so only reset the decoder here.
        if (isFinal) this._resetDecode();
      }
      if (lifecycle !== this._lifecycle) return;
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

  async _process(audio, isFinal) {
    const lifecycle = this._lifecycle;
    const { data, nFrames } = computeLogMel(audio, { normalize: this.normalize });
    const avail = nFrames - this._consumed;
    const count = isFinal ? avail : Math.floor((avail - EDGE_MARGIN_FRAMES) / CHUNK_FRAMES) * CHUNK_FRAMES;
    if (count > 0) {
      const consumed = await this._model.pushFrames(
        data, nFrames, this._consumed, count, this._promptIndex,
        (id) => this._uttTokens.push(id),
        () => lifecycle !== this._lifecycle,
      );
      this._consumed += consumed;
      if (consumed < count) return; // stopped mid-utterance; nothing to emit
    } else if (!isFinal) {
      return;
    }
    const text = this._model.tokenizer.decode(this._uttTokens);
    if (isFinal) {
      this.onTranscript({ text, isFinal: true });
    } else {
      this.onTranscript({ text, isFinal: false });
    }
  }
}

export { computeLogMel, N_MELS, HOP, SR };
