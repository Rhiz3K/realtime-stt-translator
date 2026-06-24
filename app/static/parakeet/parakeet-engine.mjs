// Browser-side Parakeet-tdt-0.6b-v3 engine (onnxruntime-web + WebGPU).
//
// Sibling of the Nemotron engine, reusing its log-mel front-end. Unlike Nemotron
// (cache-aware streaming), the int8 Parakeet export is OFFLINE: the encoder takes
// a whole mel buffer and returns the whole encoded sequence (no streaming cache).
// So this engine re-encodes the growing utterance buffer for each interim and on
// the final, TDT-greedy-decoding from scratch each time — Whisper-style, but with
// a transducer decoder. Audio never leaves the client; text -> /ws (translation).
//
// Verified I/O (scripts/prepare_parakeet_onnx.py --inspect-only):
//   encoder-int8.onnx: audio_signal[1,128,T] f32 + length[1] i64
//                      -> outputs[1,1024,T'] f32 + encoded_lengths
//   decoder_joint-int8.onnx: see tdt.js. vocab.txt: 8193 tokens, blank <blk>=8192.

import { computeLogMel, N_MELS, HOP, SR } from "../nemotron/mel.js";
import { TdtDecoder } from "./tdt.js";
import { Tokenizer } from "./tokenizer.js";

const PARAKEET_MODEL_ASSET_VERSION = "1";
const ENCODER_FILE = "encoder-int8.onnx";
const ENCODER_DATA_FILE = "encoder-int8.onnx.data";
const DECODER_FILE = "decoder_joint-int8.onnx";
const VOCAB_FILE = "vocab.txt";
const BLANK_ID = 8192;
const VOCAB_SIZE = 8193;   // token logits incl. blank
const NUM_DURATIONS = 5;

function emitStatus(onStatus, text, progress, indeterminate = false) {
  onStatus({ text, progress, indeterminate });
}

export function parakeetModelAssetUrl(dir, file) {
  const sep = String(dir).includes("?") ? "&" : "?";
  return `${dir}/${file}${sep}v=${PARAKEET_MODEL_ASSET_VERSION}`;
}

export class ParakeetModel {
  constructor(ort, { encSession, decSession, tokenizer, normalize }) {
    this.ort = ort;
    this.enc = encSession;
    this.tokenizer = tokenizer;
    this.normalize = normalize; // "none" | "per_feature"
    this.decoder = new TdtDecoder(ort, decSession, {
      blankId: BLANK_ID,
      vocabSize: VOCAB_SIZE,
      durations: Array.from({ length: NUM_DURATIONS }, (_, i) => i), // [0,1,2,3,4]
    });
  }

  static async load(ort, { dir = "/static/parakeet/models", device = "auto", normalize = "none", onStatus = () => {} } = {}) {
    emitStatus(onStatus, "Checking Parakeet model assets...", 10, true);
    const tokenizer = await Tokenizer.load(parakeetModelAssetUrl(dir, VOCAB_FILE));
    emitStatus(onStatus, "Parakeet model assets found", 25);

    const encOptions = {
      externalData: [{ path: ENCODER_DATA_FILE, data: parakeetModelAssetUrl(dir, ENCODER_DATA_FILE) }],
      graphOptimizationLevel: "all",
    };

    // Encoder: WebGPU with WASM fallback (the int8 weights are reused either way).
    let encSession;
    const wantGpu = device === "auto" || device === "webgpu";
    const hasGpu = typeof navigator !== "undefined" && !!navigator.gpu;
    if (wantGpu && hasGpu) {
      try {
        emitStatus(onStatus, "Loading Parakeet encoder on GPU (WebGPU)...", 35, true);
        encSession = await ort.InferenceSession.create(parakeetModelAssetUrl(dir, ENCODER_FILE), { ...encOptions, executionProviders: ["webgpu"] });
        emitStatus(onStatus, "Parakeet encoder ready (WebGPU)", 70);
      } catch (e) {
        if (device === "webgpu") throw e;
        emitStatus(onStatus, "WebGPU failed; loading Parakeet on CPU (WASM, slower)...", 35, true);
      }
    } else if (device === "webgpu") {
      throw new Error("Parakeet WebGPU requested but navigator.gpu is unavailable");
    }
    if (!encSession) {
      if (!(wantGpu && hasGpu)) emitStatus(onStatus, "Loading Parakeet encoder on CPU (WASM)...", 35, true);
      encSession = await ort.InferenceSession.create(parakeetModelAssetUrl(dir, ENCODER_FILE), { ...encOptions, executionProviders: ["wasm"] });
      emitStatus(onStatus, "Parakeet encoder ready (CPU/WASM)", 70);
    }

    // Decoder/joint: tiny, queried per token — WASM avoids per-token GPU dispatch.
    emitStatus(onStatus, "Loading Parakeet decoder...", 78, true);
    const decSession = await ort.InferenceSession.create(parakeetModelAssetUrl(dir, DECODER_FILE), {
      executionProviders: ["wasm"],
    });
    emitStatus(onStatus, "Parakeet model ready", 95);

    return new ParakeetModel(ort, { encSession, decSession, tokenizer, normalize });
  }

  /** Offline encode of the whole mel buffer (mel-major [128,T] -> encoded [1024,T']). */
  async _encode(melData, T) {
    const ort = this.ort;
    const out = await this.enc.run({
      audio_signal: new ort.Tensor("float32", melData, [1, N_MELS, T]),
      length: new ort.Tensor("int64", BigInt64Array.of(BigInt(T)), [1]),
    });
    return { encoded: out.outputs.data, Tout: out.outputs.dims[2] };
  }

  /** Mel + encode + TDT-decode a whole audio clip. Stateless (fresh per call). */
  async transcribe(audioF32) {
    const { data, nFrames } = computeLogMel(audioF32, { normalize: this.normalize });
    if (nFrames <= 0) return { ids: [], text: "" };
    const { encoded, Tout } = await this._encode(data, nFrames);
    const ids = [];
    this.decoder.reset();
    await this.decoder.decode(encoded, Tout, (id) => ids.push(id));
    return { ids, text: this.tokenizer.decode(ids) };
  }
}

// --- mic-driven wrapper (the index.html-facing engine) -------------------------
const FRAME_SAMPLES = 512;          // matches /static/whisper/pcm-worklet.js
const RMS_SPEECH = 0.012;           // normalised-RMS "loud" gate (same as Nemotron/Whisper)
const SILENCE_FRAMES_END = 28;      // ~900 ms of silence ends an utterance
const MIN_SPEECH_FRAMES = 8;        // ignore sub-256 ms blips
const PRE_ROLL_FRAMES = 8;          // lead-in kept from just before speech onset
const MAX_UTT_SAMPLES = 20 * SR;    // hard cap (~20 s) -> force a final
const INTERIM_MIN_NEW_SAMPLES = Math.floor(1.2 * SR); // re-encode for interims at most ~every 1.2 s

function rmsNorm(int16) {
  let s = 0;
  for (let i = 0; i < int16.length; i++) { const v = int16[i] / 32768; s += v * v; }
  return Math.sqrt(s / int16.length);
}

/**
 * Mic -> 16 kHz PCM worklet -> offline Parakeet (re-encode growing buffer) ->
 * growing interim text, committed final at end-of-utterance (RMS silence).
 * Mirrors NemotronLocalEngine's public surface: new ParakeetLocalEngine({...});
 * await load(); await start(stream); stop(). onTranscript({text, isFinal}).
 */
export class ParakeetLocalEngine {
  constructor({ language = "cs", device = "auto", normalize = "none", onStatus = () => {}, onTranscript = () => {} } = {}) {
    this.language = language; // multilingual model auto-detects; kept for API parity
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
    this._inSpeech = false;
    this._silence = 0;
    this._speechFrames = 0;
    this._busy = false;
    this._needsProcess = false;
    this._finalPending = false;
    this._lastInterimSamples = 0;
  }

  async load() {
    emitStatus(this.onStatus, "Loading Parakeet runtime...", 5, true);
    const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const hardwareConcurrency = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
    ort.env.wasm.numThreads =
      typeof self !== "undefined" && self.crossOriginIsolated
        ? Math.max(1, Math.min(hardwareConcurrency, 8))
        : 1;
    this._model = await ParakeetModel.load(ort, { device: this.device, normalize: this.normalize, onStatus: this.onStatus });
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
    emitStatus(this.onStatus, "Listening (local Parakeet)", 100);
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
        this._lastInterimSamples = 0;
      }
      return;
    }

    this._frames.push(f);
    if (loud) { this._silence = 0; this._speechFrames++; } else this._silence++;

    const samples = this._frames.length * FRAME_SAMPLES;
    if (samples >= MAX_UTT_SAMPLES) { this._schedule(true); return; }
    if (this._silence >= SILENCE_FRAMES_END) {
      if (this._speechFrames >= MIN_SPEECH_FRAMES) this._schedule(true);
      else this._resetUtterance(); // discard a noise blip
      return;
    }
    // Throttle interim re-encodes (the whole buffer is re-encoded each time).
    if (samples - this._lastInterimSamples >= INTERIM_MIN_NEW_SAMPLES) {
      this._lastInterimSamples = samples;
      this._schedule(false);
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
        this.onStatus("Parakeet error: " + (e && e.message ? e.message : e));
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
    const { text } = await this._model.transcribe(this._flatten());
    if (isFinal) {
      this.onTranscript({ text, isFinal: true });
      this._resetUtterance();
    } else {
      this.onTranscript({ text, isFinal: false });
    }
  }
}

export { computeLogMel, N_MELS, HOP, SR };
