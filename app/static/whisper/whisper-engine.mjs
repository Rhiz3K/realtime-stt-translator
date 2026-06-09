/**
 * Browser-local Whisper STT (similar to vilassn/whisper_android).
 * Uses @huggingface/transformers — audio stays on device; only text is sent to the server.
 */
import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0';

env.allowLocalModels = false;
// Load the ONNX Runtime wasm from the SAME transformers.js build so the JS glue
// and the wasm binaries are ABI-compatible. transformers.js@3.4.0 pins a specific
// ORT dev build (1.22.0-dev.*); pointing wasmPaths at the standalone
// onnxruntime-web@1.22.0 release ships mismatched binaries and throws
// "s._OrtGetInputName is not a function" at session creation. Keep this version
// in sync with the import URL above.
env.backends.onnx.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0/dist/';

// When the page is cross-origin isolated (COOP+COEP headers from the server),
// SharedArrayBuffer is available and ORT Web can run the WASM/CPU path
// multi-threaded — a large speedup on multi-core devices. Without isolation it
// stays single-threaded. Cap threads so we don't peg every core on a phone.
if (typeof self !== 'undefined' && self.crossOriginIsolated) {
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
  env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(cores, 8));
}

/**
 * Available models. `webgpu`/`wasm` hold the dtype used on each backend.
 * Whisper's encoder is quantization-sensitive, so on WebGPU we keep the encoder
 * at fp16 and only 4-bit-quantize the decoder (q4) for speed/size; on CPU/WASM
 * we use int8 (q8) for everything to keep the download small.
 * @type {Record<string, { id: string, label: string, multilingual: boolean, webgpu: any, wasm: any }>}
 */
// Lineup tuned for the CPU/WASM browser path: small multilingual models that
// stay responsive on a phone CPU (multi-threaded when cross-origin isolated).
// large-v3-turbo is kept for devices with WebGPU but is slow on CPU.
export const WHISPER_MODELS = {
  tiny: {
    id: 'onnx-community/whisper-tiny',
    label: 'tiny (fastest, multilingual)',
    multilingual: true,
    webgpu: 'fp16',
    wasm: 'q8',
  },
  base: {
    id: 'onnx-community/whisper-base',
    label: 'base (fast, multilingual)',
    multilingual: true,
    webgpu: 'fp16',
    wasm: 'q8',
  },
  small: {
    id: 'onnx-community/whisper-small',
    label: 'small (best multilingual quality)',
    multilingual: true,
    webgpu: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    wasm: 'q8',
  },
  'large-v3-turbo': {
    id: 'onnx-community/whisper-large-v3-turbo',
    label: 'large-v3-turbo (top quality, WebGPU only)',
    multilingual: true,
    webgpu: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    wasm: 'q8',
  },
};

const SAMPLE_RATE = 16000;
// pcm-worklet.js buffers each render quantum into FRAME_SAMPLES (512) before
// posting, so each frame the engine sees is ~32 ms at 16 kHz.
const SILENCE_FRAMES_END = 28; // ~900 ms @ 32 ms/frame
const MIN_SPEECH_FRAMES = 13; // ~400 ms of actual speech (loud frames, not the silence tail)
const MAX_SPEECH_FRAMES = 470; // ~15 s cap per utterance
// Lead-in kept from just before VAD onset so the first syllable isn't clipped
// (the VAD needs a frame or two to trigger).
const PRE_ROLL_FRAMES = 8; // ~256 ms
// Cap the transcription backlog. On slow CPUs (mobile WASM) inference can be
// slower than real time; without a cap the queue — and thus latency — grows
// without bound. We keep only the most recent utterances and drop the oldest.
const MAX_PENDING_UTTERANCES = 3;

// dtype per backend. WASM uses int8 (small download, matches the advertised
// model sizes); WebGPU uses fp16 (GPU-friendly and much faster) with a graceful
// fallback to WASM when no GPU adapter or fp16 model is available.
const WASM_DTYPE = 'q8';
const WEBGPU_DTYPE = 'fp16';

// Language codes Whisper's multilingual tokenizer accepts (the openai/whisper
// set, which transformers.js validates against). The UI passes googletrans
// codes; an unknown code (eo, ceb, …) or region variant (zh-cn, zh-tw) would
// make transformers.js throw on EVERY utterance, so anything outside this set
// falls back to Whisper auto-detect instead.
const WHISPER_LANG_CODES = new Set([
  'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl',
  'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro',
  'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy',
  'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu',
  'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km',
  'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo',
  'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg',
  'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su',
]);

// Codes googletrans spells differently from Whisper.
const WHISPER_LANG_ALIASES = { iw: 'he', fil: 'tl', jv: 'jw' };

/** Map a (googletrans) language code to a Whisper code, or null → auto-detect. */
function toWhisperLanguage(code) {
  if (typeof code !== 'string') return null;
  let c = code.trim().toLowerCase();
  if (!c) return null;
  c = WHISPER_LANG_ALIASES[c] || c;
  if (WHISPER_LANG_CODES.has(c)) return c;
  // Region variants: zh-cn → zh, pt-br → pt, …
  const base = c.split(/[-_]/)[0];
  const aliased = WHISPER_LANG_ALIASES[base] || base;
  return WHISPER_LANG_CODES.has(aliased) ? aliased : null;
}

function int16ChunksToFloat32(chunks) {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      out[offset++] = chunk[i] / (chunk[i] < 0 ? 0x8000 : 0x7fff);
    }
  }
  return out;
}

function rmsInt16(chunk) {
  if (!chunk.length) return 0;
  let sum = 0;
  for (let i = 0; i < chunk.length; i++) {
    const n = chunk[i] / 32768;
    sum += n * n;
  }
  return Math.sqrt(sum / chunk.length);
}

/** Best-effort WebGPU capability probe (requires a real adapter, not just the API). */
async function isWebGpuAvailable() {
  try {
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
      return false;
    }
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch (_e) {
    return false;
  }
}

// Silero VAD: a tiny (~2 MB) ONNX speech/non-speech classifier that runs in the
// browser. It replaces the crude RMS gate so the engine segments on actual
// speech (not just loudness), cutting false segments and Whisper hallucinations
// on noise/music. It runs on a standalone onnxruntime-web instance pointed at
// the wasm already cached for transformers.js (no extra wasm download).
const SILERO_ORT_URL =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250306-ccf8fdd9ea/+esm';
const SILERO_MODEL_URL = '/static/whisper/silero-vad.onnx';
const SILERO_SPEECH_THRESHOLD = 0.5; // speech probability gate (0..1)

class SileroVad {
  constructor() {
    this._ort = null;
    this._session = null;
    this._state = null; // recurrent state, carried across frames
    this._sr = null; // sample-rate tensor (int64 16000)
    this._f32 = new Float32Array(512); // reusable input buffer (32 ms @ 16 kHz)
  }

  async load() {
    const ort = await import(SILERO_ORT_URL);
    // Reuse the wasm already fetched for transformers.js (same dev build); a
    // single thread is plenty for this tiny model.
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0/dist/';
    ort.env.wasm.numThreads = 1;
    this._ort = ort;
    this._session = await ort.InferenceSession.create(SILERO_MODEL_URL);
    this._state = new Float32Array(2 * 1 * 128);
    this._sr = new ort.Tensor('int64', new BigInt64Array([16000n]), []);
  }

  /**
   * @param {Int16Array} int16 - 512-sample frame @ 16 kHz
   * @returns {Promise<number>} speech probability 0..1
   */
  async process(int16) {
    const ort = this._ort;
    const f32 = this._f32;
    const n = Math.min(int16.length, f32.length);
    for (let i = 0; i < n; i++) f32[i] = int16[i] / 32768;
    for (let i = n; i < f32.length; i++) f32[i] = 0;
    const input = new ort.Tensor('float32', f32, [1, f32.length]);
    const state = new ort.Tensor('float32', this._state, [2, 1, 128]);
    const out = await this._session.run({ input, state, sr: this._sr });
    this._state = out.stateN.data; // carry the updated recurrent state
    return out.output.data[0];
  }

  dispose() {
    try {
      if (this._session && this._session.release) this._session.release();
    } catch (_e) {
      /* ignore */
    }
    this._session = null;
  }
}

export class WhisperLocalEngine {
  /**
   * @param {object} options
   * @param {string} [options.modelKey] - key in WHISPER_MODELS
   * @param {string} [options.language] - ISO 639-1 or '' for auto
   * @param {number} [options.vadThreshold] - RMS gate 0..1
   * @param {(msg: string) => void} [options.onStatus]
   * @param {(payload: { text: string, isFinal: boolean }) => void} [options.onTranscript]
   */
  constructor(options = {}) {
    this.modelKey = options.modelKey || 'tiny';
    this.language = options.language || 'cs';
    // 'auto' picks WebGPU when a GPU adapter is available, else WASM.
    // 'webgpu'/'wasm' force a specific backend.
    this.device = options.device === 'webgpu' || options.device === 'wasm' ? options.device : 'auto';
    this.vadThreshold = typeof options.vadThreshold === 'number' ? options.vadThreshold : 0.012;
    this.vadSpeechThreshold =
      typeof options.vadSpeechThreshold === 'number' ? options.vadSpeechThreshold : SILERO_SPEECH_THRESHOLD;
    this.onStatus = options.onStatus || (() => {});
    this.onTranscript = options.onTranscript || (() => {});

    /** Silero VAD (null until loaded; null also means fall back to the RMS gate). */
    this._vad = null;
    /** Frames awaiting VAD classification (VAD inference is async). */
    this._vadQueue = [];
    this._vadBusy = false;

    /** @type {import('@huggingface/transformers').AutomaticSpeechRecognitionPipeline | null} */
    this._transcriber = null;
    /** Backend the model actually loaded on ('webgpu' | 'wasm' | null). */
    this._device = null;
    this._audioContext = null;
    this._processor = null;
    this._mediaStream = null;
    this._running = false;
    this._busy = false;

    /** Queue of utterances waiting to be transcribed (each is an array of Int16 chunks). */
    this._pendingUtterances = [];

    this._speechFrames = [];
    this._inSpeech = false;
    this._silenceRun = 0;
    this._speechFrameCount = 0;
    this._loudFrameCount = 0;
    /** Ring of the most recent non-speech frames, prepended to each utterance. */
    this._preRollFrames = [];
  }

  _modelSpec() {
    return WHISPER_MODELS[this.modelKey] || WHISPER_MODELS.tiny;
  }

  async _wantsWebGpu() {
    if (this.device === 'wasm') return false;
    if (this.device === 'webgpu') return true;
    return isWebGpuAvailable();
  }

  async load() {
    const spec = this._modelSpec();

    if (await this._wantsWebGpu()) {
      try {
        this.onStatus(`Loading Whisper ${spec.label} on GPU (WebGPU)…`);
        this._transcriber = await pipeline('automatic-speech-recognition', spec.id, {
          device: 'webgpu',
          dtype: spec.webgpu || WEBGPU_DTYPE,
        });
        this._device = 'webgpu';
        this.onStatus('Whisper model ready (WebGPU)');
        await this._loadVad();
        return;
      } catch (err) {
        // No GPU adapter, missing fp16 model, or a driver/runtime issue — fall
        // back to CPU so the engine still works (just slower).
        console.warn('WebGPU Whisper unavailable, falling back to CPU/WASM:', err);
        this.onStatus('WebGPU unavailable — loading on CPU (WASM)…');
        this._transcriber = null;
      }
    }

    this.onStatus(`Loading Whisper ${spec.label} on CPU (WASM)…`);
    this._transcriber = await pipeline('automatic-speech-recognition', spec.id, {
      device: 'wasm',
      dtype: spec.wasm || WASM_DTYPE,
    });
    this._device = 'wasm';
    this.onStatus('Whisper model ready (CPU/WASM)');
    await this._loadVad();
  }

  /** Load Silero VAD; on any failure leave this._vad null so the RMS gate is used. */
  async _loadVad() {
    if (this._vad) return;
    try {
      this.onStatus('Loading voice detection…');
      const vad = new SileroVad();
      await vad.load();
      this._vad = vad;
    } catch (err) {
      console.warn('Silero VAD unavailable, falling back to energy gate:', err);
      this._vad = null;
    }
  }

  async start(mediaStream) {
    // Set _running early so a Stop pressed during load()/addModule() can signal abort.
    this._running = true;
    this._mediaStream = mediaStream;
    this._resetVad();
    this._pendingUtterances = [];

    try {
      if (!this._transcriber) {
        await this.load();
      }
      if (!this._running) return;

      this._audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      const workletUrl = new URL('/static/whisper/pcm-worklet.js', window.location.origin).href;
      await this._audioContext.audioWorklet.addModule(workletUrl);

      if (!this._running) {
        try { await this._audioContext.close(); } catch (_e) { /* ignore */ }
        this._audioContext = null;
        return;
      }

      const source = this._audioContext.createMediaStreamSource(mediaStream);
      this._processor = new AudioWorkletNode(this._audioContext, 'int16-pcm-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
      });

      this._processor.port.onmessage = (event) => {
        if (!this._running) return;
        const int16 = new Int16Array(event.data);
        this._vadQueue.push(int16);
        // Safety cap (~2 s) in case VAD inference ever falls behind frame arrival.
        if (this._vadQueue.length > 64) this._vadQueue.shift();
        this._pumpVad();
      };

      source.connect(this._processor);
      this._processor.connect(this._audioContext.destination);
      this.onStatus('Listening (local Whisper)…');
    } catch (err) {
      this._running = false;
      if (this._processor) {
        try { this._processor.disconnect(); } catch (_e) { /* ignore */ }
        this._processor = null;
      }
      if (this._audioContext) {
        try { await this._audioContext.close(); } catch (_e) { /* ignore */ }
        this._audioContext = null;
      }
      throw err;
    }
  }

  stop() {
    this._running = false;
    this._pendingUtterances = [];
    this._vadQueue = [];
    this._vadBusy = false;
    if (this._vad) {
      try { this._vad.dispose(); } catch (_e) { /* ignore */ }
      this._vad = null;
    }
    if (this._processor) {
      try {
        this._processor.disconnect();
      } catch (_e) {
        /* ignore */
      }
      this._processor = null;
    }
    if (this._audioContext) {
      try {
        this._audioContext.close();
      } catch (_e) {
        /* ignore */
      }
      this._audioContext = null;
    }
    this._resetVad();
  }

  _resetVad() {
    this._speechFrames = [];
    this._inSpeech = false;
    this._silenceRun = 0;
    this._speechFrameCount = 0;
    this._loudFrameCount = 0;
    this._preRollFrames = [];
  }

  /** Serially classify queued frames (Silero if available, else RMS) and drive the VAD machine. */
  async _pumpVad() {
    if (this._vadBusy) return;
    this._vadBusy = true;
    try {
      while (this._running && this._vadQueue.length > 0) {
        const int16 = this._vadQueue.shift();
        let loud;
        if (this._vad) {
          try {
            loud = (await this._vad.process(int16)) >= this.vadSpeechThreshold;
          } catch (_e) {
            loud = rmsInt16(int16) >= this.vadThreshold; // fall back if a run fails
          }
        } else {
          loud = rmsInt16(int16) >= this.vadThreshold;
        }
        if (!this._running) break;
        this._feedFrame(int16, loud);
      }
    } finally {
      this._vadBusy = false;
    }
  }

  _feedFrame(int16, loud) {
    if (loud) {
      if (!this._inSpeech) {
        this._inSpeech = true;
        // Seed with the pre-roll so the utterance keeps the lead-in audio from
        // just before the VAD triggered.
        this._speechFrames = this._preRollFrames;
        this._preRollFrames = [];
        this._speechFrameCount = this._speechFrames.length;
        this._loudFrameCount = 0;
      }
      this._speechFrames.push(int16);
      this._speechFrameCount++;
      this._loudFrameCount++;
      this._silenceRun = 0;

      if (this._speechFrameCount >= MAX_SPEECH_FRAMES) {
        this._enqueueUtterance();
      }
      return;
    }

    if (!this._inSpeech) {
      this._preRollFrames.push(int16);
      if (this._preRollFrames.length > PRE_ROLL_FRAMES) this._preRollFrames.shift();
      return;
    }

    this._speechFrames.push(int16);
    this._speechFrameCount++;
    this._silenceRun++;

    if (this._silenceRun >= SILENCE_FRAMES_END) {
      this._enqueueUtterance();
    }
  }

  _enqueueUtterance() {
    const frames = this._speechFrames;
    const loudFrames = this._loudFrameCount;
    this._resetVad();
    // Gate on frames the VAD actually classified as speech — `frames` also
    // holds the pre-roll and the ~900 ms silence tail, so its length alone
    // would let a single noise blip through to a full Whisper inference.
    if (loudFrames < MIN_SPEECH_FRAMES) return;
    this._pendingUtterances.push(frames);
    // Stay live on slow devices: bound the backlog by dropping the oldest
    // utterances instead of letting latency grow without limit.
    if (this._pendingUtterances.length > MAX_PENDING_UTTERANCES) {
      const dropped = this._pendingUtterances.length - MAX_PENDING_UTTERANCES;
      this._pendingUtterances.splice(0, dropped);
      this.onStatus(`Whisper can't keep up — dropped ${dropped} buffered segment${dropped > 1 ? 's' : ''}`);
    }
    this._processQueue();
  }

  async _processQueue() {
    if (this._busy) return;
    this._busy = true;
    try {
      while (this._running && this._pendingUtterances.length > 0) {
        const frames = this._pendingUtterances.shift();
        await this._transcribeUtterance(frames);
      }
    } finally {
      this._busy = false;
    }
  }

  async _transcribeUtterance(frames) {
    // Skip the opening interim if the engine has already been stopped — otherwise
    // the UI would show a '…' placeholder that never gets closed.
    const openedInterim = this._running;
    if (openedInterim) {
      this.onTranscript({ text: '', isFinal: false });
    }

    try {
      const audio = int16ChunksToFloat32(frames);
      const spec = this._modelSpec();
      // no_repeat_ngram_size breaks Whisper's runaway repetition loops (e.g.
      // "ještě, že ještě, že …" on silence/uncertain audio), which otherwise
      // generate the full 448-token budget — a huge latency spike on CPU.
      // temperature 0 = deterministic greedy decoding (fastest).
      const opts = {
        task: 'transcribe',
        return_timestamps: false,
        no_repeat_ngram_size: 3,
        temperature: 0,
      };

      // Unsupported/unknown codes stay unset → Whisper auto-detects, instead of
      // transformers.js throwing on every utterance.
      const whisperLang = spec.multilingual ? toWhisperLanguage(this.language) : null;
      if (whisperLang) {
        opts.language = whisperLang;
      }

      const result = await this._transcriber(audio, opts);
      if (!this._running) return;
      const text = String(result?.text || '').trim();
      // Always dispatch a final so the UI can clear the '…' interim, even when
      // Whisper returns blank text (silence / non-speech / [BLANK_AUDIO]).
      this.onTranscript({ text, isFinal: true });
    } catch (err) {
      console.error('Whisper transcription failed:', err);
      this.onStatus(`Whisper error: ${err?.message || err}`);
      if (this._running && openedInterim) {
        this.onTranscript({ text: '', isFinal: true });
      }
    }
  }
}
