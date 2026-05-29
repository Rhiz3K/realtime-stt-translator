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

/**
 * Available models. `webgpu`/`wasm` hold the dtype used on each backend.
 * Whisper's encoder is quantization-sensitive, so on WebGPU we keep the encoder
 * at fp16 and only 4-bit-quantize the decoder (q4) for speed/size; on CPU/WASM
 * we use int8 (q8) for everything to keep the download small.
 * @type {Record<string, { id: string, label: string, multilingual: boolean, webgpu: any, wasm: any }>}
 */
export const WHISPER_MODELS = {
  tiny: {
    id: 'onnx-community/whisper-tiny',
    label: 'tiny (fastest, weak Czech)',
    multilingual: true,
    webgpu: 'fp16',
    wasm: 'q8',
  },
  base: {
    id: 'onnx-community/whisper-base',
    label: 'base (multilingual)',
    multilingual: true,
    webgpu: 'fp16',
    wasm: 'q8',
  },
  small: {
    id: 'onnx-community/whisper-small',
    label: 'small (better Czech)',
    multilingual: true,
    webgpu: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    wasm: 'q8',
  },
  'large-v3-turbo': {
    id: 'onnx-community/whisper-large-v3-turbo',
    label: 'large-v3-turbo (best Czech)',
    multilingual: true,
    webgpu: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    wasm: 'q8',
  },
  'tiny.en': {
    id: 'onnx-community/whisper-tiny.en',
    label: 'tiny.en (English only)',
    multilingual: false,
    webgpu: 'fp16',
    wasm: 'q8',
  },
};

const SAMPLE_RATE = 16000;
// pcm-worklet.js buffers each render quantum into FRAME_SAMPLES (512) before
// posting, so each frame the engine sees is ~32 ms at 16 kHz.
const SILENCE_FRAMES_END = 28; // ~900 ms @ 32 ms/frame
const MIN_SPEECH_FRAMES = 13; // ~400 ms
const MAX_SPEECH_FRAMES = 470; // ~15 s cap per utterance
// Cap the transcription backlog. On slow CPUs (mobile WASM) inference can be
// slower than real time; without a cap the queue — and thus latency — grows
// without bound. We keep only the most recent utterances and drop the oldest.
const MAX_PENDING_UTTERANCES = 3;

// dtype per backend. WASM uses int8 (small download, matches the advertised
// model sizes); WebGPU uses fp16 (GPU-friendly and much faster) with a graceful
// fallback to WASM when no GPU adapter or fp16 model is available.
const WASM_DTYPE = 'q8';
const WEBGPU_DTYPE = 'fp16';

/** ISO 639-1 → Whisper language token (multilingual models). */
const ISO_TO_WHISPER = {
  cs: 'czech',
  en: 'english',
  ru: 'russian',
  de: 'german',
  fr: 'french',
  es: 'spanish',
  pl: 'polish',
  sk: 'slovak',
  uk: 'ukrainian',
  it: 'italian',
  pt: 'portuguese',
  nl: 'dutch',
  ja: 'japanese',
  zh: 'chinese',
};

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
    this.onStatus = options.onStatus || (() => {});
    this.onTranscript = options.onTranscript || (() => {});

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
        this._feedFrame(int16);
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
  }

  _feedFrame(int16) {
    const level = rmsInt16(int16);
    const loud = level >= this.vadThreshold;

    if (loud) {
      if (!this._inSpeech) {
        this._inSpeech = true;
        this._speechFrames = [];
        this._speechFrameCount = 0;
      }
      this._speechFrames.push(int16);
      this._speechFrameCount++;
      this._silenceRun = 0;

      if (this._speechFrameCount >= MAX_SPEECH_FRAMES) {
        this._enqueueUtterance();
      }
      return;
    }

    if (!this._inSpeech) return;

    this._speechFrames.push(int16);
    this._speechFrameCount++;
    this._silenceRun++;

    if (this._silenceRun >= SILENCE_FRAMES_END) {
      this._enqueueUtterance();
    }
  }

  _enqueueUtterance() {
    const frames = this._speechFrames;
    this._resetVad();
    if (frames.length < MIN_SPEECH_FRAMES) return;
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
      const opts = { task: 'transcribe', return_timestamps: false };

      if (spec.multilingual && this.language) {
        const whisperLang = ISO_TO_WHISPER[this.language] || this.language;
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
