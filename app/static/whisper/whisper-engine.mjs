/**
 * Browser-local Whisper STT (similar to vilassn/whisper_android).
 * Uses @huggingface/transformers — audio stays on device; only text is sent to the server.
 */
import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0';

env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

/** @type {Record<string, { id: string, label: string, multilingual: boolean }>} */
export const WHISPER_MODELS = {
  tiny: { id: 'Xenova/whisper-tiny', label: 'tiny (~75 MB)', multilingual: true },
  'tiny.en': { id: 'Xenova/whisper-tiny.en', label: 'tiny.en (~75 MB)', multilingual: false },
  base: { id: 'Xenova/whisper-base', label: 'base (~145 MB)', multilingual: true },
};

const SAMPLE_RATE = 16000;
const FRAME_SAMPLES = 512;
const SILENCE_FRAMES_END = 28; // ~900 ms @ 32 ms/frame
const MIN_SPEECH_FRAMES = 13; // ~400 ms
const MAX_SPEECH_FRAMES = 470; // ~15 s cap per utterance

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
    this.vadThreshold = typeof options.vadThreshold === 'number' ? options.vadThreshold : 0.012;
    this.onStatus = options.onStatus || (() => {});
    this.onTranscript = options.onTranscript || (() => {});

    /** @type {import('@huggingface/transformers').AutomaticSpeechRecognitionPipeline | null} */
    this._transcriber = null;
    this._audioContext = null;
    this._processor = null;
    this._mediaStream = null;
    this._running = false;
    this._busy = false;

    this._speechFrames = [];
    this._inSpeech = false;
    this._silenceRun = 0;
    this._speechFrameCount = 0;
  }

  _modelSpec() {
    return WHISPER_MODELS[this.modelKey] || WHISPER_MODELS.tiny;
  }

  async load() {
    const spec = this._modelSpec();
    this.onStatus(`Loading Whisper model ${spec.label}…`);
    this._transcriber = await pipeline('automatic-speech-recognition', spec.id, {
      dtype: 'q8',
    });
    this.onStatus('Whisper model ready');
  }

  async start(mediaStream) {
    if (!this._transcriber) {
      await this.load();
    }
    this._mediaStream = mediaStream;
    this._running = true;
    this._resetVad();

    this._audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    const workletUrl = new URL('/static/whisper/pcm-worklet.js', window.location.origin).href;
    await this._audioContext.audioWorklet.addModule(workletUrl);

    const source = this._audioContext.createMediaStreamSource(mediaStream);
    this._processor = new AudioWorkletNode(this._audioContext, 'int16-pcm-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      channelCount: 1,
    });

    this._processor.port.onmessage = (event) => {
      if (!this._running || this._busy) return;
      const int16 = new Int16Array(event.data);
      this._feedFrame(int16);
    };

    source.connect(this._processor);
    this._processor.connect(this._audioContext.destination);
    this.onStatus('Listening (local Whisper)…');
  }

  stop() {
    this._running = false;
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
        this._finalizeUtterance();
      }
      return;
    }

    if (!this._inSpeech) return;

    this._speechFrames.push(int16);
    this._speechFrameCount++;
    this._silenceRun++;

    if (this._silenceRun >= SILENCE_FRAMES_END) {
      this._finalizeUtterance();
    }
  }

  async _finalizeUtterance() {
    const frames = this._speechFrames;
    this._resetVad();

    if (frames.length < MIN_SPEECH_FRAMES) return;
    if (this._busy) return;

    this._busy = true;
    this.onTranscript({ text: '', isFinal: false });

    try {
      const audio = int16ChunksToFloat32(frames);
      const spec = this._modelSpec();
      const opts = { task: 'transcribe', return_timestamps: false };

      if (spec.multilingual && this.language) {
        const whisperLang = ISO_TO_WHISPER[this.language] || this.language;
        opts.language = whisperLang;
      }

      const result = await this._transcriber(audio, opts);
      const text = String(result?.text || '').trim();
      if (text) {
        this.onTranscript({ text, isFinal: true });
      }
    } catch (err) {
      console.error('Whisper transcription failed:', err);
      this.onStatus(`Whisper error: ${err?.message || err}`);
    } finally {
      this._busy = false;
    }
  }
}
