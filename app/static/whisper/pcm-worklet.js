// 16-bit PCM mono processor for microphone capture (16 kHz).
// Buffers each AudioWorklet render quantum (128 samples) into FRAME_SAMPLES
// chunks so the engine's VAD timing constants (~32 ms/frame) match reality.
const FRAME_SAMPLES = 512;

class Int16PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(FRAME_SAMPLES);
    this._offset = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (output && output[0]) {
      output[0].fill(0);
    }

    if (!input || !input[0]) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      this._buffer[this._offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this._offset >= FRAME_SAMPLES) {
        const out = new Int16Array(this._buffer);
        this.port.postMessage(out.buffer, [out.buffer]);
        this._offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('int16-pcm-processor', Int16PCMProcessor);
