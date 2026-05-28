// 16-bit PCM mono processor for microphone capture (16 kHz).
class Int16PCMProcessor extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (output && output[0]) {
      output[0].fill(0);
    }

    if (!input || !input[0]) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    const int16Data = new Int16Array(channel.length);
    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage(int16Data.buffer, [int16Data.buffer]);
    return true;
  }
}

registerProcessor('int16-pcm-processor', Int16PCMProcessor);
