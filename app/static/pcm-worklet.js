class Pcm16CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate;
    this.source = [];
    this.position = 0;
    this.frame = new Int16Array(1600); // Fixed 100 ms at 16 kHz.
    this.frameLength = 0;
    this.active = true;
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === "flush") this.flush();
    };
  }

  encode(value) {
    const clipped = Math.max(-1, Math.min(1, value));
    return clipped < 0 ? Math.round(clipped * 32768) : Math.round(clipped * 32767);
  }

  emit(value) {
    this.frame[this.frameLength++] = this.encode(value);
    if (this.frameLength === this.frame.length) this.postFrame(this.frame);
  }

  postFrame(samples) {
    const output = samples.buffer.slice(
      samples.byteOffset,
      samples.byteOffset + samples.byteLength
    );
    this.port.postMessage(output, [output]);
    this.frameLength = 0;
  }

  resample() {
    if (this.ratio >= 1) {
      // Weighted interval averaging doubles as a small low-pass filter during
      // decimation, avoiding the strong speech-band aliasing of point/linear
      // sampling when a browser ignores the requested 16 kHz context rate.
      while (this.position + this.ratio <= this.source.length) {
        const end = this.position + this.ratio;
        const first = Math.floor(this.position);
        const last = Math.ceil(end);
        let sum = 0;
        for (let index = first; index < last; index += 1) {
          const weight = Math.min(end, index + 1) - Math.max(this.position, index);
          if (weight > 0) sum += this.source[index] * weight;
        }
        this.emit(sum / this.ratio);
        this.position = end;
      }
    } else {
      // Upsampling is unusual here; linear interpolation is appropriate.
      while (this.position + 1 < this.source.length) {
        const left = Math.floor(this.position);
        const fraction = this.position - left;
        const value = this.source[left] * (1 - fraction) + this.source[left + 1] * fraction;
        this.emit(value);
        this.position += this.ratio;
      }
    }
    const retain = this.ratio < 1 ? 1 : 0;
    const consumed = Math.min(Math.floor(this.position), Math.max(0, this.source.length - retain));
    if (consumed > 0) {
      this.source.splice(0, consumed);
      this.position -= consumed;
    }
  }

  flush() {
    if (!this.active) return;
    this.active = false;
    this.resample();
    if (this.frameLength > 0) {
      this.postFrame(this.frame.subarray(0, this.frameLength));
    }
    this.port.postMessage({type: "flushed"});
  }

  process(inputs) {
    if (!this.active) return true;
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;
    for (let index = 0; index < channel.length; index += 1) {
      this.source.push(channel[index]);
    }
    this.resample();
    return true;
  }
}

registerProcessor("pcm16-capture", Pcm16CaptureProcessor);
