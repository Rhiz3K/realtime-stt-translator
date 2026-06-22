// RNNT greedy decoder for the Nemotron-3.5-ASR engine.
//
// Ports the greedy loop from scripts/nemotron_reference.py: for each encoder time
// frame, query the fused prediction-network + joint (decoder_joint.onnx) until it
// emits blank, advancing the prediction-net state + previous token only on
// non-blank emissions. The prediction-net LSTM state and previous token persist
// across encoder chunks (reset per utterance) so streaming stays continuous.
//
// decoder_joint.onnx I/O (fp32):
//   in : encoder_outputs[1,1024,1], targets[1,1] i32, target_length[1] i32,
//        input_states_1[2,1,640], input_states_2[2,1,640]
//   out: outputs[...,13088] logits, output_states_1[2,1,640], output_states_2[2,1,640]

export class RnntDecoder {
  constructor(ort, session, { blankId, hiddenDim = 1024, stateDim = 640, stateLayers = 2, maxSymbols = 10 }) {
    this.ort = ort;
    this.session = session;
    this.blankId = blankId;
    this.hiddenDim = hiddenDim;
    this.stateDim = stateDim;
    this.stateLayers = stateLayers;
    this.maxSymbols = maxSymbols;
    this.reset();
  }

  /** Reset prediction-net state + previous token. Call at the start of each utterance. */
  reset() {
    const n = this.stateLayers * 1 * this.stateDim;
    this.s1 = new Float32Array(n);
    this.s2 = new Float32Array(n);
    this.lastToken = this.blankId; // blank acts as SOS
  }

  /**
   * Greedily decode a block of encoder frames.
   * @param {Float32Array} encoded channel-major [1, hiddenDim, T] (encoded[c*T + t])
   * @param {number} T number of time frames in this block
   * @param {(id:number)=>void} emit called with each emitted (non-blank) token id
   */
  async decode(encoded, T, emit) {
    const ort = this.ort;
    const H = this.hiddenDim;
    const stateDims = [this.stateLayers, 1, this.stateDim];
    for (let t = 0; t < T; t++) {
      const enc = new Float32Array(H);
      for (let c = 0; c < H; c++) enc[c] = encoded[c * T + t];
      const encTensor = new ort.Tensor("float32", enc, [1, H, 1]);
      for (let sym = 0; sym < this.maxSymbols; sym++) {
        const out = await this.session.run({
          encoder_outputs: encTensor,
          targets: new ort.Tensor("int32", Int32Array.of(this.lastToken), [1, 1]),
          target_length: new ort.Tensor("int32", Int32Array.of(1), [1]),
          input_states_1: new ort.Tensor("float32", this.s1, stateDims),
          input_states_2: new ort.Tensor("float32", this.s2, stateDims),
        });
        const logits = out.outputs.data; // last dim = vocab_size + 1 (blank)
        let k = 0, best = logits[0];
        for (let i = 1; i < logits.length; i++) if (logits[i] > best) { best = logits[i]; k = i; }
        if (k === this.blankId) break;
        emit(k);
        this.lastToken = k;
        this.s1 = out.output_states_1.data;
        this.s2 = out.output_states_2.data;
      }
    }
  }
}
