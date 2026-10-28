// TDT (Token-and-Duration Transducer) greedy decoder for the Parakeet-tdt-0.6b-v3
// browser engine. Adapted from ../nemotron/rnnt.js: the fused prediction-net +
// joint (decoder_joint-int8.onnx) emits, per step, BOTH a token distribution
// (vocab + blank) AND a duration distribution. Greedy decode advances the encoder
// time index by the predicted duration instead of always +1 on blank.
//
// decoder_joint-int8.onnx I/O (fp32) — verified by scripts/prepare_parakeet_onnx.py:
//   in : encoder_outputs[1,1024,1], targets[1,1] i32, target_length[1] i32,
//        input_states_1[2,1,640], input_states_2[2,1,640]
//   out: outputs[..., VOCAB_SIZE + NUM_DURATIONS], output_states_1[2,1,640],
//        output_states_2[2,1,640]
//   VOCAB_SIZE = 8193 (token logits, incl. blank id 8192); NUM_DURATIONS = 5.
//
// The duration set for NeMo parakeet-tdt models is [0,1,2,3,4]; verify on-device
// (wrong durations => systematically mistimed/garbled output).

export const DURATIONS = [0, 1, 2, 3, 4];

export class TdtDecoder {
  constructor(ort, session, {
    blankId,
    vocabSize,
    durations = DURATIONS,
    hiddenDim = 1024,
    stateDim = 640,
    stateLayers = 2,
    maxSymbols = 10,
  }) {
    this.ort = ort;
    this.session = session;
    this.blankId = blankId;
    this.vocabSize = vocabSize;        // count of token logits (incl. blank)
    this.durations = durations;
    this.hiddenDim = hiddenDim;
    this.stateDim = stateDim;
    this.stateLayers = stateLayers;
    this.maxSymbols = maxSymbols;
    this.reset();
  }

  /** Reset prediction-net state + previous token. Call at the start of each decode. */
  reset() {
    const n = this.stateLayers * 1 * this.stateDim;
    this.s1 = new Float32Array(n);
    this.s2 = new Float32Array(n);
    this.lastToken = this.blankId; // blank acts as SOS
  }

  /**
   * Greedily decode a block of encoder frames with TDT time-advance.
   * @param {Float32Array} encoded channel-major [1, hiddenDim, T] (encoded[c*T + t])
   * @param {number} T number of time frames
   * @param {(id:number)=>void} emit called with each emitted (non-blank) token id
   */
  async decode(encoded, T, emit) {
    const ort = this.ort;
    const H = this.hiddenDim;
    const V = this.vocabSize;
    const nDur = this.durations.length;
    const stateDims = [this.stateLayers, 1, this.stateDim];

    let t = 0;
    while (t < T) {
      const enc = new Float32Array(H);
      for (let c = 0; c < H; c++) enc[c] = encoded[c * T + t];
      const encTensor = new ort.Tensor("float32", enc, [1, H, 1]);

      let symbols = 0;
      let advanced = false;
      while (symbols < this.maxSymbols) {
        const out = await this.session.run({
          encoder_outputs: encTensor,
          targets: new ort.Tensor("int32", Int32Array.of(this.lastToken), [1, 1]),
          target_length: new ort.Tensor("int32", Int32Array.of(1), [1]),
          input_states_1: new ort.Tensor("float32", this.s1, stateDims),
          input_states_2: new ort.Tensor("float32", this.s2, stateDims),
        });
        const logits = out.outputs.data; // [V + nDur]

        // Token argmax over [0, V).
        let k = 0, best = logits[0];
        for (let i = 1; i < V; i++) if (logits[i] > best) { best = logits[i]; k = i; }
        // Duration argmax over [V, V + nDur).
        let di = 0, dbest = logits[V];
        for (let j = 1; j < nDur; j++) { const v = logits[V + j]; if (v > dbest) { dbest = v; di = j; } }
        const d = this.durations[di];

        if (k === this.blankId) {
          // Blank: advance time (force progress if the model predicted duration 0).
          t += d > 0 ? d : 1;
          advanced = true;
          break;
        }

        // Non-blank: emit, advance prediction-net state + previous token.
        emit(k);
        this.lastToken = k;
        this.s1 = out.output_states_1.data;
        this.s2 = out.output_states_2.data;
        symbols++;

        if (d > 0) { t += d; advanced = true; break; }
        // d === 0: emit another symbol at the same frame (bounded by maxSymbols).
      }
      if (!advanced) t += 1; // safety: maxSymbols hit with duration 0 -> force progress
    }
  }
}
