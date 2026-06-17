// Log-mel front-end for the Nemotron-3.5-ASR streaming engine.
//
// A 1:1 port of `log_mel()` in scripts/nemotron_reference.py, which mirrors
// NeMo's AudioToMelSpectrogramPreprocessor. Keep the two in sync — the Python
// reference is the ground truth the browser output is validated against.
//
// Front-end: preemphasis 0.97 -> center reflect-pad n_fft/2 -> 400-pt symmetric
// Hann centered in a 512-pt FFT -> power spectrum -> 128-bin Slaney mel ->
// log(x + 2^-24) -> optional per-feature normalisation.

export const SR = 16000;
export const N_FFT = 512;
export const HOP = 160;
export const WIN = 400;
export const N_MELS = 128;
const PREEMPH = 0.97;
const LOG_GUARD = Math.pow(2, -24);
const N_BINS = N_FFT / 2 + 1; // 257

// ---------- iterative radix-2 FFT (n must be a power of two) ----------
class FFT {
  constructor(n) {
    this.n = n;
    this.cos = new Float32Array(n / 2);
    this.sin = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
    // bit-reversal permutation table
    this.rev = new Uint32Array(n);
    let bits = 0;
    while (1 << bits < n) bits++;
    for (let i = 0; i < n; i++) {
      let x = i, r = 0;
      for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
      this.rev[i] = r;
    }
  }

  // In-place complex FFT of (re, im); imag is assumed zero for our real input.
  transform(re, im) {
    const n = this.n, rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let k = 0, idx = 0; k < half; k++, idx += step) {
          const c = this.cos[idx], s = this.sin[idx];
          const a = i + k, b = i + k + half;
          const tre = re[b] * c - im[b] * s;
          const tim = re[b] * s + im[b] * c;
          re[b] = re[a] - tre; im[b] = im[a] - tim;
          re[a] += tre; im[a] += tim;
        }
      }
    }
  }
}

// ---------- Slaney mel scale (matches librosa htk=False, norm='slaney') ----------
function hzToMel(hz) {
  const fSp = 200.0 / 3.0, minLogHz = 1000.0, minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27.0;
  return hz >= minLogHz ? minLogMel + Math.log(hz / minLogHz) / logstep : hz / fSp;
}
function melToHz(mel) {
  const fSp = 200.0 / 3.0, minLogHz = 1000.0, minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27.0;
  return mel >= minLogMel ? minLogHz * Math.exp(logstep * (mel - minLogMel)) : fSp * mel;
}

// fb[m] is a {start, vals} sparse row over FFT bins (most weights are zero).
function buildMelFilterbank(fmin = 0, fmax = SR / 2) {
  const fftFreqs = new Float32Array(N_BINS);
  for (let k = 0; k < N_BINS; k++) fftFreqs[k] = (k * SR) / N_FFT;
  const melMin = hzToMel(fmin), melMax = hzToMel(fmax);
  const hzPts = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    hzPts[i] = melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1));
  }
  const rows = [];
  for (let m = 0; m < N_MELS; m++) {
    const lo = hzPts[m], ctr = hzPts[m + 1], hi = hzPts[m + 2];
    const enorm = 2.0 / (hi - lo); // Slaney area normalisation
    const dLo = ctr - lo, dHi = hi - ctr;
    const vals = [];
    let start = -1;
    for (let k = 0; k < N_BINS; k++) {
      const f = fftFreqs[k];
      const w = Math.max(0, Math.min((f - lo) / dLo, (hi - f) / dHi)) * enorm;
      if (w > 0) { if (start < 0) start = k; vals.push(w); }
      else if (start >= 0) break; // triangular: contiguous support
    }
    rows.push({ start: start < 0 ? 0 : start, vals: new Float32Array(vals) });
  }
  return rows;
}

// 400-pt symmetric Hann, centered (zero-padded) inside the 512-pt FFT buffer.
function buildWindow() {
  const win = new Float32Array(N_FFT);
  const off = (N_FFT - WIN) >> 1; // 56
  for (let i = 0; i < WIN; i++) win[off + i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1));
  return win;
}

const _fft = new FFT(N_FFT);
const _fb = buildMelFilterbank();
const _win = buildWindow();

// Reflect-pad like numpy.pad(mode='reflect'): mirror without repeating the edge sample.
function reflectPad(x, pad) {
  const n = x.length;
  const out = new Float32Array(n + 2 * pad);
  out.set(x, pad);
  for (let i = 0; i < pad; i++) {
    out[pad - 1 - i] = x[i + 1];          // left mirror
    out[pad + n + i] = x[n - 2 - i];      // right mirror
  }
  return out;
}

/**
 * Compute log-mel features for a mono 16 kHz Float32Array in [-1, 1].
 * @returns {{data: Float32Array, nMels: number, nFrames: number}} row-major [nMels, nFrames]
 */
export function computeLogMel(audio, { normalize = "none", padMode = "reflect" } = {}) {
  // preemphasis over the whole signal (y[0] = x[0])
  const y0 = new Float32Array(audio.length);
  y0[0] = audio[0];
  for (let i = 1; i < audio.length; i++) y0[i] = audio[i] - PREEMPH * audio[i - 1];
  const pad = N_FFT >> 1;
  const y = padMode === "reflect" ? reflectPad(y0, pad) : (() => {
    const o = new Float32Array(y0.length + 2 * pad); o.set(y0, pad); return o;
  })();

  const nFrames = 1 + Math.floor((y.length - N_FFT) / HOP);
  const mel = new Float32Array(N_MELS * nFrames);
  const re = new Float32Array(N_FFT);
  const im = new Float32Array(N_FFT);
  const power = new Float32Array(N_BINS);

  for (let t = 0; t < nFrames; t++) {
    const base = t * HOP;
    for (let i = 0; i < N_FFT; i++) { re[i] = y[base + i] * _win[i]; im[i] = 0; }
    _fft.transform(re, im);
    for (let k = 0; k < N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      const row = _fb[m];
      let acc = 0;
      for (let j = 0; j < row.vals.length; j++) acc += row.vals[j] * power[row.start + j];
      mel[m * nFrames + t] = Math.log(acc + LOG_GUARD);
    }
  }

  if (normalize === "per_feature") {
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      for (let t = 0; t < nFrames; t++) sum += mel[m * nFrames + t];
      const mean = sum / nFrames;
      let varAcc = 0;
      for (let t = 0; t < nFrames; t++) { const d = mel[m * nFrames + t] - mean; varAcc += d * d; }
      const std = Math.sqrt(varAcc / (nFrames - 1)) + 1e-5;
      for (let t = 0; t < nFrames; t++) mel[m * nFrames + t] = (mel[m * nFrames + t] - mean) / std;
    }
  }
  return { data: mel, nMels: N_MELS, nFrames };
}
