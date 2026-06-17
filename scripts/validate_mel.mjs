// Validate app/static/nemotron/mel.js against the Python reference (scripts/ref_mel.npy).
// Run after scripts/nemotron_reference.py has produced ref_mel.npy:
//   node scripts/validate_mel.mjs [normalize]
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dir = dirname(fileURLToPath(import.meta.url));
const NORMALIZE = process.argv[2] || "none";

function parseWav(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let off = 12;
  while (off + 8 <= dv.byteLength) {
    const id = String.fromCharCode(dv.getUint8(off), dv.getUint8(off + 1), dv.getUint8(off + 2), dv.getUint8(off + 3));
    const sz = dv.getUint32(off + 4, true);
    if (id === "data") {
      const n = sz >> 1;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = dv.getInt16(off + 8 + i * 2, true) / 32768;
      return out;
    }
    off += 8 + sz + (sz & 1);
  }
  throw new Error("no data chunk");
}

// Minimal .npy reader for little-endian float32 (descr '<f4').
function parseNpy(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const hlen = dv.getUint16(8, true);
  const header = new TextDecoder().decode(buf.subarray(10, 10 + hlen));
  if (!header.includes("<f4")) throw new Error("expected <f4 npy, got: " + header);
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const dataOff = 10 + hlen;
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = dv.getFloat32(dataOff + i * 4, true);
  return { shape, data };
}

const { computeLogMel } = await import(join(__dir, "..", "app", "static", "nemotron", "mel.js"));

const wav = parseWav(readFileSync(join(__dir, "sample_16k.wav")));
const { shape, data: ref } = parseNpy(readFileSync(join(__dir, "ref_mel.npy")));
const js = computeLogMel(wav, { normalize: NORMALIZE });

console.log(`ref shape [${shape}]  js [${js.nMels}, ${js.nFrames}]  normalize=${NORMALIZE}`);
if (shape[0] !== js.nMels || shape[1] !== js.nFrames) {
  console.log(`SHAPE MISMATCH (ref frames ${shape[1]} vs js ${js.nFrames})`);
}
const n = Math.min(ref.length, js.data.length);
let maxAbs = 0, sumAbs = 0;
for (let i = 0; i < n; i++) {
  const d = Math.abs(ref[i] - js.data[i]);
  if (d > maxAbs) maxAbs = d;
  sumAbs += d;
}
console.log(`maxAbsDiff=${maxAbs.toExponential(3)}  meanAbsDiff=${(sumAbs / n).toExponential(3)}`);
console.log(maxAbs < 1e-2 ? "PASS (mel.js matches reference)" : "FAIL (mel mismatch — check preemph/pad/window/mel-norm)");
