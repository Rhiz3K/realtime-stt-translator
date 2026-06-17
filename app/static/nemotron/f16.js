// float32 <-> float16 helpers. The encoder ONNX is uniformly fp16, so its float
// inputs/outputs are exchanged as Float16Array — which is what onnxruntime-web
// requires for a 'float16' tensor (it rejects a raw Uint16Array). Float16Array is
// the WebGPU-era baseline (Chrome 138+ / the WebGPU-capable browsers we target).

const HAS_F16 = typeof Float16Array !== "undefined";

function ensure() {
  if (!HAS_F16) {
    throw new Error(
      "Float16Array is required for the local Nemotron engine. Use a current Chrome/Edge (138+) with WebGPU."
    );
  }
}

/** Float32Array -> Float16Array (values converted, not bit-reinterpreted). */
export function f32ToF16(f32) {
  ensure();
  return new Float16Array(f32);
}

/** Float16Array (ORT output) -> Float32Array. */
export function f16ToF32(f16) {
  return new Float32Array(f16);
}

/** Zero-filled fp16 tensor payload of `n` elements. */
export function zerosF16(n) {
  ensure();
  return new Float16Array(n);
}
