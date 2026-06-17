# Nemotron-3.5-ASR browser engine

Fully on-device streaming STT in the browser via **onnxruntime-web + WebGPU**, a
sibling of the local Whisper engine: mic → `pcm-worklet.js` (16 kHz int16) → ONNX
in the browser → text → `/ws` (translation only). Audio never leaves the client.

Model: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(cache-aware FastConformer encoder + RNN-T decoder, 40 language-locales),
re-exported to ONNX by [`altunenes/parakeet-rs`](https://huggingface.co/altunenes/parakeet-rs).

## Files

| File | Role |
|------|------|
| `nemotron-engine.mjs` | `NemotronModel` (load + streaming encode + RNN-T decode) and `NemotronLocalEngine` (mic + RMS VAD + streaming interim/final). |
| `mel.js` | Log-mel front-end (NeMo-compatible: preemph 0.97, 400-win/160-hop Hann, n_fft 512, 128-bin Slaney mel, `log(x+2⁻²⁴)`). |
| `rnnt.js` | RNN-T greedy decode over `decoder_joint.onnx`. |
| `tokenizer.js` | SentencePiece detokenisation from `vocab.json`. |
| `f16.js` | float32 ↔ float16 (`Float16Array`) for the fp16 encoder I/O. |
| `models/` | **Generated, gitignored (~1.2 GB).** Build with the script below. |

## Building the model assets

The engine needs `models/{encoder_fp16.onnx,encoder_fp16.onnx.data,decoder_joint.onnx,config.json,vocab.json}`.
These are large and generated, not committed. Build them once:

```bash
python3.12 -m venv .venv-nemotron-prep
.venv-nemotron-prep/bin/pip install -r requirements-nemotron-prep.txt
.venv-nemotron-prep/bin/python scripts/prepare_nemotron_onnx.py
```

The script downloads the fp32 ONNX export (~2.6 GB), converts the encoder to fp16
(~1.2 GB — fits the browser's ~2 GB ArrayBuffer limit and halves WebGPU VRAM),
copies the fp32 decoder, and extracts `vocab.json`.

## Requirements & performance

- **WebGPU strongly recommended** (real-time). Desktop Chrome/Edge 138+ or Android
  Chrome with WebGPU. Needs a secure context (HTTPS or `localhost`) for the mic.
- **CPU/WASM fallback** works but is **not real-time** for this 600 M model (≈10× slower
  than real-time single-threaded; the app's COOP/COEP headers enable multi-threaded WASM,
  which helps but still won't reach real-time on most CPUs).
- First run downloads ~1.2 GB (then cached `immutable` by the server).

## Validating a change

`scripts/nemotron_reference.py` is the Python ground truth (mel + transcript);
`scripts/validate_mel.mjs` checks `mel.js` against it; `scripts/nemotron_poc.html`
(served by `scripts/serve_poc.py`) runs the whole pipeline in a real browser.

## Keep in sync

- onnxruntime-web version is pinned in **both** the CSP in `app/main.py` (`_CSPMiddleware`)
  and the import URL + `env.wasm.wasmPaths` in `nemotron-engine.mjs`.
- Bump the `?v=N` cache-buster on the `import('/static/nemotron/nemotron-engine.mjs?v=N')`
  in `index.html` when changing the engine.
- `_ALL_ENGINES` in `app/main.py` ↔ the engine `<option>` + dispatch in `index.html`.

## License

The model is NVIDIA's; check the upstream model card's license (OpenMDW-1.1) before
redistributing weights.
