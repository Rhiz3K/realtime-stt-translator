# Parakeet-TDT-v3 browser engine

Fully on-device STT in the browser via **onnxruntime-web + WebGPU**, reusing the
Nemotron front-end: mic → `pcm-worklet.js` (16 kHz int16) → `../nemotron/mel.js`
log-mel → ONNX in the browser → text → `/ws` (translation only). Audio never
leaves the client.

Model: [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(FastConformer encoder + **TDT** decoder with a duration head, 25 European
languages incl. Czech), int8-quantised ONNX export from
[`nasedkinpv/parakeet-tdt-0.6b-v3-onnx-int8`](https://huggingface.co/nasedkinpv/parakeet-tdt-0.6b-v3-onnx-int8).

Unlike Nemotron, this export is **offline** — there is no streaming cache, so the
engine re-encodes the growing buffer for each interim and commits a final on an
RMS-VAD pause. That makes interims more expensive than Nemotron's but keeps the
decode path much simpler.

## Files

| File | Role |
|------|------|
| `parakeet-engine.mjs` | `ParakeetModel` (load + offline encode + TDT decode) and `ParakeetLocalEngine` (mic + RMS VAD + interim/final). |
| `tdt.js` | TDT greedy decode over `decoder_joint-int8.onnx` (token head + duration head, which lets the decoder skip frames). |
| `models/` | **Generated, gitignored (~930 MB).** Build with the script below. |

The log-mel front-end (`../nemotron/mel.js`, 128-bin) and the PCM worklet
(`/static/whisper/pcm-worklet.js`) are shared, not duplicated here.

## Building the model assets

The engine needs `models/{encoder-int8.onnx,encoder-int8.onnx.data,decoder_joint-int8.onnx,vocab.txt}`.
These are large and generated, not committed. Build them once:

```bash
python3.12 -m venv .venv-parakeet-prep
.venv-parakeet-prep/bin/pip install -r requirements-parakeet-prep.txt
.venv-parakeet-prep/bin/python scripts/prepare_parakeet_onnx.py
```

The script downloads the int8 export from a pinned upstream revision and installs
it atomically (write to `.tmp`, then `os.replace`), so an interrupted run cannot
leave a truncated model in place. `python scripts/prepare_parakeet_onnx.py --inspect-only`
prints the ONNX graph I/O without downloading the weights — use it after moving
the revision pin to confirm the I/O contract the engine is written against still
holds. Set `PARAKEET_MODEL_DIR=/path/to/models` to write somewhere else.

Unlike Nemotron there is **no** auto-prepare on container start; build the assets
before enabling `parakeet` in `ENABLED_ENGINES`.

## Requirements & performance

- **WebGPU recommended.** Desktop Chrome/Edge 138+ or Android Chrome with WebGPU.
  Needs a secure context (HTTPS or `localhost`) for the mic.
- **WASM fallback** works but is slower; the app's COOP/COEP headers enable
  multi-threaded WASM, which helps.
- Because interims re-encode the whole utterance, interim latency grows with
  utterance length — the RMS-VAD pause that commits a final also resets that cost.

## Keep in sync

- onnxruntime-web version is pinned in **both** the CSP in `app/main.py`
  (`_CSPMiddleware`) and the import URL + `env.wasm.wasmPaths` in
  `parakeet-engine.mjs` — the same pin Nemotron uses, so bump them together.
- Bump the `?v=N` cache-buster on `import('/static/parakeet/parakeet-engine.mjs?v=N')`
  in `index.html` when changing the engine.
- The model file names are duplicated in `parakeet-engine.mjs` and
  `scripts/prepare_parakeet_onnx.py` — change both.
- `_ALL_ENGINES` in `app/main.py` ↔ the engine `<option>` + dispatch in `index.html`.

## License

The model is NVIDIA's; check the upstream model card's license (CC-BY-4.0) before
redistributing weights.
