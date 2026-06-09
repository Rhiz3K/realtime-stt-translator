# app/static/whisper

Support files for the browser-local Whisper STT engine, served at
`/static/whisper/*` with `Cache-Control: no-cache` (set by `_CSPMiddleware`
in `app/main.py`).

- `whisper-engine.mjs` — the engine module (Transformers.js + ONNX Runtime Web,
  Silero VAD segmentation). Loaded on demand from `index.html` via dynamic
  `import('/static/whisper/whisper-engine.mjs?v=N')`; bump `N` whenever this
  file or its pinned transformers.js/ORT versions change.
- `pcm-worklet.js` — AudioWorklet converting mic input to 16-bit / 16 kHz mono
  PCM in 512-sample frames (~32 ms), matching the engine's VAD timing
  constants.
- `silero-vad.onnx` — [Silero VAD](https://github.com/snakers4/silero-vad)
  speech/non-speech classifier (MIT license), v5 ONNX export
  (inputs `input`/`state`/`sr`, outputs `output`/`stateN`).
  sha256: `a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808`.
  To update, take `src/silero_vad/data/silero_vad.onnx` from a silero-vad
  release; keep the v5 I/O signature expected by `SileroVad` in
  `whisper-engine.mjs`.
