#!/usr/bin/env python3
"""Reference CPU inference for the Nemotron-3.5-ASR streaming ONNX export.

This is the *ground truth* for the browser (JS) port: it implements the exact
log-mel front-end + cache-aware streaming encoder loop + RNNT greedy decode in
NumPy/onnxruntime, prints the transcript, and dumps the mel + a few intermediate
values so the JS engine can be checked for numeric parity.

The log-mel front-end mirrors NeMo's AudioToMelSpectrogramPreprocessor so the JS
mel.js can be transliterated 1:1. Normalisation is a flag because the export's
config says ``normalize: "NA"`` while the parakeet-rs Rust reference does
per-feature — we resolve it empirically (whichever yields a correct transcript).

    python scripts/nemotron_reference.py --wav scripts/sample_16k.wav --lang en --normalize none
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "app" / "static" / "nemotron" / "models"

# --- preprocessor constants (from config.json / NeMo) ---
SR = 16000
N_FFT = 512
HOP = 160
WIN = 400
N_MELS = 128
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24

# --- streaming constants ---
CHUNK_FRAMES = 56        # new mel frames per chunk (= 8960 samples / 160)
PRE_ENCODE = 9           # mel frames carried from previous chunk's tail
BLANK = 13087
MAX_SYM_PER_FRAME = 10


# ---------- audio ----------
def read_wav(path: Path) -> np.ndarray:
    w = wave.open(str(path))
    assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16kHz mono 16-bit"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


# ---------- Slaney mel filterbank (matches librosa htk=False, norm='slaney') ----------
def _hz_to_mel(hz):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.asarray(hz, dtype=np.float64)
    return np.where(hz >= min_log_hz, min_log_mel + np.log(np.maximum(hz, 1e-12) / min_log_hz) / logstep, hz / f_sp)


def _mel_to_hz(mel):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel >= min_log_mel, min_log_hz * np.exp(logstep * (mel - min_log_mel)), f_sp * mel)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=0.0, fmax=SR / 2):
    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_bins)
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fftfreqs[None, :]
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        fb[i] = np.maximum(0, np.minimum(lower, upper))
        enorm = 2.0 / (hz_pts[i + 2] - hz_pts[i])  # Slaney area normalisation
        fb[i] *= enorm
    return fb.astype(np.float32)


def log_mel(audio: np.ndarray, normalize: str, pad_mode: str) -> np.ndarray:
    # preemphasis over the whole signal (y[0] = x[0])
    y = audio.copy()
    y[1:] = audio[1:] - PREEMPH * audio[:-1]
    # center pad by n_fft//2
    pad = N_FFT // 2
    y = np.pad(y, pad, mode=pad_mode)
    # 400-pt symmetric Hann placed centered in the 512-pt FFT buffer
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / (WIN - 1))
    win = np.zeros(N_FFT, dtype=np.float64)
    off = (N_FFT - WIN) // 2
    win[off:off + WIN] = hann
    n_frames = 1 + (len(y) - N_FFT) // HOP
    fb = mel_filterbank()
    mel = np.empty((N_MELS, n_frames), dtype=np.float32)
    for t in range(n_frames):
        frame = y[t * HOP: t * HOP + N_FFT] * win
        power = np.abs(np.fft.rfft(frame, n=N_FFT)) ** 2
        mel[:, t] = fb @ power.astype(np.float32)
    mel = np.log(mel + LOG_GUARD)
    if normalize == "per_feature":
        m = mel.mean(axis=1, keepdims=True)
        s = mel.std(axis=1, ddof=1, keepdims=True) + 1e-5
        mel = (mel - m) / s
    return mel  # [n_mels, T]


# ---------- streaming encode + RNNT greedy ----------
def run(wav: Path, lang: str, normalize: str, pad_mode: str):
    cfg = json.loads((MODELS / "config.json").read_text())
    prompt_index = cfg["prompt_dictionary"].get(lang, cfg["prompt_dictionary"].get("auto", 0))
    cs = cfg["cache_shapes"]
    print(f"lang={lang} -> prompt_index={prompt_index}; normalize={normalize}; pad_mode={pad_mode}")

    # Compute + dump mel first so mel.js can be validated even if the encoder fails.
    audio = read_wav(wav)
    mel = log_mel(audio, normalize, pad_mode)
    np.save(ROOT / "scripts" / "ref_mel.npy", mel)
    print(f"mel shape={mel.shape} mean={mel.mean():.4f} std={mel.std():.4f} "
          f"first3={mel[:3,0].tolist()}")

    so = ort.SessionOptions()
    enc = ort.InferenceSession(str(MODELS / "encoder_fp16.onnx"), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(MODELS / "decoder_joint.onnx"), so, providers=["CPUExecutionProvider"])

    # encoder cache — fp16 I/O (the encoder graph is uniformly fp16)
    c_chan = np.zeros(cs["cache_last_channel"], dtype=np.float16)
    c_time = np.zeros(cs["cache_last_time"], dtype=np.float16)
    c_len = np.zeros(cs["cache_last_channel_len"], dtype=np.int64)

    enc_frames = []
    T = mel.shape[1]
    prev_tail = np.zeros((N_MELS, PRE_ENCODE), dtype=np.float32)  # zeros for first chunk
    start = 0
    while start < T:
        main = mel[:, start:start + CHUNK_FRAMES]
        chunk = np.concatenate([prev_tail, main], axis=1)  # [128, <=65]
        prev_tail = main[:, -PRE_ENCODE:] if main.shape[1] >= PRE_ENCODE else main
        feed = {
            "processed_signal": chunk[None].astype(np.float16),
            "processed_signal_length": np.array([chunk.shape[1]], dtype=np.int64),
            "cache_last_channel": c_chan,
            "cache_last_time": c_time,
            "cache_last_channel_len": c_len,
        }
        if "prompt_index" in {i.name for i in enc.get_inputs()}:
            feed["prompt_index"] = np.array([prompt_index], dtype=np.int64)
        out = enc.run(None, feed)
        names = [o.name for o in enc.get_outputs()]
        o = dict(zip(names, out))
        enc_frames.append(o["encoded"])  # [1,1024,t_out]
        c_chan = o["cache_last_channel_next"]
        c_time = o["cache_last_time_next"]
        c_len = o["cache_last_channel_len_next"]
        start += CHUNK_FRAMES

    encoded = np.concatenate(enc_frames, axis=2)  # [1,1024,total]
    print(f"encoded shape={encoded.shape}")

    # RNNT greedy
    s1 = np.zeros((2, 1, 640), dtype=np.float32)
    s2 = np.zeros((2, 1, 640), dtype=np.float32)
    last = BLANK
    ids = []
    Tout = encoded.shape[2]
    for t in range(Tout):
        enc_t = encoded[:, :, t:t + 1].astype(np.float32)
        for _ in range(MAX_SYM_PER_FRAME):
            douts = dec.run(None, {
                "encoder_outputs": enc_t,
                "targets": np.array([[last]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": s1,
                "input_states_2": s2,
            })
            dn = [o.name for o in dec.get_outputs()]
            dd = dict(zip(dn, douts))
            logits = dd["outputs"].reshape(-1)
            k = int(np.argmax(logits))
            if k == BLANK:
                break
            ids.append(k)
            last = k
            s1 = dd["output_states_1"]
            s2 = dd["output_states_2"]

    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(ROOT / "scripts" / "sample_16k.wav"))
    ap.add_argument("--lang", default="en")
    ap.add_argument("--normalize", choices=["none", "per_feature"], default="none")
    ap.add_argument("--pad-mode", choices=["reflect", "constant"], default="reflect")
    args = ap.parse_args()

    ids = run(Path(args.wav), args.lang, args.normalize, args.pad_mode)

    # detokenise with the vocab.json we extracted (mirrors the JS detok path):
    # skip control/special tokens (<unk>, language tags like <en-US>).
    import re
    vocab = json.loads((MODELS / "vocab.json").read_text(encoding="utf-8"))
    keep = [vocab[i] for i in ids if 0 <= i < len(vocab) and not re.fullmatch(r"<[^>]*>", vocab[i])]
    text = re.sub(r"\s+", " ", "".join(keep).replace("▁", " ")).strip()
    print(f"\nemitted {len(ids)} tokens")
    print(f"TRANSCRIPT: {text!r}")
    (ROOT / "scripts" / "ref_transcript.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
