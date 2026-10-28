#!/usr/bin/env python3
"""Prepare Parakeet-tdt-0.6b-v3 ONNX assets for the browser engine.

Unlike the Nemotron script, no conversion is needed: a ready int8 ONNX export
(``nasedkinpv/parakeet-tdt-0.6b-v3-onnx-int8``) is downloaded as-is into
``app/static/parakeet/models/`` (gitignored), in the same encoder /
decoder_joint / vocab layout the Nemotron engine already uses. It also prints
the *exact* ONNX I/O contract so the JS engine (``parakeet-engine.mjs`` +
``tdt.js``) can be written against the real tensor names/shapes — in particular
the decoder_joint's token vs. duration outputs (TDT, not plain RNN-T).

Parakeet-v3 is multilingual incl. Czech (~11% FLEURS WER, CC-BY-4.0).

Dev-only.

    python scripts/prepare_parakeet_onnx.py                # download + introspect + install
    python scripts/prepare_parakeet_onnx.py --inspect-only  # fetch only the small graph files, print I/O, stop
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "nasedkinpv/parakeet-tdt-0.6b-v3-onnx-int8"
# Pinned for the same reason as the Nemotron prep script: a community repo feeding
# an auto-prepare path must not change under us. To move the pin:
# `HfApi().repo_info(REPO_ID).sha`, bump, then re-run `--inspect-only` and re-verify I/O.
REVISION = "f27d0efd32282dcb8124a4ddd2e09ce42eca03c6"
ENCODER = "encoder-int8.onnx"
ENCODER_DATA = "encoder-int8.onnx.data"
DECODER = "decoder_joint-int8.onnx"
VOCAB = "vocab.txt"

# The big weights live in ENCODER_DATA (~880 MB); the .onnx files carry only the
# graph, so I/O can be inspected without downloading the weights.
INSPECT_FILES = [ENCODER, DECODER, VOCAB]
ALL_FILES = [ENCODER, ENCODER_DATA, DECODER, VOCAB]

ROOT = Path(__file__).resolve().parent.parent


def _out_dir_from_env() -> Path:
    raw = os.getenv("PARAKEET_MODEL_DIR")
    if not raw:
        return ROOT / "app" / "static" / "parakeet" / "models"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


OUT_DIR = _out_dir_from_env()


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


def download(patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[1/3] Downloading {REPO_ID} ({', '.join(patterns)}; cached after first run)…")
    local = snapshot_download(repo_id=REPO_ID, revision=REVISION, allow_patterns=patterns)
    src = Path(local)
    for f in patterns:
        p = src / f
        if not p.exists():
            sys.exit(f"  ! missing expected file: {p}")
        print(f"      {f:28s} {_human(p.stat().st_size)}")
    return src


def _describe(model_path: Path, label: str) -> None:
    import onnx

    # Header-only load: we only need the graph I/O, not the (huge) external weights.
    m = onnx.load(str(model_path), load_external_data=False)

    def _io(values):
        out = []
        for v in values:
            t = v.type.tensor_type
            elem = onnx.TensorProto.DataType.Name(t.elem_type)
            dims = [d.dim_param or (d.dim_value if d.HasField("dim_value") else "?") for d in t.shape.dim]
            out.append(f"      {v.name:28s} {elem:8s} {dims}")
        return "\n".join(out)

    print(f"\n--- {label}: {model_path.name} ---")
    print("    inputs:")
    print(_io(m.graph.input))
    print("    outputs:")
    print(_io(m.graph.output))


def inspect(src: Path) -> None:
    print("\n[2/3] ONNX I/O contract (write parakeet-engine.mjs / tdt.js against this):")
    _describe(src / ENCODER, "ENCODER")
    _describe(src / DECODER, "DECODER_JOINT (TDT: expect token + duration outputs)")
    vocab = src / VOCAB
    if vocab.exists():
        n = sum(1 for _ in vocab.open("r", encoding="utf-8", errors="replace"))
        print(f"\n    vocab: {VOCAB} — {n} tokens")


def install(src: Path) -> None:
    print(f"\n[3/3] Copying assets to {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in ALL_FILES:
        dst = OUT_DIR / f
        tmp = dst.with_name(dst.name + ".tmp")
        try:
            shutil.copyfile(src / f, tmp)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f"Downloaded model asset is empty: {f}")
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"      {f:28s} {_human(dst.stat().st_size)}")


def _sweep_stale_temporaries() -> None:
    """Remove `.tmp` debris from a run that was killed mid-write (SIGKILL/OOM),
    which try/finally cannot clean up. These are hundreds of MB each."""
    if not OUT_DIR.exists():
        return
    for leftover in sorted(OUT_DIR.glob("*.tmp")):
        print(f"  removing stale temporary {leftover.name}")
        leftover.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect-only", action="store_true", help="download only the small graph files, print I/O, then stop")
    args = ap.parse_args()

    _sweep_stale_temporaries()

    if args.inspect_only:
        src = download(INSPECT_FILES)
        inspect(src)
        print("\n(inspect-only) done.")
        return

    src = download(ALL_FILES)
    inspect(src)
    try:
        install(src)
    except RuntimeError as exc:
        # Same clean one-line failure as download()'s sys.exit, without a traceback.
        sys.exit(f"  ! {exc}")
    print("\nDone. Set ENABLED_ENGINES to include 'parakeet' to expose the engine.")


if __name__ == "__main__":
    main()
