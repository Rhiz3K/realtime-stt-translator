#!/usr/bin/env python3
"""Prepare Nemotron-3.5-ASR-streaming ONNX assets for the browser engine.

Downloads the community ONNX export (encoder / decoder_joint / tokenizer) from
``altunenes/parakeet-rs``, prints the *exact* ONNX I/O contract (so we can verify
it against the JS engine's assumptions), converts the ~2.45 GB fp32 encoder to
fp16 (~1.2 GB — fits the browser's ~2 GB ArrayBuffer limit and halves WebGPU VRAM,
while keeping fp32 model I/O via ``keep_io_types`` so the JS side stays simple),
copies the small fp32 decoder_joint as-is, and extracts a flat ``vocab.json``
(id -> SentencePiece piece) for detokenisation.

Dev-only. Outputs land in ``app/static/nemotron/models/`` (gitignored).

    python scripts/prepare_nemotron_onnx.py            # download + introspect + convert + vocab
    python scripts/prepare_nemotron_onnx.py --inspect-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "altunenes/parakeet-rs"
SUBDIR = "nemotron-3.5-asr-streaming-0.6b-onnx"
FILES = ["encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model", "config.json"]

ROOT = Path(__file__).resolve().parent.parent


def _out_dir_from_env() -> Path:
    raw = os.getenv("NEMOTRON_MODEL_DIR")
    if not raw:
        return ROOT / "app" / "static" / "nemotron" / "models"
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


def download() -> Path:
    from huggingface_hub import snapshot_download

    print(f"[1/4] Downloading {REPO_ID}/{SUBDIR}/* (~2.6 GB, cached after first run)…")
    local = snapshot_download(repo_id=REPO_ID, allow_patterns=[f"{SUBDIR}/*"])
    src = Path(local) / SUBDIR
    for f in FILES:
        p = src / f
        if not p.exists():
            sys.exit(f"  ! missing expected file: {p}")
        print(f"      {f:24s} {_human(p.stat().st_size)}")
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
    print("\n[2/4] ONNX I/O contract (verify against the JS engine assumptions):")
    _describe(src / "encoder.onnx", "ENCODER")
    _describe(src / "decoder_joint.onnx", "DECODER_JOINT")


def _to_fp16_uniform(model) -> None:
    """In-place uniform fp32 -> fp16 conversion we fully control.

    The onnxconverter_common converters all mishandle this Conformer's
    length/shape-arithmetic Cast nodes (retyping Cast outputs to fp16 while
    leaving `to`=FLOAT) and bloat the saved file. Instead we convert *everything*
    float to fp16 — initializers, Constant-node tensors, Cast `to` attributes, and
    value_info / graph I/O — so the graph is uniformly fp16 with no mixed-type
    nodes. The browser feeds fp16 (f16.js) and casts `encoded` back to fp32 for the
    fp32 decoder_joint. Integer/length math stays integer; the few length-math
    Casts now target fp16, exact for our small (<=65) frame counts.
    """
    import numpy as np
    from onnx import TensorProto, numpy_helper

    F32, F16 = TensorProto.FLOAT, TensorProto.FLOAT16
    for init in model.graph.initializer:
        if init.data_type == F32:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in model.graph.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == F32:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16)))
        elif node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == F32:
                    a.i = F16
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.type.tensor_type.elem_type == F32:
            vi.type.tensor_type.elem_type = F16


def convert_encoder_fp16(src: Path) -> None:
    import onnx
    import onnxruntime as ort

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_onnx = OUT_DIR / "encoder_fp16.onnx"
    out_data = "encoder_fp16.onnx.data"

    print("\n[3/4] Converting encoder fp32 -> fp16 (uniform, in-house)…")
    # The HF cache stores encoder.onnx.data as a symlink to a blob, which onnx's
    # external-data loader rejects. Materialise real copies into a temp dir first.
    tmp = OUT_DIR / "_fp32tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    print("      materialising real fp32 files (resolving HF symlinks)…")
    shutil.copy2(src / "encoder.onnx", tmp / "encoder.onnx")
    shutil.copy2(src / "encoder.onnx.data", tmp / "encoder.onnx.data")

    print("      loading fp32 (~2.5 GB) + converting every float tensor to fp16…")
    model = onnx.load(str(tmp / "encoder.onnx"), load_external_data=True)
    _to_fp16_uniform(model)

    # onnx.save's external-data writer bloats this model to ~10 GB (stale offsets ->
    # huge gaps), so write the weights file ourselves: just the fp16 raw bytes,
    # back to back. Result == sum of weight sizes (~1.2 GB).
    from onnx import TensorProto

    print("      writing external fp16 weights (manual, gap-free)…")
    offset = 0
    with open(OUT_DIR / out_data, "wb") as f:
        for init in model.graph.initializer:
            if init.HasField("raw_data") and len(init.raw_data) >= 1024:
                raw = init.raw_data
                f.write(raw)
                init.data_location = TensorProto.EXTERNAL
                del init.external_data[:]
                for k, v in (("location", out_data), ("offset", str(offset)), ("length", str(len(raw)))):
                    e = init.external_data.add()
                    e.key, e.value = k, v
                init.ClearField("raw_data")
                offset += len(raw)
    onnx.save_model(model, str(out_onnx))  # graph + small inline tensors; refs our .data

    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    itypes = {i.name: i.type for i in sess.get_inputs()}
    otypes = {o.name: o.type for o in sess.get_outputs()}
    del sess
    print(f"      LOADS in ORT  (fp16 .data = {_human((OUT_DIR / out_data).stat().st_size)})")
    print(f"        inputs : {itypes}")
    print(f"        outputs: {otypes}")
    print("      => uniform fp16 I/O (JS feeds fp16 via f16.js; encoded cast to fp32 for the decoder)")

    shutil.rmtree(tmp, ignore_errors=True)  # drop the 2.3 GB fp32 working copy

    # decoder_joint stays fp32 (small; runs on WASM per-token).
    shutil.copy2(src / "decoder_joint.onnx", OUT_DIR / "decoder_joint.onnx")
    shutil.copy2(src / "config.json", OUT_DIR / "config.json")
    print(f"      copied decoder_joint.onnx ({_human((OUT_DIR / 'decoder_joint.onnx').stat().st_size)}) + config.json")


def extract_vocab(src: Path) -> None:
    import sentencepiece as spm

    print("\n[4/4] Extracting vocab.json from tokenizer.model…")
    sp = spm.SentencePieceProcessor(model_file=str(src / "tokenizer.model"))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    (OUT_DIR / "vocab.json").write_text(json.dumps(pieces, ensure_ascii=False), encoding="utf-8")
    cfg = json.loads((src / "config.json").read_text())
    print(f"      sentencepiece pieces: {len(pieces)}  (config vocab_size={cfg.get('vocab_size')}, "
          f"blank_id={cfg.get('blank_id')})")
    print(f"      sample pieces: {pieces[:5]} … unk={sp.id_to_piece(sp.unk_id())!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect-only", action="store_true", help="download + print ONNX I/O, then stop")
    args = ap.parse_args()

    src = download()
    inspect(src)
    if args.inspect_only:
        print("\n(inspect-only) done.")
        return
    convert_encoder_fp16(src)
    extract_vocab(src)
    print(f"\nDone. Assets in {OUT_DIR.relative_to(ROOT)}/:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name:28s} {_human(p.stat().st_size)}")


if __name__ == "__main__":
    main()
