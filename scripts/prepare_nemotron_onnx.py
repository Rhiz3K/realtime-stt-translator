#!/usr/bin/env python3
"""Prepare Nemotron-3.5-ASR-streaming ONNX assets for the browser engine.

Downloads the community ONNX export (encoder / decoder_joint / tokenizer) from
``altunenes/parakeet-rs``, prints the *exact* ONNX I/O contract (so we can verify
it against the JS engine's assumptions), converts the ~2.45 GB fp32 encoder to
fp16 (~1.2 GB — fits the browser's ~2 GB ArrayBuffer limit and halves WebGPU VRAM,
while keeping fp32 model I/O via ``keep_io_types`` so the JS side stays simple),
copies the small fp32 decoder_joint as-is, writes small Concat-limited encoder
graph variants for lower WebGPU storage-buffer limits, and extracts a flat
``vocab.json`` (id -> SentencePiece piece) for detokenisation.

Dev-only. Outputs land in ``app/static/nemotron/models/`` (gitignored).

    python scripts/prepare_nemotron_onnx.py            # download + introspect + convert + vocab
    python scripts/prepare_nemotron_onnx.py --inspect-only
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "altunenes/parakeet-rs"
SUBDIR = "nemotron-3.5-asr-streaming-0.6b-onnx"
FILES = ["encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model", "config.json"]
ENCODER_BASE = "encoder_fp16.onnx"
ENCODER_DATA = "encoder_fp16.onnx.data"
ENCODER_CONCAT_VARIANTS = (
    (16, "encoder_fp16_concat16.onnx"),
    (8, "encoder_fp16_concat8.onnx"),
)

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


@dataclass(frozen=True)
class ConcatSplitPlanNode:
    inputs: list[str]
    output: str


def _plan_concat_split(inputs: list[str], *, output_name: str, max_inputs: int, name_prefix: str) -> list[ConcatSplitPlanNode]:
    """Plan a tree of Concat nodes where no node has more than max_inputs.

    WebGPU storage-buffer limits count Concat inputs plus the output buffer in
    onnxruntime-web's generated shader. A device limit of 16 therefore needs
    Concat nodes with at most 15 inputs; limit 8 needs at most 7 inputs.
    """
    if max_inputs < 2:
        raise ValueError("max_inputs must be at least 2")

    pending = list(inputs)
    nodes: list[ConcatSplitPlanNode] = []
    level = 0
    while len(pending) > max_inputs:
        next_pending: list[str] = []
        for chunk_index, start in enumerate(range(0, len(pending), max_inputs)):
            chunk = pending[start:start + max_inputs]
            if len(chunk) == 1:
                next_pending.append(chunk[0])
                continue
            out = f"{output_name}__{name_prefix}_part_{level}_{chunk_index}"
            nodes.append(ConcatSplitPlanNode(inputs=chunk, output=out))
            next_pending.append(out)
        pending = next_pending
        level += 1

    nodes.append(ConcatSplitPlanNode(inputs=pending, output=output_name))
    return nodes


def _make_concat_node_like(source_node, inputs: list[str], outputs: list[str], name: str):
    from onnx import helper

    node = helper.make_node(
        source_node.op_type,
        inputs,
        outputs,
        name=name,
        domain=source_node.domain,
    )
    node.attribute.extend(source_node.attribute)
    return node


def _split_large_concat_nodes(model, *, max_storage_buffers: int) -> int:
    max_inputs = max_storage_buffers - 1
    rewritten = 0
    nodes = []

    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Concat" or len(node.input) <= max_inputs:
            nodes.append(node)
            continue

        output_name = node.output[0]
        prefix = f"concat{max_storage_buffers}_{node_index}"
        plan = _plan_concat_split(
            list(node.input),
            output_name=output_name,
            max_inputs=max_inputs,
            name_prefix=prefix,
        )
        for plan_index, planned in enumerate(plan):
            is_final = plan_index == len(plan) - 1
            name_base = node.name or output_name
            name = name_base if is_final else f"{name_base}__{prefix}_{plan_index}"
            nodes.append(_make_concat_node_like(node, planned.inputs, [planned.output], name))
        rewritten += 1

    if rewritten:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return rewritten


def _write_concat_limited_encoder_variants(model) -> None:
    import onnx

    for max_storage_buffers, filename in ENCODER_CONCAT_VARIANTS:
        variant = copy.deepcopy(model)
        rewritten = _split_large_concat_nodes(variant, max_storage_buffers=max_storage_buffers)
        onnx.save_model(variant, str(OUT_DIR / filename))
        print(
            f"      wrote {filename} for WebGPU storage-buffer limit {max_storage_buffers} "
            f"({rewritten} Concat node(s) split)"
        )


def _base_browser_assets_exist() -> bool:
    return all((OUT_DIR / name).is_file() for name in (ENCODER_BASE, ENCODER_DATA, "decoder_joint.onnx", "config.json", "vocab.json"))


def _missing_concat_variant_files() -> list[str]:
    return [filename for _limit, filename in ENCODER_CONCAT_VARIANTS if not (OUT_DIR / filename).is_file()]


def _write_missing_concat_variants_from_existing() -> bool:
    missing = _missing_concat_variant_files()
    if not missing or not _base_browser_assets_exist():
        return False

    import onnx

    print("\n[preflight] Existing fp16 assets found; generating missing WebGPU Concat variants…")
    model = onnx.load(str(OUT_DIR / ENCODER_BASE), load_external_data=False)
    _write_concat_limited_encoder_variants(model)
    return True


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
    out_onnx = OUT_DIR / ENCODER_BASE
    out_data = ENCODER_DATA

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
    print("      writing Concat-limited encoder graph variants for lower WebGPU limits…")
    _write_concat_limited_encoder_variants(model)

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

    if not args.inspect_only and _write_missing_concat_variants_from_existing():
        print(f"\nDone. Assets in {OUT_DIR.relative_to(ROOT)}/:")
        for p in sorted(OUT_DIR.iterdir()):
            print(f"  {p.name:28s} {_human(p.stat().st_size)}")
        return

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
