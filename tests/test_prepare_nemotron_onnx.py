import re
import sys
import types

import scripts.prepare_nemotron_onnx as prepare


_plan_concat_split = prepare._plan_concat_split


def test_plan_concat_split_keeps_all_concat_inputs_within_storage_buffer_limit_16():
    inputs = [f"in_{i}" for i in range(24)]

    nodes = _plan_concat_split(inputs, output_name="out", max_inputs=15, name_prefix="concat")

    assert nodes[-1].output == "out"
    assert max(len(node.inputs) for node in nodes) <= 15
    assert len(nodes) == 3


def test_plan_concat_split_keeps_all_concat_inputs_within_storage_buffer_limit_8():
    inputs = [f"in_{i}" for i in range(24)]

    nodes = _plan_concat_split(inputs, output_name="out", max_inputs=7, name_prefix="concat")

    assert nodes[-1].output == "out"
    assert max(len(node.inputs) for node in nodes) <= 7
    assert len(nodes) == 5


def test_zero_length_concat_variant_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "OUT_DIR", tmp_path)
    for _limit, filename in prepare.ENCODER_CONCAT_VARIANTS:
        (tmp_path / filename).write_text("ok")
    zero_length = prepare.ENCODER_CONCAT_VARIANTS[0][1]
    (tmp_path / zero_length).write_bytes(b"")

    assert prepare._missing_concat_variant_files() == [zero_length]


def test_nemotron_download_pins_the_model_revision(tmp_path, monkeypatch):
    """Auto-prepare runs on container start; it must fetch a fixed commit."""
    assert re.fullmatch(r"[0-9a-f]{40}", prepare.REVISION), prepare.REVISION

    snapshot = tmp_path / "snapshot"
    (snapshot / prepare.SUBDIR).mkdir(parents=True)
    for name in prepare.FILES:
        (snapshot / prepare.SUBDIR / name).write_bytes(b"x")
    seen = {}

    def fake_snapshot_download(repo_id, revision, allow_patterns):
        seen.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    assert prepare.download() == snapshot / prepare.SUBDIR
    assert seen["revision"] == prepare.REVISION
    assert seen["repo_id"] == prepare.REPO_ID


def test_nemotron_sweeps_temporaries_and_the_fp32_working_copy(tmp_path, monkeypatch):
    output = tmp_path / "models"
    output.mkdir()
    stale_tmp = output / "encoder_fp16.onnx.data.tmp"
    stale_tmp.write_bytes(b"partial")
    workdir = output / "_fp32tmp"
    workdir.mkdir()
    (workdir / "encoder.onnx").write_bytes(b"2.3 GB stand-in")
    keeper = output / "config.json"
    keeper.write_bytes(b"{}")
    monkeypatch.setattr(prepare, "OUT_DIR", output)

    prepare._sweep_stale_temporaries()

    assert not stale_tmp.exists()
    assert not workdir.exists()
    assert keeper.read_bytes() == b"{}"
