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
