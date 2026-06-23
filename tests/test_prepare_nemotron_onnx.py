from scripts.prepare_nemotron_onnx import _plan_concat_split


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
