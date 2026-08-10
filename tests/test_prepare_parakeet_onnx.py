import shutil
from pathlib import Path

import pytest

import scripts.prepare_parakeet_onnx as prepare


def _write_source_assets(directory: Path) -> None:
    directory.mkdir()
    for name in prepare.ALL_FILES:
        (directory / name).write_bytes(("new-" + name).encode())


def test_parakeet_install_replaces_complete_assets_atomically(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source_assets(source)
    monkeypatch.setattr(prepare, "OUT_DIR", output)

    prepare.install(source)

    for name in prepare.ALL_FILES:
        assert (output / name).read_bytes() == ("new-" + name).encode()
        assert not (output / f"{name}.tmp").exists()


def test_parakeet_install_keeps_previous_asset_when_copy_fails(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source_assets(source)
    output.mkdir()
    destination = output / prepare.ENCODER
    destination.write_bytes(b"previous-complete-model")
    monkeypatch.setattr(prepare, "OUT_DIR", output)

    real_copyfile = shutil.copyfile

    def failing_copyfile(src, dst):
        if Path(src).name == prepare.ENCODER:
            Path(dst).write_bytes(b"partial")
            raise OSError("disk full")
        return real_copyfile(src, dst)

    monkeypatch.setattr(prepare.shutil, "copyfile", failing_copyfile)

    with pytest.raises(OSError, match="disk full"):
        prepare.install(source)

    assert destination.read_bytes() == b"previous-complete-model"
    assert not (output / f"{prepare.ENCODER}.tmp").exists()
