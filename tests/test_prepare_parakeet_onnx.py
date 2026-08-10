import re
import sys
import types
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


def test_parakeet_download_pins_the_model_revision(tmp_path, monkeypatch):
    """The auto-prepare path must not follow whatever HEAD upstream happens to be."""
    assert re.fullmatch(r"[0-9a-f]{40}", prepare.REVISION), prepare.REVISION

    source = tmp_path / "snapshot"
    _write_source_assets(source)
    seen = {}

    def fake_snapshot_download(repo_id, revision, allow_patterns):
        seen.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        return str(source)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    assert prepare.download([prepare.VOCAB]) == source
    assert seen["revision"] == prepare.REVISION
    assert seen["repo_id"] == prepare.REPO_ID


def test_parakeet_install_rejects_an_empty_downloaded_asset(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source_assets(source)
    (source / prepare.ENCODER).write_bytes(b"")
    output.mkdir()
    destination = output / prepare.ENCODER
    destination.write_bytes(b"previous-complete-model")
    monkeypatch.setattr(prepare, "OUT_DIR", output)

    with pytest.raises(RuntimeError, match="empty"):
        prepare.install(source)

    # The good copy survives a truncated download.
    assert destination.read_bytes() == b"previous-complete-model"
    assert not (output / f"{prepare.ENCODER}.tmp").exists()
