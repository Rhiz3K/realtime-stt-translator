from pathlib import Path

import pytest


def test_ensure_nemotron_assets_skips_when_engine_disabled(tmp_path: Path):
    from app.nemotron_assets import ensure_nemotron_assets

    calls = []

    ensure_nemotron_assets(
        enabled_engines="webspeech,deepgram",
        model_dir=tmp_path / "models",
        run_prepare=lambda: calls.append("prepare"),
    )

    assert calls == []
    assert not (tmp_path / "models").exists()


def test_ensure_nemotron_assets_prepares_missing_files_when_enabled(tmp_path: Path):
    from app.nemotron_assets import REQUIRED_MODEL_FILES, ensure_nemotron_assets

    model_dir = tmp_path / "models"
    calls = []

    def fake_prepare():
        calls.append("prepare")
        for name in REQUIRED_MODEL_FILES:
            (model_dir / name).write_text("ok")

    ensure_nemotron_assets(
        enabled_engines="webspeech,nemotron",
        model_dir=model_dir,
        run_prepare=fake_prepare,
    )

    assert calls == ["prepare"]
    assert model_dir.is_dir()


def test_ensure_nemotron_assets_does_not_prepare_when_files_exist(tmp_path: Path):
    from app.nemotron_assets import REQUIRED_MODEL_FILES, ensure_nemotron_assets

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (model_dir / name).write_text("ok")

    calls = []

    ensure_nemotron_assets(
        enabled_engines="webspeech,nemotron",
        model_dir=model_dir,
        run_prepare=lambda: calls.append("prepare"),
    )

    assert calls == []


def test_ensure_nemotron_assets_fails_when_auto_prepare_disabled(tmp_path: Path):
    from app.nemotron_assets import ensure_nemotron_assets

    with pytest.raises(RuntimeError, match="Nemotron model assets are missing"):
        ensure_nemotron_assets(
            enabled_engines="nemotron",
            auto_prepare="false",
            model_dir=tmp_path / "models",
            run_prepare=lambda: None,
        )
