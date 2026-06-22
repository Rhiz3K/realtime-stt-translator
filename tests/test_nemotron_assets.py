from pathlib import Path
import logging
import subprocess

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

    with pytest.raises(RuntimeError, match="Nemotron model assets are missing") as exc_info:
        ensure_nemotron_assets(
            enabled_engines="nemotron",
            auto_prepare="false",
            model_dir=tmp_path / "models",
            run_prepare=lambda: None,
        )

    assert "mount or generate the model files" in str(exc_info.value)


def test_ensure_nemotron_assets_accepts_boolean_auto_prepare(tmp_path: Path):
    from app.nemotron_assets import ensure_nemotron_assets

    with pytest.raises(RuntimeError, match="Nemotron model assets are missing"):
        ensure_nemotron_assets(
            enabled_engines="nemotron",
            auto_prepare=False,
            model_dir=tmp_path / "models",
            run_prepare=lambda: None,
        )


def test_run_prepare_script_uses_configured_timeout(monkeypatch):
    import app.nemotron_assets as assets

    calls = []

    def fake_run(cmd, *, cwd, env, check, timeout):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env, "check": check, "timeout": timeout})

    monkeypatch.setenv("NEMOTRON_PREPARE_TIMEOUT_SECONDS", "123")
    monkeypatch.setattr(assets.subprocess, "run", fake_run)

    model_dir = Path("/tmp/nemotron-custom-models")
    assets._run_prepare_script(model_dir)

    assert calls == [
        {
            "cmd": [assets.sys.executable, str(assets.PREPARE_SCRIPT)],
            "cwd": str(assets.ROOT),
            "env": {
                **assets.os.environ,
                "NEMOTRON_MODEL_DIR": str(model_dir),
            },
            "check": True,
            "timeout": 123.0,
        }
    ]


def test_run_prepare_script_reports_timeout(tmp_path: Path, monkeypatch):
    import app.nemotron_assets as assets

    def fake_run(cmd, *, cwd, env, check, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setenv("NEMOTRON_PREPARE_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(assets.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exceeded timeout"):
        assets._run_prepare_script(tmp_path / "models")


def test_ensure_nemotron_assets_passes_model_dir_to_default_runner(tmp_path: Path, monkeypatch):
    import app.nemotron_assets as assets

    calls = []

    def fake_run(cmd, *, cwd, env, check, timeout):
        calls.append({"env_model_dir": env["NEMOTRON_MODEL_DIR"]})
        model_dir = Path(env["NEMOTRON_MODEL_DIR"])
        for name in assets.REQUIRED_MODEL_FILES:
            (model_dir / name).write_text("ok")

    monkeypatch.setattr(assets.subprocess, "run", fake_run)

    model_dir = tmp_path / "custom-models"
    assets.ensure_nemotron_assets(enabled_engines="nemotron", model_dir=model_dir)

    assert calls == [{"env_model_dir": str(model_dir)}]


def test_main_logs_runtime_errors_without_traceback(monkeypatch, caplog):
    import app.nemotron_assets as assets

    def fail():
        raise RuntimeError("expected startup failure")

    monkeypatch.setattr(assets, "ensure_nemotron_assets", fail)
    caplog.set_level(logging.ERROR)

    assert assets.main() == 1
    assert caplog.records[-1].message == "expected startup failure"
    assert caplog.records[-1].exc_info is None


def test_main_logs_unexpected_errors_with_traceback(monkeypatch, caplog):
    import app.nemotron_assets as assets

    def fail():
        raise ValueError("unexpected startup failure")

    monkeypatch.setattr(assets, "ensure_nemotron_assets", fail)
    caplog.set_level(logging.ERROR)

    assert assets.main() == 1
    assert caplog.records[-1].message == "Nemotron asset preparation failed"
    assert caplog.records[-1].exc_info is not None
