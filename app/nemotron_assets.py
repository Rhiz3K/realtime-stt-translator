from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "app" / "static" / "nemotron" / "models"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_nemotron_onnx.py"

REQUIRED_MODEL_FILES = (
    "encoder_fp16.onnx",
    "encoder_fp16.onnx.data",
    "decoder_joint.onnx",
    "config.json",
    "vocab.json",
)


def _split_engines(enabled_engines: str | Iterable[str] | None) -> list[str]:
    if enabled_engines is None:
        enabled_engines = os.getenv("ENABLED_ENGINES", "webspeech")
    if isinstance(enabled_engines, str):
        raw_values = enabled_engines.split(",")
    else:
        raw_values = enabled_engines
    return [engine.strip().lower() for engine in raw_values if engine and engine.strip()]


def _truthy_or_default_true(value: str | None) -> bool:
    if value is None:
        value = os.getenv("NEMOTRON_AUTO_PREPARE", "true")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def nemotron_enabled(enabled_engines: str | Iterable[str] | None = None) -> bool:
    return "nemotron" in _split_engines(enabled_engines)


def missing_model_files(model_dir: Path = DEFAULT_MODEL_DIR) -> list[str]:
    return [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]


def _run_prepare_script() -> None:
    subprocess.run([sys.executable, str(PREPARE_SCRIPT)], cwd=str(ROOT), check=True)


def ensure_nemotron_assets(
    *,
    enabled_engines: str | Iterable[str] | None = None,
    auto_prepare: str | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    run_prepare: Callable[[], None] | None = None,
) -> None:
    if not nemotron_enabled(enabled_engines):
        logging.info("Nemotron is not enabled; skipping model asset preparation.")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_model_files(model_dir)
    if not missing:
        logging.info("Nemotron model assets are present in %s.", model_dir)
        return

    if not _truthy_or_default_true(auto_prepare):
        raise RuntimeError(
            "Nemotron model assets are missing in "
            f"{model_dir}: {', '.join(missing)}. "
            "Set NEMOTRON_AUTO_PREPARE=true or mount/generated the model files."
        )

    logging.warning(
        "Nemotron is enabled but model assets are missing in %s: %s. "
        "Preparing them now; this downloads ~2.6 GB and writes ~1.3 GB.",
        model_dir,
        ", ".join(missing),
    )
    (run_prepare or _run_prepare_script)()

    missing = missing_model_files(model_dir)
    if missing:
        raise RuntimeError(
            "Nemotron model asset preparation finished but files are still missing in "
            f"{model_dir}: {', '.join(missing)}"
        )
    logging.info("Nemotron model assets prepared in %s.", model_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        ensure_nemotron_assets()
    except Exception as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
