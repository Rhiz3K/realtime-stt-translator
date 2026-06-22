from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_does_not_block_uvicorn_on_nemotron_prepare():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "--start-period=30m" not in dockerfile
    assert "python -m app.nemotron_assets && exec uvicorn" not in dockerfile
    assert "python -m app.nemotron_assets & exec uvicorn" in dockerfile


def test_dockerignore_excludes_generated_nemotron_models():
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()

    assert "app/static/nemotron/models/" in dockerignore
