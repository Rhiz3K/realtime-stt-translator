from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_is_small_runtime_only_image():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "requirements.txt" in dockerfile
    assert "requirements-nemotron-prep" not in dockerfile
    assert "requirements-parakeet-prep" not in dockerfile
    assert "prepare_" not in dockerfile
    assert "app.nemotron_assets" not in dockerfile
    assert "/health/ready" in dockerfile
    assert "USER appuser" in dockerfile
    assert "uvicorn app.main:app" in dockerfile


def test_dockerignore_excludes_secrets_and_development_state():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored
    assert ".git/" in ignored
    assert ".venv/" in ignored
    assert "tests/" in ignored
    assert not any("nemotron" in line or "parakeet" in line for line in ignored)
