from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import pytest

from app import healthcheck


@pytest.mark.parametrize("port", [None, "9000"])
@pytest.mark.parametrize("status", [200, 503])
def test_readiness_uses_runtime_port_and_bounded_request(monkeypatch, port, status):
    if port is None:
        monkeypatch.delenv("PORT", raising=False)
    else:
        monkeypatch.setenv("PORT", port)
    response = MagicMock()
    response.__enter__.return_value.status = status
    request = MagicMock(return_value=response)
    monkeypatch.setattr(healthcheck, "urlopen", request)

    assert healthcheck.main() == (0 if status == 200 else 1)
    request.assert_called_once_with(
        f"http://127.0.0.1:{port or '8000'}/health/ready", timeout=3
    )
    response.__exit__.assert_called_once()


@pytest.mark.parametrize("port", ["", "invalid", "0", "65536"])
def test_invalid_port_never_opens_a_connection(monkeypatch, port):
    monkeypatch.setenv("PORT", port)
    request = MagicMock()
    monkeypatch.setattr(healthcheck, "urlopen", request)

    assert healthcheck.main() == 1
    request.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("private response"),
        URLError("private response"),
        HTTPError("http://127.0.0.1", 503, "private response", {}, None),
    ],
)
def test_failure_is_nonzero_and_does_not_log_payload(monkeypatch, capsys, error):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(healthcheck, "urlopen", MagicMock(side_effect=error))

    assert healthcheck.main() == 1
    output = capsys.readouterr()
    assert not output.out and not output.err
