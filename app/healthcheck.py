"""Readiness command for Docker and Coolify, without curl or application imports."""

import os
from urllib.request import urlopen


def main() -> int:
    try:
        port = int(os.environ.get("PORT", "8000"))
        if not 1 <= port <= 65535:
            return 1
        with urlopen(f"http://127.0.0.1:{port}/health/ready", timeout=3) as response:
            return 0 if response.status == 200 else 1
    except (OSError, ValueError):
        # Healthcheck logs must not include response bodies or environment data.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
