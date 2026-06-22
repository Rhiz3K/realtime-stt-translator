#!/usr/bin/env python3
"""Tiny static server for the Nemotron browser PoC (scripts/nemotron_poc.html).

Serves the repo root with the right MIME types for ES modules / wasm / onnx so
onnxruntime-web can load. WebGPU does not require cross-origin isolation and the
PoC runs the decoder single-threaded, so no COOP/COEP is set (which also avoids
COEP blocking the CDN-hosted onnxruntime-web import).

    python scripts/serve_poc.py            # -> http://127.0.0.1:8077/scripts/nemotron_poc.html
"""
import http.server
import socketserver
from pathlib import Path

PORT = 8077
ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".data": "application/octet-stream",
        ".json": "application/json",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass  # quiet


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"serving {ROOT} on http://127.0.0.1:{PORT}/")
        print(f"PoC: http://127.0.0.1:{PORT}/scripts/nemotron_poc.html")
        httpd.serve_forever()
