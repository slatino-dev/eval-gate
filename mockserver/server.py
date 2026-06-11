"""
Minimal in-repo OpenAI-compatible mock server.

Returns a canned /v1/chat/completions response — no real model needed.
Run:  python -m mockserver.server [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

CANNED_RESPONSE: dict = {
    "id": "chatcmpl-mock-0001",
    "object": "chat.completion",
    "created": 0,  # filled at runtime
    "model": "mock-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a canned mock response from evalgate's in-repo test server.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 18,
        "total_tokens": 28,
    },
}


class MockHandler(BaseHTTPRequestHandler):
    """Handle POST /v1/chat/completions and GET /health."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Quiet by default; set verbose=True on the server to enable.
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"}, 200)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)  # consume body, ignore for mock
            payload = dict(CANNED_RESPONSE)
            payload["id"] = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
            payload["created"] = int(time.time())
            self._send_json(payload, 200)
        else:
            self._send_json({"error": "not found"}, 404)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(port: int = 8765, verbose: bool = False) -> None:
    server = HTTPServer(("127.0.0.1", port), MockHandler)
    server.verbose = verbose  # type: ignore[attr-defined]
    print(f"Mock server listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evalgate mock OpenAI server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(port=args.port, verbose=args.verbose)
