"""
Minimal REST server to receive progress callbacks for local testing.

Run:
    pip install flask
    python utils/mock_progress_server.py

POST to:
    http://localhost:8000/progress
with JSON:
    {"run_id": "run-123", "status": "running", "progress": 42}
"""

from __future__ import annotations

import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("progress_server")


@app.route("/progress", methods=["POST"])
def progress() -> tuple:
    payload = request.get_json(force=True, silent=True) or {}
    run_id = payload.get("run_id")
    status = payload.get("status")
    progress_value = payload.get("progress")
    logger.info(
        "[progress] run_id=%s status=%s progress=%s payload=%s",
        run_id,
        status,
        progress_value,
        payload,
    )
    return jsonify({"ok": True, "received": payload}), 200


@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Mock progress server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True)

