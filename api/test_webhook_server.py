"""
Tiny webhook receiver for manual testing of progress/result posts.

Run:
    uvicorn api.test_webhook_server:app --host 0.0.0.0 --port 55812 --reload

It logs query params and JSON body for any POST to /api/internal/job/ or /api/internal/job/result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request

app = FastAPI(title="Webhook Test Receiver", version="0.1.0")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _dump_request(request: Request) -> Dict[str, Any]:
    body: Optional[Any]
    try:
        body = await request.json()
    except Exception:
        body = None
    return {
        "method": request.method,
        "url": str(request.url),
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
        "json": body,
        "received_at": _timestamp(),
    }


@app.post("/api/internal/job/")
async def receive_progress(request: Request) -> Dict[str, Any]:
    payload = await _dump_request(request)
    print(f"[{payload['received_at']}] /job/ {payload}")  # stdout logging
    return {"status": "ok", "received": payload}


@app.post("/api/internal/job/result")
async def receive_result(request: Request) -> Dict[str, Any]:
    payload = await _dump_request(request)
    print(f"[{payload['received_at']}] /job/result {payload}")  # stdout logging
    return {"status": "ok", "received": payload}

