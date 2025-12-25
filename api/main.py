"""Minimal FastAPI server to trigger LLM-Diagnose evaluations for testing."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from llm_diagnose import ConfigLoader, run_from_config

DEFAULT_CONFIG_PATH = Path(
    os.getenv(
        "DIAGNOSE_CONFIG",
        Path(__file__).resolve().parents[1] / "examples" / "config.xboundary-qwen2.5-7b-instruct.yaml",
    )
)
DEFAULT_DRY_RUN = os.getenv("DIAGNOSE_DRY_RUN", "false").lower() == "true"
DEFAULT_OUTPUT_DIR = Path(os.getenv("DIAGNOSE_OUTPUT_DIR", "results"))
API_TOKEN = os.getenv("API_KEY", "iM1b1sxY8yCYCACqA7lvHEdh1XjpKgS4")


class EvaluationState(BaseModel):
    run_id: str
    status: str  # pending | running | completed | failed
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


app = FastAPI(title="LLM-Diagnose Test API", version="0.1.0")

security = HTTPBearer(auto_error=True)
_runs: Dict[str, EvaluationState] = {}
_run_tasks: Dict[str, asyncio.Task] = {}
_jobs_lock = asyncio.Lock()
_config_loader: Optional[ConfigLoader] = None


def _get_config_loader() -> ConfigLoader:
    global _config_loader
    if _config_loader is None:
        if not DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Config file not found at {DEFAULT_CONFIG_PATH}. "
                "Set DIAGNOSE_CONFIG to point at a valid YAML/JSON config."
            )
        _config_loader = ConfigLoader.from_file(str(DEFAULT_CONFIG_PATH))
    return _config_loader


async def _execute_run(run_id: str) -> None:
    async with _jobs_lock:
        state = _runs[run_id]
        state.status = "running"
        state.started_at = datetime.now(timezone.utc)

    try:
        cfg = _get_config_loader()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                run_from_config,
                cfg,
                dry_run=DEFAULT_DRY_RUN,
                output_dir=str(DEFAULT_OUTPUT_DIR),
                run_id=run_id,
            ),
        )

        payload: Dict[str, Any] = result or {"message": "dry run finished", "dry_run": True}

        async with _jobs_lock:
            state = _runs[run_id]
            state.result = payload
            state.status = "completed"
            state.finished_at = datetime.now(timezone.utc)
    except Exception as exc:  # pragma: no cover - defensive guard
        async with _jobs_lock:
            state = _runs[run_id]
            state.error = str(exc)
            state.status = "failed"
            state.finished_at = datetime.now(timezone.utc)


async def _wait_for_run(run_id: str) -> EvaluationState:
    # Await the task if it is still running; no-op otherwise.
    async with _jobs_lock:
        task = _run_tasks.get(run_id)
    if task:
        await asyncio.shield(task)
    async with _jobs_lock:
        state = _runs.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return state


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def _require_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    if credentials is None or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")


@app.post("/evaluations", status_code=status.HTTP_202_ACCEPTED, response_model=EvaluationState)
async def create_evaluation(
    _: None = Depends(_require_api_key),
) -> EvaluationState:
    """
    Create a new evaluation run. Always returns immediately (202 Accepted).
    """
    run_id = str(uuid.uuid4())
    state = EvaluationState(run_id=run_id, status="pending")

    async with _jobs_lock:
        _runs[run_id] = state
        _run_tasks[run_id] = asyncio.create_task(_execute_run(run_id))

    return state


@app.get("/evaluations/{run_id}", response_model=EvaluationState)
async def get_evaluation(
    run_id: str,
    _: None = Depends(_require_api_key),
) -> EvaluationState:
    async with _jobs_lock:
        state = _runs.get(run_id)
    if state is None:
        # Fallback: try to read from disk if API restarted after completion
        data = _load_run_from_disk(run_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return EvaluationState(
            run_id=run_id,
            status="completed",
            started_at=None,
            finished_at=None,
            result=data,
            error=None,
        )

    if state.status in {"pending", "running"}:
        state = await _wait_for_run(run_id)
    return state


@app.get("/evaluations")
async def list_evaluations(
    _: None = Depends(_require_api_key),
) -> Dict[str, List[str]]:
    async with _jobs_lock:
        return {"run_ids": list(_runs.keys())}


def _load_run_from_disk(run_id: str) -> Optional[Dict[str, Any]]:
    run_dir = DEFAULT_OUTPUT_DIR / run_id
    results_path = run_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@app.get("/runs/{run_id}", response_model=EvaluationState)
async def get_run(
    run_id: str,
    _: None = Depends(_require_api_key),
) -> EvaluationState:
    # Back-compat alias for evaluations/{run_id}
    return await get_evaluation(run_id)

