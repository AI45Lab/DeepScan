"""Minimal FastAPI server to trigger LLM-Diagnose evaluations for testing."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_diagnose import ConfigLoader, run_from_config

DEFAULT_CONFIG_PATH = Path(
    os.getenv(
        "DIAGNOSE_CONFIG",
        Path(__file__).resolve().parents[1] / "examples" / "config.xboundary.tellme-qwen2.5-7b-instruct.yaml",
    )
)
DEFAULT_WEBHOOK_CONFIG_PATH = Path(
    os.getenv("DIAGNOSE_WEBHOOK_CONFIG", Path(__file__).resolve().parent / "webhook.yaml")
)
DEFAULT_DRY_RUN = os.getenv("DIAGNOSE_DRY_RUN", "false").lower() == "true"
DEFAULT_OUTPUT_DIR = Path(os.getenv("DIAGNOSE_OUTPUT_DIR", "results"))
API_TOKEN = os.getenv("API_KEY", "iM1b1sxY8yCYCACqA7lvHEdh1XjpKgS4")


class EvaluationState(BaseModel):
    run_id: str
    status: str  # pending | running | completed | failed
    diagnosis: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class InferenceParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = Field(None, alias="top-p")
    top_k: Optional[int] = Field(None, alias="top-k")
    repetition_penalty: Optional[float] = Field(None, alias="repetition-penalty")

    class Config:
        allow_population_by_field_name = True

    @property
    def resolved_repetition_penalty(self) -> Optional[float]:
        return self.repetition_penalty


class DiagnosisItem(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class EvaluationCreateRequest(BaseModel):
    """
    Request body for creating an evaluation.

    Clients send `job_id`; we map it to internal `run_id`. For backward
    compatibility we still honor `run_id` if provided.
    """

    job_id: Optional[str] = None
    run_id: Optional[str] = None
    diagnosis: Optional[Union[str, List[DiagnosisItem]]] = None
    model: Optional[str] = None
    inference_parameters: Optional[InferenceParameters] = Field(None, alias="inference-parameters")

    def resolved_run_id(self, query_job_id: Optional[str] = None) -> str:
        # Prefer job_id from query, then body.job_id, then run_id, else empty string.
        return (query_job_id or self.job_id or self.run_id or "").strip()

    def diagnosis_label(self) -> Optional[str]:
        return self.diagnosis if isinstance(self.diagnosis, str) else None

    def diagnosis_items(self) -> List[DiagnosisItem]:
        if isinstance(self.diagnosis, list):
            items: List[DiagnosisItem] = []
            for entry in self.diagnosis:
                if isinstance(entry, DiagnosisItem):
                    items.append(entry)
                else:
                    try:
                        items.append(DiagnosisItem.parse_obj(entry))
                    except Exception:
                        continue
            return items
        return []

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


app = FastAPI(title="LLM-Diagnose Test API", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

security = HTTPBearer(auto_error=True)
_runs: Dict[str, EvaluationState] = {}
_run_tasks: Dict[str, asyncio.Task] = {}
_jobs_lock = asyncio.Lock()
_config_loader: Optional[ConfigLoader] = None
_run_configs: Dict[str, ConfigLoader] = {}


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


def _deepcopy_config(cfg: ConfigLoader) -> Dict[str, Any]:
    # Avoid mutating cached ConfigLoader; serialize for a deep copy of nested structures.
    return json.loads(json.dumps(cfg.to_dict()))


def _normalize_model_tokens(model_id: Optional[str]) -> Dict[str, Optional[str]]:
    if not model_id:
        return {"generation": None, "model_name": None}
    cleaned = str(model_id).strip()
    lowered = cleaned.lower()
    generation: Optional[str] = None
    if lowered.startswith("qwen2.5"):
        generation = "qwen2.5"
    elif lowered.startswith("qwen2"):
        generation = "qwen2"
    elif lowered.startswith("qwen3"):
        generation = "qwen3"

    def _pretty_name(raw: str) -> str:
        tokens = raw.replace("_", "-").split("-")
        pretty: List[str] = []
        for tok in tokens:
            if not tok:
                continue
            if tok[-1].lower() == "b" and tok[:-1].isdigit():
                pretty.append(f"{tok[:-1]}B")
                continue
            pretty.append(tok.capitalize())
        return "-".join(pretty)

    model_name = _pretty_name(cleaned)
    return {"generation": generation, "model_name": model_name}


def _apply_model_override(
    base_model_cfg: Dict[str, Any], model_id: Optional[str], params: Optional[InferenceParameters]
) -> Dict[str, Any]:
    model_cfg = copy.deepcopy(base_model_cfg or {})
    tokens = _normalize_model_tokens(model_id)
    if tokens["generation"]:
        model_cfg["generation"] = tokens["generation"]
    if tokens["model_name"]:
        model_cfg["model_name"] = tokens["model_name"]

    if params:
        gen_cfg = dict(model_cfg.get("generation_config") or {})
        if params.temperature is not None:
            gen_cfg["temperature"] = params.temperature
        if params.resolved_repetition_penalty is not None:
            gen_cfg["repetition_penalty"] = params.resolved_repetition_penalty
        if params.top_p is not None:
            gen_cfg["top_p"] = params.top_p
        if params.top_k is not None:
            gen_cfg["top_k"] = params.top_k
        if gen_cfg:
            model_cfg["generation_config"] = gen_cfg
    return model_cfg


def _match_base_evaluator(base_evaluators: List[Dict[str, Any]], normalized_name: str) -> Dict[str, Any]:
    for candidate in base_evaluators:
        cand_type = str(candidate.get("type") or "").replace("_", "-").lower()
        cand_run = str(candidate.get("run_name") or "").replace("_", "-").lower()
        if normalized_name in {cand_type, cand_run} or normalized_name.replace("-", "") == cand_type.replace("-", ""):
            return copy.deepcopy(candidate)
    return copy.deepcopy(base_evaluators[0]) if base_evaluators else {}


def _normalize_eval_name(name: str) -> str:
    lowered = str(name or "").strip().replace("_", "-").lower()
    if "xboundary" in lowered or "x-boundary" in lowered or "x_boundary" in lowered:
        return "xboundary"
    return lowered


def _apply_diagnosis_overrides(base_cfg: Dict[str, Any], items: List[DiagnosisItem]) -> List[Dict[str, Any]]:
    base_evaluators = base_cfg.get("evaluators") or []
    root_dataset = base_cfg.get("dataset") or {}
    overrides: List[Dict[str, Any]] = []
    allowed = {"xboundary", "x-boundary", "tellme", "spin"}

    for item in items:
        normalized = _normalize_eval_name(item.name)
        # Temporarily ignore unsupported evaluators (e.g., spin, mi-peaks).
        if normalized not in allowed:
            continue
        base_eval = _match_base_evaluator(base_evaluators, normalized)
        eval_cfg: Dict[str, Any] = base_eval or {}
        eval_cfg["type"] = "xboundary" if "xboundary" in normalized else normalized
        eval_cfg.setdefault("run_name", eval_cfg.get("run_name") or eval_cfg["type"])
        if eval_cfg["type"] == "spin" and str(base_eval.get("type") or "").lower() != "spin":
            # Avoid inheriting a mismatched run_name when the base evaluator isn't SPIN.
            eval_cfg["run_name"] = "spin"

        args = item.args or {}
        target_layers = args.get("target-layers") or args.get("target_layers")
        samples = args.get("samples")
        dataset_cfg = copy.deepcopy(eval_cfg.get("dataset") or root_dataset or {})

        if eval_cfg["type"] == "xboundary":
            if target_layers is not None:
                eval_cfg["target_layers"] = target_layers
            if samples is not None:
                dataset_cfg["num_samples_per_class"] = samples
            if dataset_cfg:
                eval_cfg["dataset"] = dataset_cfg
        elif eval_cfg["type"] == "tellme":
            if args.get("batch_size") is not None:
                eval_cfg["batch_size"] = args.get("batch_size")
            if args.get("layer_ratio") is not None:
                eval_cfg["layer_ratio"] = args.get("layer_ratio")
            if target_layers:
                try:
                    eval_cfg["layer"] = int(target_layers[0])
                except Exception:
                    pass
            if samples is not None:
                dataset_cfg["max_rows"] = samples
            if dataset_cfg:
                eval_cfg["dataset"] = dataset_cfg
        elif eval_cfg["type"] == "spin":
            # Map common SPIN knobs
            if args.get("nsamples") is not None:
                eval_cfg["nsamples"] = args.get("nsamples")
            elif samples is not None:
                eval_cfg["nsamples"] = samples
            if args.get("seed") is not None:
                eval_cfg["seed"] = args.get("seed")
            if args.get("q") is not None:
                eval_cfg["q"] = args.get("q")
            if args.get("p") is not None:
                eval_cfg["p"] = args.get("p")
            if args.get("target_module") is not None:
                eval_cfg["target_module"] = args.get("target_module")
            if dataset_cfg:
                eval_cfg["dataset"] = dataset_cfg
        else:
            if dataset_cfg:
                eval_cfg["dataset"] = dataset_cfg

        overrides.append(eval_cfg)

    # If all requested evaluators were filtered out, fall back to the base evaluators.
    return overrides if overrides else base_evaluators


def _build_config_for_request(base_cfg: ConfigLoader, body: EvaluationCreateRequest) -> ConfigLoader:
    cfg_dict = _deepcopy_config(base_cfg)
    if body.model or body.inference_parameters:
        cfg_dict["model"] = _apply_model_override(cfg_dict.get("model") or {}, body.model, body.inference_parameters)

    diag_items = body.diagnosis_items()
    if diag_items:
        cfg_dict["evaluators"] = _apply_diagnosis_overrides(cfg_dict, diag_items)

    return ConfigLoader.from_dict(cfg_dict)


async def _execute_run(run_id: str) -> None:
    async with _jobs_lock:
        state = _runs[run_id]
        cfg_for_run = _run_configs.get(run_id)
        state.status = "running"
        state.started_at = datetime.now(timezone.utc)

    try:
        cfg = cfg_for_run or _get_config_loader()
        webhook_cfg_path: Optional[str] = None
        if DEFAULT_WEBHOOK_CONFIG_PATH.exists():
            webhook_cfg_path = str(DEFAULT_WEBHOOK_CONFIG_PATH)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                run_from_config,
                cfg,
                dry_run=DEFAULT_DRY_RUN,
                output_dir=str(DEFAULT_OUTPUT_DIR),
                run_id=run_id,
                webhook_config=webhook_cfg_path,
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
    finally:
        async with _jobs_lock:
            _run_configs.pop(run_id, None)


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
    body: EvaluationCreateRequest,
    job_id: Optional[str] = Query(None, alias="jobId"),
    _: None = Depends(_require_api_key),
) -> EvaluationState:
    """
    Create a new evaluation run. Always returns immediately (202 Accepted).

    `jobId` is expected as a query parameter (?jobId=...). The request body can
    carry an optional diagnosis and a legacy run_id for backward compatibility.
    """
    return await _create_run(body, job_id)


@app.post("/v1/diagnosis/", status_code=status.HTTP_202_ACCEPTED, response_model=EvaluationState)
async def create_diagnosis(
    body: EvaluationCreateRequest,
    job_id: Optional[str] = Query(None, alias="jobId"),
    _: None = Depends(_require_api_key),
) -> EvaluationState:
    """
    Preferred endpoint for launching diagnosis jobs.

    Example payload:
    {
        "model": "qwen2.5-7b-instruct",
        "inference-parameters": {
            "temperature": 0,
            "repitition-penalty": 1,
            "top-p": 0.9,
            "top-k": 50
        },
        "diagnosis": [
            {"name": "x-boundary", "args": {"target-layers": [10, 19], "samples": 100}},
            {"name": "tellme", "args": {"target-layers": [10, 19], "samples": 100}}
        ]
    }
    """
    return await _create_run(body, job_id)


async def _create_run(body: EvaluationCreateRequest, job_id: Optional[str]) -> EvaluationState:
    # Prefer query param `jobId`; fall back to body fields for backward compatibility.
    requested_run_id = body.resolved_run_id(job_id)
    run_id = requested_run_id or str(uuid.uuid4())

    async with _jobs_lock:
        if run_id in _runs:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Run id already exists",
            )

    state = EvaluationState(run_id=run_id, status="pending", diagnosis=body.diagnosis_label())
    cfg_for_run = _build_config_for_request(_get_config_loader(), body)

    async with _jobs_lock:
        _runs[run_id] = state
        _run_configs[run_id] = cfg_for_run
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

