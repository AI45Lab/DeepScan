"""
End-to-end runner that executes a diagnosis pipeline from a config file/dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, Callable, List, Tuple
import math
import argparse
import json
import logging
import gc
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path

from llm_diagnose import ConfigLoader
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry
from llm_diagnose.evaluators.registry import get_evaluator_registry
from llm_diagnose.summarizers.registry import get_summarizer_registry
from llm_diagnose.utils.progress import infer_total_items
from llm_diagnose.utils.throughput import TokenThroughputTracker
from llm_diagnose.utils.model_introspection import get_num_hidden_layers

def _maybe_cuda_cleanup() -> None:
    """
    Best-effort memory cleanup between evaluators/models.

    Notes:
    - This does NOT unload model weights.
    - PyTorch may keep freed memory in its CUDA caching allocator; empty_cache()
      releases cached blocks back to the driver so other allocations can succeed.
    """
    try:
        import torch  # type: ignore

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Collect inter-process cached blocks (best-effort).
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        # torch not installed or CUDA not available
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _as_config_loader(config: Union[str, Dict[str, Any], ConfigLoader]) -> ConfigLoader:
    if isinstance(config, ConfigLoader):
        return config
    if isinstance(config, str):
        return ConfigLoader.from_file(config)
    if isinstance(config, dict):
        return ConfigLoader.from_dict(config)
    raise TypeError("config must be a path, dict, or ConfigLoader")


# Webhook classes removed for open-source version - progress callbacks are preserved


class _PipelineProgressAdapter:
    """
    Wrap a progress sink so that:
    - Model loading contributes a fixed portion of progress.
    - Evaluation samples consume the remaining portion.
    - Total progress always sums to 100.
    """

    def __init__(self, sink: Any, *, load_fraction: float = 0.1, total_units: int = 100):
        self._sink = sink
        self.total_units = int(total_units)
        self.load_units = int(round(self.total_units * max(0.0, min(load_fraction, 1.0))))
        self.pre_load_units = min(self.load_units, max(0, int(round(self.load_units * 0.5))))
        self.eval_units = max(0, self.total_units - self.load_units)
        self._eval_total: Optional[float] = None
        self._preload_emitted = False
        self._load_emitted = False

    def set_eval_total(self, total: Optional[int]) -> None:
        if isinstance(total, (int, float)) and total > 0:
            self._eval_total = float(total)

    def _scaled_completed(self, completed: int) -> Optional[int]:
        if self._eval_total is None or self._eval_total <= 0 or self.eval_units == 0:
            return None
        bounded = min(max(float(completed), 0.0), self._eval_total)
        if self._load_emitted:
            base = self.load_units
        elif self._preload_emitted:
            base = self.pre_load_units
        else:
            base = 0
        scaled = base + self.eval_units * (bounded / self._eval_total)
        return int(round(scaled))

    def _call_sink(self, method: str, *args: Any) -> None:
        if self._sink is None:
            return
        fn = getattr(self._sink, method, None)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            logging.debug("Progress sink call failed", exc_info=True)

    def mark_model_preload(self, desc: str) -> None:
        if self._preload_emitted or self.pre_load_units == 0:
            return
        self._preload_emitted = True
        self._call_sink("on_update", self.pre_load_units, self.total_units, desc)

    def mark_model_loaded(self, desc: str) -> None:
        if self.load_units == 0:
            return
        if not self._preload_emitted and self.pre_load_units > 0:
            self.mark_model_preload(desc)
        if self._load_emitted:
            return
        self._load_emitted = True
        self._call_sink("on_update", self.load_units, self.total_units, desc)

    def on_start(self, total: Optional[int], desc: str) -> None:
        self.set_eval_total(total)
        self._call_sink("on_start", self.total_units, desc)
        # If model load already advanced progress, re-emit it so sinks don't
        # reset to 0 when evaluation starts.
        if self._load_emitted and self.load_units > 0:
            self._call_sink("on_update", self.load_units, self.total_units, desc)

    def on_update(self, completed: int, total: Optional[int], desc: str) -> None:
        self.set_eval_total(total)
        scaled = self._scaled_completed(completed)
        if scaled is None:
            self._call_sink("on_update", completed, total, desc)
        else:
            self._call_sink("on_update", scaled, self.total_units, desc)

    def on_done(self, completed: int, total: Optional[int], desc: str) -> None:
        self._call_sink("on_done", self.total_units, self.total_units, desc)

    def on_result(self, result: Dict[str, Any]) -> None:
        self._call_sink("on_result", result)

    def on_throughput(self, metrics: Dict[str, Any]) -> None:
        self._call_sink("on_throughput", metrics)


class _PhaseProgressAdapter:
    """
    Wrap a sink so repeated evaluator phases contribute to a single monotonically
    increasing progress stream (useful for multiple evaluators in one run).
    """

    def __init__(
        self,
        sink: Any,
        *,
        phase_idx: int,
        n_phases: int,
        phase_total: Optional[int],
        phase_label: str,
        global_total: Optional[int] = None,
        phase_offset: Optional[int] = None,
        equalize_mode: bool = False,
    ):
        self._sink = sink
        self.phase_idx = max(0, int(phase_idx))
        self.n_phases = max(1, int(n_phases))
        self.phase_total = phase_total if isinstance(phase_total, int) and phase_total > 0 else None
        self.phase_label = phase_label or "phase"
        self.global_total_override = global_total if isinstance(global_total, int) and global_total > 0 else None
        self.phase_offset = phase_offset if isinstance(phase_offset, int) and phase_offset >= 0 else None
        self.equalize_mode = bool(equalize_mode)
        self._phase_total_seen: Optional[int] = None

    def _call(self, method: str, *args: Any) -> None:
        if self._sink is None:
            return
        fn = getattr(self._sink, method, None)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            logging.debug("Progress sink call failed", exc_info=True)

    def on_throughput(self, metrics: Dict[str, Any]) -> None:
        """Forward throughput updates without phase scaling."""
        self._call("on_throughput", metrics)

    def _effective_phase_total(self, total: Optional[int]) -> Optional[int]:
        if self.phase_total is not None:
            return self.phase_total
        if isinstance(total, int) and total > 0:
            return total
        return self._phase_total_seen

    def _global_total(self, total: Optional[int]) -> Optional[int]:
        if self.global_total_override is not None:
            return self.global_total_override
        phase_total = self._effective_phase_total(total)
        if phase_total is None:
            return None
        return int(phase_total) * int(self.n_phases)

    def _global_completed(self, completed: int, total: Optional[int]) -> Optional[int]:
        phase_total = self._effective_phase_total(total)
        if phase_total is None:
            return None
        bounded = min(max(int(completed), 0), int(phase_total))
        if self.equalize_mode and self.global_total_override is not None:
            frac = float(bounded) / float(phase_total) if phase_total else 0.0
            base = float(self.phase_offset or 0)
            return base + frac
        if self.phase_offset is not None:
            return self.phase_offset + bounded
        return int(self.phase_idx) * int(phase_total) + bounded

    def on_start(self, total: Optional[int], desc: str) -> None:
        if isinstance(total, int) and total > 0:
            self._phase_total_seen = int(total)
        global_total = self._global_total(total)
        # Only emit a real start once to avoid resetting sinks between phases.
        if self.phase_idx == 0:
            self._call("on_start", global_total, desc)
        else:
            global_completed = self._global_completed(0, total)
            if global_completed is not None:
                self._call("on_update", global_completed, global_total, desc)

    def on_update(self, completed: int, total: Optional[int], desc: str) -> None:
        if isinstance(total, int) and total > 0:
            self._phase_total_seen = int(total)
        global_total = self._global_total(total)
        global_completed = self._global_completed(completed, total)
        if global_completed is None:
            self._call("on_update", completed, total, desc)
        else:
            self._call("on_update", global_completed, global_total, desc)

    def on_done(self, completed: int, total: Optional[int], desc: str) -> None:
        if isinstance(total, int) and total > 0:
            self._phase_total_seen = int(total)
        global_total = self._global_total(total)
        # Mark this phase completed.
        global_completed = self._global_completed(self._effective_phase_total(total) or completed, total)
        if global_completed is not None:
            self._call("on_update", global_completed, global_total, desc)
        # Only emit "done" at the end of the final phase.
        if self.phase_idx == self.n_phases - 1:
            self._call("on_done", global_total or completed, global_total, desc)


def run_from_config(
    config: Union[str, Dict[str, Any], ConfigLoader],
    *,
    dry_run: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    progress_sink: Optional[Any] = None,
    on_progress_start: Optional[Callable[[Optional[int], str], None]] = None,
    on_progress_update: Optional[Callable[[int, Optional[int], str], None]] = None,
    on_progress_done: Optional[Callable[[int, Optional[int], str], None]] = None,
    progress_equalize_evaluators: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute an end-to-end diagnosis based on the provided configuration.

    Required config structure:
        model:
          generation: qwen3          # registry key
          model_name: Qwen3-8B       # variant
          ... other runner kwargs ...
        dataset:
          name: tellme/beaver_tails_filtered
          ... dataset-specific kwargs ...
        evaluator:
          type: tellme               # evaluator registry key
          ... evaluator-specific kwargs ...

        # Or: multiple evaluators in one run
        # (either `evaluator: [ ... ]` or `evaluators: [ ... ]` is accepted)
        evaluators:
          - type: xboundary
            run_name: xb
            ... evaluator-specific kwargs ...
          - type: spin
            run_name: spin_privacy
            ... evaluator-specific kwargs ...

    Args:
        config: Path, dict, or ConfigLoader with model/dataset/evaluator sections.
        dry_run: If True, only validates registry entries and returns None.

    Returns:
        Evaluation results dictionary (or None for dry_run).
    """
    cfg = _as_config_loader(config)
    cfg_progress_equalize = cfg.get("progress_equalize_evaluators")
    model_cfg_raw = cfg.get("model", {}) or {}
    models_cfg = model_cfg_raw if isinstance(model_cfg_raw, list) else [model_cfg_raw]
    dataset_cfg = cfg.get("dataset", {}) or {}
    evaluator_cfg_raw = cfg.get("evaluators")
    if evaluator_cfg_raw is None:
        evaluator_cfg_raw = cfg.get("evaluator", {}) or {}
    evaluator_cfgs: List[Dict[str, Any]]
    if isinstance(evaluator_cfg_raw, list):
        evaluator_cfgs = evaluator_cfg_raw
    else:
        evaluator_cfgs = [evaluator_cfg_raw]
    summarizer_cfg = cfg.get("summarizer", {}) or {}

    if not models_cfg:
        raise ValueError("Config must provide at least one model entry.")

    if not evaluator_cfgs:
        raise ValueError("Config must provide evaluator (dict) or evaluators (list of dicts).")

    evaluator_types: List[str] = []
    evaluator_dataset_cfgs: List[Dict[str, Any]] = []
    evaluator_summarizer_cfgs: List[Optional[Dict[str, Any]]] = []

    def _coerce_dataset_cfg(ds_cfg: Optional[Dict[str, Any]], *, idx: int) -> Dict[str, Any]:
        if ds_cfg is None:
            return dataset_cfg
        if not isinstance(ds_cfg, dict):
            raise ValueError(f"evaluator.dataset must be a dict (got {type(ds_cfg)!r}) at index {idx}.")
        return ds_cfg

    dataset_names: List[str] = []
    for idx, e_cfg in enumerate(evaluator_cfgs):
        if not isinstance(e_cfg, dict):
            raise ValueError(f"Each evaluator entry must be a dict (got: {type(e_cfg)!r}) at index {idx}.")
        e_type = e_cfg.get("type")
        if not e_type:
            raise ValueError(f"Each evaluator entry must provide evaluator.type (missing at index {idx}).")
        evaluator_types.append(str(e_type))
        ds_cfg = _coerce_dataset_cfg(e_cfg.get("dataset"), idx=idx)
        if not ds_cfg:
            raise ValueError("Config must provide dataset (root-level) or evaluator.dataset.")
        ds_name = ds_cfg.get("name")
        if not ds_name:
            raise ValueError(f"Dataset name is required for evaluator at index {idx} (dataset.name missing).")
        evaluator_dataset_cfgs.append(ds_cfg)
        dataset_names.append(str(ds_name))
        e_sum_cfg = e_cfg.get("summarizer")
        if e_sum_cfg is not None:
            if not isinstance(e_sum_cfg, dict):
                raise ValueError(f"evaluator.summarizer must be a dict (got {type(e_sum_cfg)!r}) at index {idx}.")
            e_sum_type = e_sum_cfg.get("type")
            if not e_sum_type:
                raise ValueError(f"evaluator.summarizer.type is required (missing at index {idx}).")
            if not get_summarizer_registry().is_registered(str(e_sum_type)):
                raise RuntimeError(f"Summarizer '{e_sum_type}' is not registered.")
            evaluator_summarizer_cfgs.append(e_sum_cfg)
        else:
            evaluator_summarizer_cfgs.append(None)

    if not dataset_names:
        raise ValueError("Config must provide dataset (root-level) or evaluator.dataset.")

    evaluator_type = evaluator_types[0]

    model_registry = get_model_registry()
    dataset_registry = get_dataset_registry()
    evaluator_registry = get_evaluator_registry()
    summarizer_registry = get_summarizer_registry()

    # Validate all models are registered
    for m_cfg in models_cfg:
        model_key = m_cfg.get("generation") or m_cfg.get("name")
        if not model_key:
            raise ValueError("Each model entry must provide model.generation (or model.name).")
        if not model_registry.is_registered(model_key):
            raise RuntimeError(f"Model '{model_key}' is not registered.")

    for ds_name in dataset_names:
        if not dataset_registry.is_registered(ds_name):
            raise RuntimeError(f"Dataset '{ds_name}' is not registered.")
    for e_type in evaluator_types:
        if not evaluator_registry.is_registered(e_type):
            raise RuntimeError(f"Evaluator '{e_type}' is not registered.")

    summarizer_type = summarizer_cfg.get("type")
    if summarizer_type:
        if not summarizer_registry.is_registered(summarizer_type):
            raise RuntimeError(f"Summarizer '{summarizer_type}' is not registered.")

    if dry_run:
        return None

    # No global throttling; progress driven by sample counts.

    # Persist results with run id/timestamp and folder structure
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path("results")
    resolved_progress_equalize = (
        progress_equalize_evaluators
        if progress_equalize_evaluators is not None
        else cfg_progress_equalize
    )
    run_identifier = run_id
    if not run_identifier:
        raise ValueError("A run id is required. Provide --run-id or run_id parameter.")
    timestamp = datetime.now(timezone.utc).isoformat()
    run_dir = resolved_output_dir / run_identifier
    run_dir.mkdir(parents=True, exist_ok=True)

    all_models_results = []

    # Prepare progress sink/callbacks.
    effective_sink = progress_sink
    progress_adapter = _PipelineProgressAdapter(effective_sink) if effective_sink is not None else None
    sink_for_callbacks = progress_adapter or effective_sink
    throughput_tracker = TokenThroughputTracker(sink=sink_for_callbacks)
    try:
        # Initialize UI with a zero-rate snapshot before any tokens flow.
        throughput_tracker.emit_zero_rate(status="running")
    except Exception:
        logging.debug("Initial throughput zero emission failed", exc_info=True)

    model_summaries = [
        str(m.get("run_name") or m.get("model_name") or m.get("generation") or m.get("name") or "model")
        for m in models_cfg
    ]
    evaluator_summaries = [str(e.get("run_name") or e.get("type") or f"eval_{i}") for i, e in enumerate(evaluator_cfgs)]
    dataset_summaries = dataset_names or [dataset_cfg.get("name") or "dataset"]

    def _run_summary_message() -> str:
        return (
            "Job summary | "
            f"run_id={run_identifier} | "
            f"datasets={', '.join(dataset_summaries)} | "
            f"evaluators={', '.join(evaluator_summaries)} | "
            f"models={', '.join(model_summaries)} | "
            f"output_dir={run_dir}"
        )

    def _emit_log(message: str) -> None:
        if not message:
            return
        logging.info(message)

    _emit_log(
        (
            f"Run {run_identifier} started | "
            f"datasets: {', '.join(dataset_summaries)} | "
            f"evaluators: {', '.join(evaluator_summaries)} | "
            f"models: {', '.join(model_summaries)}"
        )
    )
    _emit_log(_run_summary_message())

    # Only use explicit callbacks when provided; sink methods remain available.
    cb_start = on_progress_start
    cb_update = on_progress_update
    cb_done = on_progress_done

    def _safe_call(fn, *args):
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            logging.debug("Progress callback failed", exc_info=True)

    def _emit_start(total: Optional[int], desc: str) -> None:
        _safe_call(cb_start, total, desc)
        target_sink = sink_for_callbacks
        if target_sink is not None and hasattr(target_sink, "on_start"):
            _safe_call(getattr(target_sink, "on_start"), total, desc)

    def _emit_update(completed: int, total: Optional[int], desc: str) -> None:
        _safe_call(cb_update, completed, total, desc)
        target_sink = sink_for_callbacks
        if target_sink is not None and hasattr(target_sink, "on_update"):
            _safe_call(getattr(target_sink, "on_update"), completed, total, desc)

    def _emit_done(completed: int, total: Optional[int], desc: str) -> None:
        _safe_call(cb_done, completed, total, desc)
        target_sink = sink_for_callbacks
        if target_sink is not None and hasattr(target_sink, "on_done"):
            _safe_call(getattr(target_sink, "on_done"), completed, total, desc)

    _emit_log("Preparing dataset (warming up caches and computing totals)")
    _emit_update(0, None, "preparing dataset")
    # Prepare evaluators (support multiple evaluators run linearly).
    evaluator_specs: List[Dict[str, Any]] = []
    for idx, e_cfg in enumerate(evaluator_cfgs):
        e_type = str(e_cfg.get("type"))
        e_run_name = e_cfg.get("run_name")
        # Ensure SPIN gets a stable default run name when none is provided.
        if not e_run_name and e_type == "spin":
            e_run_name = "spin"
            e_cfg["run_name"] = e_run_name

        e_id = str(e_run_name or e_type or f"eval_{idx}")
        if any(spec.get("id") == e_id for spec in evaluator_specs):
            e_id = f"{e_id}_{idx}"
        evaluator_kwargs = {
            k: v for k, v in e_cfg.items() if k not in {"type", "run_name", "dataset", "summarizer"}
        }
        e_dataset_cfg = evaluator_dataset_cfgs[idx]
        e_summarizer_cfg = evaluator_summarizer_cfgs[idx]
        evaluator_obj = evaluator_registry.create_evaluator(e_type, config=evaluator_kwargs)
        evaluator_specs.append(
            {
                "id": e_id,
                "type": e_type,
                "config": e_cfg,
                "instance": evaluator_obj,
                "dataset_cfg": e_dataset_cfg,
                "summarizer_cfg": e_summarizer_cfg,
            }
        )

    # Load datasets lazily per evaluator (cache identical configs).
    dataset_cache: Dict[str, Tuple[Any, Optional[int]]] = {}

    def _dataset_key(ds_cfg: Dict[str, Any]) -> str:
        try:
            return json.dumps(ds_cfg, sort_keys=True, default=str)
        except Exception:
            return str(ds_cfg)

    def _load_dataset(ds_cfg: Dict[str, Any]) -> Tuple[Any, Optional[int]]:
        key = _dataset_key(ds_cfg)
        if key in dataset_cache:
            return dataset_cache[key]
        ds_name = ds_cfg.get("name")
        ds_kwargs = {k: v for k, v in ds_cfg.items() if k != "name"}
        ds = dataset_registry.get_dataset(ds_name, **ds_kwargs)
        ds_total = infer_total_items(ds)
        dataset_cache[key] = (ds, ds_total)
        return ds, ds_total

    def _estimate_spin_total(
        dataset: Any,
        evaluator_cfg: Dict[str, Any],
        model_obj: Any,
    ) -> Optional[int]:
        """
        Best-effort progress total for SPIN.

        SPIN progress updates once per (layer, batch) pair, so total steps are:
        num_layers * (len(dataset1) + len(dataset2)).
        """
        try:
            hf_model = getattr(model_obj, "model", model_obj)
            num_layers = get_num_hidden_layers(hf_model)
            if not isinstance(num_layers, int) or num_layers <= 0:
                return None
            if not isinstance(dataset, dict):
                return None
            # spin/csv_bundle returns {"paths": {"dataset1": ..., "dataset2": ...}}
            if "paths" in dataset:
                nsamples = evaluator_cfg.get("nsamples") or evaluator_cfg.get("samples")
                if isinstance(nsamples, int) and nsamples > 0:
                    return num_layers * (int(nsamples) * 2)
                return None
            d1_key = evaluator_cfg.get("dataset1_key") or "dataset1"
            d2_key = evaluator_cfg.get("dataset2_key") or "dataset2"
            d1 = dataset.get(d1_key)
            d2 = dataset.get(d2_key)
            if isinstance(d1, list) and isinstance(d2, list):
                return num_layers * (len(d1) + len(d2))
        except Exception:
            return None
        return None

    # Pre-load datasets to establish global totals for progress scaling.
    phase_totals: List[Optional[int]] = []
    for spec in evaluator_specs:
        _, ds_total = _load_dataset(spec["dataset_cfg"])
        phase_totals.append(ds_total if isinstance(ds_total, int) and ds_total > 0 else None)

    all_totals_known = all(t is not None for t in phase_totals) and len(phase_totals) > 0

    # Equalize mode: each evaluator gets the same weight (1 unit) when totals are known.
    if all_totals_known and resolved_progress_equalize:
        global_eval_total: Optional[int] = len(phase_totals)
        phase_offsets: List[Optional[int]] = list(range(len(phase_totals)))
    else:
        global_eval_total = sum(t for t in phase_totals if t is not None) if all_totals_known else None
        if all_totals_known:
            offset = 0
            phase_offsets = []
            for t in phase_totals:
                phase_offsets.append(offset)
                offset += int(t) if t is not None else 0
        else:
            phase_offsets = [None] * len(phase_totals)

    # Prime progress with the first evaluator's dataset for "dataset ready" feedback.
    n_evaluators = len(evaluator_specs)
    first_dataset_cfg = evaluator_specs[0]["dataset_cfg"]
    try:
        first_dataset, first_dataset_total = _load_dataset(first_dataset_cfg)
        if progress_adapter is not None:
            progress_adapter.set_eval_total(global_eval_total or first_dataset_total)
        if first_dataset_total is not None:
            logging.info("Loaded dataset '%s' with %d examples", first_dataset_cfg.get("name"), first_dataset_total)
            _emit_log(
                f"Dataset ready: {first_dataset_cfg.get('name')} "
                f"({first_dataset_total} examples, phases={n_evaluators})"
            )
            _emit_update(0, first_dataset_total, "dataset ready")
        else:
            _emit_log(f"Dataset ready: {first_dataset_cfg.get('name')} (example count unknown)")
            _emit_update(0, None, "dataset ready")
    except Exception:
        raise

    n_evaluators = len(evaluator_specs)
    for m_cfg in models_cfg:
        model_key = m_cfg.get("generation") or m_cfg.get("name")
        model_name = m_cfg.get("model_name")
        model_kwargs = {k: v for k, v in m_cfg.items() if k not in {"generation", "model_name", "name", "run_name"}}
        model_label = m_cfg.get("run_name") or model_name or model_key
        _emit_log(
            f"[Model] Loading {model_label} "
            f"(registry_key={model_key}, model_name={model_name or 'unknown'}, evaluators={n_evaluators})"
        )
        if progress_adapter is not None:
            progress_adapter.mark_model_preload(f"loading model {model_label}")
        _emit_update(0, None, f"loading model {model_label}")
        model = model_registry.get_model(model_key, model_name=model_name, **model_kwargs)
        if progress_adapter is not None:
            progress_adapter.mark_model_loaded(f"model ready {model_label}")
        _emit_update(0, None, f"model ready {model_label}")
        _emit_log(f"[Model] Ready {model_label} (registry_key={model_key})")

        model_id = m_cfg.get("run_name") or model_name or model_key
        model_dir = run_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_evaluations: List[Dict[str, Any]] = []
        results_by_evaluator: Dict[str, Any] = {}

        for e_idx, spec in enumerate(evaluator_specs):
            e_type = spec["type"]
            e_id = spec["id"]
            evaluator = spec["instance"]
            evaluator_dir = model_dir / e_id
            evaluator_dir.mkdir(parents=True, exist_ok=True)

            dataset_for_eval, dataset_total = _load_dataset(spec["dataset_cfg"])
            if e_type == "spin":
                spin_total = _estimate_spin_total(dataset_for_eval, spec.get("config") or {}, model)
                if spin_total is not None:
                    dataset_total = spin_total
            if progress_adapter is not None:
                progress_adapter.set_eval_total(global_eval_total or dataset_total)

            phase_sink = (
                _PhaseProgressAdapter(
                    sink_for_callbacks,
                    phase_idx=e_idx,
                    n_phases=n_evaluators,
                    phase_total=dataset_total if isinstance(dataset_total, int) and dataset_total > 0 else None,
                    phase_label=e_id,
                    global_total=global_eval_total,
                    phase_offset=phase_offsets[e_idx],
                    equalize_mode=bool(resolved_progress_equalize and global_eval_total is not None),
                )
                if sink_for_callbacks is not None and n_evaluators > 1
                else sink_for_callbacks
            )

            # Wrap user callbacks so their progress is also monotonic across phases.
            phase_total = dataset_total if isinstance(dataset_total, int) and dataset_total > 0 else None
            if global_eval_total is not None:
                # In equalize mode, totals are number of evaluators; otherwise sum of samples.
                phase_global_total = global_eval_total
            else:
                phase_global_total = (phase_total * n_evaluators) if (phase_total and n_evaluators > 1) else None

            def _phase_start(total: Optional[int], desc: str, _idx: int = e_idx) -> None:
                if cb_start is None:
                    return
                if phase_total is None or n_evaluators <= 1:
                    _safe_call(cb_start, total, desc)
                    return
                if _idx == 0:
                    _safe_call(cb_start, phase_global_total, desc)

            def _phase_update(
                completed: int,
                total: Optional[int],
                desc: str,
                _idx: int = e_idx,
                _phase_total: Optional[int] = phase_total,
                _global_total: Optional[int] = phase_global_total,
            ) -> None:
                if cb_update is None:
                    return
                if _phase_total is None or n_evaluators <= 1:
                    _safe_call(cb_update, completed, total, desc)
                    return
                if global_eval_total is not None and resolved_progress_equalize:
                    frac = float(completed) / float(_phase_total) if _phase_total else 0.0
                    offset = float(phase_offsets[_idx] or 0)
                    global_completed = offset + frac
                    _safe_call(cb_update, global_completed, _global_total, desc)
                elif global_eval_total is not None:
                    bounded = min(max(int(completed), 0), int(_phase_total))
                    offset = phase_offsets[_idx] or 0
                    global_completed = offset + bounded
                    _safe_call(cb_update, global_completed, _global_total, desc)
                else:
                    bounded = min(max(int(completed), 0), int(_phase_total))
                    global_completed = int(_idx) * int(_phase_total) + bounded
                    _safe_call(cb_update, global_completed, _global_total, desc)

            def _phase_done(
                completed: int,
                total: Optional[int],
                desc: str,
                _idx: int = e_idx,
                _global_total: Optional[int] = phase_global_total,
            ) -> None:
                if cb_done is None:
                    return
                if phase_total is None or n_evaluators <= 1:
                    _safe_call(cb_done, completed, total, desc)
                    return
                if global_eval_total is not None and resolved_progress_equalize:
                    if _idx == n_evaluators - 1:
                        _safe_call(cb_done, _global_total, _global_total, desc)
                elif global_eval_total is not None:
                    if _idx == n_evaluators - 1:
                        _safe_call(cb_done, _global_total, _global_total, desc)
                else:
                    if _idx == n_evaluators - 1:
                        _safe_call(cb_done, _global_total, _global_total, desc)

            _emit_log(
                f"[Eval:{e_id}] Start on {model_label} "
                f"(type={e_type}, dataset={spec['dataset_cfg'].get('name')}, total={dataset_total or 'unknown'})"
            )
            # Avoid resetting progress when multiple evaluators share one stream.
            if n_evaluators <= 1 or e_idx == 0 or progress_adapter is None:
                _emit_update(0, dataset_total, f"running evaluator {e_id} on {model_label}")

            raw_results = evaluator.evaluate(
                model,
                dataset_for_eval,
                output_dir=str(evaluator_dir.resolve()),
                progress_sink=phase_sink,
                on_progress_start=_phase_start if cb_start is not None else None,
                on_progress_update=_phase_update if cb_update is not None else None,
                on_progress_done=_phase_done if cb_done is not None else None,
                throughput_tracker=throughput_tracker,
            )

            evaluator_payload = {
                "run_id": run_identifier,
                "timestamp": timestamp,
                "model": {
                    "registry_key": model_key,
                    "model_name": model_name,
                    "config": m_cfg,
                },
                "dataset": {
                    "name": spec["dataset_cfg"].get("name"),
                    "config": spec["dataset_cfg"],
                },
                "evaluator": {
                    "id": e_id,
                    "type": e_type,
                    "config": spec.get("config") or {},
                },
                "results": raw_results,
            }

            evaluator_results_path = evaluator_dir / "results.json"
            with open(evaluator_results_path, "w", encoding="utf-8") as f:
                json.dump(evaluator_payload, f, indent=2, ensure_ascii=False)

            # Optional per-evaluator summarizer (if provided in evaluator config).
            e_sum_cfg = spec.get("summarizer_cfg")
            if e_sum_cfg:
                e_sum_type = e_sum_cfg.get("type")
                e_sum_kwargs = {k: v for k, v in e_sum_cfg.items() if k != "type"}
                try:
                    e_summarizer = summarizer_registry.create_summarizer(e_sum_type, config=e_sum_kwargs)
                    summary_input = {
                        "run_id": run_identifier,
                        "timestamp": timestamp,
                        "dataset": {"name": spec["dataset_cfg"].get("name"), "config": spec["dataset_cfg"]},
                        "evaluator": {"id": e_id, "type": e_type, "config": spec.get("config") or {}},
                        "evaluators": [{"id": e_id, "type": e_type, "config": spec.get("config") or {}}],
                        "models": [
                            {
                                "model_id": model_id,
                                "registry_key": model_key,
                                "model_name": model_name,
                                "evaluations": [
                                    {
                                        "evaluator": {"id": e_id, "type": e_type, "config": spec.get("config") or {}},
                                        "results": raw_results,
                                        "results_path": str(evaluator_results_path.resolve()),
                                    }
                                ],
                                "results_by_evaluator": {e_id: raw_results},
                            }
                        ],
                    }
                    e_summary = e_summarizer.summarize(summary_input)
                    e_summary_path = evaluator_dir / "summary.json"
                    with open(e_summary_path, "w", encoding="utf-8") as f:
                        json.dump(e_summary, f, indent=2, ensure_ascii=False)
                    try:
                        report_md = e_summarizer.format_report(e_summary, format="markdown")
                        with open(evaluator_dir / "summary.md", "w", encoding="utf-8") as f:
                            f.write(str(report_md))
                    except Exception:
                        pass
                except Exception:
                    logging.debug("Per-evaluator summarizer failed for %s", e_id, exc_info=True)

            model_evaluations.append(
                {
                    "evaluator": {"id": e_id, "type": e_type, "config": spec.get("config") or {}},
                    "results": raw_results,
                    "results_path": str(evaluator_results_path.resolve()),
                }
            )
            results_by_evaluator[e_id] = raw_results
            _emit_log(
                f"[Eval:{e_id}] Finished on {model_label} "
                f"(results saved to {evaluator_results_path.relative_to(run_dir)})"
            )
            # Reset throughput to zero after each evaluator to avoid stale rates.
            try:
                throughput_tracker.emit_zero_rate(status="running")
            except Exception:
                logging.debug("Throughput idle emission failed", exc_info=True)
            # Best-effort CUDA cleanup between evaluators (keeps model weights loaded).
            _maybe_cuda_cleanup()

        per_model_payload = {
            "run_id": run_identifier,
            "timestamp": timestamp,
            "model": {
                "registry_key": model_key,
                "model_name": model_name,
                "config": m_cfg,
            },
            "dataset": {
                "name": first_dataset_cfg.get("name"),
                "config": first_dataset_cfg,
            },
            "evaluators": [{"id": s["id"], "type": s["type"], "config": s.get("config") or {}} for s in evaluator_specs],
            "evaluations": model_evaluations,
            "results_by_evaluator": results_by_evaluator,
        }

        model_results_path = model_dir / "results.json"
        with open(model_results_path, "w", encoding="utf-8") as f:
            json.dump(per_model_payload, f, indent=2, ensure_ascii=False)

        all_models_results.append(
            {
                "model_id": model_id,
                "registry_key": model_key,
                "model_name": model_name,
                "evaluations": model_evaluations,
                "results_by_evaluator": results_by_evaluator,
                "results_path": str(model_results_path.resolve()),
            }
        )

    throughput_summary = throughput_tracker.snapshot()

    status_text = (
        f"Run {run_identifier} completed successfully on {', '.join(dataset_summaries)} "
        f"using evaluators {', '.join(evaluator_summaries)} and models {', '.join(model_summaries)}. "
        f"Processed approximately {throughput_summary.get('tokens', 0)} tokens "
        f"(avg {round(throughput_summary.get('tokens_per_second_avg', 0.0), 2)} tokens/sec)."
    )

    # Run-level summary
    wrapped = {
        "run_id": run_identifier,
        "timestamp": timestamp,
        "dataset": {"name": first_dataset_cfg.get("name"), "config": first_dataset_cfg},
        "evaluators": [{"id": s["id"], "type": s["type"], "config": s.get("config") or {}} for s in evaluator_specs],
        "models": all_models_results,
        "throughput": throughput_summary,
        "status_text": status_text,
    }

    # Optional summarization stage (writes `summary.json` and `summary.md` to run_dir)
    if summarizer_type:
        summarizer_kwargs = {k: v for k, v in summarizer_cfg.items() if k != "type"}
        summarizer = summarizer_registry.create_summarizer(summarizer_type, config=summarizer_kwargs)
        summary = summarizer.summarize(wrapped)
        wrapped["summary"] = {
            "type": summarizer_type,
            "config": summarizer_cfg,
            "data": summary,
        }
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        # Best-effort markdown report
        try:
            report_md = summarizer.format_report(summary, format="markdown")
            with open(run_dir / "summary.md", "w", encoding="utf-8") as f:
                f.write(str(report_md))
        except Exception:  # pragma: no cover
            pass

    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(wrapped, f, indent=2, ensure_ascii=False)

    # Also persist the resolved config for traceability
    config_path = run_dir / "config.resolved.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    _emit_log(
        f"Run {run_identifier} completed | "
        f"models={len(all_models_results)} | evaluators={n_evaluators} | "
        f"throughput tokens={throughput_summary.get('tokens')} "
        f"tps_avg={round(throughput_summary.get('tokens_per_second_avg', 0), 2)}"
    )

    throughput_tracker.finalize(status="complete")
    try:
        # Emit an explicit zero-rate payload at the very end to clear UI displays.
        throughput_tracker.emit_zero_rate(status="complete")
    except Exception:
        logging.debug("Final throughput idle emission failed", exc_info=True)
    target_sink = sink_for_callbacks
    if target_sink is not None and hasattr(target_sink, "on_result"):
        _safe_call(getattr(target_sink, "on_result"), wrapped)

    return wrapped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-Diagnose pipeline from config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/registries without loading model or dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit path to save results JSON (overrides run-id folder).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base directory to store run outputs (default: results).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier for organizing output directories.",
    )
    return parser.parse_args()


def _cli() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    results = run_from_config(
        args.config,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    if results is None:
        logging.info("Dry run successful.")
        return
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info("Saved results to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    _cli()

