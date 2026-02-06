"""
End-to-end runner that executes a diagnosis pipeline from a config file/dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, List, Tuple
import argparse
import json
import logging
import gc
from datetime import datetime, timezone
from pathlib import Path

from deepscan import ConfigLoader
from deepscan.registry.model_registry import get_model_registry
from deepscan.registry.dataset_registry import get_dataset_registry
from deepscan.evaluators.registry import get_evaluator_registry
from deepscan.summarizers.registry import get_summarizer_registry

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


def run_from_config(
    config: Union[str, Dict[str, Any], ConfigLoader],
    *,
    dry_run: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
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

    resolved_output_dir = Path(output_dir) if output_dir is not None else Path("results")
    run_identifier = run_id
    if not run_identifier:
        run_identifier = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now(timezone.utc).isoformat()
    run_dir = resolved_output_dir / run_identifier
    run_dir.mkdir(parents=True, exist_ok=True)

    all_models_results = []

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
    dataset_cache: Dict[str, Any] = {}

    def _dataset_key(ds_cfg: Dict[str, Any]) -> str:
        try:
            return json.dumps(ds_cfg, sort_keys=True, default=str)
        except Exception:
            return str(ds_cfg)

    def _load_dataset(ds_cfg: Dict[str, Any]) -> Any:
        key = _dataset_key(ds_cfg)
        if key in dataset_cache:
            return dataset_cache[key]
        ds_name = ds_cfg.get("name")
        ds_kwargs = {k: v for k, v in ds_cfg.items() if k != "name"}
        ds = dataset_registry.get_dataset(ds_name, **ds_kwargs)
        dataset_cache[key] = ds
        return ds

    n_evaluators = len(evaluator_specs)
    first_dataset_cfg = evaluator_specs[0]["dataset_cfg"]
    for m_cfg in models_cfg:
        model_key = m_cfg.get("generation") or m_cfg.get("name")
        model_name = m_cfg.get("model_name")
        model_kwargs = {k: v for k, v in m_cfg.items() if k not in {"generation", "model_name", "name", "run_name"}}
        model_label = m_cfg.get("run_name") or model_name or model_key
        _emit_log(
            f"[Model] Loading {model_label} "
            f"(registry_key={model_key}, model_name={model_name or 'unknown'}, evaluators={n_evaluators})"
        )
        model = model_registry.get_model(model_key, model_name=model_name, **model_kwargs)
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

            dataset_for_eval = _load_dataset(spec["dataset_cfg"])

            _emit_log(
                f"[Eval:{e_id}] Start on {model_label} "
                f"(type={e_type}, dataset={spec['dataset_cfg'].get('name')})"
            )

            raw_results = evaluator.evaluate(
                model,
                dataset_for_eval,
                output_dir=str(evaluator_dir.resolve()),
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

    status_text = (
        f"Run {run_identifier} completed successfully on {', '.join(dataset_summaries)} "
        f"using evaluators {', '.join(evaluator_summaries)} and models {', '.join(model_summaries)}."
    )

    wrapped = {
        "run_id": run_identifier,
        "timestamp": timestamp,
        "dataset": {"name": first_dataset_cfg.get("name"), "config": first_dataset_cfg},
        "evaluators": [{"id": s["id"], "type": s["type"], "config": s.get("config") or {}} for s in evaluator_specs],
        "models": all_models_results,
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
        f"models={len(all_models_results)} | evaluators={n_evaluators}"
    )

    return wrapped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepScan pipeline from config.")
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
        help="Run identifier for output directory (default: run_<timestamp>).",
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

