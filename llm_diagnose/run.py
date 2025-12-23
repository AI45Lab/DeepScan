"""
End-to-end runner that executes a diagnosis pipeline from a config file/dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from llm_diagnose import ConfigLoader
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry
from llm_diagnose.evaluators.registry import get_evaluator_registry
from llm_diagnose.summarizers.registry import get_summarizer_registry


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
    evaluator_cfg = cfg.get("evaluator", {}) or {}
    summarizer_cfg = cfg.get("summarizer", {}) or {}

    if not models_cfg:
        raise ValueError("Config must provide at least one model entry.")

    dataset_name = dataset_cfg.get("name")
    if not dataset_name:
        raise ValueError("Config must provide dataset.name.")

    evaluator_type = evaluator_cfg.get("type")
    if not evaluator_type:
        raise ValueError("Config must provide evaluator.type.")

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

    if not dataset_registry.is_registered(dataset_name):
        raise RuntimeError(f"Dataset '{dataset_name}' is not registered.")
    if not evaluator_registry.is_registered(evaluator_type):
        raise RuntimeError(f"Evaluator '{evaluator_type}' is not registered.")

    summarizer_type = summarizer_cfg.get("type")
    if summarizer_type:
        if not summarizer_registry.is_registered(summarizer_type):
            raise RuntimeError(f"Summarizer '{summarizer_type}' is not registered.")

    if dry_run:
        return None

    dataset_kwargs = {k: v for k, v in dataset_cfg.items() if k != "name"}
    dataset = dataset_registry.get_dataset(dataset_name, **dataset_kwargs)

    evaluator_kwargs = {k: v for k, v in evaluator_cfg.items() if k != "type"}
    evaluator = evaluator_registry.create_evaluator(evaluator_type, config=evaluator_kwargs)

    # Persist results with run id/timestamp and folder structure
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path("results")
    run_identifier = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = datetime.now(timezone.utc).isoformat()
    run_dir = resolved_output_dir / run_identifier
    run_dir.mkdir(parents=True, exist_ok=True)

    all_models_results = []

    for m_cfg in models_cfg:
        model_key = m_cfg.get("generation") or m_cfg.get("name")
        model_name = m_cfg.get("model_name")
        model_kwargs = {k: v for k, v in m_cfg.items() if k not in {"generation", "model_name", "name", "run_name"}}
        model = model_registry.get_model(model_key, model_name=model_name, **model_kwargs)

        model_id = m_cfg.get("run_name") or model_name or model_key
        model_dir = run_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Provide a per-model output directory for evaluators that write artifacts (plots, metric json, etc.)
        raw_results = evaluator.evaluate(model, dataset, output_dir=str(model_dir.resolve()))

        per_model_payload = {
            "run_id": run_identifier,
            "timestamp": timestamp,
            "model": {
                "registry_key": model_key,
                "model_name": model_name,
                "config": m_cfg,
            },
            "dataset": {
                "name": dataset_name,
                "config": dataset_cfg,
            },
            "evaluator": {
                "type": evaluator_type,
                "config": evaluator_cfg,
            },
            "results": raw_results,
        }

        with open(model_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(per_model_payload, f, indent=2, ensure_ascii=False)

        all_models_results.append(
            {
                "model_id": model_id,
                "registry_key": model_key,
                "model_name": model_name,
                "results": raw_results,
                "results_path": str((model_dir / "results.json").resolve()),
            }
        )

    # Run-level summary
    wrapped = {
        "run_id": run_identifier,
        "timestamp": timestamp,
        "dataset": {"name": dataset_name, "config": dataset_cfg},
        "evaluator": {"type": evaluator_type, "config": evaluator_cfg},
        "models": all_models_results,
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
        help="Optional run id; defaults to timestamp if not provided.",
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
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    _cli()

