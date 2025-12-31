"""
End-to-end runner that executes a diagnosis pipeline from a config file/dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, Callable
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from llm_diagnose import ConfigLoader
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry
from llm_diagnose.evaluators.registry import get_evaluator_registry
from llm_diagnose.summarizers.registry import get_summarizer_registry
from llm_diagnose.utils.progress import infer_total_items


def _as_config_loader(config: Union[str, Dict[str, Any], ConfigLoader]) -> ConfigLoader:
    if isinstance(config, ConfigLoader):
        return config
    if isinstance(config, str):
        return ConfigLoader.from_file(config)
    if isinstance(config, dict):
        return ConfigLoader.from_dict(config)
    raise TypeError("config must be a path, dict, or ConfigLoader")


def _as_optional_config_dict(config: Optional[Union[str, Dict[str, Any], ConfigLoader]]) -> Dict[str, Any]:
    """Load optional config-like input into a plain dict; returns {} when absent."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if isinstance(config, ConfigLoader):
        return config.to_dict()
    if isinstance(config, str):
        try:
            return ConfigLoader.from_file(config).to_dict()
        except Exception:
            logging.warning("Failed to load webhook config from %s", config, exc_info=True)
            return {}
    logging.warning("Unsupported webhook config type: %s", type(config))
    return {}


class _WebhookSink:
    """Lightweight optional webhook sink for progress callbacks."""

    def __init__(
        self,
        url: str,
        run_id: str,
        timeout: float = 2.0,
        result_url: Optional[str] = None,
        append_run_id_query: bool = True,
    ):
        self.progress_url = self._with_run_id(url, run_id) if append_run_id_query else url
        self.result_url = (
            self._with_run_id(result_url, run_id) if (result_url and append_run_id_query) else result_url
        )
        self.run_id = run_id
        self.timeout = timeout
        try:
            import requests  # type: ignore

            self._requests = requests
        except Exception:
            self._requests = None
            logging.warning("requests is not installed; progress webhook disabled.")

    def _with_run_id(self, base_url: Optional[str], run_id: str) -> Optional[str]:
        """
        Append ?_id=<run_id> to the provided URL when not already present.
        Falls back gracefully if parsing fails.
        """
        if not base_url:
            return None
        try:
            parsed = urlsplit(base_url)
            query_pairs = dict(parse_qsl(parsed.query, keep_blank_values=True))
            if run_id and "_id" not in query_pairs:
                query_pairs["_id"] = run_id
            query = urlencode(query_pairs)
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))
        except Exception:
            # Basic fallback keeps original URL and simply appends the query string.
            suffix = f"?_id={run_id}" if run_id else ""
            return f"{base_url}{suffix}"

    def _post_json(self, url: Optional[str], payload: Dict[str, Any]) -> None:
        if self._requests is None or not url:
            return
        try:
            self._requests.post(url, json=payload, timeout=self.timeout)
        except Exception:
            logging.debug("Webhook post failed.", exc_info=True)

    def _post_progress(self, *, status: str, progress: Optional[float], pass_rate: Optional[float] = None) -> None:
        payload: Dict[str, Any] = {"status": status, "run_id": self.run_id}
        if progress is not None:
            payload["progress"] = progress
        if pass_rate is not None:
            payload["passRate"] = pass_rate
        self._post_json(self.progress_url, payload)

    @staticmethod
    def _percent(completed: int, total: Optional[int]) -> Optional[float]:
        if not total:
            return None
        try:
            return round(float(completed) / float(total) * 100.0, 2)
        except Exception:
            return None

    def on_start(self, total: Optional[int], desc: str) -> None:
        progress = 0 if total else None
        self._post_progress(status="running", progress=progress)

    def on_update(self, completed: int, total: Optional[int], desc: str) -> None:
        progress = self._percent(completed, total)
        self._post_progress(status="running", progress=progress)

    def on_done(self, completed: int, total: Optional[int], desc: str) -> None:
        progress = 100.0 if total else None
        self._post_progress(status="complete", progress=progress)

    def on_result(self, result: Dict[str, Any]) -> None:
        """Send final result payload to a (possibly different) endpoint."""
        payload = {"status": "complete", "run_id": self.run_id, "result": result}
        self._post_json(self.result_url or self.progress_url, payload)


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
    progress_webhook: Optional[str] = None,
    result_webhook: Optional[str] = None,
    webhook_config: Optional[Union[str, Dict[str, Any], ConfigLoader]] = None,
    append_run_id_query: Optional[bool] = None,
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
    webhook_cfg = _as_optional_config_dict(webhook_config)
    cfg_progress_webhook = cfg.get("progress_webhook")
    cfg_result_webhook = cfg.get("result_webhook")
    cfg_append_run_id_query = cfg.get("append_run_id_query")
    wh_progress_webhook = webhook_cfg.get("progress_webhook")
    wh_result_webhook = webhook_cfg.get("result_webhook")
    wh_append_run_id_query = webhook_cfg.get("append_run_id_query")
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
    dataset_total = infer_total_items(dataset)
    if dataset_total is not None:
        logging.info("Loaded dataset '%s' with %d examples", dataset_name, dataset_total)
        _emit_update(0, dataset_total, "dataset ready")
    else:
        _emit_update(0, None, "dataset ready")

    evaluator_kwargs = {k: v for k, v in evaluator_cfg.items() if k != "type"}
    evaluator = evaluator_registry.create_evaluator(evaluator_type, config=evaluator_kwargs)

    # Persist results with run id/timestamp and folder structure
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path("results")
    resolved_progress_webhook = progress_webhook or wh_progress_webhook or cfg_progress_webhook
    resolved_result_webhook = result_webhook or wh_result_webhook or cfg_result_webhook
    resolved_append_run_id_query = (
        append_run_id_query
        if append_run_id_query is not None
        else (wh_append_run_id_query if wh_append_run_id_query is not None else cfg_append_run_id_query)
    )
    if resolved_append_run_id_query is None:
        resolved_append_run_id_query = True

    def _extract_id_from_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        try:
            parsed = urlsplit(url)
            query_pairs = dict(parse_qsl(parsed.query, keep_blank_values=True))
            return query_pairs.get("_id")
        except Exception:
            return None

    inferred_run_id = _extract_id_from_url(resolved_progress_webhook) or _extract_id_from_url(
        resolved_result_webhook
    )
    run_identifier = run_id or inferred_run_id
    if not run_identifier:
        raise ValueError(
            "A run/job id is required. Provide --run-id or include `_id=<job_id>` in the progress/result webhook URL."
        )
    timestamp = datetime.now(timezone.utc).isoformat()
    run_dir = resolved_output_dir / run_identifier
    run_dir.mkdir(parents=True, exist_ok=True)

    all_models_results = []

    # Prepare progress sink/callbacks (webhook is opt-in).
    effective_sink = progress_sink
    if effective_sink is None and resolved_progress_webhook:
        effective_sink = _WebhookSink(
            resolved_progress_webhook,
            run_identifier,
            result_url=resolved_result_webhook,
            append_run_id_query=resolved_append_run_id_query,
        )

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
        if effective_sink is not None and hasattr(effective_sink, "on_start"):
            _safe_call(getattr(effective_sink, "on_start"), total, desc)

    def _emit_update(completed: int, total: Optional[int], desc: str) -> None:
        _safe_call(cb_update, completed, total, desc)
        if effective_sink is not None and hasattr(effective_sink, "on_update"):
            _safe_call(getattr(effective_sink, "on_update"), completed, total, desc)

    def _emit_done(completed: int, total: Optional[int], desc: str) -> None:
        _safe_call(cb_done, completed, total, desc)
        if effective_sink is not None and hasattr(effective_sink, "on_done"):
            _safe_call(getattr(effective_sink, "on_done"), completed, total, desc)

    _emit_update(0, None, "preparing dataset")
    for m_cfg in models_cfg:
        model_key = m_cfg.get("generation") or m_cfg.get("name")
        model_name = m_cfg.get("model_name")
        model_kwargs = {k: v for k, v in m_cfg.items() if k not in {"generation", "model_name", "name", "run_name"}}
        model_label = m_cfg.get("run_name") or model_name or model_key
        _emit_update(0, None, f"loading model {model_label}")
        model = model_registry.get_model(model_key, model_name=model_name, **model_kwargs)
        _emit_update(0, None, f"model ready {model_label}")

        model_id = m_cfg.get("run_name") or model_name or model_key
        model_dir = run_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Provide a per-model output directory for evaluators that write artifacts (plots, metric json, etc.)
        raw_results = evaluator.evaluate(
            model,
            dataset,
            output_dir=str(model_dir.resolve()),
            progress_sink=effective_sink,
            on_progress_start=cb_start,
            on_progress_update=cb_update,
            on_progress_done=cb_done,
        )

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

    # Final webhook for results if enabled (allows distinct endpoint from progress).
    if effective_sink is not None and hasattr(effective_sink, "on_result"):
        _safe_call(getattr(effective_sink, "on_result"), wrapped)

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
        help="Required unless webhook URL already contains _id=<job_id>; used to tag webhook posts.",
    )
    parser.add_argument(
        "--progress-webhook",
        default=None,
        help="Optional URL to POST progress updates (appends ?_id=<run_id> and sends status/progress JSON).",
    )
    parser.add_argument(
        "--result-webhook",
        default=None,
        help="Optional URL to POST final result payload (appends ?_id=<run_id>). Defaults to progress webhook when omitted.",
    )
    parser.add_argument(
        "--webhook-config",
        default=None,
        help="Optional separate YAML/JSON config for webhook settings (progress_webhook, result_webhook, run_id).",
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
        progress_webhook=args.progress_webhook,
        result_webhook=args.result_webhook,
        webhook_config=args.webhook_config,
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

