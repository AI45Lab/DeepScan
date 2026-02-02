"""
TellMe summarizer: flattens TellMe evaluator metrics per model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from deepscan.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)


def _pick_tellme_results(model_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve TellMe results from model entry, supporting both single- and multi-evaluator schemas.
    """
    if not isinstance(model_entry, dict):
        return {}

    # Preferred: explicit evaluations list
    evals = model_entry.get("evaluations") or []
    if isinstance(evals, list):
        for ev in evals:
            if not isinstance(ev, dict):
                continue
            meta = ev.get("evaluator") or {}
            if (meta.get("type") == "tellme" or meta.get("id") == "tellme") and isinstance(ev.get("results"), dict):
                return ev.get("results") or {}

    # Fallback: results_by_evaluator map.
    rbe = model_entry.get("results_by_evaluator") or {}
    if isinstance(rbe, dict):
        for k, v in rbe.items():
            if isinstance(k, str) and ("tellme" in k.lower()) and isinstance(v, dict):
                return v

    # Old schema: single evaluator output in results.
    if isinstance(model_entry.get("results"), dict):
        return model_entry.get("results") or {}

    return {}


class TellMeSummarizer(BaseSummarizer):
    """
    Summarize TellMe metrics into a compact per-model view.
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name or "tellme", config=config)

    def summarize(self, results: Dict[str, Any], benchmark: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        models = results.get("models", []) if isinstance(results, dict) else []
        per_model: Dict[str, Any] = {}

        for m in models:
            model_id = m.get("model_id") or m.get("model_name") or "unknown"
            r = _pick_tellme_results(m)
            metrics = r.get("metrics") or {}
            per_model[str(model_id)] = {
                "metrics": metrics,
                "layer": r.get("layer"),
                "num_samples": r.get("num_samples"),
            }

        return {
            "run_id": results.get("run_id"),
            "evaluator": "tellme",
            "summary": per_model,
        }


# Auto-register
try:
    from deepscan.summarizers.registry import get_summarizer_registry

    get_summarizer_registry().register_summarizer("tellme")(TellMeSummarizer)
except Exception:  # pragma: no cover
    logger.debug("Could not auto-register TellMeSummarizer with the registry.")

