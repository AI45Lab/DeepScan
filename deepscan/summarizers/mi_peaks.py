"""
MI-Peaks summarizer.

This is intentionally lightweight: it extracts the list-valued MI trajectory
metric so downstream UIs can render a line chart.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from deepscan.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)


def _pick_mi_peaks_results(model_entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(model_entry, dict):
        return {}

    evals = model_entry.get("evaluations") or []
    if isinstance(evals, list):
        for ev in evals:
            if not isinstance(ev, dict):
                continue
            meta = ev.get("evaluator") or {}
            ev_type = str(meta.get("type") or "").lower()
            ev_id = str(meta.get("id") or "").lower()
            if any(key in (ev_type, ev_id) for key in ("mi-peaks", "mi_peaks", "mipeaks")) and isinstance(ev.get("results"), dict):
                return ev.get("results") or {}

    rbe = model_entry.get("results_by_evaluator") or {}
    if isinstance(rbe, dict):
        for k, v in rbe.items():
            if isinstance(k, str) and "mi" in k.lower() and "peak" in k.lower() and isinstance(v, dict):
                return v

    if isinstance(model_entry.get("results"), dict):
        return model_entry.get("results") or {}

    return {}


class MiPeaksSummarizer(BaseSummarizer):
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name or "mi-peaks", config=config)

    def summarize(self, results: Dict[str, Any], benchmark: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        models = results.get("models", []) if isinstance(results, dict) else []
        per_model: Dict[str, Any] = {}

        for m in models:
            model_id = m.get("model_id") or m.get("model_name") or "unknown"
            r = _pick_mi_peaks_results(m) if isinstance(m, dict) else {}
            metrics = r.get("metrics") or {}

            traj: List[Any] = []
            if isinstance(metrics, dict):
                traj = metrics.get("mi_mean_trajectory") or []
            if not isinstance(traj, list):
                traj = []

            per_model[str(model_id)] = {
                # The key that downstream clients can use directly for line charts.
                "result_metric": traj,
                "target_layer": metrics.get("target_layer") if isinstance(metrics, dict) else None,
                "thinking_tokens_top": metrics.get("thinking_tokens_top") if isinstance(metrics, dict) else None,
                "artifacts": r.get("artifacts") or {},
            }

        return {"run_id": results.get("run_id"), "evaluator": "mi-peaks", "summary": per_model}


# Auto-register
try:
    from deepscan.summarizers.registry import get_summarizer_registry

    get_summarizer_registry().register_summarizer("mi-peaks")(MiPeaksSummarizer)
    get_summarizer_registry().register_summarizer("mi_peaks")(MiPeaksSummarizer)
except Exception:  # pragma: no cover
    logger.debug("Could not auto-register MiPeaksSummarizer with the registry.")

