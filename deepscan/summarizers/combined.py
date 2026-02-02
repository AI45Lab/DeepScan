"""
Combined run-level summarizer: aggregates all evaluator outputs per model.

This is evaluator-agnostic and simply collates:
- evaluator id/type
- evaluator config
- raw results (metrics/artifacts/etc.)
- results_path when present
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from deepscan.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)


class CombinedSummarizer(BaseSummarizer):
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name or "combined", config=config)

    def summarize(self, results: Dict[str, Any], benchmark: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        models = results.get("models", []) if isinstance(results, dict) else []
        per_model: Dict[str, Any] = {}

        for m in models:
            if not isinstance(m, dict):
                continue
            model_id = m.get("model_id") or m.get("model_name") or "unknown"
            evals = m.get("evaluations") or []
            grouped: Dict[str, Any] = {}
            if isinstance(evals, list):
                for ev in evals:
                    if not isinstance(ev, dict):
                        continue
                    meta = ev.get("evaluator") or {}
                    ev_id = meta.get("id") or meta.get("type") or "unknown"
                    grouped[str(ev_id)] = {
                        "type": meta.get("type"),
                        "config": meta.get("config"),
                        "results": ev.get("results"),
                        "results_path": ev.get("results_path"),
                    }
            per_model[str(model_id)] = grouped

        return {
            "run_id": results.get("run_id"),
            "summary": per_model,
        }


# Auto-register
try:
    from deepscan.summarizers.registry import get_summarizer_registry

    get_summarizer_registry().register_summarizer("combined")(CombinedSummarizer)
except Exception:  # pragma: no cover
    logger.debug("Could not auto-register CombinedSummarizer", exc_info=True)

