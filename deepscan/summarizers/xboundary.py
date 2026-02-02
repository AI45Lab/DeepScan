"""
X-Boundary summarizer.

This turns the evaluator output into a compact run-level summary:
- For each model: best layer by separation_score (higher) and best by boundary_ratio (lower)
- Preserves artifact paths (metrics_summary.json + t-SNE plots)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

from deepscan.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)


@dataclass
class _XBoundarySummarizerConfig:
    # Which layer selection to highlight
    select_by: str = "separation_score"  # or "boundary_ratio"
    # Also include the alternative selection
    include_both_selections: bool = True


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _pick_best_layer(metrics_by_layer: Dict[str, Any], metric: str) -> Tuple[Optional[str], Optional[float]]:
    """
    metrics_by_layer keys may be ints or strings. Return key as string.
    """
    if not metrics_by_layer:
        return None, None

    best_key: Optional[str] = None
    best_value: Optional[float] = None

    for k, v in metrics_by_layer.items():
        kk = str(k)
        score = _safe_float((v or {}).get(metric))
        if best_value is None:
            best_key, best_value = kk, score
            continue

        if metric == "boundary_ratio":
            # lower is better
            if score < best_value:
                best_key, best_value = kk, score
        else:
            # higher is better
            if score > best_value:
                best_key, best_value = kk, score

    return best_key, best_value


class XBoundarySummarizer(BaseSummarizer):
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name or "xboundary", config=config)
        allowed = {f.name for f in fields(_XBoundarySummarizerConfig)}
        cfg_values = {k: v for k, v in (config or {}).items() if k in allowed}
        self._cfg = _XBoundarySummarizerConfig(**cfg_values)

    def summarize(self, results: Dict[str, Any], benchmark: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        models = results.get("models", []) if isinstance(results, dict) else []
        per_model: Dict[str, Any] = {}

        def _pick_xboundary_results(model_entry: Dict[str, Any]) -> Dict[str, Any]:
            """
            Extract xboundary results from model entry.
            Supports both single-evaluator runs (stored in `model["results"]`) and
            multi-evaluator runs (stored in `model["evaluations"]` and/or `model["results_by_evaluator"]`).
            """
            if not isinstance(model_entry, dict):
                return {}

            # New schema preferred: explicit per-evaluator evaluations list.
            evals = model_entry.get("evaluations") or []
            if isinstance(evals, list):
                for ev in evals:
                    if not isinstance(ev, dict):
                        continue
                    meta = ev.get("evaluator") or {}
                    if (meta.get("type") == "xboundary" or meta.get("id") == "xboundary") and isinstance(ev.get("results"), dict):
                        return ev.get("results") or {}

            # Fallback: results_by_evaluator map keyed by evaluator id.
            rbe = model_entry.get("results_by_evaluator") or {}
            if isinstance(rbe, dict):
                for k, v in rbe.items():
                    if isinstance(k, str) and ("xboundary" in k.lower()) and isinstance(v, dict):
                        return v

            # Old schema: single evaluator output.
            if isinstance(model_entry.get("results"), dict):
                return model_entry.get("results") or {}

            return {}

        for m in models:
            model_id = m.get("model_id") or m.get("model_name") or "unknown"
            r = _pick_xboundary_results(m) if isinstance(m, dict) else {}

            metrics_by_layer = r.get("metrics_by_layer") or {}
            artifacts = r.get("artifacts") or {}

            best_sep_layer, best_sep = _pick_best_layer(metrics_by_layer, "separation_score")
            best_ratio_layer, best_ratio = _pick_best_layer(metrics_by_layer, "boundary_ratio")

            highlight_metric = self._cfg.select_by
            if highlight_metric == "boundary_ratio":
                chosen_layer, chosen_value = best_ratio_layer, best_ratio
            else:
                chosen_layer, chosen_value = best_sep_layer, best_sep

            entry: Dict[str, Any] = {
                "chosen_by": highlight_metric,
                "chosen_layer": chosen_layer,
                "chosen_value": chosen_value,
                "artifacts": artifacts,
            }

            if self._cfg.include_both_selections:
                entry["best_by_separation_score"] = {"layer": best_sep_layer, "value": best_sep}
                entry["best_by_boundary_ratio"] = {"layer": best_ratio_layer, "value": best_ratio}

            per_model[str(model_id)] = entry

        return {
            "run_id": results.get("run_id"),
            "evaluator": results.get("evaluator"),
            "summary": per_model,
        }


# Auto-register
try:
    from deepscan.summarizers.registry import get_summarizer_registry

    get_summarizer_registry().register_summarizer("xboundary")(XBoundarySummarizer)
except Exception:  # pragma: no cover
    logger.debug("Could not auto-register XBoundarySummarizer with the registry.")


