"""
SPIN summarizer.

This summarizes the diagnosis-only SPIN evaluator output:
- Total coupled count and coupled rate
- Fairness–Privacy Neurons Coupling Ratio (coupled / total neurons)
- Top layers by coupled count
- Preserves artifact paths (e.g., coupled_per_layer plot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from llm_diagnose.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)


@dataclass
class _SpinSummarizerConfig:
    top_layers: int = 5


def _pick_top_layers(layers: List[Dict[str, Any]], k: int) -> List[Tuple[int, int]]:
    scored: List[Tuple[int, int]] = []
    for entry in layers or []:
        try:
            layer_idx = int(entry.get("layer_idx"))
            coupled = int((entry.get("totals") or {}).get("coupled", 0))
        except Exception:
            continue
        scored.append((layer_idx, coupled))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(0, int(k))]


class SpinSummarizer(BaseSummarizer):
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name or "spin", config=config)
        allowed = {f.name for f in fields(_SpinSummarizerConfig)}
        cfg_values = {k: v for k, v in (config or {}).items() if k in allowed}
        self._cfg = _SpinSummarizerConfig(**cfg_values)

    def summarize(self, results: Dict[str, Any], benchmark: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        models = results.get("models", []) if isinstance(results, dict) else []
        per_model: Dict[str, Any] = {}

        def _pick_spin_results(model_entry: Dict[str, Any]) -> Dict[str, Any]:
            """
            Extract spin results from model entry.
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
                    if (meta.get("type") == "spin" or meta.get("id") == "spin") and isinstance(ev.get("results"), dict):
                        return ev.get("results") or {}

            # Fallback: results_by_evaluator map keyed by evaluator id.
            rbe = model_entry.get("results_by_evaluator") or {}
            if isinstance(rbe, dict):
                for k, v in rbe.items():
                    if isinstance(k, str) and ("spin" in k.lower()) and isinstance(v, dict):
                        return v

            # Old schema: single evaluator output.
            if isinstance(model_entry.get("results"), dict):
                return model_entry.get("results") or {}

            return {}

        for m in models:
            model_id = m.get("model_id") or m.get("model_name") or "unknown"
            r = _pick_spin_results(m) if isinstance(m, dict) else {}
            totals = r.get("totals") or {}
            layers = r.get("layers") or []
            artifacts = r.get("artifacts") or {}

            per_model[str(model_id)] = {
                "totals": {
                    "coupled": totals.get("coupled"),
                    "candidate_dataset1": totals.get("candidate_dataset1"),
                    "candidate_dataset2": totals.get("candidate_dataset2"),
                    "coupled_rate_vs_candidate_mean": totals.get("coupled_rate_vs_candidate_mean"),
                    "fairness_privacy_neurons_coupling_ratio": totals.get("fairness_privacy_neurons_coupling_ratio"),
                },
                "top_layers_by_coupled": _pick_top_layers(layers, self._cfg.top_layers),
                "artifacts": artifacts,
            }

        return {
            "run_id": results.get("run_id"),
            "evaluator": results.get("evaluator"),
            "summary": per_model,
        }


# Auto-register
try:
    from llm_diagnose.summarizers.registry import get_summarizer_registry

    get_summarizer_registry().register_summarizer("spin")(SpinSummarizer)
except Exception:  # pragma: no cover
    logger.debug("Could not auto-register SpinSummarizer with the registry.")


