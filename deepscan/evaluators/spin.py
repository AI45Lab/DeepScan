"""
SPIN diagnosis evaluator (diagnosis-only, reproduction-focused).

This module intentionally mirrors the SPIN reference repository implementation
for the "diagnostic signal" component:
- Importance score per weight: |W| * |dL/dW| (see `runners/importance_runner.py`)
- Coupling selection: intersection of top-q(dataset1) and top-q(dataset2), no
  subtraction by a general dataset

We do NOT apply masking (mitigation) here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deepscan.evaluators.base import BaseEvaluator
from deepscan.utils.throughput import TokenThroughputTracker, count_tokens_from_batch
from deepscan.utils.model_introspection import get_num_hidden_layers

logger = logging.getLogger(__name__)

def _extract_hf_model_and_tokenizer(model: Any) -> Tuple[Any, Any]:
    hf_model = getattr(model, "model", model)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("SPIN evaluator expects model to expose a tokenizer at `model.tokenizer`.")
    return hf_model, tokenizer


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SPIN evaluator requires torch. Install with `pip install torch`.") from exc
    return torch


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "For exact SPIN reproduction, install Hugging Face datasets: `pip install datasets`."
        ) from exc
    return load_dataset


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SPIN evaluator requires numpy for plotting/arrays. Install with `pip install numpy`.") from exc
    return np


def _load_and_sample_csv_items(path: str, nsamples: int, seed: int) -> List[Dict[str, str]]:
    """
    Match SPIN's `loaders/dataset_loader.py` sampling for disentangle=True:
    - load_dataset("csv", split="train")
    - shuffle(seed).select(range(nsamples))
    """
    load_dataset = _require_datasets()
    data_files = {"train": str(Path(path))}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    sampled = traindata.shuffle(seed=int(seed)).select(range(int(nsamples)))
    out: List[Dict[str, str]] = []
    for i in range(int(nsamples)):
        out.append({"prompt": sampled["prompt"][i], "response": sampled["response"][i]})
    return out


def _build_disentangled_pairs(tokenizer: Any, items: List[Dict[str, str]]) -> List[Tuple[Any, Any]]:
    """
    Match SPIN's disentangle=True packing:
    inp = concat(prompt_ids, response_ids[:, 1:])
    tar = inp; tar[:, :prompt_len] = -100
    """
    torch_mod = _require_torch()
    pairs: List[Tuple[Any, Any]] = []
    for row in items:
        trainenc_prompt = tokenizer(row["prompt"], return_tensors="pt")
        trainenc_response = tokenizer(row["response"], return_tensors="pt")
        inp = torch_mod.cat(
            (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
        )
        tar = inp.clone()
        prompt_len = trainenc_prompt.input_ids.shape[1]
        tar[:, :prompt_len] = -100
        pairs.append((inp, tar))
    return pairs


def _extract_layer_idx(name: str) -> int:
    parts = [int(part) for part in name.split(".") if part.isdigit()]
    return parts[0] if parts else 0


def _to_layer_relative_name(full_name: str, layer_idx: int) -> str:
    # SPIN masking code expects names like "self_attn.q_proj" relative to the layer.
    needle = f"layers.{layer_idx}."
    if needle in full_name:
        return full_name.split(needle, 1)[1]
    needle2 = f"model.layers.{layer_idx}."
    if needle2 in full_name:
        return full_name.split(needle2, 1)[1]
    return full_name


def _get_set_difference_mask(p: float, q: float, W_metric1, W_metric2):
    """
    Copied from SPIN `tools/spin.py` to keep exact top-k + unique behavior.
    """
    torch_mod = _require_torch()
    top_p = int(p * W_metric1.shape[1] * W_metric1.shape[0])
    top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])

    top_p_indices = torch_mod.topk(W_metric1.flatten(), top_p, largest=True)[1]
    top_q_indices = torch_mod.topk(W_metric2.flatten(), top_q, largest=True)[1]
    unique_p = torch_mod.unique(top_p_indices)
    unique_q = torch_mod.unique(top_q_indices)

    mask_only_metric1 = ~torch_mod.isin(unique_p, unique_q)
    mask_only_metric2 = ~torch_mod.isin(unique_q, unique_p)
    mask_intersection = ~(
        torch_mod.ones_like(unique_p).bool() & mask_only_metric1 & mask_only_metric2
    )

    return mask_only_metric1, mask_only_metric2, mask_intersection, unique_q


def _top_q_unique_indices(q: float, W_metric):
    """
    Return the unique flattened indices of the top-q fraction of weights.
    Used when we want the raw intersection of dataset top-q sets (no general subtraction).
    """
    torch_mod = _require_torch()
    top_q = int(q * W_metric.shape[1] * W_metric.shape[0])
    if top_q <= 0:
        return torch_mod.tensor([], device=W_metric.device, dtype=torch_mod.long)
    top_q_indices = torch_mod.topk(W_metric.flatten(), top_q, largest=True)[1]
    return torch_mod.unique(top_q_indices)


@dataclass
class _SpinConfig:
    nsamples: int = 128
    seed: int = 0
    # Ratio used to select top indices per dataset
    q: float = 5e-7  # dataset ratio
    target_module: str = "mlp"  # mlp | self_attn | all
    # Use a flexible default that respects CUDA_VISIBLE_DEVICES.
    device: str = "cuda"

    # Dataset dict keys
    dataset1_key: str = "dataset1"
    dataset2_key: str = "dataset2"

    # Artifact verbosity
    per_layer: bool = True
    per_module: bool = False
    # Optional artifact output (plot + summary json)
    save_coupling_plot: bool = True


class SpinEvaluator(BaseEvaluator):
    """
    Diagnosis-only SPIN evaluator.

    Expects:
    - model: a BaseModelRunner exposing `model` (HF) and `tokenizer`
    - dataset: a dict with keys: dataset1/dataset2, each a list of {prompt,response}
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name or "spin", config=config)
        allowed = {f.name for f in fields(_SpinConfig)}
        cfg_values = {k: v for k, v in (config or {}).items() if k in allowed}
        self.spin_config = _SpinConfig(**cfg_values)

    def evaluate(self, model: Any, dataset: Any, **kwargs: Any) -> Dict[str, Any]:
        cfg = self.spin_config
        torch_mod = _require_torch()
        # Import SPIN hook utilities lazily so `spin` can be registered in environments
        # that only need dry-run validation (no torch installed).
        from deepscan.evaluators.spin_support import (  # type: ignore
            ActLinear,
            make_act,
            no_act_recording,
            revert_act_to_linear,
        )

        try:
            q = float(cfg.q)
        except Exception as exc:
            raise ValueError(f"SPIN evaluator requires numeric q; got q={cfg.q!r}") from exc

        if cfg.target_module not in {"mlp", "self_attn", "all"}:
            raise ValueError("target_module must be one of: mlp, self_attn, all")

        if not isinstance(dataset, dict):
            raise ValueError("SpinEvaluator expects dataset to be a dict (e.g. from spin/csv_bundle).")

        # Accept either:
        # - pre-loaded rows under dataset1/dataset2 (preferred when present), or
        # - a spin/csv_bundle with `paths` (exact SPIN sampling via HF datasets).
        paths = dataset.get("paths") if isinstance(dataset.get("paths"), dict) else None
        has_rows = all(k in dataset for k in (cfg.dataset1_key, cfg.dataset2_key))
        if has_rows:
            d1 = list(dataset[cfg.dataset1_key])
            d2 = list(dataset[cfg.dataset2_key])
        elif paths is not None:
            if "dataset1" not in paths or "dataset2" not in paths:
                raise ValueError("SpinEvaluator paths must include 'dataset1' and 'dataset2'.")
            d1 = _load_and_sample_csv_items(paths["dataset1"], cfg.nsamples, cfg.seed)
            d2 = _load_and_sample_csv_items(paths["dataset2"], cfg.nsamples, cfg.seed)
        else:
            raise ValueError(
                "SpinEvaluator dataset must include either "
                f"'{cfg.dataset1_key}'/'{cfg.dataset2_key}' lists or a `paths` dict."
            )

        hf_model, tokenizer = _extract_hf_model_and_tokenizer(model)
        device = torch_mod.device(cfg.device)

        # Build SPIN-style "dataloader" (list of (inp, tar) pairs)
        batches_1 = _build_disentangled_pairs(tokenizer, d1)
        batches_2 = _build_disentangled_pairs(tokenizer, d2)

        # Match SPIN: compute importance per dataset separately with make_act wrappers.
        #
        # IMPORTANT (memory): do NOT store full importance matrices in-memory.
        # For large models, keeping (|W|*|grad|) for every linear layer will grow VRAM
        # until OOM. Instead, we keep only the top-q candidate indices per module
        # (small) and the module size for later aggregation.
        def _compute_candidate_registry(
            batches: List[Tuple[Any, Any]],
            *,
            progress=None,
            throughput_tracker: Optional[TokenThroughputTracker] = None,
        ) -> Dict[str, Any]:
            model_wrapped = make_act(hf_model, verbose=False)
            model_wrapped.eval()
            num_hidden_layers = get_num_hidden_layers(model_wrapped) or 0
            registry: Dict[str, Any] = {}

            for layer in range(num_hidden_layers):
                layer_filter_fn = lambda x, layer=layer: f"layers.{layer}." in x
                model_wrapped.zero_grad()
                model_wrapped.requires_grad_(False)

                saved_grad: Dict[str, Any] = {}
                for name, module in model_wrapped.named_modules():
                    if layer_filter_fn(name) and isinstance(module, ActLinear):
                        module.base.requires_grad_(True)
                        saved_grad[name] = torch_mod.zeros_like(
                            module.base.weight, device=module.base.weight.device
                        )
                        module.base.zero_grad()

                for inp, tar in batches:
                    if progress is not None:
                        progress.update(1)
                    if throughput_tracker is not None:
                        throughput_tracker.add_batch(inp)
                    inp, tar = inp.to(device), tar.to(device)
                    model_wrapped.zero_grad()
                    with no_act_recording(model_wrapped):
                        loss = model_wrapped(input_ids=inp, labels=tar)[0]
                    loss.backward()
                    for name, module in model_wrapped.named_modules():
                        if layer_filter_fn(name) and isinstance(module, ActLinear):
                            grad = module.base.weight.grad
                            # With MoE / sparse routing, many expert weights never receive
                            # gradients for a given batch; PyTorch leaves `.grad` as None.
                            if grad is None:
                                continue
                            saved_grad[name] += grad.abs()

                # Convert grad into importance matrix and store only the top-q indices
                # by layer-relative name (CPU).
                for name, module in model_wrapped.named_modules():
                    if layer_filter_fn(name) and isinstance(module, ActLinear):
                        layer_idx = _extract_layer_idx(name)
                        rel = _to_layer_relative_name(name, layer_idx)
                        key = f"layer_{layer_idx}:{rel}"
                        # importance score: |W| * |dL/dW|
                        grad_abs_sum = saved_grad.get(name)
                        if grad_abs_sum is None:
                            # Should not happen: we initialize saved_grad for every ActLinear in this layer.
                            cand = torch_mod.empty(0, dtype=torch_mod.long)
                        else:
                            # If a module never received any gradient signal across the dataset batches,
                            # treat it as having no candidates (avoids arbitrary top-q selection on all-zeros).
                            try:
                                has_signal = bool(torch_mod.any(grad_abs_sum))
                            except Exception:
                                has_signal = True
                            if not has_signal:
                                cand = torch_mod.empty(0, dtype=torch_mod.long)
                            else:
                                importance = (module.base.weight.data.abs() * grad_abs_sum).detach()
                                cand = _top_q_unique_indices(q, importance).detach().cpu()
                                # Best-effort: free large temporary tensor early.
                                del importance
                        registry[key] = {"candidate_indices": cand, "numel": int(module.base.weight.numel())}
                        saved_grad.pop(name, None)

            revert_act_to_linear(model_wrapped)
            model_wrapped.zero_grad()
            return registry

        total_steps = int(get_num_hidden_layers(hf_model) or 0) * (len(batches_1) + len(batches_2))
        progress_sink = kwargs.get("progress_sink")
        with self.progress(
            dataset=None,
            total=total_steps,
            desc="SPIN diagnosis (importance scoring)",
            progress_sink=progress_sink,
            on_start=kwargs.get("on_progress_start"),
            on_update=kwargs.get("on_progress_update"),
            on_done=kwargs.get("on_progress_done"),
        ) as progress:
            imp_1 = _compute_candidate_registry(
                batches_1, progress=progress, throughput_tracker=kwargs.get("throughput_tracker")
            )
            imp_2 = _compute_candidate_registry(
                batches_2, progress=progress, throughput_tracker=kwargs.get("throughput_tracker")
            )

        results: Dict[str, Any] = {
            "evaluator": {"id": "spin", "type": "spin"},
            "config": asdict(cfg),
            "dataset_paths": paths,
            "nsamples": int(cfg.nsamples),
            "totals": {
                "candidate_dataset1": 0,
                "candidate_dataset2": 0,
                "coupled": 0,
                "total_neurons": 0,
            },
            "layers": [],
            "artifacts": {},
        }

        # Aggregate per-layer coupling stats using SPIN's exact selection logic.
        num_layers = int(get_num_hidden_layers(hf_model) or 0)
        coupled_per_layer: List[int] = [0 for _ in range(num_layers)]

        for layer_idx in range(num_layers):
            layer_totals = {
                "candidate_dataset1": 0,
                "candidate_dataset2": 0,
                "coupled": 0,
                "total_neurons": 0,
            }
            if cfg.per_module:
                modules_out: List[Dict[str, Any]] = []

            # enumerate modules present in both dataset registries for this layer
            prefix = f"layer_{layer_idx}:"
            keys = sorted(
                k for k in imp_1.keys()
                if k.startswith(prefix) and k in imp_2
            )

            for k in keys:
                rel_name = k.split(prefix, 1)[1]
                if cfg.target_module == "mlp" and "self_attn" in rel_name:
                    continue
                if cfg.target_module == "self_attn" and "mlp" in rel_name:
                    continue

                e1 = imp_1[k] or {}
                e2 = imp_2[k] or {}
                cand1 = e1.get("candidate_indices")
                cand2 = e2.get("candidate_indices")
                module_neurons = int(e1.get("numel") or 0)
                if cand1 is None or cand2 is None or module_neurons <= 0:
                    continue

                # Coupled = intersection of the candidate sets (no general subtraction)
                # Note: candidate indices are stored on CPU to avoid VRAM blow-up.
                coupled = torch_mod.isin(cand1, cand2)
                coupled_count = int(coupled.sum().item())

                layer_totals["candidate_dataset1"] += int(cand1.numel())
                layer_totals["candidate_dataset2"] += int(cand2.numel())
                layer_totals["coupled"] += coupled_count
                layer_totals["total_neurons"] += module_neurons

                if cfg.per_module:
                    modules_out.append(
                        {
                            "name": rel_name,
                            "candidate_dataset1": int(cand1.numel()),
                            "candidate_dataset2": int(cand2.numel()),
                            "coupled": coupled_count,
                            "total_neurons": module_neurons,
                            # Fairness–Privacy Neurons Coupling Ratio: coupled / all neurons
                            "fairness_privacy_neurons_coupling_ratio": (
                                coupled_count / module_neurons if module_neurons > 0 else float("nan")
                            ),
                        }
                    )

            coupled_per_layer[layer_idx] = int(layer_totals["coupled"])
            results["totals"]["candidate_dataset1"] += layer_totals["candidate_dataset1"]
            results["totals"]["candidate_dataset2"] += layer_totals["candidate_dataset2"]
            results["totals"]["coupled"] += layer_totals["coupled"]
            results["totals"]["total_neurons"] += layer_totals["total_neurons"]

            if cfg.per_layer:
                entry: Dict[str, Any] = {
                    "layer_idx": int(layer_idx),
                    "totals": layer_totals,
                    # Fairness–Privacy Neurons Coupling Ratio: coupled / all neurons
                    "fairness_privacy_neurons_coupling_ratio": (
                        layer_totals["coupled"] / layer_totals["total_neurons"]
                        if layer_totals["total_neurons"] > 0
                        else float("nan")
                    ),
                }
                if cfg.per_module:
                    entry["modules"] = modules_out
                results["layers"].append(entry)

        denom = float(results["totals"]["candidate_dataset1"] + results["totals"]["candidate_dataset2"]) / 2.0
        results["totals"]["coupled_rate_vs_candidate_mean"] = (
            float(results["totals"]["coupled"]) / denom if denom > 0 else float("nan")
        )
        # Fairness–Privacy Neurons Coupling Ratio: coupled / all neurons
        results["totals"]["fairness_privacy_neurons_coupling_ratio"] = (
            float(results["totals"]["coupled"]) / float(results["totals"]["total_neurons"])
            if results["totals"]["total_neurons"] > 0
            else float("nan")
        )

        # Optional plot artifact written to output_dir (provided by run.py)
        output_dir = kwargs.get("output_dir")
        if cfg.save_coupling_plot and output_dir:
            try:
                np = _require_numpy()
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore

                out_path = Path(output_dir) / "spin_coupled_per_layer.png"
                plt.figure(figsize=(10, 4))
                plt.plot(np.arange(num_layers), coupled_per_layer)
                plt.xlabel("Layer")
                plt.ylabel("Coupled count")
                plt.title("SPIN diagnosis: coupled weights per layer")
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()
                results["artifacts"]["coupled_per_layer_plot"] = str(out_path.resolve())
            except Exception:
                pass

        return results


# Auto-register evaluator for registry users
try:
    from deepscan.evaluators.registry import get_evaluator_registry

    get_evaluator_registry().register_evaluator("spin")(SpinEvaluator)
except Exception:  # pragma: no cover
    logger.debug("Could not auto-register SpinEvaluator with the registry.")


