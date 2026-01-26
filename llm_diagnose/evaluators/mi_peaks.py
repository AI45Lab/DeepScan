"""
MI-Peaks diagnosis evaluator (diagnosis-only, reproduction-focused).

This evaluator mirrors the core MI-Peaks implementation:
- Extract activations for generated reasoning tokens ("problem" column) by
  registering forward hooks on transformer blocks and capturing the last-token
  hidden state at each generation step.
- Extract ground-truth activations ("solution" column) via a single forward pass
  and use the last-token hidden state as the MI reference.
- Compute token-wise MI using the HSIC estimator from MI-Peaks (`sigma=50` by default).

We do NOT implement any mitigation/training methods (RR/TTTS/etc.).
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from llm_diagnose.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MI-Peaks evaluator requires torch. Install with `pip install torch`.") from exc
    return torch


class ActivationHook:
    """
    Copied from MI-Peaks `src/tools/hooks.py` for exact behavior.
    """

    def __init__(self, token_position: int = -1):
        self.token_position = token_position
        self.tokens_embeddings: List[Any] = []

    def __call__(self, module, module_inputs, module_outputs):  # pragma: no cover
        torch_mod = _require_torch()
        hidden_states = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs
        emb = hidden_states[0, self.token_position].detach().cpu()
        # Ensure CPU tensor for deterministic serialization and MI kernel behavior.
        if hasattr(torch_mod, "Tensor") and isinstance(emb, torch_mod.Tensor):
            emb = emb.cpu()
        self.tokens_embeddings.append(emb)


def distmat(X):
    """
    Copied from MI-Peaks `src/tools/mi_estimators.py` for exact behavior.
    """
    torch_mod = _require_torch()
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    r = torch_mod.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch_mod.mm(X, torch_mod.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch_mod.transpose(r, 0, 1).expand_as(a)
    D = torch_mod.abs(D)
    return D


def sigma_estimation(X, Y):
    """
    Copied from MI-Peaks `src/tools/mi_estimators.py` for exact behavior.
    """
    import numpy as np  # type: ignore

    D = distmat(_require_torch().cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1e-2:
        med = 1e-2
    return med


def kernelmat(X, sigma, ktype: str = "gaussian"):
    """
    Copied from MI-Peaks `src/tools/mi_estimators.py` for exact behavior.
    """
    torch_mod = _require_torch()
    if len(X.shape) == 1:
        X = X.view(-1, 1)

    m = int(X.size()[0])
    H = torch_mod.eye(m) - (1.0 / m) * torch_mod.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2.0 * sigma * sigma * X.size()[1]
            Kx = torch_mod.exp(-Dxx / variance).type(torch_mod.FloatTensor)
        else:
            sx = sigma_estimation(X, X)
            Kx = torch_mod.exp(-Dxx / (2.0 * sx * sx)).type(torch_mod.FloatTensor)
    elif ktype == "linear":
        Kx = torch_mod.mm(X, X.T).type(torch_mod.FloatTensor)
    elif ktype == "IMQ":
        Dxx = distmat(X)
        Kx = 1 * torch_mod.rsqrt(Dxx + 1)
    else:
        raise ValueError(f"Unsupported ktype: {ktype}")

    Kxc = torch_mod.mm(Kx, H)
    return Kxc


def hsic_normalized_cca(x, y, sigma: float = 50.0, ktype: str = "gaussian"):
    """
    Copied from MI-Peaks `src/tools/mi_estimators.py` for exact behavior.
    """
    torch_mod = _require_torch()
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma, ktype=ktype)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1e-5
    K_I = torch_mod.eye(m)
    Kxc_i = torch_mod.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch_mod.inverse(Kyc + epsilon * m * K_I)
    Rx = Kxc.mm(Kxc_i)
    Ry = Kyc.mm(Kyc_i)
    Pxy = torch_mod.sum(torch_mod.mul(Rx, Ry.t()))
    return Pxy


def estimate_mi_hsic(x, y, ktype: str = "gaussian", sigma: float = 50.0):
    """
    Copied from MI-Peaks `src/tools/mi_estimators.py` for exact behavior.
    """
    return hsic_normalized_cca(x, y, ktype=ktype, sigma=sigma)


def _get_gt_reference(sample: Any, layer: int):
    """
    Copied from MI-Peaks `src/runners/mi_pipeline.py` for exact behavior.
    """
    if isinstance(sample, dict) and "reps" in sample:
        return sample["reps"][layer][0]
    return sample[layer][0]


def _infer_model_tag(model_runner: Any) -> str:
    hf_model = getattr(model_runner, "model", model_runner)
    name = getattr(hf_model, "name_or_path", None) or getattr(getattr(hf_model, "config", None), "_name_or_path", None)
    if name:
        return str(name).split("/")[-1]
    return str(getattr(model_runner, "model_name", None) or "model")


def _resolve_layer_module(hf_model: Any, layer_idx: int) -> Any:
    """
    MI-Peaks hooks `model.model.layers[layer]`.

    The exact attribute chain varies across HF architectures (Qwen2/2.5, LLaMA,
    GPT-NeoX, etc.). We try a set of common paths first, then fall back to a
    heuristic scan for the most plausible `torch.nn.ModuleList` of blocks.
    """
    torch_mod = _require_torch()

    def _get_chain(obj: Any, dotted: str) -> Any:
        cur = obj
        for part in dotted.split("."):
            cur = getattr(cur, part)
        return cur

    # Common explicit paths across HF models
    paths = [
        "model.layers",          # e.g. Qwen2ForCausalLM.model.layers
        "model.model.layers",    # some wrappers nest an extra `.model`
        "layers",                # some architectures expose blocks at top-level
        "transformer.h",         # GPT-2 style
        "transformer.layers",
        "gpt_neox.layers",
        "decoder.layers",
        "model.decoder.layers",
        "model.transformer.h",
    ]
    for p in paths:
        try:
            seq = _get_chain(hf_model, p)
            if isinstance(seq, torch_mod.nn.ModuleList):
                return seq[layer_idx]
            # also accept list/tuple of modules
            if isinstance(seq, (list, tuple)) and seq:
                return seq[layer_idx]
        except Exception:
            continue

    # Heuristic scan: pick the "best" ModuleList that looks like transformer blocks.
    best_name = None
    best_list = None
    best_score = -1
    for name, module in hf_model.named_modules():
        if not isinstance(module, torch_mod.nn.ModuleList):
            continue
        if len(module) <= 0:
            continue
        score = 0
        lowered = name.lower()
        if lowered.endswith(("layers", ".h", "blocks")) or ".layers" in lowered or lowered.endswith(".h"):
            score += 10
        score += min(len(module), 200)  # prefer larger blocklists
        # Prefer modulelists whose elements look like transformer blocks
        try:
            child0 = module[0]
            for attr in ("self_attn", "mlp", "attention", "attn"):
                if hasattr(child0, attr):
                    score += 3
        except Exception:
            pass
        if score > best_score:
            best_score = score
            best_name = name
            best_list = module

    if best_list is not None:
        logger.info("MI-Peaks: using transformer block list '%s' (len=%d)", best_name, len(best_list))
        try:
            return best_list[layer_idx]
        except Exception as exc:
            raise IndexError(
                f"Requested layer {layer_idx} is out of range for detected block list "
                f"'{best_name}' (len={len(best_list)})."
            ) from exc

    raise AttributeError(
        "Could not locate transformer layers for MI-Peaks hooks. "
        "Expected one of: `model.model.layers`, `model.layers`, `model.transformer.h` "
        "(or a detectable ModuleList of blocks)."
    )


def _infer_input_device(hf_model: Any) -> Any:
    """
    Best-effort device inference (mirrors X-Boundary helper).
    """
    torch_mod = _require_torch()
    try:
        first_param = next(hf_model.parameters())
        dev = getattr(first_param, "device", None)
        if dev is not None and getattr(dev, "type", None) != "meta":
            return dev
    except Exception:
        pass

    devmap = getattr(hf_model, "hf_device_map", None)
    if isinstance(devmap, dict):
        for v in devmap.values():
            if isinstance(v, str) and v not in {"cpu", "disk"}:
                try:
                    return torch_mod.device(v)
                except Exception:
                    continue

    return torch_mod.device("cpu")


def _collect_activations(
    texts: Sequence[str],
    tokenizer: Any,
    hf_model: Any,
    layers: List[int],
    *,
    token_pos: int = -1,
    mode: str = "generate",
    max_new_tokens: int = 512,
    progress: Optional[Any] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Mirrors MI-Peaks `collect_activations` in `src/runners/activation_pipeline.py`.
    """
    torch_mod = _require_torch()
    handles, hooks = [], []
    for layer in layers:
        hook = ActivationHook(token_position=token_pos)
        handle = _resolve_layer_module(hf_model, layer).register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)

    acts: Dict[int, Dict[str, Any]] = {
        idx: {"reps": {layer: [] for layer in layers}, "token_ids": []}
        for idx in range(len(texts))
    }

    for idx, text in enumerate(texts):
        if progress is not None:
            progress.update(1)
        input_ids = tokenizer.encode(text, return_tensors="pt")
        try:
            # For non-sharded models, ensure inputs are on the model's entry device.
            input_ids = input_ids.to(_infer_input_device(hf_model))
        except Exception:
            pass
        with torch_mod.no_grad():
            if mode == "generate":
                outputs = hf_model.generate(
                    input_ids,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                token_slice = outputs[0][:, input_ids.shape[1] : -1]
            else:
                hf_model(input_ids)
                token_slice = input_ids

        acts[idx]["token_ids"] = token_slice.squeeze().cpu()
        for layer, hook in zip(layers, hooks):
            layer_embs = torch_mod.stack(hook.tokens_embeddings).float()
            acts[idx]["reps"][layer] = layer_embs
            hook.tokens_embeddings = []

    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass

    return acts


def _calculate_mi(
    acts: Dict[int, Dict[str, Any]],
    gt_acts: Dict[int, Dict[str, Any]],
    *,
    layers: List[int],
    num_samples: int,
    save_path: Path,
    sigma: float = 50.0,
    ktype: str = "gaussian",
    reuse_cache: bool = True,
    progress: Optional[Any] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Mirrors MI-Peaks `calculate_mi` in `src/runners/mi_pipeline.py` (including resume).

    Note:
        - When `reuse_cache=True`, we resume from an existing `save_path` if present.
        - When `reuse_cache=False`, we ignore any existing file and overwrite it.
    """
    torch_mod = _require_torch()
    num_samples = len(acts) if num_samples < 0 else min(int(num_samples), len(acts))

    if len(layers) == 0:
        layers = list(acts[0]["reps"].keys())

    if reuse_cache and save_path.exists():
        final_mi_dict = torch_mod.load(save_path)
    else:
        final_mi_dict = {
            idx: {"reps": {layer: [] for layer in layers}, "total_tokens": -1}
            for idx in range(num_samples)
        }

    for idx in range(num_samples):
        if progress is not None:
            progress.update(1)
        if final_mi_dict[idx].get("total_tokens", -1) > 0:
            continue

        # Original MI-Peaks behavior:
        # - `token_ids` holds generated tokens (prompt excluded)
        # - `reps[layer]` may include an extra prompt-boundary embedding at index 0
        # - MI is computed over *all* reps entries (including that possible extra)
        final_mi_dict[idx]["total_tokens"] = int(acts[idx]["token_ids"].shape[0])
        for layer in layers:
            here_num_tokens = int(acts[idx]["reps"][layer].shape[0])
            layer_mi_list = torch_mod.zeros(here_num_tokens)
            for token_idx in range(here_num_tokens):
                layer_mi_list[token_idx] = estimate_mi_hsic(
                    acts[idx]["reps"][layer][token_idx],
                    _get_gt_reference(gt_acts[idx], layer),
                    ktype=ktype,
                    sigma=float(sigma),
                )
            final_mi_dict[idx]["reps"][layer] = layer_mi_list

        os.makedirs(save_path.parent, exist_ok=True)
        torch_mod.save(final_mi_dict, save_path)

    return final_mi_dict


def _mean_ragged_trajectory(seqs: List[List[float]]) -> List[float]:
    if not seqs:
        return []
    max_len = max((len(s) for s in seqs), default=0)
    out: List[float] = []
    for t in range(max_len):
        vals = [s[t] for s in seqs if len(s) > t]
        if not vals:
            break
        out.append(float(sum(vals) / len(vals)))
    return out


@dataclass
class _MiPeaksConfig:
    # Evaluation
    layers: List[int] = field(default_factory=lambda: [31])
    # Convenience override: use a single layer. If set, overrides `layers`.
    # Supports negative indexing: -1 means last layer.
    target_layer: Optional[int] = None
    # Optional cap on how many dataset examples to process for MI.
    # If None, we use the dataset size (after dataset loader truncation).
    sample_num: Optional[int] = 10
    max_new_tokens: int = 512
    token_position: int = -1
    # MI estimator
    sigma: float = 50.0
    ktype: str = "gaussian"
    # Peak/tokens summary
    top_k: int = 20
    top_tokens: int = 15
    # Caching / artifact roots
    artifact_root: Optional[str] = None
    reuse_cache: bool = True


class MiPeaksEvaluator(BaseEvaluator):
    """
    Diagnosis-only MI-Peaks evaluator.

    Expects dataset dict with keys:
    - items: [{"problem": str, "solution": str}, ...]
      or problems/solutions lists.
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name or "mi-peaks", config=config)
        allowed = {f.name for f in fields(_MiPeaksConfig)}
        raw_cfg = config or {}
        cfg_values = {k: v for k, v in raw_cfg.items() if k in allowed}
        self._cfg = _MiPeaksConfig(**cfg_values)

    def evaluate(self, model: Any, dataset: Any, **kwargs: Any) -> Dict[str, Any]:
        cfg = self._cfg
        torch_mod = _require_torch()

        if not hasattr(model, "model") or not hasattr(model, "tokenizer"):
            raise RuntimeError("MiPeaksEvaluator expects a model runner exposing `.model` and `.tokenizer`.")
        if model.tokenizer is None:
            raise RuntimeError("MiPeaksEvaluator requires a tokenizer on the model runner.")
        hf_model = model.model
        tokenizer = model.tokenizer

        if not isinstance(dataset, dict):
            raise ValueError("MiPeaksEvaluator expects dataset to be a dict (from mi-peaks dataset loaders).")

        if "items" in dataset and isinstance(dataset["items"], list):
            items = dataset["items"]
            problems = [str((row or {}).get("problem", "")) for row in items]
            solutions = [str((row or {}).get("solution", "")) for row in items]
        else:
            problems = list(dataset.get("problems") or [])
            solutions = list(dataset.get("solutions") or [])

        if not problems or not solutions or len(problems) != len(solutions):
            raise ValueError(
                "MiPeaksEvaluator requires paired problems/solutions of equal length "
                "(keys: items or problems+solutions)."
            )

        dataset_tag = str(dataset.get("dataset") or dataset.get("name") or "dataset")
        model_tag = _infer_model_tag(model)

        # Decide how many examples to run. Dataset loader can already truncate; this cap is for MI compute.
        dataset_n = int(min(len(problems), len(solutions)))
        run_n = dataset_n if cfg.sample_num is None else max(0, min(int(cfg.sample_num), dataset_n))
        problems = problems[:run_n]
        solutions = solutions[:run_n]

        output_dir = kwargs.get("output_dir")
        root = Path(cfg.artifact_root) if cfg.artifact_root else (Path(output_dir) / "mi_peaks" if output_dir else Path("mi_peaks"))
        acts_root = root / "acts"
        results_root = root / "results" / "mi"
        acts_reason_path = acts_root / "reasoning_evolve" / f"{dataset_tag}_{model_tag}.pth"
        acts_gt_path = acts_root / "gt" / f"{dataset_tag}_{model_tag}.pth"
        mi_tensor_path = results_root / f"{dataset_tag}_gtmodel={model_tag}_testmodel={model_tag}.pth"
        acts_reason_path.parent.mkdir(parents=True, exist_ok=True)
        acts_gt_path.parent.mkdir(parents=True, exist_ok=True)
        results_root.mkdir(parents=True, exist_ok=True)

        # Resolve target layers.
        layers = [int(x) for x in (cfg.layers or [])]
        if cfg.target_layer is not None:
            layers = [int(cfg.target_layer)]

        if not layers:
            raise ValueError("MiPeaksEvaluator requires non-empty `layers` (or set `target_layer`).")

        # Determine number of transformer blocks (best-effort).
        num_blocks: Optional[int] = None
        try:
            # Prefer explicit config when present.
            num_blocks = int(getattr(getattr(hf_model, "config", None), "num_hidden_layers"))
        except Exception:
            num_blocks = None

        # If we can't infer from config, infer from the blocklist heuristic.
        if num_blocks is None:
            torch_mod = _require_torch()
            best_list = None
            best_score = -1
            for name, modulelist in hf_model.named_modules():
                if not isinstance(modulelist, torch_mod.nn.ModuleList) or len(modulelist) <= 0:
                    continue
                score = 0
                lowered = name.lower()
                if lowered.endswith(("layers", ".h", "blocks")) or ".layers" in lowered or lowered.endswith(".h"):
                    score += 10
                score += min(len(modulelist), 200)
                if score > best_score:
                    best_score = score
                    best_list = modulelist
            if best_list is not None:
                num_blocks = int(len(best_list))

        # Expand "all layers" sentinel (kept for back-compat with our earlier config behavior).
        if layers == [-1] and cfg.target_layer is None:
            if num_blocks is None:
                raise ValueError("layers=[-1] requires hf_model.config.num_hidden_layers or detectable ModuleList.")
            layers = list(range(num_blocks))

        # Support negative indexing for explicit layers (including target_layer=-1 meaning last layer).
        if num_blocks is not None:
            normalized_layers: List[int] = []
            for layer in layers:
                if layer < 0:
                    normalized_layers.append(num_blocks + layer)
                else:
                    normalized_layers.append(layer)
            layers = normalized_layers

        # Validate bounds up-front so we emit a helpful error (instead of ModuleList IndexError).
        if num_blocks is not None:
            bad = [l for l in layers if l < 0 or l >= num_blocks]
            if bad:
                raise ValueError(
                    "MI-Peaks evaluator: requested layer index(es) out of range. "
                    f"requested={layers}, num_layers={num_blocks}. "
                    "If you want the last layer, set `target_layer: -1` "
                    f"(or use `layers: [{num_blocks - 1}]`)."
                )

        def _is_complete_acts(obj: Any, needed: int) -> bool:
            if not isinstance(obj, dict):
                return False
            try:
                return all(i in obj for i in range(int(needed)))
            except Exception:
                return False

        def _slice_indexed_dict(obj: Dict[int, Any], needed: int) -> Dict[int, Any]:
            # Keep a stable 0..needed-1 indexing (MI-Peaks relies on integer keys).
            return {int(i): obj[int(i)] for i in range(int(needed)) if int(i) in obj}

        # 1) Activations (computed over the same capped set used for MI)
        acts: Dict[int, Any]
        gt_acts: Dict[int, Any]
        loaded_acts = None
        loaded_gt = None
        if cfg.reuse_cache and acts_reason_path.exists() and acts_gt_path.exists():
            try:
                loaded_acts = torch_mod.load(acts_reason_path)
                loaded_gt = torch_mod.load(acts_gt_path)
            except Exception:
                loaded_acts, loaded_gt = None, None

        if _is_complete_acts(loaded_acts, run_n) and _is_complete_acts(loaded_gt, run_n):
            acts = _slice_indexed_dict(loaded_acts, run_n)
            gt_acts = _slice_indexed_dict(loaded_gt, run_n)
        else:
            total = len(problems) + len(solutions)
            with self.progress(
                dataset=None,
                total=total,
                desc="MI-Peaks diagnosis (activation extraction)",
                progress_sink=kwargs.get("progress_sink"),
                on_start=kwargs.get("on_progress_start"),
                on_update=kwargs.get("on_progress_update"),
                on_done=kwargs.get("on_progress_done"),
            ) as progress:
                acts = _collect_activations(
                    problems,
                    tokenizer,
                    hf_model,
                    layers,
                    token_pos=int(cfg.token_position),
                    mode="generate",
                    max_new_tokens=int(cfg.max_new_tokens),
                    progress=progress,
                )
                gt_acts = _collect_activations(
                    solutions,
                    tokenizer,
                    hf_model,
                    layers,
                    token_pos=int(cfg.token_position),
                    mode="forward",
                    max_new_tokens=int(cfg.max_new_tokens),
                    progress=progress,
                )

            torch_mod.save(acts, acts_reason_path)
            torch_mod.save(gt_acts, acts_gt_path)

        # 2) MI tensor (cached/resumable)
        def _is_complete_mi(obj: Any, needed: int) -> bool:
            if not isinstance(obj, dict):
                return False
            for i in range(int(needed)):
                if i not in obj:
                    return False
                try:
                    if (obj[i] or {}).get("total_tokens", -1) <= 0:
                        return False
                except Exception:
                    return False
            return True

        loaded_mi = None
        if cfg.reuse_cache and mi_tensor_path.exists():
            try:
                loaded_mi = torch_mod.load(mi_tensor_path)
            except Exception:
                loaded_mi = None

        if _is_complete_mi(loaded_mi, run_n):
            mi_tensor = loaded_mi
        else:
            with self.progress(
                dataset=None,
                total=len(problems),
                desc="MI-Peaks diagnosis (MI computation)",
                progress_sink=kwargs.get("progress_sink"),
                on_start=kwargs.get("on_progress_start"),
                on_update=kwargs.get("on_progress_update"),
                on_done=kwargs.get("on_progress_done"),
            ) as progress:
                mi_tensor = _calculate_mi(
                    acts,
                    gt_acts,
                    layers=layers,
                    num_samples=len(problems),
                    save_path=mi_tensor_path,
                    sigma=float(cfg.sigma),
                    ktype=str(cfg.ktype),
                    reuse_cache=bool(cfg.reuse_cache),
                    progress=progress,
                )

        # 3) Derive diagnosis metrics (peaks + mean trajectory)
        target_layer = layers[0]
        seqs: List[List[float]] = []
        for idx in sorted(mi_tensor.keys()):
            try:
                mi_values_raw = mi_tensor[idx]["reps"][target_layer]
                if hasattr(mi_values_raw, "tolist"):
                    mi_values = [float(x) for x in mi_values_raw.tolist()]
                else:
                    mi_values = [float(x) for x in list(mi_values_raw)]

                # Option A (minimal divergence from original MI-Peaks):
                # Keep cached MI tensor unchanged, but drop the first point *only for reporting*
                # when it looks like the prompt-boundary extra rep is present.
                try:
                    token_len = int(acts[int(idx)]["token_ids"].shape[0])
                    if len(mi_values) == token_len + 1:
                        mi_values = mi_values[1:]
                except Exception:
                    pass

                seqs.append(mi_values)
            except Exception:
                continue

        mean_traj = _mean_ragged_trajectory(seqs)

        english_pattern = re.compile(r"^[a-zA-Z]+$")
        tokens_at_peaks: List[str] = []
        # Token-at-peak stats: iterate the same number of samples used for compute.
        for idx in range(min(len(problems), len(seqs))):
            try:
                mi_values = seqs[idx]
                token_ids = acts[idx]["token_ids"].tolist()
                # Guard: ensure MI indices are valid for token_ids.
                usable_len = min(len(mi_values), len(token_ids))
                mi_values = mi_values[:usable_len]
                top_indices = sorted(range(len(mi_values)), key=lambda i: mi_values[i], reverse=True)[: int(cfg.top_k)]
                token_ids = acts[idx]["token_ids"].tolist()
                # Ensure EOS for stable decoding. Do NOT hardcode eos_token_id=2:
                # Llama-3.x and other tokenizers can differ.
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is None:
                    eos_id = 2
                token_ids.append(int(eos_id))
                decoded = tokenizer.batch_decode([token_ids[i] for i in top_indices], skip_special_tokens=False)
                for tok in decoded:
                    tok = str(tok).strip()
                    if english_pattern.match(tok):
                        tokens_at_peaks.append(tok)
            except Exception:
                continue

        token_freq = Counter(tokens_at_peaks).most_common(int(cfg.top_tokens))

        metrics: Dict[str, Any] = {
            "dataset": dataset_tag,
            "model_tag": model_tag,
            "target_layer": int(target_layer),
            "mi_mean_trajectory": mean_traj,
            "thinking_tokens_top": [{"token": t, "count": int(c)} for t, c in token_freq],
            "charts": [{"type": "line", "data": mean_traj}],
        }

        return {
            "evaluator": {"id": "mi-peaks", "type": "mi-peaks"},
            "config": asdict(cfg),
            "num_samples": int(len(problems)),
            "metrics": metrics,
            "artifacts": {
                "artifact_root": str(root.resolve()),
                "acts_reasoning_path": str(acts_reason_path.resolve()),
                "acts_gt_path": str(acts_gt_path.resolve()),
                "mi_tensor_path": str(mi_tensor_path.resolve()),
            },
        }


# Auto-register evaluator
try:
    from llm_diagnose.evaluators.registry import get_evaluator_registry

    get_evaluator_registry().register_evaluator("mi-peaks")(MiPeaksEvaluator)
    get_evaluator_registry().register_evaluator("mi_peaks")(MiPeaksEvaluator)
except Exception:  # pragma: no cover
    logger.debug("Could not auto-register MiPeaksEvaluator with the registry.")

