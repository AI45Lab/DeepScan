#!/usr/bin/env python3
"""
Summarize experiment results under `results/exp/` to CSV files.

Writes 1 CSV per evaluator (e.g. tellme.csv, spin.csv, x-boundary.csv).

The first columns match the requested schema:
  model name, release month and year, organization, country,
  open-source statues, reasoning capability, metric 1, metric2, ...

Model metadata is joined from a YAML mapping (optional).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import yaml


BASE_COLS = [
    "model name",
    "release month and year",
    "organization",
    "country",
    "open-source statues",
    "reasoning capability",
]


@dataclass(frozen=True)
class ModelInfo:
    run_name: str
    model_name: str


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"metadata YAML must be a mapping, got: {type(data)}")
            return data
    except FileNotFoundError:
        return {}


def _sanitize_filename(name: str) -> str:
    # Keep it readable; avoid path separators and special chars.
    name = name.strip().replace(os.sep, "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _looks_like_path(s: str) -> bool:
    # Cheap heuristic: absolute/relative paths or common artifact suffixes.
    if "/" in s or "\\" in s:
        return True
    lower = s.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".pdf", ".json", ".md"))


def _flatten(
    obj: Any,
    prefix: str = "",
    *,
    out: Optional[MutableMapping[str, Any]] = None,
    skip_keys: Tuple[str, ...] = ("artifacts",),
) -> Dict[str, Any]:
    """
    Flattens nested dicts into dot-keys. Keeps scalars; encodes lists as JSON strings.
    Skips any subtree whose key is in `skip_keys`.
    """
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(v, key, out=out, skip_keys=skip_keys)
        return dict(out)

    if isinstance(obj, list):
        # Keep list values but store as a stable JSON string.
        out[prefix] = json.dumps(obj, ensure_ascii=False)
        return dict(out)

    if _is_scalar(obj):
        # Avoid polluting CSV with long artifact-like strings outside 'artifacts'.
        if isinstance(obj, str) and _looks_like_path(obj) and (
            "artifact" in prefix.lower() or "path" in prefix.lower() or "dir" in prefix.lower()
        ):
            return dict(out)
        out[prefix] = obj
        return dict(out)

    # Fallback: stringify unknown objects
    out[prefix] = str(obj)
    return dict(out)


def _load_models_from_config_resolved(config_path: Path) -> List[ModelInfo]:
    cfg = _read_json(config_path)
    if not cfg:
        return []
    models: List[ModelInfo] = []
    for m in cfg.get("model", []) or []:
        if not isinstance(m, dict):
            continue
        run_name = str(m.get("run_name") or "").strip()
        model_name = str(m.get("model_name") or "").strip()
        if run_name:
            models.append(ModelInfo(run_name=run_name, model_name=model_name or run_name))
    return models


def _guess_models_from_dirs(exp_dir: Path) -> List[ModelInfo]:
    """
    Fallback: if config.resolved.json is missing, infer model runs from subdirectories
    that contain a `results.json`.
    """
    models: List[ModelInfo] = []
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        res = _read_json(child / "results.json")
        if not res:
            continue
        # Prefer canonical sources if present; otherwise use folder name.
        run_name = (
            (res.get("model") or {}).get("config", {}).get("run_name")
            or (res.get("model") or {}).get("config", {}).get("model_name")
            or child.name
        )
        model_name = (res.get("model") or {}).get("model_name") or child.name
        models.append(ModelInfo(run_name=str(run_name), model_name=str(model_name)))
    return models


def _find_model_dir(exp_dir: Path, run_name: str) -> Optional[Path]:
    direct = exp_dir / run_name
    if direct.is_dir():
        return direct
    # fallback: locate by inspecting results.json files
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        res = _read_json(child / "results.json")
        if not res:
            continue
        rn = (res.get("model") or {}).get("config", {}).get("run_name")
        if rn == run_name:
            return child
    return None


def _evaluator_id_from_summary(summary: Mapping[str, Any], fallback: str) -> str:
    ev = summary.get("evaluator")
    if isinstance(ev, str):
        return ev
    if isinstance(ev, dict):
        if isinstance(ev.get("id"), str) and ev["id"].strip():
            return ev["id"].strip()
        if isinstance(ev.get("type"), str) and ev["type"].strip():
            return ev["type"].strip()
    return fallback


def _extract_metrics_from_summary(summary_json: Mapping[str, Any], run_name: str) -> Dict[str, Any]:
    """
    Returns a flat dict of metrics for this evaluator/model.
    """
    summary = summary_json.get("summary")
    if not isinstance(summary, dict):
        return {}
    entry = summary.get(run_name)
    if not isinstance(entry, dict):
        # Some summaries may key by model_name; try the single entry if unambiguous.
        if len(summary) == 1:
            only = next(iter(summary.values()))
            if isinstance(only, dict):
                entry = only
            else:
                return {}
        else:
            return {}

    # Special-case SPIN: make top_layers_by_coupled human-friendly + add ratios.
    if isinstance(entry.get("top_layers_by_coupled"), list):
        try:
            pairs = entry["top_layers_by_coupled"]
            if all(isinstance(p, list) and len(p) == 2 for p in pairs):
                entry = dict(entry)
                entry["top_layers_by_coupled"] = ";".join(f"{p[0]}:{p[1]}" for p in pairs)
        except Exception:
            pass

    flat = _flatten(entry)

    # SPIN: add coupled ratios.
    # We only have candidate counts (per dataset) in the summary, so define:
    # - coupled_over_candidate_mean = coupled / mean(candidate_dataset1, candidate_dataset2)
    # - coupled_over_candidate_union = coupled / (candidate_dataset1 + candidate_dataset2 - coupled)
    try:
        coupled = flat.get("totals.coupled")
        cand1 = flat.get("totals.candidate_dataset1")
        cand2 = flat.get("totals.candidate_dataset2")
        if isinstance(coupled, (int, float)) and isinstance(cand1, (int, float)) and isinstance(
            cand2, (int, float)
        ):
            denom_mean = (cand1 + cand2) / 2.0
            denom_union = cand1 + cand2 - coupled
            if denom_mean > 0:
                flat["totals.coupled_over_candidate_mean"] = coupled / denom_mean
            if denom_union > 0:
                flat["totals.coupled_over_candidate_union"] = coupled / denom_union
    except Exception:
        pass

    # SPIN: log-scale extremely small ratios for readability.
    try:
        r = flat.get("totals.fairness_privacy_neurons_coupling_ratio")
        if isinstance(r, (int, float)) and r > 0:
            flat["totals.ln_fairness_privacy_neurons_coupling_ratio"] = math.log(float(r))
            flat["totals.log10_fairness_privacy_neurons_coupling_ratio"] = math.log10(float(r))
    except Exception:
        pass

    # Remove any empty keys (shouldn't happen, but be safe)
    flat.pop("", None)
    return flat


def _mean_numeric_leaves_across_layers(per_layer: Mapping[str, Any]) -> Dict[str, float]:
    """
    Given {layer: metrics_dict}, compute mean for each numeric leaf key across layers.
    Returns flattened keys (e.g. "separation_score", "details.dist_bound_safe").
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for _, metrics in per_layer.items():
        if not isinstance(metrics, dict):
            continue
        flat = _flatten(metrics)
        for k, v in flat.items():
            if isinstance(v, bool):
                # Don't average booleans.
                continue
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

    means: Dict[str, float] = {}
    for k, s in sums.items():
        c = counts.get(k, 0)
        if c > 0:
            means[k] = s / c
    return means


def _xboundary_avg_metrics_for_model(evaluator_dir: Path) -> Dict[str, Any]:
    """
    X-Boundary: compute average of each metric across all evaluated layers.
    Prefers `metrics_summary.json` (if referenced in summary.json), otherwise uses evaluator results.json.
    """
    # 1) Try metrics_summary.json path from summary.json artifacts
    summary_json = _read_json(evaluator_dir / "summary.json") or {}
    artifacts = None
    if isinstance(summary_json.get("summary"), dict) and len(summary_json["summary"]) >= 1:
        # artifacts live under summary[run_name].artifacts, but run_name isn't available here.
        # So scan entries and take the first artifacts block that contains metrics_summary_json.
        for v in summary_json["summary"].values():
            if isinstance(v, dict) and isinstance(v.get("artifacts"), dict):
                artifacts = v["artifacts"]
                break
    metrics_path = None
    if isinstance(artifacts, dict) and isinstance(artifacts.get("metrics_summary_json"), str):
        metrics_path = Path(artifacts["metrics_summary_json"])
        if not metrics_path.is_absolute():
            metrics_path = (evaluator_dir / metrics_path).resolve()

    if metrics_path and metrics_path.exists():
        ms = _read_json(metrics_path)
        if isinstance(ms, dict):
            means = _mean_numeric_leaves_across_layers(ms)
            return {f"avg.{k}": v for k, v in means.items()}

    # 2) Fallback to evaluator results.json
    r = _read_json(evaluator_dir / "results.json") or {}
    metrics = (r.get("results") or {}).get("metrics")
    if isinstance(metrics, dict):
        per_layer = metrics.get("per_layer") or metrics.get("metrics_by_layer")
        if isinstance(per_layer, dict):
            means = _mean_numeric_leaves_across_layers(per_layer)
            return {f"avg.{k}": v for k, v in means.items()}

    return {}


def _extract_metrics_fallback_from_results(model_dir: Path, run_name: str) -> Dict[str, Any]:
    """
    If `summary.json` is missing, try evaluator `results.json` and flatten the `results.metrics`.
    """
    out: Dict[str, Any] = {}
    # evaluator dirs are children that contain results.json
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        r = _read_json(child / "results.json")
        if not r:
            continue
        # Ensure this is the right model
        rn = (r.get("model") or {}).get("config", {}).get("run_name")
        if rn and rn != run_name:
            continue
        metrics = (r.get("results") or {}).get("metrics")
        if isinstance(metrics, dict):
            out.update({f"results.metrics.{k}": v for k, v in _flatten(metrics).items()})
    return out


def _get_model_metadata(
    metadata: Mapping[str, Any],
    *,
    run_name: str,
    model_name: str,
) -> Dict[str, str]:
    """
    Joins optional metadata. Returns normalized dict for the BASE_COLS (excluding model name).
    """
    candidates = [run_name, model_name]
    record: Optional[Mapping[str, Any]] = None
    for key in candidates:
        v = metadata.get(key)
        if isinstance(v, dict):
            record = v
            break

    def _normalize_yes_no(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, float)):
            if v == 1:
                return "yes"
            if v == 0:
                return "no"
        s = str(v).strip().lower()
        if s in {"yes", "y", "true", "t", "1", "open", "opensource", "open-source"}:
            return "yes"
        if s in {"no", "n", "false", "f", "0", "closed", "proprietary"}:
            return "no"
        # If user puts something else, keep it (but they asked for yes/no).
        return str(v).strip()

    def get_str(field: str) -> str:
        if not record:
            return ""
        v = record.get(field)
        return "" if v is None else str(v)

    return {
        "release month and year": get_str("release_month_year"),
        "organization": get_str("organization"),
        "country": get_str("country"),
        "open-source statues": _normalize_yes_no(record.get("open_source_status") if record else None),
        "reasoning capability": _normalize_yes_no(
            record.get("reasoning_capability") if record else None
        ),
    }


def summarize(exp_root: Path, out_dir: Path, metadata_path: Path) -> None:
    metadata = _read_yaml(metadata_path)

    rows_by_eval: Dict[str, List[Dict[str, Any]]] = {}
    missing_meta: List[Tuple[str, str]] = []

    exp_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    for exp_dir in exp_dirs:
        config_path = exp_dir / "config.resolved.json"
        models = _load_models_from_config_resolved(config_path)
        if not models:
            models = _guess_models_from_dirs(exp_dir)

        for m in models:
            model_dir = _find_model_dir(exp_dir, m.run_name)
            if not model_dir:
                continue

            base = {"model name": m.model_name}
            base.update(_get_model_metadata(metadata, run_name=m.run_name, model_name=m.model_name))
            if not any(base.get(k, "") for k in BASE_COLS[1:]):
                missing_meta.append((m.run_name, m.model_name))

            # For each evaluator directory, prefer `summary.json`.
            evaluator_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
            for ev_dir in evaluator_dirs:
                summary_path = ev_dir / "summary.json"
                if summary_path.exists():
                    sj = _read_json(summary_path) or {}
                    ev_id = _evaluator_id_from_summary(sj, fallback=ev_dir.name)
                    metrics = _extract_metrics_from_summary(sj, m.run_name)
                else:
                    # fallback to evaluator results.json (less standardized)
                    ev_id = ev_dir.name
                    metrics = _extract_metrics_fallback_from_results(model_dir, m.run_name)
                    if not metrics:
                        continue

                # X-Boundary: add average metrics across evaluated layers.
                if ev_id in {"x-boundary", "xboundary"}:
                    metrics.update(_xboundary_avg_metrics_for_model(ev_dir))

                row = dict(base)
                row.update(metrics)
                rows_by_eval.setdefault(ev_id, []).append(row)

    out_dir.mkdir(parents=True, exist_ok=True)

    for ev_id, rows in rows_by_eval.items():
        # Union all metric keys (exclude BASE_COLS)
        metric_keys = sorted({k for r in rows for k in r.keys() if k not in BASE_COLS})
        header = BASE_COLS + metric_keys
        out_path = out_dir / f"{_sanitize_filename(ev_id)}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                # Ensure all base columns exist
                for k in BASE_COLS:
                    r.setdefault(k, "")
                writer.writerow(r)

    # ----------------------------
    # Filtered/processed CSVs
    # ----------------------------
    # Per-evaluator metric selection (output_column -> source_key_in_row)
    filtered_specs: Dict[str, List[Tuple[str, str]]] = {
        "x-boundary": [
            ("avg.separation_score", "avg.separation_score"),
            ("avg.boundary_ratio", "avg.boundary_ratio"),
            ("avg.details.dist_bound_safe", "avg.details.dist_bound_safe"),
            ("avg.details.dist_bound_harmful", "avg.details.dist_bound_harmful"),
        ],
        "tellme": [
            ("L2", "metrics.L2"),
            ("L1", "metrics.L1"),
            ("cos_sim", "metrics.cos_sim"),
            ("hausdorff", "metrics.hausdorff"),
            ("R_same", "metrics.R_same"),
            ("R_diff", "metrics.R_diff"),
            ("R_gap", "metrics.R_gap"),
            ("e_rank", "metrics.erank"),
        ],
        "spin": [
            (
                "ln_fairness_privacy_neurons_coupling_ratio",
                "totals.ln_fairness_privacy_neurons_coupling_ratio",
            )
        ],
    }

    for ev_id, spec in filtered_specs.items():
        rows = rows_by_eval.get(ev_id, [])
        if not rows:
            continue
        header = BASE_COLS + [out_k for out_k, _ in spec]
        out_path = out_dir / f"{_sanitize_filename(ev_id)}_filtered.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                row_out: Dict[str, Any] = {k: r.get(k, "") for k in BASE_COLS}
                for out_k, src_k in spec:
                    row_out[out_k] = r.get(src_k, "")
                w.writerow(row_out)

    # Aggregated CSV (one row per model) using user-selected "main" metrics.
    # Keys are the column names written in the per-evaluator CSVs.
    selected: Dict[str, Tuple[str, str]] = {
        "tellme": ("tellme.metrics.L2", "metrics.L2"),
        "spin": (
            "spin.totals.ln_fairness_privacy_neurons_coupling_ratio",
            "totals.ln_fairness_privacy_neurons_coupling_ratio",
        ),
        "x-boundary": ("x-boundary.avg.separation_score", "avg.separation_score"),
        "xboundary": ("x-boundary.avg.separation_score", "avg.separation_score"),
    }

    # Build a stable model index.
    by_model: Dict[str, Dict[str, Any]] = {}
    for ev_id, rows in rows_by_eval.items():
        for r in rows:
            model = str(r.get("model name") or "").strip()
            if not model:
                continue
            agg = by_model.setdefault(model, {k: r.get(k, "") for k in BASE_COLS})
            # Prefer filling missing base fields if other evaluator has it.
            for k in BASE_COLS:
                if not agg.get(k) and r.get(k):
                    agg[k] = r.get(k)

            if ev_id in selected:
                out_key, src_key = selected[ev_id]
                agg[out_key] = r.get(src_key, "")

    aggregate_header = BASE_COLS + [
        "tellme.metrics.L2",
        "spin.totals.ln_fairness_privacy_neurons_coupling_ratio",
        "x-boundary.avg.separation_score",
    ]
    aggregate_path = out_dir / "aggregate_main_metrics.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=aggregate_header, extrasaction="ignore")
        w.writeheader()
        for model in sorted(by_model.keys()):
            row = by_model[model]
            for k in aggregate_header:
                row.setdefault(k, "")
            w.writerow(row)

    # Filtered aggregate (uses the requested metric names, in a compact form)
    filtered_agg_header = BASE_COLS + [
        "L2",
        "ln_fairness_privacy_neurons_coupling_ratio",
        "avg.separation_score",
    ]
    filtered_agg_path = out_dir / "aggregate_main_metrics_filtered.csv"
    with filtered_agg_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=filtered_agg_header, extrasaction="ignore")
        w.writeheader()
        for model in sorted(by_model.keys()):
            src = by_model[model]
            row = {k: src.get(k, "") for k in BASE_COLS}
            row["L2"] = src.get("tellme.metrics.L2", "")
            row["ln_fairness_privacy_neurons_coupling_ratio"] = src.get(
                "spin.totals.ln_fairness_privacy_neurons_coupling_ratio", ""
            )
            row["avg.separation_score"] = src.get("x-boundary.avg.separation_score", "")
            w.writerow(row)

    if missing_meta:
        uniq = sorted(set(missing_meta))
        print(
            f"[warn] Missing model metadata for {len(uniq)} model(s). "
            f"Fill `release_month_year/organization/country/open_source_status/reasoning_capability` in: {metadata_path}"
        )
        for run_name, model_name in uniq:
            print(f"  - key candidates: {run_name!r} or {model_name!r}")

    print(f"[ok] Wrote {len(rows_by_eval)} CSV file(s) + aggregate to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-root",
        default="results/exp",
        help="Root directory containing exp runs (default: results/exp).",
    )
    ap.add_argument(
        "--out-dir",
        default="results/exp_csv",
        help="Output directory for per-evaluator CSV files (default: results/exp_csv).",
    )
    ap.add_argument(
        "--metadata",
        default="tools/model_metadata.yaml",
        help="YAML file mapping model run_name/model_name -> metadata fields.",
    )
    args = ap.parse_args()

    summarize(Path(args.exp_root), Path(args.out_dir), Path(args.metadata))


if __name__ == "__main__":
    main()

