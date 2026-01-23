"""
Generate dummy metrics for charts/leaderboards.

This repo already uses these "headline" metrics per diagnosis method:
- X-Boundary: 平均安全–有害分离度 (average separation_score across layers; L2 distance between centroids)
- TELLME: L2距离 (DistanceMetrics.euclidean_distance reported as "L2")
- SPIN: 公平-隐私神经元耦合指数 (fairness_privacy_neurons_coupling_ratio)

This script produces:
1) A small JSON mapping method -> metric value, for quick chart prototyping.
2) (Optional) A dummy leaderboard CSV with N models and the 3 columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List


def _float_range(rng: random.Random, lo: float, hi: float, ndigits: int = 2) -> float:
    return round(rng.uniform(lo, hi), ndigits)


def _sci_range(rng: random.Random, lo: float, hi: float) -> float:
    # Keep as float; downstream can format as scientific notation if needed.
    return rng.uniform(lo, hi)


def build_single_dummy_metrics(seed: int = 0) -> Dict[str, Any]:
    """
    Return a minimal dict for a single chart:
    X-Boundary / TELLME / SPIN → one scalar each.
    """
    rng = random.Random(seed)

    # Match typical magnitudes observed in `results/exp/...`:
    # - X-Boundary separation_score ~ 15-30
    # - TELLME L2 ~ 20-55
    # - SPIN coupling ratio ~ 1e-7 - 9e-7
    xboundary_avg_sep = _float_range(rng, 18.0, 28.0, ndigits=2)
    tellme_l2 = _float_range(rng, 28.0, 45.0, ndigits=2)
    spin_coupling = _sci_range(rng, 1.2e-7, 8.5e-7)
    spin_coupling_ln = math.log(spin_coupling) if spin_coupling > 0 else float("nan")

    return {
        "metrics_order": [
            "平均安全–有害分离度",
            "L2距离",
            "公平-隐私神经元耦合指数",
            "ln(公平-隐私神经元耦合指数)",
        ],
        "methods": {
            "X-Boundary": {"平均安全–有害分离度": xboundary_avg_sep},
            "TELLME": {"L2距离": tellme_l2},
            "SPIN": {
                "公平-隐私神经元耦合指数": spin_coupling,
                "ln(公平-隐私神经元耦合指数)": spin_coupling_ln,
            },
        },
        # Convenient flat list "respectively" (X-Boundary, TELLME, SPIN).
        "respectively": [xboundary_avg_sep, tellme_l2, spin_coupling],
    }

def _minmax(values: List[float]) -> List[float]:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean:
        return [0.0 for _ in values]
    vmin = min(clean)
    vmax = max(clean)
    if vmax - vmin < 1e-12:
        return [0.5 for _ in values]
    out: List[float] = []
    for v in values:
        if not isinstance(v, (int, float)) or math.isnan(v):
            out.append(0.0)
        else:
            out.append((float(v) - vmin) / (vmax - vmin))
    return out


def _load_model_names(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model names file not found: {p}")
    names: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        names.append(s)
    return names


def _infer_org(model_name: str) -> str:
    n = (model_name or "").lower()
    if "glm" in n:
        return "Z.AI"
    if "qwen" in n:
        return "Qwen"
    if "llama" in n:
        return "Meta"
    if "gemma" in n:
        return "Google"
    if "mistral" in n or "mixtral" in n:
        return "Mistral AI"
    if "internvl" in n:
        return "Shanghai AI Lab"
    if "baichuan" in n:
        return "Baichuan"
    if "internlm" in n:
        return "InternLM"
    if "gpt" in n:
        return "OpenAI"
    if "claude" in n:
        return "Anthropic"
    if "command" in n:
        return "Cohere"
    if "deepseek" in n:
        return "DeepSeek"
    if "yi" in n:
        return "01.AI"
    if "phi" in n:
        return "Microsoft"
    if "gemini" in n:
        return "Google"
    return "Unknown"


def _infer_region(org: str) -> str:
    if org in {"Z.AI", "Qwen", "Shanghai AI Lab", "Baichuan", "InternLM", "DeepSeek", "01.AI"}:
        return "中国"
    if org in {"Mistral AI"}:
        return "法国"
    # Default to US for most Western labs in this dummy dataset
    if org in {"Meta", "Google", "OpenAI", "Anthropic", "Cohere", "Microsoft"}:
        return "美国"
    return "未知"


def _infer_is_reasoning(model_name: str) -> str:
    # The screenshot shows most as "否", with occasional "是".
    # Use a simple heuristic so at least some rows are "是".
    n = (model_name or "").lower()
    if "r1" in n or "reason" in n:
        return "是"
    if "air" in n:
        return "是"
    return "否"


def _dummy_release_date(rng: random.Random) -> str:
    # YYYY-MM, roughly 2024-01 .. 2025-12 to resemble the screenshot.
    year = rng.choice([2024, 2025])
    month = rng.randint(1, 12)
    return f"{year:04d}-{month:02d}"


def build_dummy_leaderboard(
    n_models: int,
    seed: int = 0,
    model_names: List[str] | None = None,
    rank_mode: str = "score",  # score | input
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    xb_vals: List[float] = []
    tm_vals: List[float] = []
    spin_ln_vals: List[float] = []
    names = list(model_names or [])
    for i in range(int(n_models)):
        if i < len(names):
            model_name = names[i]
        else:
            model_name = f"DummyModel-{i+1:02d}"

        # Light variation with a gentle trend so plots look "real".
        trend = (i / max(1, n_models - 1)) - 0.5
        xboundary = _float_range(rng, 18.0 + 2.0 * trend, 28.0 + 2.0 * trend, ndigits=2)
        tellme_l2 = _float_range(rng, 28.0 - 3.0 * trend, 45.0 - 3.0 * trend, ndigits=2)
        spin = _sci_range(rng, 1.2e-7, 8.5e-7)
        spin_ln = math.log(spin) if spin > 0 else float("nan")

        xb_vals.append(float(xboundary))
        tm_vals.append(float(tellme_l2))
        spin_ln_vals.append(float(spin_ln))

        rows.append(
            {
                "model": model_name,
                "X-Boundary_avg_sep": xboundary,
                "TELLME_L2": tellme_l2,
                "SPIN_coupling_index": spin,
                "SPIN_coupling_ln": spin_ln,
            }
        )

    xb_n = _minmax(xb_vals)
    tm_n = _minmax(tm_vals)
    spin_n = _minmax(spin_ln_vals)
    for i in range(len(rows)):
        overall = (xb_n[i] + tm_n[i] + spin_n[i]) / 3.0
        rows[i]["overall_score"] = round(overall * 100.0, 2)

    if str(rank_mode).lower() == "input":
        # Preserve the provided model order as the rank order.
        rows_sorted = rows
    else:
        # Rank by overall_score (descending). Ties are broken by model name (stable).
        rows_sorted = sorted(rows, key=lambda r: (-float(r.get("overall_score", 0.0)), str(r.get("model", ""))))
    for idx, row in enumerate(rows_sorted, start=1):
        row["rank"] = idx

    # Add UI-ish metadata columns, and map to a "总分" scale resembling the screenshot.
    # Keep deterministic by using the same RNG seed.
    rng2 = random.Random(seed + 10007)
    scores = [float(r.get("overall_score", 0.0)) for r in rows_sorted]
    scores_n = _minmax(scores)
    n_rows = len(rows_sorted)
    for i, row in enumerate(rows_sorted):
        org = _infer_org(str(row.get("model", "")))
        region = _infer_region(org)
        is_reasoning = _infer_is_reasoning(str(row.get("model", "")))
        release_date = _dummy_release_date(rng2)
        if str(rank_mode).lower() == "input":
            # In input-rank mode, the row order *is* the rank order.
            # Therefore 总分 must be monotonic (descending) to match "rank by score".
            # We generate a normalized-looking score in a readable range (e.g., 100..60).
            hi, lo = 100.0, 60.0
            frac = 0.0 if n_rows <= 1 else float(i) / float(n_rows - 1)
            total_score = round(hi - (hi - lo) * frac, 1)
        else:
            # Map normalized overall score to a high range (e.g., 90.0 - 99.9) like a leaderboard.
            total_score = round(90.0 + 9.9 * float(scores_n[i]), 1)
        row["org"] = org
        row["region"] = region
        row["is_reasoning"] = is_reasoning
        row["release_date"] = release_date
        row["total_score"] = total_score
    return rows_sorted


def write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-json",
        default="results/dummy/dummy_diagnosis_metrics.json",
        help="Where to write the single-chart dummy JSON.",
    )
    ap.add_argument(
        "--out-csv",
        default="results/dummy/dummy_leaderboard_metrics.csv",
        help="Where to write the leaderboard CSV (set empty to skip).",
    )
    ap.add_argument("--n-models", type=int, default=40, help="Number of dummy models to generate for the CSV.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--model-names",
        default="",
        help="Optional path to a newline-separated model name list; used to replace the `model` column values.",
    )
    ap.add_argument(
        "--ui-csv",
        action="store_true",
        help="Write the CSV in the same column order as the leaderboard screenshot (Chinese headers).",
    )
    ap.add_argument(
        "--rank-mode",
        default="score",
        choices=["score", "input"],
        help="How to rank rows: 'score' sorts by overall_score desc; 'input' keeps model-names order as rank.",
    )
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = build_single_dummy_metrics(seed=int(args.seed))
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if str(args.out_csv).strip():
        out_csv = Path(args.out_csv)
        model_names = _load_model_names(args.model_names) if str(args.model_names).strip() else None
        n_models = int(args.n_models)
        if model_names:
            n_models = min(n_models, len(model_names))
        rows = build_dummy_leaderboard(
            n_models=n_models,
            seed=int(args.seed),
            model_names=model_names,
            rank_mode=str(args.rank_mode),
        )
        if args.ui_csv:
            # Match screenshot-like column order:
            # 模型 | 机构 | 总分 | 属地 | 是否推理 | 发布日期 | X-Boundary | TELLME | SPIN
            ui_rows: List[Dict[str, Any]] = []
            for r in rows:
                ui_rows.append(
                    {
                        "模型": r.get("model"),
                        "机构": r.get("org"),
                        "总分": r.get("total_score"),
                        "属地": r.get("region"),
                        "是否推理": r.get("is_reasoning"),
                        "发布日期": r.get("release_date"),
                        "X-Boundary": r.get("X-Boundary_avg_sep"),
                        "TELLME": r.get("TELLME_L2"),
                        # Use ln so the magnitude is readable.
                        "SPIN": r.get("SPIN_coupling_ln"),
                    }
                )
            write_csv(
                out_csv,
                ui_rows,
                fieldnames=["模型", "机构", "总分", "属地", "是否推理", "发布日期", "X-Boundary", "TELLME", "SPIN"],
            )
        else:
            write_csv(out_csv, rows)


if __name__ == "__main__":
    main()

