"""
SPIN diagnostic dataset loader.

For exact reproduction with the SPIN reference code, we prefer to keep this loader
"thin" and pass paths through to the evaluator, because the reference repo uses
Hugging Face `datasets` for deterministic `shuffle(seed).select(range(nsamples))`
sampling on CSV inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_diagnose.registry.dataset_registry import get_dataset_registry


@dataclass(frozen=True)
class SpinCsvBundle:
    dataset1_path: str
    dataset2_path: str
    general_path: str
    dataset1_name: str = "dataset1"
    dataset2_name: str = "dataset2"
    general_name: str = "general"
    max_rows: Optional[int] = None


def _read_prompt_response_csv(path: str, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Lightweight CSV reader fallback (only used when the evaluator wants pre-loaded rows).
    """
    try:
        import csv
    except Exception as exc:  # pragma: no cover
        raise ImportError("CSV loader requires Python's stdlib csv module.") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SPIN CSV not found: {p}")

    items: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"SPIN CSV has no header row: {p}")
        if "prompt" not in reader.fieldnames or "response" not in reader.fieldnames:
            raise ValueError(
                f"SPIN CSV must contain 'prompt' and 'response' columns. "
                f"Got columns={reader.fieldnames!r} in {p}"
            )

        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            response = (row.get("response") or "").strip()
            if not prompt or not response:
                continue
            items.append({"prompt": prompt, "response": response})
            if max_rows is not None and len(items) >= int(max_rows):
                break

    if not items:
        raise ValueError(f"SPIN CSV produced 0 usable prompt/response rows: {p}")
    return items


def load_spin_csv_bundle(
    *,
    dataset1_path: str,
    dataset2_path: str,
    general_path: str,
    dataset1_name: str = "dataset1",
    dataset2_name: str = "dataset2",
    general_name: str = "general",
    max_rows: Optional[int] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Load 3 CSVs for SPIN-style diagnosis:
    - dataset1: e.g. privacy
    - dataset2: e.g. fairness
    - general: general capability baseline (e.g. Alpaca)
    """

    bundle = SpinCsvBundle(
        dataset1_path=str(dataset1_path),
        dataset2_path=str(dataset2_path),
        general_path=str(general_path),
        dataset1_name=str(dataset1_name),
        dataset2_name=str(dataset2_name),
        general_name=str(general_name),
        max_rows=max_rows,
    )

    return {
        "type": "spin/csv_bundle",
        "bundle": bundle,
        # Pass through paths; the evaluator will perform SPIN-style deterministic
        # sampling using HF `datasets` where available.
        "paths": {
            "dataset1": bundle.dataset1_path,
            "dataset2": bundle.dataset2_path,
            "general": bundle.general_path,
        },
        # Backward compatible optional preload (off by default).
        "preloaded": None,
    }


def register_spin_dataset() -> None:
    registry = get_dataset_registry()
    registry.register_dataset(
        "spin/csv_bundle",
        dataset_family="spin",
        dataset_split="diagnostic",
        description="SPIN CSV bundle (dataset1 + dataset2 + general), each with prompt/response columns.",
    )(load_spin_csv_bundle)


