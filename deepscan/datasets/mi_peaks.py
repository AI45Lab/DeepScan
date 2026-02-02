"""
MI-Peaks dataset loaders.

MI-Peaks expects a CSV with at least:
- problem: prompt text to generate reasoning from
- solution: ground-truth solution text (used as MI reference via forward pass)

This module registers:
- `mi-peaks/csv` (generic): provide `csv_path`
- `mi-peaks/math_train_12k` (default): attempts to locate MI-Peaks' bundled CSV
  in this workspace, or accepts overrides via `data_root`/`csv_path`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepscan.registry.dataset_registry import get_dataset_registry

logger = logging.getLogger(__name__)


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MI-Peaks CSV loading requires `pandas`. Install with `pip install pandas` "
            "(or install the framework extra: `pip install -e '.[mi_peaks]'`)."
        ) from exc
    return pd


def _find_default_mi_peaks_data_root() -> Optional[Path]:
    """
    Best-effort locator for the MI-Peaks repo in this workspace.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "dataset" / "mi_peaks",
        Path("/root/code/MI-Peaks/src/data"),
        Path.cwd() / "MI-Peaks" / "src" / "data",
        Path.cwd().parent / "MI-Peaks" / "src" / "data",
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


@dataclass
class MiPeaksDataset:
    """
    Lightweight dataset container to make the evaluator contract explicit.
    """

    dataset: str
    items: List[Dict[str, str]]
    config: Dict[str, Any]


def _load_mi_peaks_csv(
    *,
    csv_path: str,
    sample_num: Optional[int] = None,
    problem_column: str = "problem",
    solution_column: str = "solution",
    dataset_name: str = "mi-peaks/csv",
    **_: Any,
) -> Dict[str, Any]:
    pd = _require_pandas()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"MI-Peaks CSV not found at {path}")

    df = pd.read_csv(path)
    required = {problem_column, solution_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"MI-Peaks CSV missing required columns: {sorted(missing)} "
            f"(found: {sorted(df.columns)})"
        )

    if sample_num is not None and int(sample_num) > 0:
        df = df.head(int(sample_num))

    items = [
        {"problem": str(p), "solution": str(s)}
        for p, s in zip(df[problem_column].tolist(), df[solution_column].tolist())
    ]
    return {
        "dataset": dataset_name,
        "items": items,
        "problems": [row["problem"] for row in items],
        "solutions": [row["solution"] for row in items],
        "config": {
            "csv_path": str(path),
            "sample_num": sample_num,
            "problem_column": problem_column,
            "solution_column": solution_column,
        },
    }


def _load_math_train_12k(
    *,
    csv_path: Optional[str] = None,
    data_root: Optional[str] = None,
    sample_num: Optional[int] = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Mirror MI-Peaks default dataset pathing and truncation behavior.
    """
    if csv_path:
        path = Path(csv_path)
    else:
        root = Path(data_root) if data_root else _find_default_mi_peaks_data_root()
        if root is None:
            raise FileNotFoundError(
                "Could not locate MI-Peaks `src/data/`. Provide `data_root` or `csv_path` "
                "for dataset `mi-peaks/math_train_12k`."
            )
        path = root / "math_train_12k.csv"

    return _load_mi_peaks_csv(
        csv_path=str(path),
        sample_num=sample_num,
        dataset_name="math_train_12k",
        **kwargs,
    )


def register_mi_peaks_datasets() -> None:
    registry = get_dataset_registry()

    registry.register_dataset(
        "mi-peaks/csv",
        factory=_load_mi_peaks_csv,
        dataset_family="mi-peaks",
        dataset_split="custom",
        description="MI-Peaks CSV dataset with 'problem' and 'solution' columns.",
        required_files=["csv_path"],
        optional_files=["sample_num", "problem_column", "solution_column"],
        expected_columns=["problem", "solution"],
    )

    registry.register_dataset(
        "mi-peaks/math_train_12k",
        factory=_load_math_train_12k,
        dataset_family="mi-peaks",
        dataset_split="default",
        description="MI-Peaks math_train_12k dataset (CSV), mirroring the reference repo defaults.",
        required_files=[],
        optional_files=["csv_path", "data_root", "sample_num", "problem_column", "solution_column"],
        expected_columns=["problem", "solution"],
    )

    logger.info("Registered MI-Peaks dataset loaders")

