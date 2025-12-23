"""
TELLME dataset registration helpers.

This registers a lightweight CSV-based loader for the filtered BeaverTails split
used by TELLME. The loader expects the CSV to contain at least the following
columns (when using local CSV inputs):

- prompt: user prompt text
- response: model response text
- is_safe: integer label (1 safe, 0 unsafe)

It also supports an optional `raw` mode that pulls the original dataset from
Hugging Face (or a local HF cache path). This keeps parity with other diagnosis
methods that operate directly on HF datasets before preprocessing. In `raw`
mode, column validation is skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from llm_diagnose.registry.dataset_registry import get_dataset_registry

logger = logging.getLogger(__name__)

TELLME_DATASET_NAME = "tellme/beaver_tails_filtered"
DEFAULT_HF_ID = "PKU-Alignment/BeaverTails"

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TELLME CSV loading requires `pandas`. Install with `pip install pandas` "
            "(or install the framework extra: `pip install -e '.[tellme]'`)."
        ) from exc
    return pd


def _load_csv(path: str, max_rows: Optional[int] = None):
    pd = _require_pandas()
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"TELLME CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"prompt", "response", "is_safe"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"TELLME CSV missing required columns: {sorted(missing)} "
            f"(found: {sorted(df.columns)})"
        )

    if max_rows:
        df = df.head(max_rows)

    return df


def _load_tellme_dataset(
    test_path: str,
    train_path: Optional[str] = None,
    max_rows: Optional[int] = None,
    hf_id: str = DEFAULT_HF_ID,
    hf_split: str = "test",
    raw: bool = False,
    hf_cache_dir: Optional[str] = None,
    **_: Any,
) -> Dict[str, pd.DataFrame]:
    """
    Load filtered BeaverTails CSVs used by TELLME.

    Args:
        test_path: Path to the filtered test CSV (required unless raw=True).
        train_path: Optional path to a filtered train CSV.
        max_rows: Optional cap on rows (applied to both splits).
        hf_id: Hugging Face dataset id for raw loading (when raw=True).
        hf_split: Split name for raw loading.
        raw: If True, load raw HF dataset (no column validation) for downstream preprocessing.
        hf_cache_dir: Optional cache dir passed to datasets.load_dataset.

    Returns:
        Dict containing splits: {"test": DataFrame, "train": DataFrame?} for CSV mode,
        or {"raw": datasets.Dataset} for raw HF mode.
    """
    if raw:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "The 'datasets' library is required for raw HF loading. Install with `pip install datasets`."
            ) from exc

        ds = load_dataset(hf_id, split=hf_split, cache_dir=hf_cache_dir)
        return {"raw": ds}

    dataset: Dict[str, Any] = {"test": _load_csv(test_path, max_rows=max_rows)}
    if train_path:
        dataset["train"] = _load_csv(train_path, max_rows=max_rows)

    logger.info(
        "Loaded TELLME dataset (test=%d rows%s)",
        len(dataset["test"]),
        f", train={len(dataset['train'])}" if "train" in dataset else "",
    )
    return dataset


def register_tellme_dataset() -> None:
    """
    Register the filtered BeaverTails CSV loader under the dataset registry.
    """
    registry = get_dataset_registry()

    registry.register_dataset(
        TELLME_DATASET_NAME,
        factory=_load_tellme_dataset,
        dataset_family="tellme",
        dataset_split="test",
        description="Filtered BeaverTails CSV used by TELLME metrics.",
        required_files=["test_path"],
        optional_files=["train_path", "hf_id", "hf_split", "hf_cache_dir", "raw"],
        expected_columns=["prompt", "response", "is_safe"],
    )

    logger.info("Registered TELLME filtered BeaverTails dataset loader")


