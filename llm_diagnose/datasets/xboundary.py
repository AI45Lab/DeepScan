"""
X-Boundary diagnostic dataset loader.

This dataset mirrors the sampling logic in the X-Boundary repository:
- Harmful (Erase): `circuit_breakers_train_2400.json` (prompt/output)
- Safe (Retain): UltraChat (local `datasets` disk cache preferred)
- Boundary-Safe: `ORbench_retain_set.json` filtered to `status == "1_full_compliance"`

Unlike the original X-Boundary script (which bakes a tokenizer into the Dataset),
the framework dataset returns **chat messages** + labels. The evaluator is
responsible for applying the model tokenizer's chat template.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_diagnose.registry.dataset_registry import get_dataset_registry

logger = logging.getLogger(__name__)

XBOUNDARY_DATASET_NAME = "xboundary/diagnostic"


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"X-Boundary file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_ultrachat_messages(
    *,
    local_path: Path,
    hf_id: str,
    hf_split: str,
    num_samples: int,
    hf_cache_dir: Optional[str] = None,
    hf_revision: Optional[str] = None,
) -> List[List[Dict[str, str]]]:
    """
    Load UltraChat chat messages using `datasets` if available.
    Returns list of `messages`, where each element is `[{"role":..., "content":...}, ...]`.
    """
    try:
        from datasets import load_dataset, load_from_disk  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "X-Boundary safe set requires `datasets`. Install with `pip install datasets` "
            "or provide `safe_messages_path` as a JSONL/JSON file via your own dataset loader."
        ) from exc

    if local_path.exists():
        logger.info("Loading UltraChat safe set from disk: %s", str(local_path))
        try:
            # Preferred: a real `datasets.save_to_disk()` directory
            ds = load_from_disk(str(local_path))
        except Exception as exc_disk:  # broader than FileNotFoundError; local arrow may be incomplete
            logger.warning(
                "Failed to load local dataset with `load_from_disk` (%s); trying Arrow shards or HF Hub.",
                exc_disk,
            )
            # Common alternative: raw Arrow shards (e.g., `data-00000-of-00001.arrow`)
            arrow_files = sorted(local_path.glob("*.arrow"))
            if arrow_files:
                logger.info(
                    "Local path is not a datasets save_to_disk directory; loading Arrow shards: %s",
                    ", ".join([f.name for f in arrow_files]),
                )
                try:
                    ds = load_dataset(
                        "arrow",
                        data_files=[str(p) for p in arrow_files],
                        split="train",
                        cache_dir=hf_cache_dir,
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning(
                        "Failed to load local Arrow shards (%s); falling back to HF Hub: %s (%s)",
                        exc,
                        hf_id,
                        hf_split,
                    )
                    ds = load_dataset(
                        hf_id,
                        split=hf_split,
                        cache_dir=hf_cache_dir,
                        revision=hf_revision,
                    )
            else:
                logger.info("No Arrow shards found; falling back to HF Hub: %s (%s)", hf_id, hf_split)
                ds = load_dataset(hf_id, split=hf_split, cache_dir=hf_cache_dir, revision=hf_revision)
    else:
        logger.info("Loading UltraChat safe set from HF: %s (%s)", hf_id, hf_split)
        ds = load_dataset(hf_id, split=hf_split, cache_dir=hf_cache_dir, revision=hf_revision)

    def stable_messages_id(messages: List[Dict[str, str]]) -> str:
        normalized = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages]
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # Deterministic selection: keep N smallest content-hashes.
    # This ensures HF vs local Arrow copies use the same records (assuming identical content).
    heap: List[tuple[int, str, List[Dict[str, str]]]] = []  # entries: (-key_int, hash_hex, messages)
    for example in ds:
        messages = example.get("messages")
        if not messages or not isinstance(messages, list) or len(messages) < 2:
            continue
        hid = stable_messages_id(messages)
        key = int(hid, 16)
        entry = (-key, hid, messages)
        if len(heap) < num_samples:
            heapq.heappush(heap, entry)
        else:
            heapq.heappushpop(heap, entry)

    heap.sort(key=lambda x: x[1])
    return [messages for _, __, messages in heap]


def _load_xboundary_dataset(
    *,
    data_dir: Optional[str] = None,
    num_samples_per_class: int = 200,
    harmful_file: str = "circuit_breakers_train_2400.json",
    boundary_file: str = "ORbench_retain_set.json",
    ultrachat_local_dir: str = "ultrachat_200k_local",
    ultrachat_hf_id: str = "HuggingFaceH4/ultrachat_200k",
    ultrachat_hf_split: str = "test_sft",
    ultrachat_hf_revision: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "items": [{"messages": [...], "label": int, "source": str}, ...],
          "label_map": {0: "...", 1: "...", 2: "..."},
          "config": {...}
        }
    """
    if num_samples_per_class < 50 or num_samples_per_class > 400:
        raise ValueError("num_samples_per_class must be between 50 and 400 (X-Boundary default).")

    # Original X-Boundary code assumes relative `data/` from repo root.
    # For framework usage, allow explicit `data_dir`, otherwise try `./data`,
    # and finally fall back to the common absolute path in this workspace.
    candidates: List[Path] = []
    if data_dir:
        candidates.append(Path(data_dir))
    candidates.append(Path("data"))
    candidates.append(Path("/mnt/shared-storage-user/guojiaxuan/code/X-Boundary/data"))

    base_dir = next((p for p in candidates if p.exists()), None)
    if base_dir is None:
        raise FileNotFoundError(
            "Could not locate X-Boundary data directory. "
            "Provide `dataset.data_dir` pointing to the X-Boundary `data/` folder."
        )

    label_map = {
        0: "Harmful (Erase)",
        1: "Safe (Retain)",
        2: "Boundary-Safe",
    }

    items: List[Dict[str, Any]] = []

    # 1) Harmful (Erase)
    harmful_path = base_dir / harmful_file
    harmful_data = _read_json(harmful_path)[:num_samples_per_class]
    for row in harmful_data:
        items.append(
            {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["output"]},
                ],
                "label": 0,
                "source": "harmful",
            }
        )

    # 2) Safe (Retain) - UltraChat
    ultrachat_local_path = base_dir / ultrachat_local_dir
    safe_messages = _load_ultrachat_messages(
        local_path=ultrachat_local_path,
        hf_id=ultrachat_hf_id,
        hf_split=ultrachat_hf_split,
        num_samples=num_samples_per_class,
        hf_cache_dir=hf_cache_dir,
        hf_revision=ultrachat_hf_revision,
    )
    for messages in safe_messages:
        items.append({"messages": messages, "label": 1, "source": "safe"})

    # 3) Boundary-Safe
    boundary_path = base_dir / boundary_file
    boundary_data = _read_json(boundary_path)
    boundary_data = [row for row in boundary_data if row.get("status") == "1_full_compliance"]
    boundary_data = boundary_data[:num_samples_per_class]
    for row in boundary_data:
        items.append(
            {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ],
                "label": 2,
                "source": "boundary",
            }
        )

    logger.info(
        "Loaded X-Boundary dataset: harmful=%d safe=%d boundary=%d total=%d",
        len(harmful_data),
        len(safe_messages),
        len(boundary_data),
        len(items),
    )

    return {
        "items": items,
        "label_map": label_map,
        "config": {
            "data_dir": str(base_dir),
            "num_samples_per_class": num_samples_per_class,
            "harmful_file": harmful_file,
            "boundary_file": boundary_file,
            "ultrachat_local_dir": ultrachat_local_dir,
            "ultrachat_hf_id": ultrachat_hf_id,
            "ultrachat_hf_split": ultrachat_hf_split,
            "ultrachat_hf_revision": ultrachat_hf_revision,
        },
    }


def register_xboundary_dataset() -> None:
    registry = get_dataset_registry()
    registry.register_dataset(
        XBOUNDARY_DATASET_NAME,
        factory=_load_xboundary_dataset,
        dataset_family="xboundary",
        dataset_split="diagnostic",
        description="X-Boundary diagnostic dataset (harmful/safe/boundary) as chat messages + labels.",
        required_files=[],
        optional_files=[
            "data_dir",
            "num_samples_per_class",
            "harmful_file",
            "boundary_file",
            "ultrachat_local_dir",
            "ultrachat_hf_id",
            "ultrachat_hf_split",
            "ultrachat_hf_revision",
            "hf_cache_dir",
        ],
        expected_columns=["messages", "label"],
    )
    logger.info("Registered X-Boundary diagnostic dataset loader")


