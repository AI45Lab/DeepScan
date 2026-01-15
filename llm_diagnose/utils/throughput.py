"""
Token throughput tracking utilities.

Used to count how many tokens are processed (not generated) during evaluation
and to optionally emit best-effort throughput metrics to any sink exposing
`on_throughput(payload)`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

try:  # Optional heavy deps: only used when available.
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


def _sum_numeric(obj: Any) -> int:
    """Best-effort numeric sum for torch/np/python containers."""
    try:
        if torch is not None and isinstance(obj, torch.Tensor):
            return int(obj.detach().sum().item())
    except Exception:
        pass
    try:
        if np is not None and isinstance(obj, np.ndarray):
            return int(np.sum(obj).item())
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        total = 0
        for item in obj:
            total += _sum_numeric(item)
        return total
    if isinstance(obj, (int, float)):
        try:
            return int(obj)
        except Exception:
            return 0
    return 0


def _count_from_shape(obj: Any) -> int:
    """Estimate token count from tensor/array shapes when masks are missing."""
    # Handle plain Python nested lists/tuples first.
    if isinstance(obj, (list, tuple)):
        try:
            # Assume rectangular: count first-level length * inner length.
            if len(obj) == 0:
                return 0
            first = obj[0]
            if isinstance(first, (list, tuple)):
                return int(len(obj) * len(first))
            return int(len(obj))
        except Exception:
            return 0
    try:
        shape = getattr(obj, "shape", None)
        if shape is None:
            return 0
        if hasattr(shape, "__iter__"):
            total = 1
            for dim in shape:
                try:
                    total *= int(dim)
                except Exception:
                    return 0
            return int(total)
    except Exception:
        return 0
    return 0


def count_tokens_from_batch(batch: Any) -> int:
    """
    Estimate how many tokens are present in a typical model input batch.

    Priority:
    1) attention_mask sum (accounts for padding)
    2) input_ids shape (falls back to number of ids)
    3) nested structures (lists/tuples of the above)
    """
    if batch is None:
        return 0

    # Fast-path raw tensors/arrays: estimate from shape when no masks are present.
    try:
        if torch is not None and isinstance(batch, torch.Tensor):
            return _count_from_shape(batch)
    except Exception:
        pass
    try:
        if np is not None and isinstance(batch, np.ndarray):
            return _count_from_shape(batch)
    except Exception:
        pass

    # BatchEncoding-like objects expose dict-style accessors and attributes.
    if isinstance(batch, Mapping):
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            return _sum_numeric(batch["attention_mask"])
        if "input_ids" in batch and batch["input_ids"] is not None:
            return _count_from_shape(batch["input_ids"])

    if isinstance(batch, (list, tuple)):
        return sum(count_tokens_from_batch(item) for item in batch)

    attn = getattr(batch, "attention_mask", None)
    if attn is not None:
        tokens = _sum_numeric(attn)
        if tokens > 0:
            return tokens

    ids = getattr(batch, "input_ids", None)
    if ids is not None:
        tokens = _count_from_shape(ids)
        if tokens > 0:
            return tokens

    return 0


class TokenThroughputTracker:
    """
    Simple token counter with optional periodic sink emission.
    """

    def __init__(
        self,
        sink: Optional[Any] = None,
        *,
        min_interval_seconds: float = 1.0,
        min_tokens_delta: int = 128,
    ):
        self.start_time = time.time()
        self.total_tokens = 0
        self._sink = sink
        self._min_interval = max(0.1, float(min_interval_seconds))
        self._min_tokens_delta = max(0, int(min_tokens_delta))
        self._last_emit_time = self.start_time
        self._last_emit_tokens = 0

    def add_tokens(self, tokens: int) -> None:
        try:
            tokens_int = int(tokens)
        except Exception:
            return
        if tokens_int <= 0:
            return
        self.total_tokens += tokens_int
        self._maybe_emit()

    def add_batch(self, batch: Any) -> None:
        """Convenience to count tokens from a batch-like payload."""
        self.add_tokens(count_tokens_from_batch(batch))

    def snapshot(self) -> dict:
        now = time.time()
        elapsed = max(now - self.start_time, 1e-6)
        interval_elapsed = max(now - self._last_emit_time, 1e-6)
        token_delta = self.total_tokens - self._last_emit_tokens
        # Instantaneous rate since last emit; cumulative average also available.
        tps_instant = float(token_delta) / float(interval_elapsed) if interval_elapsed > 0 else 0.0
        tps_avg = float(self.total_tokens) / float(elapsed) if elapsed > 0 else 0.0
        return {
            "tokens": int(self.total_tokens),
            "elapsed_seconds": elapsed,
            "tokens_per_second": tps_instant,
            "tokens_per_second_avg": tps_avg,
        }

    # Public finalize to push a last update even if thresholds weren't hit.
    def finalize(self, *, status: Optional[str] = None) -> None:
        self._emit(force=True, status=status or "complete")

    def emit_zero_rate(self, *, status: str = "running") -> None:
        """
        Emit a zero-rate snapshot (useful after an evaluator finishes) so that
        downstream sinks stop showing the last observed throughput.
        """
        if self._sink is None:
            return
        now = time.time()
        elapsed = max(now - self.start_time, 1e-6)
        snapshot = {
            "tokens": int(self.total_tokens),
            "elapsed_seconds": elapsed,
            # Explicitly zero out both instantaneous and average rates to clear UI.
            "tokens_per_second": 0.0,
            "tokens_per_second_avg": 0.0,
            "_status": status,
        }
        self._last_emit_tokens = self.total_tokens
        self._last_emit_time = now
        try:
            fn = getattr(self._sink, "on_throughput", None)
            if fn is not None:
                fn(snapshot)
            logger.info("Throughput zeroed | status=%s | tokens=%s", status, snapshot["tokens"])
        except Exception:
            logger.debug("Throughput sink call failed", exc_info=True)

    def _maybe_emit(self) -> None:
        if self._sink is None:
            return
        now = time.time()
        token_delta = self.total_tokens - self._last_emit_tokens
        if token_delta < self._min_tokens_delta and (now - self._last_emit_time) < self._min_interval:
            return
        self._emit()

    def _emit(self, *, force: bool = False, status: Optional[str] = None) -> None:
        if self._sink is None:
            return
        now = time.time()
        token_delta = self.total_tokens - self._last_emit_tokens
        time_delta = now - self._last_emit_time
        if not force:
            if token_delta < self._min_tokens_delta and time_delta < self._min_interval:
                return
        snapshot = self.snapshot()
        if status is not None:
            snapshot["_status"] = status
        # Advance counters after computing snapshot to reflect this emission window.
        self._last_emit_tokens = self.total_tokens
        self._last_emit_time = now
        try:
            fn = getattr(self._sink, "on_throughput", None)
            if fn is not None:
                fn(snapshot)
        except Exception:
            logger.debug("Throughput sink call failed", exc_info=True)


__all__ = ["TokenThroughputTracker", "count_tokens_from_batch"]
