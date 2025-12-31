"""
Lightweight progress reporting utilities.

This module centralizes progress tracking so evaluators can emit consistent
dataset-level progress information without depending on a specific UI. It will
use `tqdm` when available and fall back to periodic logging otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


def _is_sized(obj: Any) -> bool:
    """Return True for objects that support len() and are not plain strings."""
    if obj is None:
        return False
    if isinstance(obj, (str, bytes)):
        return False
    return hasattr(obj, "__len__")


def infer_total_items(dataset: Any) -> Optional[int]:
    """
    Best-effort estimate of dataset length for progress reporting.

    Supports common dataset shapes used in this repo:
    - dicts with an "items" list (e.g., xboundary)
    - dicts with split DataFrames/arrays (e.g., tellme)
    - Hugging Face Dataset objects (len works; also expose .num_rows)
    - any object implementing __len__
    """
    if dataset is None:
        return None

    # Dicts: prefer common payload keys over the dict size itself.
    if isinstance(dataset, dict):
        preferred_keys = ("items", "data", "examples", "records", "rows")
        for key in preferred_keys:
            if key in dataset and _is_sized(dataset[key]):
                try:
                    return len(dataset[key])
                except Exception:
                    pass
        # Fall back to the first sized value.
        for value in dataset.values():
            if _is_sized(value):
                try:
                    return len(value)
                except Exception:
                    continue
        return None

    # Hugging Face datasets expose num_rows even if len() fails.
    num_rows = getattr(dataset, "num_rows", None)
    if isinstance(num_rows, int):
        return num_rows

    if _is_sized(dataset):
        try:
            return len(dataset)
        except Exception:
            return None

    return None


class ProgressReporter:
    """
    Thin wrapper around tqdm with logging fallback.

    - Uses tqdm if installed; otherwise emits periodic log lines.
    - Can be used as a context manager; always safe to call update/close.
    """

    def __init__(
        self,
        *,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        logger_: Optional[logging.Logger] = None,
        log_ratio: float = 0.1,
        progress_sink: Optional[Any] = None,
        on_start: Optional[Callable[[Optional[int], str], None]] = None,
        on_update: Optional[Callable[[int, Optional[int], str], None]] = None,
        on_done: Optional[Callable[[int, Optional[int], str], None]] = None,
    ):
        self.total = total if total is None or total >= 0 else None
        self.desc = desc or "Progress"
        self.logger = logger_ or logger
        self._log_ratio = max(0.01, log_ratio)
        self._log_every = None
        self._bar = None
        self._started = False
        self._completed = 0
        self._sink = progress_sink
        self._on_start = on_start
        self._on_update = on_update
        self._on_done = on_done

    def __enter__(self) -> "ProgressReporter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        self._notify_start()

        if self.total:
            try:
                from tqdm import tqdm

                self._bar = tqdm(total=self.total, desc=self.desc)
            except Exception:
                self._bar = None
                self._compute_log_every()
                self.logger.info("%s: started (total=%d)", self.desc, self.total)
        else:
            self._compute_log_every()
            self.logger.info("%s: started", self.desc)

    def update(self, n: int = 1) -> None:
        self._completed += max(0, int(n))
        if self._bar:
            self._bar.update(n)
            self._notify_update()
            return

        if self._log_every is None:
            self._compute_log_every()

        should_log = False
        if self.total:
            should_log = self._completed >= self.total or (self._completed % self._log_every == 0)
            msg_total = f"{self._completed}/{self.total}"
        else:
            should_log = self._completed % self._log_every == 0
            msg_total = f"{self._completed}"

        if should_log:
            self.logger.info("%s: %s", self.desc, msg_total)
            self._notify_update()

    def close(self) -> None:
        if self._bar:
            self._bar.close()
            self._bar = None
        else:
            if self._started:
                if self.total:
                    self.logger.info("%s: done (%d/%d)", self.desc, self._completed, self.total)
                else:
                    self.logger.info("%s: done (%d items)", self.desc, self._completed)
        self._notify_done()

    def _compute_log_every(self) -> None:
        if self.total:
            self._log_every = max(1, int(self.total * self._log_ratio))
        else:
            self._log_every = 50

    def _notify_start(self) -> None:
        self._safe_call(self._on_start, self.total, self.desc)
        self._safe_sink("on_start", self.total, self.desc)

    def _notify_update(self) -> None:
        self._safe_call(self._on_update, self._completed, self.total, self.desc)
        self._safe_sink("on_update", self._completed, self.total, self.desc)

    def _notify_done(self) -> None:
        self._safe_call(self._on_done, self._completed, self.total, self.desc)
        self._safe_sink("on_done", self._completed, self.total, self.desc)

    def _safe_call(self, fn: Optional[Callable], *args) -> None:
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            self.logger.debug("Progress callback raised; ignoring.", exc_info=True)

    def _safe_sink(self, method: str, *args) -> None:
        if self._sink is None:
            return
        fn = getattr(self._sink, method, None)
        if fn is None:
            return
        self._safe_call(fn, *args)


def progress_for_dataset(
    *,
    dataset: Any = None,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    logger_: Optional[logging.Logger] = None,
    progress_sink: Optional[Any] = None,
    on_start: Optional[Callable[[Optional[int], str], None]] = None,
    on_update: Optional[Callable[[int, Optional[int], str], None]] = None,
    on_done: Optional[Callable[[int, Optional[int], str], None]] = None,
) -> ProgressReporter:
    """
    Convenience helper to create a progress reporter using dataset-aware totals.
    """
    inferred_total = total if total is not None else infer_total_items(dataset)
    return ProgressReporter(
        total=inferred_total,
        desc=desc,
        logger_=logger_,
        progress_sink=progress_sink,
        on_start=on_start,
        on_update=on_update,
        on_done=on_done,
    )


__all__ = ["ProgressReporter", "infer_total_items", "progress_for_dataset"]

