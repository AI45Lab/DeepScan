"""
Dataset implementations and registrations for the framework.

Datasets defined here are automatically registered with the global dataset
registry when this module is imported, so they are available out of the box.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# NOTE: Each dataset family may have optional dependencies. Import/register them
# independently so one missing dependency does not prevent others from registering.

__all__ = []

try:  # BeaverTails (optional: `datasets`)
    from llm_diagnose.datasets.beaver_tails import register_beaver_tails_datasets

    register_beaver_tails_datasets()
    __all__.append("register_beaver_tails_datasets")
except Exception as exc:  # pragma: no cover
    logger.debug("Skipping BeaverTails dataset registration: %s", exc)

try:  # TELLME (optional: `pandas` for CSV mode)
    from llm_diagnose.datasets.tellme import register_tellme_dataset

    register_tellme_dataset()
    __all__.append("register_tellme_dataset")
except Exception as exc:  # pragma: no cover
    logger.debug("Skipping TELLME dataset registration: %s", exc)

try:  # X-Boundary (loads `datasets` only when you actually load the dataset)
    from llm_diagnose.datasets.xboundary import register_xboundary_dataset

    register_xboundary_dataset()
    __all__.append("register_xboundary_dataset")
except Exception as exc:  # pragma: no cover
    logger.debug("Skipping X-Boundary dataset registration: %s", exc)

try:  # SPIN (no extra deps)
    from llm_diagnose.datasets.spin import register_spin_dataset

    register_spin_dataset()
    __all__.append("register_spin_dataset")
except Exception as exc:  # pragma: no cover
    logger.debug("Skipping SPIN dataset registration: %s", exc)

