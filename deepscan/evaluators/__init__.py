"""
Evaluator modules for different evaluation strategies.
"""

from __future__ import annotations

import logging

from deepscan.evaluators.base import BaseEvaluator
from deepscan.evaluators.registry import EvaluatorRegistry

logger = logging.getLogger(__name__)

# Import optional evaluators defensively so missing heavy deps don't block others.
try:
    from deepscan.evaluators.neuron_attribution import NeuronAttributionEvaluator
except Exception as exc:  # pragma: no cover
    NeuronAttributionEvaluator = None  # type: ignore
    logger.debug("NeuronAttributionEvaluator unavailable: %s", exc)

try:
    from deepscan.evaluators.representation_engineering import RepresentationEngineeringEvaluator
except Exception as exc:  # pragma: no cover
    RepresentationEngineeringEvaluator = None  # type: ignore
    logger.debug("RepresentationEngineeringEvaluator unavailable: %s", exc)

try:
    from deepscan.evaluators.tellme import TellMeEvaluator
except Exception as exc:  # pragma: no cover
    TellMeEvaluator = None  # type: ignore
    logger.debug("TellMeEvaluator unavailable: %s", exc)

# SPIN evaluator is lightweight at import-time (heavy deps loaded lazily).
try:
    from deepscan.evaluators.spin import SpinEvaluator
except Exception as exc:  # pragma: no cover
    SpinEvaluator = None  # type: ignore
    logger.debug("SpinEvaluator unavailable: %s", exc)

# MI-Peaks evaluator is lightweight at import-time (heavy deps loaded lazily).
try:
    from deepscan.evaluators.mi_peaks import MiPeaksEvaluator
except Exception as exc:  # pragma: no cover
    MiPeaksEvaluator = None  # type: ignore
    logger.debug("MiPeaksEvaluator unavailable: %s", exc)

# X-Boundary evaluator is lightweight at import-time (heavy deps loaded lazily).
from deepscan.evaluators.xboundary import XBoundaryEvaluator

__all__ = [
    "BaseEvaluator",
    "XBoundaryEvaluator",
    "EvaluatorRegistry",
]

if NeuronAttributionEvaluator is not None:  # pragma: no cover
    __all__.append("NeuronAttributionEvaluator")
if RepresentationEngineeringEvaluator is not None:  # pragma: no cover
    __all__.append("RepresentationEngineeringEvaluator")
if TellMeEvaluator is not None:  # pragma: no cover
    __all__.append("TellMeEvaluator")
if SpinEvaluator is not None:  # pragma: no cover
    __all__.append("SpinEvaluator")
if MiPeaksEvaluator is not None:  # pragma: no cover
    __all__.append("MiPeaksEvaluator")

