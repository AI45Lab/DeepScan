"""
Summarizer modules for different benchmarks and evaluation results.
"""

from deepscan.summarizers.base import BaseSummarizer
from deepscan.summarizers.registry import SummarizerRegistry
from deepscan.summarizers.xboundary import XBoundarySummarizer
from deepscan.summarizers.tellme import TellMeSummarizer
from deepscan.summarizers.combined import CombinedSummarizer

try:
    from deepscan.summarizers.spin import SpinSummarizer
except Exception:  # pragma: no cover
    SpinSummarizer = None  # type: ignore

try:
    from deepscan.summarizers.mi_peaks import MiPeaksSummarizer
except Exception:  # pragma: no cover
    MiPeaksSummarizer = None  # type: ignore

__all__ = ["BaseSummarizer", "SummarizerRegistry", "XBoundarySummarizer", "TellMeSummarizer", "CombinedSummarizer"]
if SpinSummarizer is not None:  # pragma: no cover
    __all__.append("SpinSummarizer")
if MiPeaksSummarizer is not None:  # pragma: no cover
    __all__.append("MiPeaksSummarizer")

