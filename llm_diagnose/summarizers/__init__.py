"""
Summarizer modules for different benchmarks and evaluation results.
"""

from llm_diagnose.summarizers.base import BaseSummarizer
from llm_diagnose.summarizers.registry import SummarizerRegistry
from llm_diagnose.summarizers.xboundary import XBoundarySummarizer

try:
    from llm_diagnose.summarizers.spin import SpinSummarizer
except Exception:  # pragma: no cover
    SpinSummarizer = None  # type: ignore

__all__ = ["BaseSummarizer", "SummarizerRegistry", "XBoundarySummarizer"]
if SpinSummarizer is not None:  # pragma: no cover
    __all__.append("SpinSummarizer")

