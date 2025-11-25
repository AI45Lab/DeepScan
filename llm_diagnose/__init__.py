"""
LLM-Diagnose Framework: A flexible and extensible framework for diagnosing LLMs and MLLMs.
"""

__version__ = "0.1.0"

from llm_diagnose.registry import ModelRegistry, DatasetRegistry
from llm_diagnose.config import ConfigLoader
from llm_diagnose.evaluators import (
    BaseEvaluator,
    NeuronAttributionEvaluator,
    RepresentationEngineeringEvaluator,
)
from llm_diagnose.summarizers import BaseSummarizer

# Auto-register supported models (Qwen, etc.)
# This makes models available immediately without manual registration
try:
    import llm_diagnose.models  # noqa: F401
except ImportError:  # pragma: no cover
    pass  # Models module may not be available in all installations

# Auto-register supported datasets (MMLU, etc.)
try:
    import llm_diagnose.datasets  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "ModelRegistry",
    "DatasetRegistry",
    "ConfigLoader",
    "BaseEvaluator",
    "NeuronAttributionEvaluator",
    "RepresentationEngineeringEvaluator",
    "BaseSummarizer",
]

