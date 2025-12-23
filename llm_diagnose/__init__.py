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
    TellMeEvaluator,
)
from llm_diagnose.summarizers import BaseSummarizer
from llm_diagnose.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    PromptMessage,
    PromptContent,
)

# NOTE: `python -m llm_diagnose.run` should not warn about the module being pre-imported.
# Keep `run_from_config` available on the package, but import it lazily.
def run_from_config(*args, **kwargs):  # type: ignore
    from llm_diagnose.run import run_from_config as _run_from_config

    return _run_from_config(*args, **kwargs)

# Auto-register supported models (Qwen, etc.)
# This makes models available immediately without manual registration
try:
    import llm_diagnose.models  # noqa: F401
except ImportError:  # pragma: no cover
    pass  # Models module may not be available in all installations

# Auto-register supported datasets (BeaverTails, TELLME, etc.)
try:
    import llm_diagnose.datasets  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "ModelRegistry",
    "DatasetRegistry",
    "ConfigLoader",
    "BaseModelRunner",
    "GenerationRequest",
    "GenerationResponse",
    "PromptMessage",
    "PromptContent",
    "BaseEvaluator",
    "NeuronAttributionEvaluator",
    "RepresentationEngineeringEvaluator",
    "TellMeEvaluator",
    "BaseSummarizer",
    "run_from_config",
]

