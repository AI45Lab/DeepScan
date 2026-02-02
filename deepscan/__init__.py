"""
DeepScan: A flexible and extensible framework for diagnosing LLMs and MLLMs.
"""

__version__ = "0.1.0"

from deepscan.registry import ModelRegistry, DatasetRegistry
from deepscan.config import ConfigLoader
from deepscan.evaluators import (
    BaseEvaluator,
    NeuronAttributionEvaluator,
    RepresentationEngineeringEvaluator,
    TellMeEvaluator,
)
from deepscan.summarizers import BaseSummarizer
from deepscan.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    PromptMessage,
    PromptContent,
)

# NOTE: `python -m deepscan.run` should not warn about the module being pre-imported.
# Keep `run_from_config` available on the package, but import it lazily.
def run_from_config(*args, **kwargs):  # type: ignore
    from deepscan.run import run_from_config as _run_from_config

    return _run_from_config(*args, **kwargs)

# Auto-register supported models (Qwen, etc.)
# This makes models available immediately without manual registration
try:
    import deepscan.models  # noqa: F401
except ImportError:  # pragma: no cover
    pass  # Models module may not be available in all installations

# Auto-register supported datasets (BeaverTails, TELLME, etc.)
try:
    import deepscan.datasets  # noqa: F401
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

