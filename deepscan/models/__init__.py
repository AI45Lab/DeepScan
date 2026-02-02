"""
Model implementations and registrations for the framework.

This module contains pre-configured model registrations for popular model families.
Models are automatically registered when this module is imported.
"""

from deepscan.models.qwen import register_qwen_models
from deepscan.models.llama import register_llama_models
from deepscan.models.mistral import register_mistral_models
from deepscan.models.internlm import register_internlm_models
from deepscan.models.gemma import register_gemma_models
from deepscan.models.internvl import register_internvl_models
from deepscan.models.glm import register_glm_models

# Auto-register supported models
register_qwen_models()
register_llama_models()
register_mistral_models()
register_internlm_models()
register_gemma_models()
register_internvl_models()
register_glm_models()

__all__ = [
    "register_qwen_models",
    "register_llama_models",
    "register_mistral_models",
    "register_internlm_models",
    "register_gemma_models",
    "register_internvl_models",
    "register_glm_models",
]

