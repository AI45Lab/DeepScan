"""
Model implementations and registrations for the framework.

This module contains pre-configured model registrations for popular model families.
Models are automatically registered when this module is imported.
"""

from llm_diagnose.models.qwen import register_qwen_models

# Auto-register supported models
register_qwen_models()

__all__ = ["register_qwen_models"]

