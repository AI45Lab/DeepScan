"""
Dataset implementations and registrations for the framework.

Datasets defined here are automatically registered with the global dataset
registry when this module is imported, so they are available out of the box.
"""

from llm_diagnose.datasets.mmlu import register_mmlu_datasets

# Auto-register supported datasets
register_mmlu_datasets()

__all__ = ["register_mmlu_datasets"]

