"""
Dataset implementations and registrations for the framework.

Datasets defined here are automatically registered with the global dataset
registry when this module is imported, so they are available out of the box.
"""

from llm_diagnose.datasets.beaver_tails import register_beaver_tails_datasets
from llm_diagnose.datasets.tellme import register_tellme_dataset

# Auto-register supported datasets
register_beaver_tails_datasets()
register_tellme_dataset()

__all__ = ["register_beaver_tails_datasets", "register_tellme_dataset"]

