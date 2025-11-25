"""
MMLU dataset registration helpers.

The Massive Multitask Language Understanding (MMLU) benchmark contains
57 subjects that cover STEM, humanities, social sciences, and other domains.
This module registers every subject along with a convenience factory that can
load multiple subjects at once.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Union, Dict, Any

from llm_diagnose.registry.dataset_registry import get_dataset_registry

logger = logging.getLogger(__name__)

MMLU_SUBJECTS: List[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MMLU_AVAILABLE_SPLITS: Sequence[str] = ("validation", "test")
DEFAULT_SPLIT = "validation"
HUGGING_FACE_DATASET_ID = "cais/mmlu"


def register_mmlu_datasets() -> None:
    """
    Register the MMLU dataset (all subjects + per-subject factories).
    """

    registry = get_dataset_registry()

    def _load_subject(subject: str, split: str, **kwargs: Any):
        """
        Lazily load a single MMLU subject from Hugging Face datasets.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "The 'datasets' library is required to load MMLU. "
                "Install it with `pip install datasets`."
            ) from exc

        if split not in MMLU_AVAILABLE_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Available splits: {MMLU_AVAILABLE_SPLITS}"
            )

        return load_dataset(
            HUGGING_FACE_DATASET_ID,
            subject,
            split=split,
            **kwargs,
        )

    def _create_subject_factory(subject: str):
        def factory(split: str = DEFAULT_SPLIT, **kwargs: Any):
            return _load_subject(subject, split=split, **kwargs)

        return factory

    def _normalize_subjects(
        subjects: Optional[Union[str, Iterable[str]]]
    ) -> List[str]:
        if subjects is None:
            return list(MMLU_SUBJECTS)

        if isinstance(subjects, str):
            normalized = [subjects]
        else:
            normalized = [s for s in subjects]

        invalid = sorted(set(normalized) - set(MMLU_SUBJECTS))
        if invalid:
            raise ValueError(
                f"Unknown MMLU subject(s): {invalid}. "
                f"Available subjects: {MMLU_SUBJECTS}"
            )

        return normalized

    # Register per-subject datasets (e.g., mmlu/astronomy)
    for subject in MMLU_SUBJECTS:
        registry.register_dataset(
            f"mmlu/{subject}",
            factory=_create_subject_factory(subject),
            dataset_family="mmlu",
            dataset_subject=subject,
            available_splits=list(MMLU_AVAILABLE_SPLITS),
            huggingface_id=HUGGING_FACE_DATASET_ID,
            description=f"MMLU subject: {subject.replace('_', ' ').title()}",
        )

    @registry.register_dataset(
        "mmlu",
        dataset_family="mmlu",
        description="Massive Multitask Language Understanding benchmark (all subjects).",
        available_subjects=list(MMLU_SUBJECTS),
        available_splits=list(MMLU_AVAILABLE_SPLITS),
        huggingface_id=HUGGING_FACE_DATASET_ID,
    )
    def create_mmlu_dataset(
        subjects: Optional[Union[str, Iterable[str]]] = None,
        split: str = DEFAULT_SPLIT,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """
        Load one or more MMLU subjects.

        Args:
            subjects: Subject name or iterable of subjects. Defaults to all subjects.
            split: Dataset split to load ("validation" or "test").
            return_dict: If True, always return a dict keyed by subject.
                Otherwise, when a single subject is requested, the dataset object
                is returned directly for convenience.
            **kwargs: Forwarded to `datasets.load_dataset`.
        """

        subject_list = _normalize_subjects(subjects)

        datasets_by_subject: Dict[str, Any] = {}
        for subject in subject_list:
            datasets_by_subject[subject] = _load_subject(subject, split=split, **kwargs)

        if return_dict or len(datasets_by_subject) != 1:
            return datasets_by_subject

        # Return the single dataset object directly for convenience
        return next(iter(datasets_by_subject.values()))

    logger.info("Registered MMLU datasets (%d subjects)", len(MMLU_SUBJECTS))

