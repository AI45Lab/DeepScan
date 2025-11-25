"""
End-to-end example: evaluate Qwen3 on the MMLU benchmark.

This script demonstrates how to:
1. Fetch a pre-registered Qwen3 model.
2. Load the bundled MMLU dataset (astronomy + philosophy subjects).
3. Run a neuron attribution evaluator.

The script defaults to a dry-run that only checks the pipeline wiring without
actually loading the model/dataset (to keep it lightweight). Pass --run to
perform the real evaluation (requires 'transformers' and 'datasets' installed).
"""

from __future__ import annotations

import argparse
import logging

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llm_diagnose  # noqa: F401 - triggers auto-registration

# Ensure model/dataset registries are populated even if auto-registration changes
try:  # pragma: no cover - defensive import
    import llm_diagnose.models  # noqa: F401
    import llm_diagnose.datasets  # noqa: F401
except ImportError:
    pass

from llm_diagnose import ConfigLoader, NeuronAttributionEvaluator
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 on MMLU.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "examples" / "config.qwen3_mmlu.yaml"),
        help="Path to a YAML/JSON config file (defaults to bundled example).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the evaluation (loads the model/dataset).",
    )
    return parser.parse_args()


def load_config(path: str) -> ConfigLoader:
    logger.info("Loading configuration from %s", path)
    return ConfigLoader.from_file(path)


def validate_config(config: ConfigLoader) -> None:
    subjects = config.get("dataset.subjects") or []
    if not isinstance(subjects, (list, tuple)) or not subjects:
        raise ValueError(
            "Config error: 'dataset.subjects' must be a non-empty list "
            "of MMLU subjects."
        )


def dry_run(config: ConfigLoader) -> None:
    """
    Verify that registries contain the requested resources.
    """
    model_registry = get_model_registry()
    dataset_registry = get_dataset_registry()

    # Ensure model entry exists
    if not model_registry.is_registered("qwen3"):
        raise RuntimeError("Qwen3 generation is not registered. Did auto-registration fail?")
    metadata = model_registry.get_metadata("qwen3")
    model_name = config.get("model.model_name")
    if model_name not in metadata.get("available_models", []):
        raise ValueError(f"Model {model_name} is not available in the registry.")

    # Ensure dataset entries exist
    dataset_name = config.get("dataset.name")
    if not dataset_registry.is_registered(dataset_name):
        raise RuntimeError(f"Dataset '{dataset_name}' is not registered.")

    subjects = config.get("dataset.subjects")
    for subject in subjects:
        if not dataset_registry.is_registered(f"{dataset_name}/{subject}"):
            raise RuntimeError(f"MMLU subject '{subject}' is not registered.")

    logger.info("Dry-run successful: model/dataset entries found in registries.")


def run_evaluation(config: ConfigLoader) -> None:
    """
    Execute the actual evaluation (requires heavy dependencies).
    """
    model_registry = get_model_registry()
    dataset_registry = get_dataset_registry()

    model_name = config.get("model.model_name", "Qwen3-7B")
    try:
        model = model_registry.get_model(
            "qwen3",
            model_name=model_name,
            device=config.get("model.device", "cuda"),
            torch_dtype=config.get("model.torch_dtype", "float16"),
            load_in_8bit=config.get("model.load_in_8bit", False),
        )
        dataset = dataset_registry.get_dataset(
            config.get("dataset.name"),
            subjects=config.get("dataset.subjects"),
            split=config.get("dataset.split"),
            return_dict=config.get("dataset.return_dict", False),
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency required for this evaluation. "
            "Install 'transformers' (for models) and 'datasets' (for MMLU) "
            "before running with --run."
        ) from exc

    evaluator = NeuronAttributionEvaluator(
        name=f"qwen3_{model_name}_mmlu",
        config=config.get("evaluator"),
    )
    results = evaluator.evaluate(
        model,
        dataset,
        target_layers=config.get("evaluator.target_layers"),
        top_k=config.get("evaluator.top_k"),
    )
    statistics = results.get("statistics")
    if statistics:
        logger.info("Evaluation statistics: %s", statistics)
    elif "top_neurons" in results:
        logger.info(
            "Evaluation completed. Logged top neurons for %d layers.",
            len(results["top_neurons"]),
        )
    else:
        logger.info("Evaluation finished. Available result keys: %s", list(results.keys()))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)

    validate_config(config)
    dry_run(config)

    if args.run:
        logger.info("Running full evaluation (this may take a while)...")
        run_evaluation(config)
    else:
        logger.info("Dry run only. Pass --run to execute the full evaluation.")


if __name__ == "__main__":
    main()

