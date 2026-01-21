"""
Smoke regression tests for example configs and multi-evaluator wiring.

These run in dry-run mode to validate:
- Registry entries exist for models, datasets, evaluators, and summarizers.
- Per-evaluator dataset wiring and per-evaluator summarizers parse correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_diagnose.run import run_from_config


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    "config_path",
    [
        EXAMPLES_DIR / "config.xboundary.tellme-qwen2.5-7b-instruct.yaml",
        EXAMPLES_DIR / "config.mi_peaks.yaml",
    ],
)
def test_example_configs_dry_run(config_path: Path) -> None:
    """
    Ensure bundled example configs stay valid (registry + schema) for dry-run.
    """
    assert config_path.exists(), f"Missing example config: {config_path}"
    # dry_run skips loading model/dataset weights; only registry/shape validation.
    run_from_config(str(config_path), dry_run=True)


def test_dual_evaluators_tellme_xboundary_dry_run() -> None:
    """
    Validate combined tellme + xboundary config parses and registers in dry-run.
    Uses per-evaluator datasets and custom run_name for x-boundary.
    """
    cfg = {
        "model": {
            "generation": "qwen2.5",
            "model_name": "Qwen2.5-7B-Instruct",
            "device": "cuda",
            "dtype": "float16",
            "path": "/root/models/Qwen2.5-7B-Instruct",
        },
        "evaluators": [
            {
                "type": "tellme",
                "run_name": "tellme",
                "batch_size": 4,
                "token_position": -1,
                "dataset": {
                    "name": "tellme/beaver_tails_filtered",
                    # Path need not exist for dry_run; present to match schema.
                    "test_path": "/root/code/TELLME/test.csv",
                },
            },
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 8,
                "max_length": 512,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 50,
                },
            },
        ],
    }

    # dry_run avoids model/data loading but exercises registry validation and
    # evaluator wiring (per-evaluator datasets and ids).
    run_from_config(cfg, dry_run=True)
