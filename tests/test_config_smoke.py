"""
Smoke regression tests for example configs and multi-evaluator wiring.

These run in dry-run mode to validate:
- Registry entries exist for models, datasets, evaluators, and summarizers.
- Per-evaluator dataset wiring and per-evaluator summarizers parse correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deepscan.run import run_from_config


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


def test_mistral_model_registered_dry_run() -> None:
    """
    Ensure the Mistral registry key and model name are wired (dry-run).
    """
    cfg = {
        "model": {
            "generation": "mistral",
            "model_name": "Mistral-Small-24B-Instruct-2501",
            "device": "cuda",
            "dtype": "bfloat16",
            "path": "/mnt/shared-storage-user/ai4good2-share/models/mistralai/Mistral-Small-24B-Instruct-2501",
        },
        "evaluators": [
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 2,
                "max_length": 128,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 2,
                },
            }
        ],
    }
    run_from_config(cfg, dry_run=True)


def test_internlm3_model_registered_dry_run() -> None:
    """
    Ensure the InternLM3 registry key and model name are wired (dry-run).
    """
    cfg = {
        "model": {
            "generation": "internlm3",
            "model_name": "Internlm3-8b-Instruct",
            "device": "cuda",
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "path": "/mnt/shared-storage-user/ai4good2-share/models/internlm/internlm3-8b-instruct",
        },
        "evaluators": [
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 2,
                "max_length": 128,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 2,
                },
            }
        ],
    }
    run_from_config(cfg, dry_run=True)


def test_gemma3_model_registered_dry_run() -> None:
    """
    Ensure the Gemma3 registry key and model name are wired (dry-run).
    """
    cfg = {
        "model": {
            "generation": "gemma3",
            "model_name": "gemma-3-27b-it",
            "device": "cuda",
            "dtype": "bfloat16",
            "path": "/mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it",
            "trust_remote_code": False,
        },
        "evaluators": [
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 2,
                "max_length": 128,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 2,
                },
            }
        ],
    }
    run_from_config(cfg, dry_run=True)


def test_internvl35_model_registered_dry_run() -> None:
    """
    Ensure InternVL3.5 registry key and model name are wired (dry-run).
    """
    cfg = {
        "model": {
            "generation": "internvl3.5",
            "model_name": "InternVL3.5-14B",
            "device": "cuda",
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "path": "OpenGVLab/InternVL3_5-14B",
        },
        "evaluators": [
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 2,
                "max_length": 128,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 2,
                },
            }
        ],
    }
    run_from_config(cfg, dry_run=True)


def test_internvl35_241b_model_registered_dry_run() -> None:
    """
    Ensure InternVL3.5 241B-A28B registry key and model name are wired (dry-run).
    """
    cfg = {
        "model": {
            "generation": "internvl3.5",
            "model_name": "InternVL3.5-241B-A28B",
            "device": "cuda",
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "path": "/mnt/shared-storage-user/ai4good2-share/models/OpenGVLab/InternVL3_5-241B-A28B",
        },
        "evaluators": [
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 2,
                "max_length": 128,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 2,
                },
            }
        ],
    }
    run_from_config(cfg, dry_run=True)


def test_ministral3_model_registered_dry_run() -> None:
    """
    Ensure Ministral3 registry key and model name are wired (dry-run).
    """
    cfg = {
        "model": {
            "generation": "ministral3",
            "model_name": "Ministral-3-14B-Instruct-2512",
            "device": "auto",
            "dtype": "bfloat16",
            "trust_remote_code": False,
            "path": "mistralai/Ministral-3-14B-Instruct-2512",
        },
        "evaluators": [
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 2,
                "max_length": 128,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 2,
                },
            }
        ],
    }
    run_from_config(cfg, dry_run=True)


def test_glm45_air_model_registered_dry_run() -> None:
    """
    Ensure GLM registry key and model name are wired (dry-run).
    """
    cfg = {
        "model": {
            "generation": "glm",
            "model_name": "GLM-4.5-Air",
            "device": "cuda",
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "path": "zai-org/GLM-4.5-Air",
        },
        "evaluators": [
            {
                "type": "xboundary",
                "run_name": "x-boundary",
                "batch_size": 2,
                "max_length": 128,
                "dataset": {
                    "name": "xboundary/diagnostic",
                    "data_dir": "/root/code/X-Boundary/data",
                    "num_samples_per_class": 2,
                },
            }
        ],
    }
    run_from_config(cfg, dry_run=True)
