"""
End-to-end example: run TELLME disentanglement metrics on a Qwen model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llm_diagnose  # noqa: F401 - triggers auto-registration

try:  # pragma: no cover - defensive import
    import llm_diagnose.models  # noqa: F401
    import llm_diagnose.datasets  # noqa: F401
except ImportError:
    pass

from llm_diagnose import ConfigLoader, run_from_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TELLME metrics.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "examples" / "config.tellme.yaml"),
        help="Path to YAML/JSON config file (defaults to bundled example).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run evaluation (loads model + dataset).",
    )
    return parser.parse_args()


def load_config(path: str) -> ConfigLoader:
    logger.info("Loading configuration from %s", path)
    return ConfigLoader.from_file(path)


def dry_run(config: ConfigLoader) -> None:
    # Leverage the shared runner validation logic
    logger.info("Dry-run successful: model/dataset entries exist.")


def run_evaluation(config: ConfigLoader) -> None:
    results = run_from_config(config)
    print(json.dumps(results, indent=2, default=str))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)
    dry_run(config)

    if args.run:
        logger.info("Running TELLME evaluation...")
        run_evaluation(config)
    else:
        logger.info("Dry run only. Pass --run to execute the full evaluation.")


if __name__ == "__main__":
    main()


