"""
Quick smoke check: ensure MI-Peaks `sample_num` wiring works (no torch/model required).

Usage:
  python utils/smoke_mipeaks_nsamples.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    # 1) Evaluator config should use `sample_num` (canonical).
    from llm_diagnose.evaluators.mi_peaks import MiPeaksEvaluator

    e0 = MiPeaksEvaluator(config={"sample_num": 2})
    assert getattr(e0, "_cfg").sample_num == 2

    # 2) API diagnosis override should map sample_num into mi-peaks evaluator config.
    from api.main import EvaluationCreateRequest, _apply_diagnosis_overrides

    base_cfg = {
        "model": {"generation": "qwen2.5", "model_name": "Qwen2.5-7B-Instruct"},
        "evaluators": [{"type": "mi-peaks", "run_name": "mi-peaks", "dataset": {"name": "mi-peaks/math_train_12k"}}],
    }
    body = EvaluationCreateRequest.parse_obj({"diagnosis": [{"name": "mi-peaks", "args": {"sample_num": 7}}]})
    overrides = _apply_diagnosis_overrides(base_cfg, body.diagnosis_items())
    mp = next(e for e in overrides if e["type"] == "mi-peaks")
    assert mp["sample_num"] == 7
    assert mp["dataset"]["sample_num"] == 7

    body2 = EvaluationCreateRequest.parse_obj({"diagnosis": [{"name": "mi-peaks", "args": {}}]})
    overrides2 = _apply_diagnosis_overrides(base_cfg, body2.diagnosis_items())
    mp2 = next(e for e in overrides2 if e["type"] == "mi-peaks")
    assert mp2["sample_num"] == 10
    assert mp2["dataset"]["sample_num"] == 10
    assert mp2["dataset"]["name"].startswith("mi-peaks/")

    # 3) If base config does NOT include mi-peaks, requesting it should still select a mi-peaks dataset.
    base_cfg2 = {
        "model": {"generation": "qwen2.5", "model_name": "Qwen2.5-7B-Instruct"},
        "evaluators": [
            {"type": "tellme", "run_name": "tellme", "dataset": {"name": "tellme/beaver_tails_filtered"}},
        ],
    }
    body3 = EvaluationCreateRequest.parse_obj({"diagnosis": [{"name": "mi-peaks", "args": {}}]})
    overrides3 = _apply_diagnosis_overrides(base_cfg2, body3.diagnosis_items())
    mp3 = next(e for e in overrides3 if e["type"] == "mi-peaks")
    assert mp3["run_name"] == "mi-peaks"
    assert mp3["dataset"]["name"] == "mi-peaks/math_train_12k"
    assert mp3["sample_num"] == 10

    print("OK: mi-peaks sample_num wiring looks good.")


if __name__ == "__main__":
    main()

