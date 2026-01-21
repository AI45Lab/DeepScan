from __future__ import annotations

from api.main import (
    DiagnosisItem,
    EvaluationCreateRequest,
    InferenceParameters,
    _apply_diagnosis_overrides,
    _apply_model_override,
)


def _base_config() -> dict:
    return {
        "model": {"generation": "qwen2.5", "model_name": "Qwen2.5-7B-Instruct"},
        "evaluators": [
            {"type": "xboundary", "run_name": "x-boundary", "dataset": {"name": "xboundary/diagnostic"}},
            {"type": "tellme", "run_name": "tellme", "dataset": {"name": "tellme/beaver_tails_filtered"}},
        ],
    }


def test_inference_parameters_normalization() -> None:
    params = InferenceParameters.parse_obj(
        {"temperature": 0.0, "repetition-penalty": 1.1, "top-p": 0.9, "top-k": 42}
    )
    model_cfg = _apply_model_override(_base_config()["model"], "qwen2.5-7b-instruct", params)

    assert model_cfg["generation"] == "qwen2.5"
    assert model_cfg["model_name"].startswith("Qwen")
    gen_cfg = model_cfg["generation_config"]
    assert gen_cfg["repetition_penalty"] == 1.1
    assert gen_cfg["top_p"] == 0.9
    assert gen_cfg["top_k"] == 42


def test_diagnosis_overrides_mapping() -> None:
    base_cfg = _base_config()
    payload = {
        "diagnosis": [
            {"name": "x-boundary", "args": {"target-layers": [9, 18], "samples": 200}},
            {
                "name": "tellme",
                "args": {
                    "target-layers": [10, 19],
                    "samples": 100,
                    "batch_size": 4,
                    "layer_ratio": 0.66666,
                },
            },
        ]
    }
    body = EvaluationCreateRequest.parse_obj(payload)
    overrides = _apply_diagnosis_overrides(base_cfg, body.diagnosis_items())

    xb = next(e for e in overrides if e["type"] == "xboundary")
    assert xb["target_layers"] == [9, 18]
    assert xb["dataset"]["num_samples_per_class"] == 200

    tm = next(e for e in overrides if e["type"] == "tellme")
    assert tm["layer"] == 10
    assert tm["dataset"]["max_rows"] == 100
    assert tm["batch_size"] == 4
    assert tm["layer_ratio"] == 0.66666


def test_diagnosis_overrides_mapping_mi_peaks_sample_num() -> None:
    base_cfg = {
        "model": {"generation": "qwen2.5", "model_name": "Qwen2.5-7B-Instruct"},
        "evaluators": [
            {"type": "mi-peaks", "run_name": "mi-peaks", "dataset": {"name": "mi-peaks/math_train_12k"}},
        ],
    }
    payload = {"diagnosis": [{"name": "mi-peaks", "args": {"sample_num": 7}}]}
    body = EvaluationCreateRequest.parse_obj(payload)
    overrides = _apply_diagnosis_overrides(base_cfg, body.diagnosis_items())

    mp = next(e for e in overrides if e["type"] == "mi-peaks")
    assert mp["sample_num"] == 7
    # API also mirrors this into dataset.sample_num unless explicitly set.
    assert mp["dataset"]["sample_num"] == 7


def test_diagnosis_overrides_mapping_mi_peaks_defaults_sample_num_10() -> None:
    base_cfg = {
        "model": {"generation": "qwen2.5", "model_name": "Qwen2.5-7B-Instruct"},
        "evaluators": [
            {"type": "mi-peaks", "run_name": "mi-peaks", "dataset": {"name": "mi-peaks/math_train_12k"}},
        ],
    }
    payload = {"diagnosis": [{"name": "mi-peaks", "args": {}}]}
    body = EvaluationCreateRequest.parse_obj(payload)
    overrides = _apply_diagnosis_overrides(base_cfg, body.diagnosis_items())

    mp = next(e for e in overrides if e["type"] == "mi-peaks")
    assert mp["sample_num"] == 10
    assert mp["dataset"]["sample_num"] == 10
