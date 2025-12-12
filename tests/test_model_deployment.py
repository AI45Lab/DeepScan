import os

import pytest

from llm_diagnose.models.qwen import register_qwen_models
from llm_diagnose.registry.model_registry import get_model_registry


def test_qwen3_answers_who_are_you():
    """
    Integration test that loads the smallest Qwen3 model and verifies it
    produces a non-empty answer to a simple prompt.
    """
    register_qwen_models()
    registry = get_model_registry()

    runner = registry.get_model(
        "qwen3",
        model_name="Qwen3-8B",
        generation="qwen3",
        device="cuda",
        path="/mnt/shared-storage-user/guojiaxuan/data/models/models--Qwen--Qwen3-8B",
        load_tokenizer=True,
        trust_remote_code=True,
        dtype="auto",
        generation_config={"max_new_tokens": 32},
    )
    # Align input handling with the official chat template usage:
    # pass a plain prompt to the runner and let it build inputs internally.
    prompt = "Give me a short introduction to large language model."
    response = runner.generate(prompt, max_new_tokens=128, temperature=0.2)
    print("Qwen3 response:", response.text)
    assert isinstance(response.text, str)
    assert response.text.strip()

