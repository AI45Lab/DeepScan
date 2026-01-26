"""
Minimal Gemma3 multimodal demo using the LLM-Diagnose model registry/runner.

This mirrors the official HF model card flow but routes through our runner:
- build chat messages with an image + text
- runner handles processor + image preprocessing

Usage (example):
  /root/miniconda3/envs/diagnosis/bin/python examples/gemma3_multimodal_demo.py \
    --model-path /mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it \
    --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg \
    --prompt "Describe this image in detail."
"""

from __future__ import annotations

import argparse

from llm_diagnose.models.base_runner import PromptContent, PromptMessage, GenerationRequest
from llm_diagnose.registry.model_registry import get_model_registry


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Local path to gemma-3-27b-it checkpoint")
    p.add_argument("--image", required=True, help="Image URL or local path")
    p.add_argument("--prompt", required=True, help="User text prompt")
    p.add_argument("--device", default="auto", help="Transformers device_map value (e.g., auto/cuda/cpu)")
    p.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g., bfloat16/float16/auto)")
    p.add_argument("--max-new-tokens", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model = get_model_registry().get_model(
        "gemma3",
        model_name="gemma-3-27b-it",
        path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=False,
    )

    messages = [
        PromptMessage(
            role="system",
            content=[PromptContent(type="text", text="You are a helpful assistant.")],
        ),
        PromptMessage(
            role="user",
            content=[
                PromptContent(type="image", data=args.image),
                PromptContent(type="text", text=args.prompt),
            ],
        ),
    ]
    req = GenerationRequest.from_messages(messages, max_new_tokens=args.max_new_tokens, do_sample=False)
    out = model.generate(req)
    print(out.text)


if __name__ == "__main__":
    main()

