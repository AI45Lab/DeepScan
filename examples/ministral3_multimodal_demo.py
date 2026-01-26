"""
Minimal Ministral3 multimodal demo using the LLM-Diagnose model registry/runner.

Mirrors the HuggingFace reference flow:
- build chat messages with image_url + text
- tokenizer.apply_chat_template(...) returns input_ids + pixel_values
- model.generate(..., image_sizes=..., max_new_tokens=...)

Usage (example):
  python examples/ministral3_multimodal_demo.py \
    --model-path /mnt/shared-storage-user/ai4good2-share/models/mistralai/Ministral-3-14B-Instruct-2512 \
    --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg \
    --prompt "Describe this image in detail."
"""

from __future__ import annotations

import argparse

from llm_diagnose.models.base_runner import PromptContent, PromptMessage, GenerationRequest
from llm_diagnose.registry.model_registry import get_model_registry


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Local path to Ministral-3-14B-Instruct-2512 checkpoint")
    p.add_argument("--image", required=True, help="Image URL (recommended) or local path (if supported by your HF build)")
    p.add_argument("--prompt", required=True, help="User text prompt")
    p.add_argument("--device", default="auto", help="Transformers device_map value (e.g., auto/cuda/cpu)")
    p.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g., bfloat16/float16/auto)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    runner = get_model_registry().get_model(
        "ministral3",
        model_name="Ministral-3-14B-Instruct-2512",
        path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=False,
    )

    messages = [
        PromptMessage(
            role="user",
            content=[
                PromptContent(type="text", text=args.prompt),
                PromptContent(type="image", data=args.image),
            ],
        ),
    ]
    req = GenerationRequest.from_messages(messages, max_new_tokens=args.max_new_tokens, do_sample=False)
    out = runner.generate(req)
    print(out.text)


if __name__ == "__main__":
    main()

