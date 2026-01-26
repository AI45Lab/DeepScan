"""
Lightweight model introspection helpers.

Goal: keep evaluators robust across different HF model wrappers (LLM vs VLM).
Many VLMs (e.g., Gemma3ForConditionalGeneration) store text-transformer layer
counts under nested configs such as `config.text_config.num_hidden_layers`.
"""

from __future__ import annotations

from typing import Any, Optional


def get_num_hidden_layers(model: Any) -> Optional[int]:
    """
    Best-effort inference of the *text* transformer depth.

    Returns:
        int when found, else None.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    # Common direct field (most decoder-only LLMs)
    val = getattr(cfg, "num_hidden_layers", None)
    if isinstance(val, int) and val > 0:
        return val

    # Common nested configs (multimodal or wrapped architectures)
    for attr in ("text_config", "language_config", "decoder_config", "llm_config"):
        sub = getattr(cfg, attr, None)
        sub_val = getattr(sub, "num_hidden_layers", None) if sub is not None else None
        if isinstance(sub_val, int) and sub_val > 0:
            return sub_val

    return None

