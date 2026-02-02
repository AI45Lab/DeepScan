"""
Llama model registry organized by major generation.

This module registers Meta-Llama (Llama-2/3/3.1/3.2/3.3) style causal models.
We keep the registration API aligned with `deepscan.models.qwen` so configs
can swap `model.generation` with minimal changes.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

from deepscan.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    UnsupportedContentError,
)
from deepscan.registry.model_registry import get_model_registry


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is required to run generation. Install with `pip install torch`."
        ) from exc
    return torch


def _coerce_torch_dtype(value: Any) -> Any:
    """
    Normalize dtype values coming from YAML/JSON configs.

    Transformers generally accepts `torch_dtype` as a torch.dtype or the string "auto".
    Example configs often specify strings like "float16"/"bfloat16"; convert them to
    actual torch dtypes for robustness across transformers versions.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"auto"}:
            return "auto"
        torch_mod = _require_torch()
        mapping = {
            "float16": torch_mod.float16,
            "fp16": torch_mod.float16,
            "half": torch_mod.float16,
            "bfloat16": torch_mod.bfloat16,
            "bf16": torch_mod.bfloat16,
            "float32": torch_mod.float32,
            "fp32": torch_mod.float32,
            "float": torch_mod.float32,
        }
        if cleaned in mapping:
            return mapping[cleaned]
        return value
    return value


LLAMA_MODELS = {
    # Keep dot in registry key to mirror existing Qwen usage (e.g., qwen2.5).
    "llama3.3": {
        "Llama-3.3-70B-Instruct": {
            "path": "meta-llama/Llama-3.3-70B-Instruct",
            "params": "70B",
            "description": "Meta Llama 3.3 70B Instruct model",
        }
    }
}


class LlamaCausalModelRunner(BaseModelRunner):
    """
    Thin wrapper that normalizes access to `generate` for Llama causal models.

    Important Llama-specific handling:
    - Many Llama tokenizers do not define a pad token; for generation we default
      pad_token_id to eos_token_id to avoid warnings and ensure consistent behavior.
    - Prefer Hugging Face chat templates when available (`apply_chat_template`).
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        tokenizer: Optional[Any],
        default_generation: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_name=model_name,
            supports_chat=bool(tokenizer and hasattr(tokenizer, "apply_chat_template")),
            supports_multimodal=False,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.default_generation = default_generation or {}
        self._hf_device_map = getattr(model, "hf_device_map", None)

        # Llama models often lack a pad token; make generation safe by defaulting
        # pad_token_id to eos_token_id when available.
        try:
            if self.tokenizer is not None:
                if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            model_config = getattr(self.model, "config", None)
            if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
                eos_id = None
                if self.tokenizer is not None:
                    eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is None:
                    eos_id = getattr(model_config, "eos_token_id", None)
                if eos_id is not None:
                    model_config.pad_token_id = eos_id
        except Exception:
            # Best-effort only; don't fail model construction.
            pass

    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        torch_mod = _require_torch()
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer was not loaded for this runner. "
                "Set load_tokenizer=True when creating the model."
            )

        tokenized_inputs, prompt_length = self._build_inputs(request)
        gen_kwargs = {**self.default_generation, **request.generation_kwargs}
        gen_kwargs.setdefault("max_new_tokens", 256)
        # Ensure pad_token_id is always defined (Llama often needs this).
        if "pad_token_id" not in gen_kwargs:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.tokenizer, "eos_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id

        inputs = self._maybe_move_to_device(tokenized_inputs)

        with torch_mod.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, prompt_length:]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        metadata = {
            "model_name": self.model_name,
            "prompt_length": int(prompt_length),
            "tokens_generated": int(generated_ids.shape[-1]),
        }

        return GenerationResponse(
            text=text,
            raw_output=output_ids,
            request=request,
            metadata=metadata,
            generation_kwargs=gen_kwargs,
        )

    def _build_inputs(self, request: GenerationRequest) -> Tuple[Dict[str, Any], int]:
        if request.is_chat():
            if not self.supports_chat:
                raise UnsupportedContentError(
                    f"Model '{self.model_name}' does not expose a chat template."
                )
            chat_messages = request.to_hf_chat_messages()
            tokenized = self.tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return {"input_ids": tokenized}, tokenized.shape[-1]

        prompt_text = request.ensure_text_prompt()
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            add_special_tokens=True,
        )
        prompt_length = encoded["input_ids"].shape[-1]
        return encoded, prompt_length

    def _maybe_move_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # If a multi-device map is present, let transformers handle placement.
        if self._hf_device_map and not self._is_single_device_map():
            return inputs

        device = self._infer_device()
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _infer_device(self):
        torch_mod = _require_torch()
        if hasattr(self.model, "device"):
            return self.model.device  # type: ignore[attr-defined]
        try:
            return next(self.model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            return torch_mod.device("cpu")

    @property
    def device(self):
        return self._infer_device()

    def _is_single_device_map(self) -> bool:
        if not self._hf_device_map:
            return False
        if isinstance(self._hf_device_map, str):
            return True
        if isinstance(self._hf_device_map, dict):
            devices = {str(v) for v in self._hf_device_map.values() if v is not None}
            return len(devices) == 1
        return False


def register_llama_models():
    """
    Register all Llama models organized by major generation.

    Usage:
        from deepscan.models.llama import register_llama_models
        register_llama_models()
        runner = get_model_registry().get_model(
            "llama3.3",
            model_name="Llama-3.3-70B-Instruct",
            path="/path/to/local/checkpoint",
            dtype="bfloat16",
        )
    """
    registry = get_model_registry()

    def _create_factory(model_name: str, model_path: str, description: str):
        def factory(device: str = "cuda", **kwargs):
            generation_config = kwargs.pop("generation_config", None)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "transformers library is required. Install with: pip install transformers"
                ) from exc

            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
            model_kwargs: Dict[str, Any] = {
                "device_map": device,
                "torch_dtype": torch_dtype,
                # Meta-Llama models generally do not require remote code; default False.
                "trust_remote_code": kwargs.get("trust_remote_code", False),
            }

            # Add quantization options if specified
            if kwargs.get("load_in_8bit", False):
                model_kwargs["load_in_8bit"] = True
            elif kwargs.get("load_in_4bit", False):
                model_kwargs["load_in_4bit"] = True
                if "bitsandbytes" in kwargs:
                    model_kwargs["bnb_4bit_compute_dtype"] = _coerce_torch_dtype(
                        kwargs.get("bnb_4bit_compute_dtype", "float16")
                    )

            # Add any other kwargs
            for key in ["max_memory", "offload_folder", "low_cpu_mem_usage", "attn_implementation"]:
                if key in kwargs:
                    model_kwargs[key] = kwargs[key]

            model_source = kwargs.pop("path", None) or model_path

            model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

            tokenizer = None
            if kwargs.get("load_tokenizer", True):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    trust_remote_code=kwargs.get("trust_remote_code", False),
                )

            return LlamaCausalModelRunner(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                default_generation=generation_config,
            )

        return factory

    for generation, models in LLAMA_MODELS.items():
        available_models = list(models.keys())

        for model_name, config in models.items():
            registry_name = f"{generation}/{model_name}"
            registry.register_model(
                registry_name,
                factory=_create_factory(
                    model_name=model_name,
                    model_path=config["path"],
                    description=config["description"],
                ),
                model_family="llama",
                model_generation=generation,
                model_name=model_name,
                model_type="llm",
                parameters=config["params"],
                description=config["description"],
            )

        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="llama",
                model_generation=gen,
                model_type="llm",
                available_models=available,
                description=f"Llama {gen} model family factory",
            )
            def create_llama_generation(
                model_name: str,
                device: str = "cuda",
                **kwargs,
            ):
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. "
                        f"Available models: {list(models_dict.keys())}"
                    )
                model_config = models_dict[model_name]
                model_path = model_config["path"]

                generation_config = kwargs.pop("generation_config", None)
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                except ImportError as exc:
                    raise ImportError(
                        "transformers library is required. Install with: pip install transformers"
                    ) from exc

                torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
                model_kwargs: Dict[str, Any] = {
                    "device_map": device,
                    "torch_dtype": torch_dtype,
                    "trust_remote_code": kwargs.get("trust_remote_code", False),
                }

                if kwargs.get("load_in_8bit", False):
                    model_kwargs["load_in_8bit"] = True
                elif kwargs.get("load_in_4bit", False):
                    model_kwargs["load_in_4bit"] = True
                    if "bitsandbytes" in kwargs:
                        model_kwargs["bnb_4bit_compute_dtype"] = _coerce_torch_dtype(
                            kwargs.get("bnb_4bit_compute_dtype", "float16")
                        )

                for key in ["max_memory", "offload_folder", "low_cpu_mem_usage", "attn_implementation"]:
                    if key in kwargs:
                        model_kwargs[key] = kwargs[key]

                model_source = kwargs.pop("path", None) or model_path
                model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

                tokenizer = None
                if kwargs.get("load_tokenizer", True):
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_source,
                        trust_remote_code=kwargs.get("trust_remote_code", False),
                    )

                return LlamaCausalModelRunner(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    default_generation=generation_config,
                )

            return create_llama_generation

        _create_generation_factory(generation, models, available_models)

