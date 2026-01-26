"""
InternLM model registry organized by generation.

InternLM3 checkpoints typically require `trust_remote_code=True` because they provide
custom config/tokenizer/model implementations via `auto_map` and local python modules.

Evaluators in this repo expect the returned object to expose:
- `.model` (HF model)
- `.tokenizer`
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple

from llm_diagnose.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    UnsupportedContentError,
)
from llm_diagnose.registry.model_registry import get_model_registry

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required to run generation. Install with `pip install torch`.") from exc
    return torch


def _coerce_torch_dtype(value: Any) -> Any:
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


def _patch_transformers_for_internlm3() -> None:
    """
    InternLM3 remote code has been observed to import `LossKwargs` from
    `transformers.utils`, but newer Transformers versions may no longer export it.

    Example failure:
        ImportError: cannot import name 'LossKwargs' from 'transformers.utils'

    We provide a minimal shim so the module import can proceed.
    """
    try:
        import transformers  # type: ignore
        from transformers import utils as tutils  # type: ignore
    except Exception:
        return
    if hasattr(tutils, "LossKwargs"):
        return
    try:
        # InternLM3 defines:
        #   class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...
        # where FlashAttentionKwargs is a TypedDict. Therefore LossKwargs must also
        # be a TypedDict (NOT a plain dict), otherwise Python raises:
        #   TypeError: cannot inherit from both a TypedDict type and a non-TypedDict base class
        from typing import TypedDict  # py3.8+ typing.TypedDict is available

        class _LossKwargs(TypedDict, total=False):
            pass

        tutils.LossKwargs = _LossKwargs  # type: ignore[attr-defined]
        logger.warning(
            "Patched transformers.utils.LossKwargs for InternLM3 compatibility "
            "(transformers=%s). Consider pinning transformers~=4.47.1 for a cleaner setup.",
            getattr(transformers, "__version__", "unknown"),
        )
    except Exception:
        # Best-effort only.
        return


INTERNLM_MODELS = {
    "internlm3": {
        # Keep name aligned with user-facing label
        "Internlm3-8b-Instruct": {
            "path": "internlm/internlm3-8b-instruct",
            "params": "8B",
            "description": "InternLM3 8B Instruct model",
        }
    }
}


class InternLMCausalModelRunner(BaseModelRunner):
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

        # Ensure pad token exists for evaluator tokenization with padding
        try:
            if self.tokenizer is not None:
                if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            if getattr(getattr(self.model, "config", None), "pad_token_id", None) is None:
                eos_id = None
                if self.tokenizer is not None:
                    eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is None:
                    eos_id = getattr(getattr(self.model, "config", None), "eos_token_id", None)
                if eos_id is not None and getattr(self.model, "config", None) is not None:
                    self.model.config.pad_token_id = eos_id
        except Exception:
            pass

    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        torch_mod = _require_torch()
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer was not loaded for this runner. Set load_tokenizer=True.")

        tokenized_inputs, prompt_length = self._build_inputs(request)
        gen_kwargs = {**self.default_generation, **request.generation_kwargs}
        gen_kwargs.setdefault("max_new_tokens", 256)

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

        return GenerationResponse(
            text=text,
            raw_output=output_ids,
            request=request,
            metadata={
                "model_name": self.model_name,
                "prompt_length": int(prompt_length),
                "tokens_generated": int(generated_ids.shape[-1]),
            },
            generation_kwargs=gen_kwargs,
        )

    def _build_inputs(self, request: GenerationRequest) -> Tuple[Dict[str, Any], int]:
        if request.is_chat():
            if not self.supports_chat:
                raise UnsupportedContentError(f"Model '{self.model_name}' does not expose a chat template.")
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
        if self._hf_device_map and not self._is_single_device_map():
            return inputs
        device = self._infer_device()
        moved = {}
        for k, v in inputs.items():
            moved[k] = v.to(device) if hasattr(v, "to") else v
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


def register_internlm_models() -> None:
    registry = get_model_registry()

    def _create_factory(model_name: str, model_path: str, description: str):
        def factory(device: str = "cuda", **kwargs):
            generation_config = kwargs.pop("generation_config", None)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise ImportError("transformers library is required. Install with: pip install transformers") from exc

            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
            # InternLM3 generally requires trust_remote_code=True due to auto_map.
            trust_remote_code = kwargs.get("trust_remote_code", True)

            model_kwargs: Dict[str, Any] = {
                "device_map": device,
                "torch_dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
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
            _patch_transformers_for_internlm3()
            model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

            tokenizer = None
            if kwargs.get("load_tokenizer", True):
                tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=trust_remote_code)

            return InternLMCausalModelRunner(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                default_generation=generation_config,
            )

        return factory

    for generation, models in INTERNLM_MODELS.items():
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
                model_family="internlm",
                model_generation=generation,
                model_name=model_name,
                model_type="llm",
                parameters=config["params"],
                description=config["description"],
            )

        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="internlm",
                model_generation=gen,
                model_type="llm",
                available_models=available,
                description=f"InternLM {gen} model family factory",
            )
            def create_internlm_generation(model_name: str, device: str = "cuda", **kwargs):
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. Available models: {list(models_dict.keys())}"
                    )
                model_config = models_dict[model_name]
                model_path = model_config["path"]
                generation_config = kwargs.pop("generation_config", None)
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                except ImportError as exc:
                    raise ImportError("transformers library is required. Install with: pip install transformers") from exc

                torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
                trust_remote_code = kwargs.get("trust_remote_code", True)
                model_kwargs: Dict[str, Any] = {
                    "device_map": device,
                    "torch_dtype": torch_dtype,
                    "trust_remote_code": trust_remote_code,
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
                _patch_transformers_for_internlm3()
                model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

                tokenizer = None
                if kwargs.get("load_tokenizer", True):
                    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=trust_remote_code)

                return InternLMCausalModelRunner(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    default_generation=generation_config,
                )

            return create_internlm_generation

        _create_generation_factory(generation, models, available_models)

