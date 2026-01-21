"""
Qwen model registry organized by generation.

This module registers Qwen models by generation (qwen, qwen2, qwen3) since
different generations may have different configurations even at the same size.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

from llm_diagnose.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    UnsupportedContentError,
)
from llm_diagnose.registry.model_registry import get_model_registry
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
        # Fall back to transformers' own parsing if it supports additional strings.
        return value
    return value




# Qwen model configurations organized by generation
QWEN_MODELS = {
    "qwen": {
        "Qwen-0.5B": {
            "path": "Qwen/Qwen-0.5B",
            "params": "0.5B",
            "description": "Qwen 0.5B parameter model",
        },
        "Qwen-1.8B": {
            "path": "Qwen/Qwen-1.8B",
            "params": "1.8B",
            "description": "Qwen 1.8B parameter model",
        },
        "Qwen-7B": {
            "path": "Qwen/Qwen-7B",
            "params": "7B",
            "description": "Qwen 7B parameter model",
        },
        "Qwen-14B": {
            "path": "Qwen/Qwen-14B",
            "params": "14B",
            "description": "Qwen 14B parameter model",
        },
        "Qwen-72B": {
            "path": "Qwen/Qwen-72B",
            "params": "72B",
            "description": "Qwen 72B parameter model",
        },
    },
    "qwen2": {
        "Qwen2-0.5B": {
            "path": "Qwen/Qwen2-0.5B",
            "params": "0.5B",
            "description": "Qwen2 0.5B parameter model",
        },
        "Qwen2-1.5B": {
            "path": "Qwen/Qwen2-1.5B",
            "params": "1.5B",
            "description": "Qwen2 1.5B parameter model",
        },
        "Qwen2-7B": {
            "path": "Qwen/Qwen2-7B",
            "params": "7B",
            "description": "Qwen2 7B parameter model",
        },
        "Qwen2-72B": {
            "path": "Qwen/Qwen2-72B",
            "params": "72B",
            "description": "Qwen2 72B parameter model",
        },
    },
    "qwen2.5": {
        "Qwen2.5-7B-Instruct": {
            "path": "/root/models/Qwen2.5-7B-Instruct",
            "params": "7B",
            "description": "Qwen2.5 7B Instruct model",
        },
    },
    "qwen3": {
        "Qwen3-0.6B": {
            "path": "Qwen/Qwen3-0.6B",
            "params": "0.6B",
            "description": "Qwen3 0.6B parameter model",
        },
        "Qwen3-1.5B": {
            "path": "Qwen/Qwen3-1.5B",
            "params": "1.5B",
            "description": "Qwen3 1.5B parameter model",
        },
        "Qwen3-2B": {
            "path": "Qwen/Qwen3-2B",
            "params": "2B",
            "description": "Qwen3 2B parameter model",
        },
        "Qwen3-8B": {
            "path": "Qwen/Qwen3-8B",
            "params": "8B",
            "description": "Qwen3 8B parameter model",
        },
        "Qwen3-14B": {
            "path": "Qwen/Qwen3-14B",
            "params": "14B",
            "description": "Qwen3 14B parameter model",
        },
        "Qwen3-32B": {
            "path": "Qwen/Qwen3-32B",
            "params": "32B",
            "description": "Qwen3 32B parameter model",
        },
    },
}


class QwenCausalModelRunner(BaseModelRunner):
    """
    Thin wrapper that normalizes access to `generate` for Qwen causal models.
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

    def _build_inputs(
        self, request: GenerationRequest
    ) -> Tuple[Dict[str, Any], int]:
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
        """
        Expose the model device for convenience (e.g., tests moving tensors).
        """
        return self._infer_device()

    def _is_single_device_map(self) -> bool:
        """
        Detect whether hf_device_map effectively points to a single device.
        """
        if not self._hf_device_map:
            return False
        if isinstance(self._hf_device_map, str):
            return True
        if isinstance(self._hf_device_map, dict):
            devices = {str(v) for v in self._hf_device_map.values() if v is not None}
            return len(devices) == 1
        return False


def register_qwen_models():
    """
    Register all Qwen models organized by generation.
    
    This function registers:
    1. Individual model variants (qwen/Qwen-7B, qwen2/Qwen2-7B, qwen3/Qwen3-7B, etc.)
    2. Generation factories (qwen, qwen2, qwen3) that can create any variant in that generation
    
    Usage:
        from llm_diagnose.models import register_qwen_models
        register_qwen_models()
        
        # Then use:
        from llm_diagnose.registry.model_registry import get_model_registry
        registry = get_model_registry()
        
        # Individual variant
        model = registry.get_model("qwen3/Qwen3-7B")
        
        # Or generation factory
        model = registry.get_model("qwen3", model_name="Qwen3-7B")
    """
    registry = get_model_registry()
    
    def _create_factory(generation: str, model_name: str, model_path: str, model_params: str, description: str):
        """Create a factory function for a specific Qwen model."""
        def factory(device: str = "cuda", **kwargs):
            """
            Create a Qwen model.
            
            Args:
                device: Device to load model on (default: "cuda")
                **kwargs: Additional arguments for model loading
                    - torch_dtype / dtype: Data type for model weights
                    - trust_remote_code: Whether to trust remote code
                    - load_in_8bit: Load in 8-bit mode
                    - load_in_4bit: Load in 4-bit mode
                    - load_tokenizer: Whether to load tokenizer (default: True)
                    - path: Optional override path to a local checkpoint directory
            
            Returns:
                Loaded Qwen model (and tokenizer if load_tokenizer=True)
            """
            generation_config = kwargs.pop("generation_config", None)

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except ImportError:
                raise ImportError(
                    "transformers library is required. Install with: pip install transformers"
                )
            
            # Default arguments
            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
            model_kwargs = {
                "device_map": device,
                "torch_dtype": torch_dtype,
                # Qwen tokenizers/models frequently require remote code; default True for reproducibility.
                "trust_remote_code": kwargs.get("trust_remote_code", True),
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
            for key in ["max_memory", "offload_folder", "low_cpu_mem_usage"]:
                if key in kwargs:
                    model_kwargs[key] = kwargs[key]
            
            model_source = kwargs.pop("path", None) or model_path

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **model_kwargs
            )
            
            tokenizer = None
            if kwargs.get("load_tokenizer", True):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    trust_remote_code=kwargs.get("trust_remote_code", True),
                )

            return QwenCausalModelRunner(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                default_generation=generation_config,
            )
        
        return factory
    
    # Register individual models and generation factories
    for generation, models in QWEN_MODELS.items():
        available_models = list(models.keys())
        
        # Register individual model variants
        for model_name, config in models.items():
            registry_name = f"{generation}/{model_name}"
            
            registry.register_model(
                registry_name,
                factory=_create_factory(
                    generation,
                    model_name,
                    config["path"],
                    config["params"],
                    config["description"],
                ),
                model_family="qwen",
                model_generation=generation,
                model_name=model_name,
                model_type="llm",
                parameters=config["params"],
                description=config["description"],
            )
        
        # Register generation factory (need to capture generation and models in closure)
        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="qwen",
                model_generation=gen,
                model_type="llm",
                available_models=available,
                description=f"Qwen {gen} model family factory",
            )
            def create_qwen_generation(
                model_name: str,
                device: str = "cuda",
                **kwargs
            ):
                """
                Create a Qwen model from the specified generation.
                
                Args:
                    model_name: Model name (e.g., "Qwen3-7B", "Qwen2-7B")
                    device: Device to load model on (default: "cuda")
                **kwargs: Additional arguments for model loading
                        - torch_dtype / dtype: Data type for model weights
                        - trust_remote_code: Whether to trust remote code
                        - load_in_8bit: Load in 8-bit mode
                        - load_in_4bit: Load in 4-bit mode
                        - load_tokenizer: Whether to load tokenizer (default: True)
                        - path: Optional override path to a local checkpoint directory
                
                Returns:
                    Loaded Qwen model (and tokenizer if load_tokenizer=True)
                    
                Raises:
                    ValueError: If model_name is not available in this generation
                """
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. "
                        f"Available models: {list(models_dict.keys())}"
                    )
                
                model_config = models_dict[model_name]
                model_path = model_config["path"]
                
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                except ImportError:
                    raise ImportError(
                        "transformers library is required. Install with: pip install transformers"
                    )
                
                # Default arguments
                generation_config = kwargs.pop("generation_config", None)

                torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
                model_kwargs = {
                    "device_map": device,
                    "torch_dtype": torch_dtype,
                    "trust_remote_code": kwargs.get("trust_remote_code", True),
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
                for key in ["max_memory", "offload_folder", "low_cpu_mem_usage"]:
                    if key in kwargs:
                        model_kwargs[key] = kwargs[key]
                
                # Load model
                model_source = kwargs.pop("path", None) or model_path

                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    **model_kwargs
                )
                
                tokenizer = None
                if kwargs.get("load_tokenizer", True):
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_source,
                        trust_remote_code=kwargs.get("trust_remote_code", True),
                    )
                
                return QwenCausalModelRunner(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    default_generation=generation_config,
                )
            
            return create_qwen_generation
        
        _create_generation_factory(generation, models, available_models)

