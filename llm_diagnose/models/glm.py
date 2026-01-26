"""
GLM model registry (Z.ai / zai-org).

GLM-4.5 checkpoints may use newer/less common Transformers model classes (glm4_moe)
and sometimes rely on remote code. This runner is intentionally conservative:
- Prefer HF chat templates (`tokenizer.apply_chat_template`) when available.
- Provide best-effort fallbacks for older GLM/ChatGLM-style tokenizers.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

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


GLM_MODELS = {
    # Keep registry keys aligned with other families (qwen2.5, llama3.3, internvl3.5, etc.)
    "glm4.5": {
        "GLM-4.5-Air": {
            "path": "zai-org/GLM-4.5-Air",
            "params": "MoE",
            "description": "GLM-4.5-Air (Z.ai) MoE causal LM",
        },
    }
}


class GLMCausalModelRunner(BaseModelRunner):
    """
    Thin wrapper that normalizes access to `generate` for GLM causal models.
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

        # Best-effort: ensure pad token exists so evaluator tokenization with padding works.
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
            raise RuntimeError(
                "Tokenizer was not loaded for this runner. Set load_tokenizer=True when creating the model."
            )

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
            return self._build_chat_inputs(request)

        prompt_text = request.ensure_text_prompt()
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            add_special_tokens=True,
        )
        prompt_length = encoded["input_ids"].shape[-1]
        return encoded, prompt_length

    def _build_chat_inputs(self, request: GenerationRequest) -> Tuple[Dict[str, Any], int]:
        # Prefer HF chat templates (modern GLM checkpoints should provide them).
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer missing.")
        chat_messages = request.to_hf_chat_messages()
        if hasattr(self.tokenizer, "apply_chat_template"):
            tokenized = self.tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return {"input_ids": tokenized}, tokenized.shape[-1]

        # Fallback for older GLM/ChatGLM-style tokenizers.
        # Many expose `build_chat_input(query, history=...)` where history is list[tuple[str, str]].
        if hasattr(self.tokenizer, "build_chat_input"):
            system_prefix = ""
            history: List[Tuple[str, str]] = []
            last_user: Optional[str] = None
            pending_user: Optional[str] = None
            for msg in chat_messages:
                role = msg.get("role")
                content = msg.get("content")
                text = content if isinstance(content, str) else ""
                if role == "system":
                    system_prefix = (system_prefix + "\n" + text).strip() if text else system_prefix
                elif role == "user":
                    pending_user = text
                    last_user = text
                elif role == "assistant":
                    if pending_user is not None:
                        history.append((pending_user, text))
                        pending_user = None
            query = last_user or ""
            if system_prefix:
                query = f"{system_prefix}\n{query}".strip()
            built = self.tokenizer.build_chat_input(query, history=history)  # type: ignore[attr-defined]
            # Common return shape is dict with input_ids (and maybe attention_mask, position_ids).
            if isinstance(built, dict) and "input_ids" in built:
                prompt_length = built["input_ids"].shape[-1]
                return built, prompt_length

        raise UnsupportedContentError(
            f"Model '{self.model_name}' does not expose a supported chat template/builder."
        )

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


def register_glm_models() -> None:
    """
    Register GLM models and generation factories.

    Provides:
    - Individual keys: `glm4.5/GLM-4.5-Air` (and alias `glm/GLM-4.5-Air`)
    - Family keys: `glm4.5` and `glm` (both accept model_name=...)
    """

    registry = get_model_registry()

    def _create_factory(model_name: str, model_path: str, description: str):
        def factory(device: str = "cuda", **kwargs):
            generation_config = kwargs.pop("generation_config", None)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "transformers is required. Install with: pip install transformers "
                    "(GLM-4.5 may require a recent transformers build with glm4_moe support)."
                ) from exc

            trust_remote_code = kwargs.get("trust_remote_code", True)
            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
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

            for key in ["max_memory", "offload_folder", "low_cpu_mem_usage", "attn_implementation", "revision"]:
                if key in kwargs:
                    model_kwargs[key] = kwargs[key]

            model_source = kwargs.pop("path", None) or model_path
            model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

            tokenizer = None
            if kwargs.get("load_tokenizer", True):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    trust_remote_code=trust_remote_code,
                )

            return GLMCausalModelRunner(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                default_generation=generation_config,
            )

        return factory

    # Register all explicit generations in the map (e.g., glm4.5).
    for generation, models in GLM_MODELS.items():
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
                model_family="glm",
                model_generation=generation,
                model_name=model_name,
                model_type="llm",
                parameters=config.get("params"),
                description=config["description"],
            )

        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="glm",
                model_generation=gen,
                model_type="llm",
                available_models=available,
                description=f"GLM {gen} model family factory",
            )
            def create_glm_generation(model_name: str, device: str = "cuda", **kwargs):
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. Available models: {list(models_dict.keys())}"
                    )
                model_config = models_dict[model_name]
                return _create_factory(
                    model_name=model_name,
                    model_path=model_config["path"],
                    description=model_config["description"],
                )(device=device, **kwargs)

            return create_glm_generation

        _create_generation_factory(generation, models, available_models)

    # Convenience alias: `glm` points at the same available models (currently glm4.5 only).
    # This lets users set `generation: glm` in configs without thinking about sub-versioning.
    alias_models: Dict[str, Any] = {}
    for _gen, models in GLM_MODELS.items():
        alias_models.update(models)

    if alias_models:
        available_models = list(alias_models.keys())

        @registry.register_model(
            "glm",
            model_family="glm",
            model_generation="glm",
            model_type="llm",
            available_models=available_models,
            description="GLM model family factory (alias across GLM generations)",
        )
        def create_glm(model_name: str, device: str = "cuda", **kwargs):
            if model_name not in alias_models:
                raise ValueError(
                    f"Model '{model_name}' not found in glm. Available models: {available_models}"
                )
            cfg = alias_models[model_name]
            return _create_factory(
                model_name=model_name,
                model_path=cfg["path"],
                description=cfg["description"],
            )(device=device, **kwargs)

        for model_name, config in alias_models.items():
            registry.register_model(
                f"glm/{model_name}",
                factory=_create_factory(model_name=model_name, model_path=config["path"], description=config["description"]),
                model_family="glm",
                model_generation="glm",
                model_name=model_name,
                model_type="llm",
                parameters=config.get("params"),
                description=config["description"],
            )

