"""
Gemma model registry.

Gemma 3 "it" checkpoints (e.g., gemma-3-27b-it) are multimodal (image+text → text).

Integration principles for multimodal models in this framework:
- Provide a runner that supports both plain text prompts and chat-style prompts.
- When image content is present, use `AutoProcessor` (not only `AutoTokenizer`) to build inputs.
- Keep `.tokenizer` available on the runner for compatibility with existing text-only evaluators.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

from llm_diagnose.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    UnsupportedContentError,
    PromptContent,
    PromptMessage,
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


def _require_pil_image():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PIL is required for multimodal image inputs. Install with `pip install pillow`."
        ) from exc
    return Image


def _load_image_from_content(content: PromptContent) -> Any:
    """
    Best-effort image loader for PromptContent.

    Supported:
    - content.data as PIL.Image
    - content.data as local file path (str)
    - content.data as bytes

    (We intentionally do NOT fetch remote URLs in this framework runner.)
    """
    Image = _require_pil_image()
    data = content.data
    if data is None:
        raise UnsupportedContentError("Image content is missing `data`.")
    # PIL Image
    if hasattr(data, "size") and hasattr(data, "mode"):
        return data
    # file path or URL
    if isinstance(data, str):
        s = data.strip()
        if s.startswith("http://") or s.startswith("https://"):
            # Best-effort fetch for model-card style usage. Users can also pass a local path.
            import io
            import urllib.request

            with urllib.request.urlopen(s) as resp:  # nosec - intended for user-provided URLs
                raw = resp.read()
            return Image.open(io.BytesIO(raw)).convert("RGB")
        return Image.open(s).convert("RGB")
    # bytes
    if isinstance(data, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(bytes(data))).convert("RGB")
    raise UnsupportedContentError(f"Unsupported image payload type: {type(data)!r}")


def _extract_images(messages: List[PromptMessage]) -> List[Any]:
    images: List[Any] = []
    for msg in messages:
        for part in msg.content:
            if part.type == "image":
                images.append(_load_image_from_content(part))
    return images


GEMMA_MODELS = {
    "gemma3": {
        "gemma-3-27b-it": {
            "path": "google/gemma-3-27b-it",
            "params": "27B",
            "description": "Gemma 3 27B IT (multimodal) model",
        }
    }
}


class Gemma3MultimodalRunner(BaseModelRunner):
    """
    Runner for Gemma3 multimodal checkpoints.

    Exposes:
    - `.model`: HF model
    - `.processor`: HF processor (tokenizer + image preprocessor)
    - `.tokenizer`: processor.tokenizer for compatibility with text-only evaluators
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        processor: Any,
        default_generation: Optional[Dict[str, Any]] = None,
    ):
        tokenizer = getattr(processor, "tokenizer", None)
        super().__init__(
            model_name=model_name,
            supports_chat=bool(tokenizer and hasattr(tokenizer, "apply_chat_template")),
            supports_multimodal=True,
        )
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.default_generation = default_generation or {}
        self._hf_device_map = getattr(model, "hf_device_map", None)

        # Best-effort pad token safety.
        try:
            if self.tokenizer is not None:
                if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            model_config = getattr(self.model, "config", None)
            if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
                eos_id = getattr(self.tokenizer, "eos_token_id", None) if self.tokenizer is not None else None
                if eos_id is None:
                    eos_id = getattr(model_config, "eos_token_id", None)
                if eos_id is not None:
                    model_config.pad_token_id = eos_id
        except Exception:
            pass

    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        torch_mod = _require_torch()
        if self.tokenizer is None:
            raise RuntimeError("Gemma3 runner requires a tokenizer (via processor.tokenizer).")

        inputs, prompt_length = self._build_inputs(request)
        gen_kwargs = {**self.default_generation, **request.generation_kwargs}
        gen_kwargs.setdefault("max_new_tokens", 256)

        inputs = self._maybe_move_to_device(inputs)
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
        """
        Build Gemma3 inputs.

        - Text-only: use tokenizer(...) like other runners.
        - Multimodal chat: use processor.apply_chat_template(...) and pass images into processor(...).
        """
        torch_mod = _require_torch()
        if request.is_chat():
            if not self.supports_chat:
                raise UnsupportedContentError(f"Model '{self.model_name}' does not expose a chat template.")
            if request.messages is None:
                raise UnsupportedContentError("Chat request missing messages.")

            hf_messages = request.to_hf_chat_messages()
            images = _extract_images(request.messages)

            # processor.apply_chat_template can directly build tensors when tokenize=True.
            # We prefer tokenize=False and then call processor(...) so images are handled consistently.
            rendered = self.processor.apply_chat_template(
                hf_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if images:
                enc = self.processor(
                    text=rendered,
                    images=images,
                    return_tensors="pt",
                )
            else:
                enc = self.tokenizer(rendered, return_tensors="pt", padding=False, add_special_tokens=True)

            prompt_length = int(enc["input_ids"].shape[-1])
            return dict(enc), prompt_length

        prompt_text = request.ensure_text_prompt()
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            add_special_tokens=True,
        )
        prompt_length = int(enc["input_ids"].shape[-1])
        return dict(enc), prompt_length

    def _maybe_move_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # If a multi-device map is present, let transformers handle placement.
        if self._hf_device_map and not self._is_single_device_map():
            return inputs
        device = self._infer_device()
        torch_mod = _require_torch()
        # Use model dtype (or fallback) for floating tensors (e.g., pixel_values).
        float_dtype = getattr(self.model, "dtype", None) or torch_mod.bfloat16
        moved: Dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                try:
                    if hasattr(v, "dtype") and hasattr(v, "is_floating_point") and v.is_floating_point():
                        moved[k] = v.to(device=device, dtype=float_dtype)
                    else:
                        moved[k] = v.to(device=device)
                except Exception:
                    moved[k] = v.to(device=device)
            else:
                moved[k] = v
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


def register_gemma_models() -> None:
    """
    Register Gemma models, including multimodal Gemma3.
    """
    registry = get_model_registry()

    def _create_factory(model_name: str, model_path: str, description: str):
        def factory(device: str = "cuda", **kwargs):
            generation_config = kwargs.pop("generation_config", None)
            try:
                from transformers import AutoProcessor  # type: ignore
            except ImportError as exc:
                raise ImportError("transformers is required. Install with: pip install transformers") from exc

            # Prefer Gemma3's explicit conditional generation class when available.
            # (Some transformers builds do not expose AutoModelForConditionalGeneration at top-level.)
            try:
                from transformers import Gemma3ForConditionalGeneration  # type: ignore
            except Exception:  # pragma: no cover
                Gemma3ForConditionalGeneration = None  # type: ignore
                from transformers import AutoModelForCausalLM  # type: ignore

            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
            trust_remote_code = kwargs.get("trust_remote_code", False)
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

            processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=trust_remote_code)

            if Gemma3ForConditionalGeneration is not None:
                model = Gemma3ForConditionalGeneration.from_pretrained(model_source, **model_kwargs)
            else:  # pragma: no cover
                model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

            return Gemma3MultimodalRunner(
                model_name=model_name,
                model=model,
                processor=processor,
                default_generation=generation_config,
            )

        return factory

    for generation, models in GEMMA_MODELS.items():
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
                model_family="gemma",
                model_generation=generation,
                model_name=model_name,
                model_type="mllm",
                parameters=config["params"],
                description=config["description"],
            )

        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="gemma",
                model_generation=gen,
                model_type="mllm",
                available_models=available,
                description=f"Gemma {gen} model family factory",
            )
            def create_gemma_generation(model_name: str, device: str = "cuda", **kwargs):
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. Available models: {list(models_dict.keys())}"
                    )
                model_config = models_dict[model_name]
                model_path = model_config["path"]
                generation_config = kwargs.pop("generation_config", None)
                try:
                    from transformers import AutoProcessor  # type: ignore
                except ImportError as exc:
                    raise ImportError("transformers is required. Install with: pip install transformers") from exc
                try:
                    from transformers import Gemma3ForConditionalGeneration  # type: ignore
                except ImportError as exc:
                    raise ImportError(
                        "Your installed transformers does not expose Gemma3ForConditionalGeneration. "
                        "Upgrade transformers to a version that supports Gemma3 (and/or install the gemma3 model package)."
                    ) from exc

                torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
                trust_remote_code = kwargs.get("trust_remote_code", False)
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
                processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=trust_remote_code)
                model = Gemma3ForConditionalGeneration.from_pretrained(model_source, **model_kwargs)
                return Gemma3MultimodalRunner(
                    model_name=model_name,
                    model=model,
                    processor=processor,
                    default_generation=generation_config,
                )

            return create_gemma_generation

        _create_generation_factory(generation, models, available_models)

