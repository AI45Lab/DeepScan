"""
Mistral model registry organized by generation.

This module registers Mistral-style causal LMs (Mistral / Mixtral / Ministral / etc.)
using Hugging Face `AutoModelForCausalLM` + `AutoTokenizer`.

Evaluators in this repo (TellMe/XBoundary/SPIN/MI-Peaks) expect the returned object to
expose:
- `.model` (the underlying HF model)
- `.tokenizer`
- `.device` (optional convenience property)

We also do best-effort pad token handling (some Mistral tokenizers omit pad_token).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List, Union

from deepscan.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    UnsupportedContentError,
    PromptMessage,
    PromptContent,
)
from deepscan.registry.model_registry import get_model_registry


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required to run generation. Install with `pip install torch`.") from exc
    return torch


def _coerce_torch_dtype(value: Any) -> Any:
    """
    Normalize dtype values coming from YAML/JSON configs.
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


MISTRAL_MODELS = {
    # Registry key (used as `model.generation` in YAML configs)
    "mistral": {
        "Mistral-Small-24B-Instruct-2501": {
            "path": "mistralai/Mistral-Small-24B-Instruct-2501",
            "params": "24B",
            "description": "Mistral Small 24B Instruct (2501)",
        }
    }
}

MINISTRAL3_MODELS = {
    # Ministral 3 is multimodal (image+text -> text) and uses dedicated HF classes.
    "ministral3": {
        "Ministral-3-14B-Instruct-2512": {
            "path": "mistralai/Ministral-3-14B-Instruct-2512",
            "params": "14B",
            "description": "Ministral 3 14B Instruct (2512) multimodal model",
        }
    }
}


class MistralCausalModelRunner(BaseModelRunner):
    """
    Thin wrapper that normalizes access to `generate` for Mistral causal models.
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


class Ministral3MultimodalRunner(BaseModelRunner):
    """
    Runner for Ministral 3 multimodal checkpoints (text + image_url -> text).

    Uses:
    - tokenizer: transformers.MistralCommonBackend
    - model: transformers.Mistral3ForConditionalGeneration
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        tokenizer: Any,
        default_generation: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name=model_name, supports_chat=True, supports_multimodal=True)
        self.model = model
        self.tokenizer = tokenizer
        self.default_generation = default_generation or {}
        self._hf_device_map = getattr(model, "hf_device_map", None)

    @property
    def device(self):
        torch_mod = _require_torch()
        if hasattr(self.model, "device"):
            return self.model.device  # type: ignore[attr-defined]
        try:
            return next(self.model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            return torch_mod.device("cpu")

    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        torch_mod = _require_torch()

        hf_messages = self._build_mistral_messages(request)
        tokenized = self.tokenizer.apply_chat_template(
            hf_messages,
            return_tensors="pt",
            return_dict=True,
        )
        if not isinstance(tokenized, dict) or "input_ids" not in tokenized:
            raise RuntimeError("Unexpected output from MistralCommonBackend.apply_chat_template(...)")

        prompt_length = int(tokenized["input_ids"].shape[-1])
        inputs = self._maybe_move_to_device(tokenized)

        gen_kwargs = {**self.default_generation, **request.generation_kwargs}
        gen_kwargs.setdefault("max_new_tokens", 256)

        image_sizes = self._infer_image_sizes(inputs)
        if image_sizes is not None:
            gen_kwargs.setdefault("image_sizes", image_sizes)

        with torch_mod.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # output_ids can be (batch, seq) or (seq,)
        if getattr(output_ids, "dim", lambda: 0)() == 1:
            generated_ids = output_ids[prompt_length:]
        else:
            generated_ids = output_ids[0, prompt_length:]

        text = self._decode(generated_ids)
        return GenerationResponse(
            text=text,
            raw_output=output_ids,
            request=request,
            metadata={
                "model_name": self.model_name,
                "prompt_length": int(prompt_length),
                "tokens_generated": int(getattr(generated_ids, "shape", [0])[-1]),
            },
            generation_kwargs=gen_kwargs,
        )

    def _decode(self, token_ids: Any) -> str:
        # MistralCommonBackend.decode(...) signature differs across versions.
        try:
            return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        except TypeError:
            return str(self.tokenizer.decode(token_ids))

    def _build_mistral_messages(self, request: GenerationRequest) -> List[Dict[str, Any]]:
        if request.is_chat():
            if request.messages is None:
                raise UnsupportedContentError("Chat request missing messages.")
            return [self._message_to_mistral_dict(m) for m in request.messages]

        prompt_text = request.ensure_text_prompt()
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]

    def _message_to_mistral_dict(self, msg: PromptMessage) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for part in msg.content:
            if part.type == "text":
                if part.text and part.text.strip():
                    content.append({"type": "text", "text": part.text})
                continue
            if part.type == "image":
                url = self._coerce_image_to_url(part)
                content.append({"type": "image_url", "image_url": {"url": url}})
                continue
            raise UnsupportedContentError(
                f"Ministral3 runner does not support content type: {part.type!r}"
            )
        if not content:
            raise UnsupportedContentError(f"Message from role '{msg.role}' has no supported content.")
        return {"role": msg.role, "content": content}

    def _coerce_image_to_url(self, part: PromptContent) -> str:
        data = part.data
        if isinstance(data, str) and data.strip():
            return data.strip()
        raise UnsupportedContentError(
            "Ministral3 runner expects image content to be a URL/path string in PromptContent.data."
        )

    def _infer_image_sizes(self, inputs: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return None
        try:
            if hasattr(pixel_values, "dim") and pixel_values.dim() == 4:
                # (n_images, 3, H, W) or (1, 3, H, W)
                n = int(pixel_values.shape[0])
                h, w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
                return [(h, w)] * max(1, n)
            if hasattr(pixel_values, "dim") and pixel_values.dim() == 5:
                # (batch, n_images, 3, H, W) - we only support batch=1 here.
                b = int(pixel_values.shape[0])
                if b != 1:
                    raise UnsupportedContentError("Ministral3 runner only supports batch size 1 for images.")
                n = int(pixel_values.shape[1])
                h, w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
                return [(h, w)] * max(1, n)
        except Exception:
            return None
        return None

    def _maybe_move_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # If a multi-device map is present, let transformers handle placement.
        if self._hf_device_map and not self._is_single_device_map():
            return inputs

        device = self.device
        torch_mod = _require_torch()
        float_dtype = getattr(self.model, "dtype", None) or torch_mod.bfloat16
        moved: Dict[str, Any] = {}
        for k, v in inputs.items():
            if not hasattr(v, "to"):
                moved[k] = v
                continue
            # Match HF sample: pixel_values in bf16 (or model dtype) on CUDA.
            if k == "pixel_values":
                try:
                    moved[k] = v.to(device=device, dtype=float_dtype)
                except Exception:
                    moved[k] = v.to(device=device)
            else:
                moved[k] = v.to(device=device)
        return moved

    def _is_single_device_map(self) -> bool:
        if not self._hf_device_map:
            return False
        if isinstance(self._hf_device_map, str):
            return True
        if isinstance(self._hf_device_map, dict):
            devices = {str(v) for v in self._hf_device_map.values() if v is not None}
            return len(devices) == 1
        return False


def register_mistral_models() -> None:
    """
    Register Mistral models and generation factories.
    """

    registry = get_model_registry()

    def _create_factory(model_name: str, model_path: str, description: str):
        def factory(device: str = "cuda", **kwargs):
            generation_config = kwargs.pop("generation_config", None)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise ImportError("transformers library is required. Install with: pip install transformers") from exc

            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
            model_kwargs: Dict[str, Any] = {
                "device_map": device,
                "torch_dtype": torch_dtype,
                # Mistral models generally do not require remote code.
                "trust_remote_code": kwargs.get("trust_remote_code", False),
            }

            if kwargs.get("load_in_8bit", False):
                model_kwargs["load_in_8bit"] = True
            elif kwargs.get("load_in_4bit", False):
                model_kwargs["load_in_4bit"] = True
                if "bitsandbytes" in kwargs:
                    model_kwargs["bnb_4bit_compute_dtype"] = _coerce_torch_dtype(kwargs.get("bnb_4bit_compute_dtype", "float16"))

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

            return MistralCausalModelRunner(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                default_generation=generation_config,
            )

        return factory

    for generation, models in MISTRAL_MODELS.items():
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
                model_family="mistral",
                model_generation=generation,
                model_name=model_name,
                model_type="llm",
                parameters=config["params"],
                description=config["description"],
            )

        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="mistral",
                model_generation=gen,
                model_type="llm",
                available_models=available,
                description=f"Mistral {gen} model family factory",
            )
            def create_mistral_generation(model_name: str, device: str = "cuda", **kwargs):
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. Available models: {list(models_dict.keys())}"
                    )
                model_config = models_dict[model_name]
                factory = _create_factory(
                    model_name=model_name,
                    model_path=model_config["path"],
                    description=model_config["description"],
                )
                return factory(device=device, **kwargs)

            return create_mistral_generation

        _create_generation_factory(generation, models, available_models)

    # Ministral3 multimodal family (separate from causal Mistral runners).
    def _create_ministral3_factory(model_name: str, model_path: str, description: str):
        def factory(device: str = "auto", **kwargs):
            generation_config = kwargs.pop("generation_config", None)
            try:
                from transformers import Mistral3ForConditionalGeneration  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Ministral3 requires a `transformers` build that exposes "
                    "`Mistral3ForConditionalGeneration`. Upgrade with: pip install -U transformers"
                ) from exc

            # Tokenizer/backend name differs across transformers versions:
            # - Newer builds: `MistralCommonBackend`
            # - Older builds: `MistralCommonTokenizer` (in tokenization_mistral_common), which requires `mistral-common`.
            try:
                from transformers import MistralCommonBackend as _MistralBackend  # type: ignore
            except Exception:
                # Avoid importing `transformers.tokenization_mistral_common` unless we have a sufficiently new
                # `mistral-common` installed; otherwise that module can error during import with missing symbols.
                try:
                    from importlib.metadata import version as _pkg_version  # type: ignore
                except Exception:  # pragma: no cover
                    _pkg_version = None  # type: ignore

                def _parse_version_tuple(v: str):
                    parts = []
                    for seg in str(v).split("."):
                        if seg.isdigit():
                            parts.append(int(seg))
                        else:
                            # strip dev/rc tags (best-effort)
                            num = ""
                            for ch in seg:
                                if ch.isdigit():
                                    num += ch
                                else:
                                    break
                            parts.append(int(num) if num else 0)
                    return tuple(parts)

                mistral_common_ver = None
                if _pkg_version is not None:
                    try:
                        mistral_common_ver = _pkg_version("mistral-common")
                    except Exception:
                        mistral_common_ver = None

                min_required = (1, 8, 6)
                if mistral_common_ver is None or _parse_version_tuple(mistral_common_ver) < min_required:
                    have_txt = f" (installed: mistral-common=={mistral_common_ver})" if mistral_common_ver else ""
                    raise ImportError(
                        "Ministral3 (Transformers) requires `mistral-common>=1.8.6` for the tokenizer backend"
                        f"{have_txt}.\n"
                        "Fix:\n"
                        "  pip install -U \"mistral-common>=1.8.6\"\n"
                        "If your PyPI mirror is pinned, install directly from GitHub:\n"
                        "  pip install -U \"mistral-common @ git+https://github.com/mistralai/mistral-common.git\"\n"
                        "For FP8 + latest support, install transformers from main:\n"
                        "  pip install -U git+https://github.com/huggingface/transformers"
                    )
                try:
                    from transformers.tokenization_mistral_common import (  # type: ignore
                        MistralCommonTokenizer as _MistralBackend,
                    )
                except Exception as exc:  # pragma: no cover
                    try:
                        from transformers.utils.import_utils import is_mistral_common_available  # type: ignore
                    except Exception:
                        is_mistral_common_available = None  # type: ignore

                    needs_mistral_common = bool(is_mistral_common_available and not is_mistral_common_available())
                    if needs_mistral_common:
                        raise ImportError(
                            "Ministral3 requires the optional dependency `mistral-common` for the Mistral tokenizer. "
                            "Install it with: pip install mistral-common\n"
                            "If you still see import errors, also upgrade transformers: pip install -U transformers"
                        ) from exc
                    # mistral-common is present but may be too old for this transformers build.
                    try:
                        from importlib.metadata import version as _pkg_version  # type: ignore
                    except Exception:  # pragma: no cover
                        _pkg_version = None  # type: ignore
                    mistral_common_ver = None
                    if _pkg_version is not None:
                        try:
                            mistral_common_ver = _pkg_version("mistral-common")
                        except Exception:
                            mistral_common_ver = None

                    # Check for a couple of symbols that `transformers.tokenization_mistral_common` expects.
                    missing_symbols: List[str] = []
                    try:
                        from mistral_common.tokens.tokenizers import base as _mc_base  # type: ignore
                        if not hasattr(_mc_base, "SpecialTokenPolicy"):
                            missing_symbols.append("SpecialTokenPolicy")
                    except Exception:
                        missing_symbols.append("mistral_common.tokens.tokenizers.base")
                    try:
                        from mistral_common.tokens.tokenizers.image import MultiModalVersion  # type: ignore
                        _ = MultiModalVersion
                    except Exception:
                        missing_symbols.append("mistral_common.tokens.tokenizers.image.MultiModalVersion")

                    if missing_symbols:
                        ver_txt = f" (installed: mistral-common=={mistral_common_ver})" if mistral_common_ver else ""
                        raise ImportError(
                            "Ministral3 tokenizer backend requires a newer `mistral-common` than the one available"
                            f"{ver_txt}. Missing: {', '.join(missing_symbols)}.\n"
                            "Fix (preferred): upgrade from a source that has a newer mistral-common:\n"
                            "  pip install -U \"mistral-common>=1.8.6\"\n"
                            "If your internal PyPI mirror is pinned, install directly from GitHub:\n"
                            "  pip install -U \"mistral-common @ git+https://github.com/mistralai/mistral-common.git\"\n"
                            "For FP8 + latest support, install transformers from main:\n"
                            "  pip install -U git+https://github.com/huggingface/transformers"
                        ) from exc
                    raise ImportError(
                        "Ministral3 tokenizer backend could not be imported. "
                        "Try installing `mistral-common` and upgrading transformers:\n"
                        "  pip install -U \"mistral-common>=1.8.6\" transformers\n"
                        "Or transformers main:\n"
                        "  pip install -U git+https://github.com/huggingface/transformers"
                    ) from exc

            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
            model_kwargs: Dict[str, Any] = {
                "device_map": device,
                "torch_dtype": torch_dtype,
                "trust_remote_code": kwargs.get("trust_remote_code", False),
            }
            for key in ["max_memory", "offload_folder", "low_cpu_mem_usage", "attn_implementation"]:
                if key in kwargs:
                    model_kwargs[key] = kwargs[key]

            model_source = kwargs.pop("path", None) or model_path
            tokenizer = _MistralBackend.from_pretrained(model_source)
            model = Mistral3ForConditionalGeneration.from_pretrained(model_source, **model_kwargs)
            return Ministral3MultimodalRunner(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                default_generation=generation_config,
            )

        return factory

    for generation, models in MINISTRAL3_MODELS.items():
        available_models = list(models.keys())
        for model_name, config in models.items():
            registry_name = f"{generation}/{model_name}"
            registry.register_model(
                registry_name,
                factory=_create_ministral3_factory(
                    model_name=model_name,
                    model_path=config["path"],
                    description=config["description"],
                ),
                model_family="mistral",
                model_generation=generation,
                model_name=model_name,
                model_type="mllm",
                parameters=config["params"],
                description=config["description"],
            )

        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="mistral",
                model_generation=gen,
                model_type="mllm",
                available_models=available,
                description=f"Mistral {gen} multimodal model family factory",
            )
            def create_ministral3_generation(model_name: str, device: str = "auto", **kwargs):
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. Available models: {list(models_dict.keys())}"
                    )
                model_config = models_dict[model_name]
                kwargs = dict(kwargs)
                kwargs["path"] = kwargs.get("path") or model_config["path"]
                return _create_ministral3_factory(model_name, model_config["path"], model_config["description"])(
                    device=device, **kwargs
                )

            return create_ministral3_generation

        _create_generation_factory(generation, models, available_models)
