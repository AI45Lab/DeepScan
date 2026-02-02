"""
InternVL multimodal model registry (InternVL2.5 / InternVL3.5 style).

Key differences vs standard HF VLMs (e.g., Gemma3):
- InternVL "GitHub format" uses custom remote code and exposes `model.chat(...)`
  / `model.batch_chat(...)` instead of a pure `generate()` + `AutoProcessor` flow.
- Image preprocessing uses dynamic tiling (min/max patches) and IMAGENET normalization.

Design goals in this framework:
- Provide a runner that can handle text-only generation and image+text generation.
- Keep heavy deps optional and imported lazily (torchvision/decord).
- Expose `.tokenizer` and a text-backbone `.model` for compatibility with existing
  text-only evaluators (XBoundary/TellMe/SPIN/MI-Peaks).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List, Union

from deepscan.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    UnsupportedContentError,
    PromptContent,
    PromptMessage,
)
from deepscan.registry.model_registry import get_model_registry


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required to run InternVL. Install with `pip install torch`.") from exc
    return torch


def _require_pil_image():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PIL is required for image inputs. Install with `pip install pillow`.") from exc
    return Image


def _require_torchvision():
    try:
        import torchvision.transforms as T  # type: ignore
        from torchvision.transforms.functional import InterpolationMode  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torchvision is required for InternVL image preprocessing. Install with `pip install torchvision`."
        ) from exc
    return T, InterpolationMode


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


# Model-card defaults
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    T, InterpolationMode = _require_torchvision()
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img),
            T.Resize((int(input_size), int(input_size)), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_image_from_content(content: PromptContent) -> Any:
    """
    Supported:
    - torch Tensor already preprocessed (N,3,H,W) or (3,H,W)
    - PIL.Image
    - local path str
    - http(s) URL str
    - bytes
    """
    torch_mod = _require_torch()
    data = content.data
    if data is None:
        raise UnsupportedContentError("Image content is missing `data`.")
    if isinstance(data, torch_mod.Tensor):
        return data
    Image = _require_pil_image()
    if hasattr(data, "size") and hasattr(data, "mode"):
        return data
    if isinstance(data, str):
        s = data.strip()
        if s.startswith("http://") or s.startswith("https://"):
            import io
            import urllib.request

            with urllib.request.urlopen(s) as resp:  # nosec - intended for user-provided URLs
                raw = resp.read()
            return Image.open(io.BytesIO(raw)).convert("RGB")
        return Image.open(s).convert("RGB")
    if isinstance(data, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(bytes(data))).convert("RGB")
    raise UnsupportedContentError(f"Unsupported image payload type: {type(data)!r}")


def _extract_images(messages: List[PromptMessage]) -> List[Union[Any, "torch.Tensor"]]:
    images: List[Any] = []
    for msg in messages:
        for part in msg.content:
            if part.type == "image":
                images.append(_load_image_from_content(part))
    return images


def _preprocess_images_to_pixel_values(
    images: List[Any],
    *,
    input_size: int,
    max_num: int,
    min_num: int = 1,
    use_thumbnail: bool = True,
) -> Tuple[Any, List[int]]:
    """
    Returns:
      pixel_values: torch.Tensor with shape (sum_patches, 3, H, W)
      num_patches_list: per-image patch counts
    """
    torch_mod = _require_torch()

    # If caller already provided pixel_values tensors, accept them directly.
    # For multiple images, user can provide multiple tensors and we'll concat.
    if images and all(isinstance(x, torch_mod.Tensor) for x in images):
        tensors = []
        patch_counts = []
        for t in images:
            if t.dim() == 3:
                t = t.unsqueeze(0)
            if t.dim() != 4:
                raise UnsupportedContentError("Preprocessed pixel_values tensor must be 3D or 4D.")
            tensors.append(t)
            patch_counts.append(int(t.shape[0]))
        return torch_mod.cat(tensors, dim=0), patch_counts

    transform = _build_transform(input_size)
    tensors_out = []
    patch_counts = []
    for img in images:
        tiles = _dynamic_preprocess(
            img,
            min_num=min_num,
            max_num=max_num,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
        )
        pix = [transform(tile) for tile in tiles]
        pix = torch_mod.stack(pix, dim=0)
        tensors_out.append(pix)
        patch_counts.append(int(pix.shape[0]))
    if not tensors_out:
        return torch_mod.empty((0, 3, input_size, input_size)), []
    return torch_mod.cat(tensors_out, dim=0), patch_counts


INTERNVL_MODELS = {
    # Use dot in generation key (consistent with qwen2.5, internlm3.5 naming).
    "internvl3.5": {
        # GitHub format names (as in HF model cards)
        "InternVL3.5-14B": {
            "path": "OpenGVLab/InternVL3_5-14B",
            "params": "14B",
            "description": "InternVL3.5 14B (GitHub format) multimodal model",
        },
        "InternVL3.5-14B-Instruct": {
            "path": "OpenGVLab/InternVL3_5-14B-Instruct",
            "params": "14B",
            "description": "InternVL3.5 14B Instruct (GitHub format) multimodal model",
        },
        "InternVL3.5-241B-A28B": {
            "path": "OpenGVLab/InternVL3_5-241B-A28B",
            "params": "241B-A28B",
            "description": "InternVL3.5 241B-A28B (GitHub format) multimodal model",
        },
    }
}


class InternVLChatRunner(BaseModelRunner):
    """
    Runner for InternVLChatModel.

    - Uses `vlm.chat(...)` for generation.
    - Exposes `.model` as the underlying LLM (`vlm.language_model`) so existing
      text-only evaluators can still operate on it when used with text datasets.
    """

    def __init__(
        self,
        model_name: str,
        vlm: Any,
        tokenizer: Any,
        *,
        default_generation: Optional[Dict[str, Any]] = None,
        input_size: int = 448,
        max_num: int = 12,
        min_num: int = 1,
        use_thumbnail: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            supports_chat=bool(tokenizer and hasattr(tokenizer, "apply_chat_template")),
            supports_multimodal=True,
        )
        self.vlm = vlm
        # expose text backbone for evaluators
        self.model = getattr(vlm, "language_model", vlm)
        self.tokenizer = tokenizer
        self.default_generation = default_generation or {}
        self.input_size = int(input_size)
        self.max_num = int(max_num)
        self.min_num = int(min_num)
        self.use_thumbnail = bool(use_thumbnail)

    @property
    def device(self):
        torch_mod = _require_torch()
        try:
            return self.vlm.device
        except Exception:
            return torch_mod.device("cpu")

    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        torch_mod = _require_torch()
        gen_cfg = {**self.default_generation, **request.generation_kwargs}
        gen_cfg.setdefault("max_new_tokens", 256)

        if request.is_chat():
            if request.messages is None:
                raise UnsupportedContentError("Chat request missing messages.")
            # InternVL's chat API accepts pixel_values for the current turn; be cautious:
            # if images appear in earlier turns, we reject (would require replaying per-turn pixels).
            if any(
                any(part.type == "image" for part in msg.content)
                for msg in request.messages[:-1]
            ):
                raise UnsupportedContentError(
                    "InternVL runner currently supports images only in the latest user message."
                )

            # Build history from prior user/assistant turns.
            history: List[Tuple[str, str]] = []
            last_user: Optional[str] = None
            last_assistant: Optional[str] = None
            for msg in request.messages:
                if msg.role == "user":
                    last_user = msg.as_plain_text()
                elif msg.role == "assistant":
                    last_assistant = msg.as_plain_text()
                    if last_user is not None:
                        history.append((last_user, last_assistant))
                        last_user = None
                        last_assistant = None
                elif msg.role == "system":
                    # InternVL uses internal templates; prepend system message to first user turn if present.
                    pass

            # Current question is the last user message text (with image placeholders injected by internvl).
            question = request.messages[-1].as_plain_text()

            images = _extract_images(request.messages[-1:])
            pixel_values, num_patches_list = _preprocess_images_to_pixel_values(
                images,
                input_size=self.input_size,
                max_num=self.max_num,
                min_num=self.min_num,
                use_thumbnail=self.use_thumbnail,
            )
            if pixel_values.numel() == 0:
                pixel_values = None
                num_patches_list = None
            else:
                pixel_values = pixel_values.to(dtype=torch_mod.bfloat16).to(self.device)

            # InternVL's chat will add "<image>\\n" when history is None and pixel_values is not None.
            response = self.vlm.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_cfg,
                history=history if history else None,
                return_history=False,
                num_patches_list=num_patches_list,
            )
            text = str(response)
            return GenerationResponse(text=text, raw_output=response, request=request, generation_kwargs=gen_cfg)

        # Text-only prompt
        prompt = request.ensure_text_prompt()
        response = self.vlm.chat(self.tokenizer, None, prompt, gen_cfg, history=None, return_history=False)
        return GenerationResponse(text=str(response), raw_output=response, request=request, generation_kwargs=gen_cfg)


def register_internvl_models() -> None:
    registry = get_model_registry()

    def _create_factory(model_name: str, model_path: str, description: str):
        def factory(device: str = "cuda", **kwargs):
            generation_config = kwargs.pop("generation_config", None)
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise ImportError("transformers is required. Install with: pip install transformers") from exc

            torch_dtype = _coerce_torch_dtype(kwargs.get("torch_dtype", kwargs.get("dtype", "auto")))
            trust_remote_code = kwargs.get("trust_remote_code", True)

            # InternVL supports these extra kwargs (passed through to from_pretrained):
            # low_cpu_mem_usage, use_flash_attn, device_map, load_in_8bit, etc.
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
                "device_map": device,
            }
            for key in [
                "low_cpu_mem_usage",
                "use_flash_attn",
                "load_in_8bit",
                "load_in_4bit",
                "max_memory",
                "offload_folder",
            ]:
                if key in kwargs:
                    model_kwargs[key] = kwargs[key]

            model_source = kwargs.pop("path", None) or model_path

            vlm = AutoModel.from_pretrained(model_source, **model_kwargs).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=trust_remote_code,
                use_fast=kwargs.get("use_fast", False),
            )

            # Default preprocess params from config if present
            cfg = getattr(vlm, "config", None)
            input_size = int(getattr(cfg, "force_image_size", 448) or 448)
            max_num = int(getattr(cfg, "max_dynamic_patch", 12) or 12)
            min_num = int(getattr(cfg, "min_dynamic_patch", 1) or 1)
            use_thumbnail = bool(getattr(cfg, "use_thumbnail", True))

            return InternVLChatRunner(
                model_name=model_name,
                vlm=vlm,
                tokenizer=tokenizer,
                default_generation=generation_config,
                input_size=input_size,
                max_num=max_num,
                min_num=min_num,
                use_thumbnail=use_thumbnail,
            )

        return factory

    for generation, models in INTERNVL_MODELS.items():
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
                model_family="internvl",
                model_generation=generation,
                model_name=model_name,
                model_type="mllm",
                parameters=config["params"],
                description=config["description"],
            )

        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="internvl",
                model_generation=gen,
                model_type="mllm",
                available_models=available,
                description=f"InternVL {gen} model family factory",
            )
            def create_internvl_generation(model_name: str, device: str = "cuda", **kwargs):
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. Available models: {list(models_dict.keys())}"
                    )
                model_config = models_dict[model_name]
                kwargs = dict(kwargs)
                kwargs["path"] = kwargs.get("path") or model_config["path"]
                return _create_factory(model_name, model_config["path"], model_config["description"])(device=device, **kwargs)

            return create_internvl_generation

        _create_generation_factory(generation, models, available_models)

