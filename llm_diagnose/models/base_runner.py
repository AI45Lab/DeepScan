"""
Abstract model runner interfaces for unified generation across model families.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from abc import ABC, abstractmethod

PromptRole = Literal["system", "user", "assistant"]
ContentType = Literal["text", "image", "audio", "video"]


class UnsupportedContentError(ValueError):
    """Raised when a prompt contains content that the runner cannot handle."""


@dataclass
class PromptContent:
    """
    A single piece of content inside a prompt message.

    Supports text by default and leaves room for additional modalities.
    """

    type: ContentType = "text"
    text: Optional[str] = None
    data: Optional[Any] = None
    mime_type: Optional[str] = None
    description: Optional[str] = None

    def is_text(self) -> bool:
        return self.type == "text"


@dataclass
class PromptMessage:
    """Structured chat/message representation."""

    role: PromptRole
    content: List[PromptContent]

    def as_plain_text(self) -> str:
        """Concatenate textual parts for backends that only accept raw text."""
        parts = [part.text.strip() for part in self.content if part.is_text() and part.text]
        return " ".join(parts).strip()

    def to_hf_dict(self) -> Dict[str, Any]:
        """
        Convert to the Hugging Face chat template format.

        Non-text payloads are emitted as dictionaries, relying on downstream
        tokenizers to understand them.
        """
        serialized_content: List[Union[str, Dict[str, Any]]] = []
        for part in self.content:
            if part.is_text():
                if part.text is not None:
                    serialized_content.append(part.text)
                continue
            payload: Dict[str, Any] = {"type": part.type}
            if part.data is not None:
                payload["data"] = part.data
            if part.mime_type:
                payload["mime_type"] = part.mime_type
            if part.description:
                payload["description"] = part.description
            serialized_content.append(payload)

        if not serialized_content:
            raise UnsupportedContentError(
                f"Message from role '{self.role}' has no supported content."
            )

        if len(serialized_content) == 1 and isinstance(serialized_content[0], str):
            content: Union[str, List[Union[str, Dict[str, Any]]]] = serialized_content[0]
        else:
            content = serialized_content

        return {"role": self.role, "content": content}


@dataclass
class GenerationRequest:
    """
    Normalized representation of a generation call.
    """

    prompt: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)

    def is_chat(self) -> bool:
        return bool(self.messages)

    def with_updates(self, **generation_kwargs: Any) -> "GenerationRequest":
        merged_kwargs = {**self.generation_kwargs, **generation_kwargs}
        return replace(self, generation_kwargs=merged_kwargs)

    def ensure_text_prompt(self) -> str:
        if self.prompt:
            return self.prompt
        if self.messages:
            # Flatten chat messages into a single plain-text prompt for
            # backends that only understand raw text input.
            text_segments = [msg.as_plain_text() for msg in self.messages]
            text = " ".join(filter(None, text_segments)).strip()
            if text:
                return text
        raise UnsupportedContentError(
            "GenerationRequest does not contain a plain-text prompt."
        )

    def to_hf_chat_messages(self) -> List[Dict[str, Any]]:
        if not self.messages:
            raise UnsupportedContentError("No chat messages are present in the request.")
        return [msg.to_hf_dict() for msg in self.messages]

    @classmethod
    def from_text(cls, prompt: str, **generation_kwargs: Any) -> "GenerationRequest":
        return cls(prompt=prompt, generation_kwargs=dict(generation_kwargs))

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[PromptMessage],
        **generation_kwargs: Any,
    ) -> "GenerationRequest":
        return cls(messages=list(messages), generation_kwargs=dict(generation_kwargs))


@dataclass
class GenerationResponse:
    """Standardized generation output."""

    text: str
    raw_output: Any
    request: Optional[GenerationRequest] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseModelRunner(ABC):
    """
    Abstract interface that adapters must implement to expose generation.
    """

    def __init__(
        self,
        model_name: str,
        *,
        supports_chat: bool = False,
        supports_multimodal: bool = False,
    ):
        self.model_name = model_name
        self.supports_chat = supports_chat
        self.supports_multimodal = supports_multimodal

    def generate(
        self,
        request: Union[GenerationRequest, str],
        **generation_kwargs: Any,
    ) -> GenerationResponse:
        """
        Entry-point for running inference.

        Accepts either a pre-built GenerationRequest or a raw string prompt
        for convenience.
        """
        normalized_request = self._coerce_request(request, generation_kwargs)
        return self._generate(normalized_request)

    def _coerce_request(
        self,
        request: Union[GenerationRequest, str],
        generation_kwargs: Dict[str, Any],
    ) -> GenerationRequest:
        if isinstance(request, GenerationRequest):
            return request.with_updates(**generation_kwargs)
        prompt = str(request)
        return GenerationRequest.from_text(prompt, **generation_kwargs)

    @abstractmethod
    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        """Model-specific implementation."""
        raise NotImplementedError


