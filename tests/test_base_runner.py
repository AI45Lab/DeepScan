import pytest

from llm_diagnose.models.base_runner import (
    BaseModelRunner,
    GenerationRequest,
    GenerationResponse,
    PromptContent,
    PromptMessage,
    UnsupportedContentError,
)


class DummyRunner(BaseModelRunner):
    """Minimal runner used to test the BaseModelRunner contract."""

    def __init__(self):
        super().__init__(model_name="dummy", supports_chat=True)

    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="dummy-output",
            raw_output={
                "prompt": request.prompt,
                "messages": request.messages,
                "generation_kwargs": request.generation_kwargs,
            },
            request=request,
            generation_kwargs=request.generation_kwargs,
        )


def test_generate_wraps_string_into_request():
    runner = DummyRunner()
    response = runner.generate("Hello world", temperature=0.5)

    assert response.text == "dummy-output"
    assert response.request is not None
    assert response.request.prompt == "Hello world"
    assert response.request.generation_kwargs["temperature"] == 0.5
    assert response.raw_output["prompt"] == "Hello world"


def test_generate_merges_kwargs_for_existing_request():
    runner = DummyRunner()
    request = GenerationRequest.from_text("ping", max_new_tokens=32)

    response = runner.generate(request, temperature=0.1)

    assert response.request.generation_kwargs == {
        "max_new_tokens": 32,
        "temperature": 0.1,
    }


def test_chat_request_supports_plain_text_projection():
    runner = DummyRunner()
    chat_request = GenerationRequest.from_messages(
        [
            PromptMessage(
                role="system",
                content=[PromptContent(text="You are concise.")],
            ),
            PromptMessage(
                role="user",
                content=[
                    PromptContent(text="Summarize the registry pattern.")
                ],
            ),
        ],
        max_new_tokens=16,
    )

    response = runner.generate(chat_request)

    assert response.request.is_chat()
    assert chat_request.ensure_text_prompt() == (
        "You are concise. Summarize the registry pattern."
    )
    assert response.raw_output["messages"] == chat_request.messages


def test_prompt_message_without_text_raises():
    message = PromptMessage(role="user", content=[PromptContent(text=None)])
    request = GenerationRequest.from_messages([message])

    with pytest.raises(UnsupportedContentError):
        request.to_hf_chat_messages()


