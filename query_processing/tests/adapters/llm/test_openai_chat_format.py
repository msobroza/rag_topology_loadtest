"""The openai chat format: a pure data transformation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from query_processing.adapters.llm.api.openai_chat_format import OpenAIChatFormat
from query_processing.domain.components.component_requests import GenerateAnswer


def test_chat_request_has_the_openai_shape() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    body = OpenAIChatFormat().request_body("gpt", messages, 0.3)
    assert body == {"model": "gpt", "messages": messages, "temperature": 0.3}


def test_the_format_leaves_an_unstated_cap_out_of_the_body() -> None:
    assert "max_tokens" not in OpenAIChatFormat().request_body("m", [], 0.0)
    body = OpenAIChatFormat().request_body("m", [], 0.0, max_tokens=128)
    assert body["max_tokens"] == 128


def test_a_cap_and_a_response_format_go_in_one_body() -> None:
    response_format: dict[str, object] = {"type": "json_object"}
    body = OpenAIChatFormat().request_body(
        "m", [], 0.0, response_format=response_format, max_tokens=64
    )
    assert body == {
        "model": "m",
        "messages": [],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 64,
    }


@pytest.mark.parametrize(("content", "expected"), [("Paris.", "Paris."), (None, "")])
def test_chat_answer_reads_the_first_choice(content: str | None, expected: str) -> None:
    body = {"choices": [{"message": {"content": content}}, {"message": {}}]}
    assert OpenAIChatFormat().answer(body).text == expected


@pytest.mark.parametrize(
    "body", [{"choices": []}, {}, {"choices": [{}]}], ids=["empty", "none", "no-msg"]
)
def test_chat_malformed_bodies_fail_validation(body: object) -> None:
    with pytest.raises(ValidationError):
        OpenAIChatFormat().answer(body)


def test_the_format_reads_the_answer_and_its_token_counts() -> None:
    body = {
        "choices": [{"message": {"content": "Paris."}}],
        "usage": {"prompt_tokens": 829, "completion_tokens": 128, "total_tokens": 957},
    }
    assert OpenAIChatFormat().answer(body) == GenerateAnswer(
        text="Paris.", prompt_tokens=829, completion_tokens=128
    )


@pytest.mark.parametrize(
    ("usage", "prompt_tokens"),
    [("absent", None), (None, None), ({}, None), ({"prompt_tokens": 829}, 829)],
    ids=["absent", "null", "empty", "partial"],
)
def test_an_answer_with_no_or_partial_usage_is_still_an_answer(
    usage: object, prompt_tokens: int | None
) -> None:
    body: dict[str, object] = {"choices": [{"message": {"content": "Paris."}}]}
    if usage != "absent":
        body["usage"] = usage
    assert OpenAIChatFormat().answer(body) == GenerateAnswer(
        text="Paris.", prompt_tokens=prompt_tokens
    )
