"""The openai chat format: a pure data transformation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_load_test.adapters.llm.api.openai_chat_format import OpenAIChatFormat
from rag_load_test.domain.generation import Generation


def test_chat_request_has_the_openai_shape() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    body = OpenAIChatFormat().request_body("gpt", messages, 0.3, 0)
    assert body == {"model": "gpt", "messages": messages, "temperature": 0.3}


@pytest.mark.parametrize(("content", "expected"), [("Paris.", "Paris."), (None, "")])
def test_chat_answer_reads_the_first_choice(content: str | None, expected: str) -> None:
    body = {"choices": [{"message": {"content": content}}, {"message": {}}]}
    assert OpenAIChatFormat().generation(body).text == expected


@pytest.mark.parametrize(
    "body", [{"choices": []}, {}, {"choices": [{}]}], ids=["empty", "none", "no-msg"]
)
def test_chat_malformed_bodies_fail_validation(body: object) -> None:
    with pytest.raises(ValidationError):
        OpenAIChatFormat().generation(body)


def test_the_format_leaves_a_cap_of_zero_out_of_the_body() -> None:
    assert "max_tokens" not in OpenAIChatFormat().request_body("m", [], 0.0, 0)
    assert OpenAIChatFormat().request_body("m", [], 0.0, 128)["max_tokens"] == 128


def test_the_format_reads_the_answer_and_its_token_counts() -> None:
    body = {
        "choices": [{"message": {"content": "Paris."}}],
        "usage": {"prompt_tokens": 829, "completion_tokens": 128, "total_tokens": 957},
    }
    assert OpenAIChatFormat().generation(body) == Generation("Paris.", 829, 128)


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 829}])
def test_an_answer_with_no_or_partial_usage_is_still_an_answer(
    usage: dict[str, int] | None,
) -> None:
    body = {"choices": [{"message": {"content": "Paris."}}], "usage": usage}
    assert OpenAIChatFormat().generation(body).text == "Paris."
    assert OpenAIChatFormat().generation(body).completion_tokens is None
