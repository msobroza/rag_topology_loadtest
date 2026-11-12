"""The openai chat format: a pure data transformation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_load_test.adapters.llm.api.openai_chat_format import OpenAIChatFormat


def test_chat_request_has_the_openai_shape() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    body = OpenAIChatFormat().request_body("gpt", messages, 0.3)
    assert body == {"model": "gpt", "messages": messages, "temperature": 0.3}


@pytest.mark.parametrize(("content", "expected"), [("Paris.", "Paris."), (None, "")])
def test_chat_answer_reads_the_first_choice(content: str | None, expected: str) -> None:
    body = {"choices": [{"message": {"content": content}}, {"message": {}}]}
    assert OpenAIChatFormat().answer(body) == expected


@pytest.mark.parametrize(
    "body", [{"choices": []}, {}, {"choices": [{}]}], ids=["empty", "none", "no-msg"]
)
def test_chat_malformed_bodies_fail_validation(body: object) -> None:
    with pytest.raises(ValidationError):
        OpenAIChatFormat().answer(body)
