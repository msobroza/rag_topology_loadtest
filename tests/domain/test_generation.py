"""``Generation``: an answer's text and the token counts its endpoint reported."""

from __future__ import annotations

from rag_load_test.domain.generation import Generation


def test_a_generation_holds_its_text_and_its_counts() -> None:
    assert Generation("Paris.") == Generation("Paris.", None, None)
    generation = Generation("Paris.", prompt_tokens=829, completion_tokens=128)
    assert (generation.prompt_tokens, generation.completion_tokens) == (829, 128)
