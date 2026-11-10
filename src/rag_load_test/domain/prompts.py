"""The prompt the workflow sends to the chat model."""

from __future__ import annotations

from collections.abc import Sequence

from .passages import ScoredPassage

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Cite passages as [n]. If the context is insufficient, say so."
)


def rag_messages(
    question: str, passages: Sequence[ScoredPassage]
) -> list[dict[str, str]]:
    """OpenAI-style messages: system prompt + numbered context + question."""
    context = "\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
