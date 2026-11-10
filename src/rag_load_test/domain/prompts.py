"""The prompts the workflow sends to the chat model, as a value object.

The texts live in ``config/prompts.yaml`` and reach the workflow through
``RagSettings``; this module only knows how to turn them into messages.

Example::

    prompts = RagPrompts(system="Cite [n].", user="{context}\\n\\nQ: {question}")
    prompts.messages("How many vacation days?", passages)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .passages import ScoredPassage

_PLACEHOLDERS = ("{context}", "{question}")


@dataclass(frozen=True)
class RagPrompts:
    """The system prompt and the user template (holding ``{context}`` and
    ``{question}``; any other literal brace must be doubled, as in ``str.format``)."""

    system: str
    user: str

    def __post_init__(self) -> None:
        for placeholder in _PLACEHOLDERS:
            if placeholder not in self.user:
                raise ValueError(
                    f"user prompt lacks the {placeholder} placeholder: {self.user!r}"
                )

    def messages(
        self, question: str, passages: Sequence[ScoredPassage]
    ) -> list[dict[str, str]]:
        """OpenAI-style messages: the system prompt, then the filled user template
        with the passages numbered ``[1] text`` one per line.

        Example::

            prompts.messages("What is alpha?", [alpha])[1]["content"]
            # "Context:\\n[1] Alpha text.\\n\\nQuestion: What is alpha?"
        """
        context = "\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, start=1))
        user = self.user.format(context=context, question=question)
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]
