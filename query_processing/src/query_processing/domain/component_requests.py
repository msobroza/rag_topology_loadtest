"""The request and the answer of each of the four existing component types.

Each is a frozen model that refuses an unknown key, so a request is what its
component type says and nothing else. A vector store's request also refuses the
fields its retrieval mode cannot do without: a dense search with no vector, a
lexical one with no text, a hybrid one with no rank constant for its fusion.

Example:
    >>> answer = await store.ainvoke(SearchRequest(
    ...     embedding=vector, text="vacation", top_k=20, retrieval_mode="hybrid",
    ...     rrf_k=60))
    >>> [p.id for p in answer.passages]
    ['p3', 'p1', ...]
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .passages import ScoredPassage
from .retrieval_mode import RetrievalMode


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbedRequest(_FrozenStrictModel):
    """A query to embed; the embedder puts its model's query prefix in front."""

    text: str


class EmbedAnswer(_FrozenStrictModel):
    """The query's vector."""

    embedding: list[float]


class RerankRequest(_FrozenStrictModel):
    """Retrieval candidates to re-score against a query, and how many to keep."""

    query: str
    passages: list[ScoredPassage]
    top_k: int = Field(ge=1)


class RerankAnswer(_FrozenStrictModel):
    """At most ``top_k`` passages, best first."""

    passages: list[ScoredPassage]


class GenerateRequest(_FrozenStrictModel):
    """An OpenAI-style message list, each message with ``role`` and ``content``."""

    messages: list[dict[str, str]]


class GenerateAnswer(_FrozenStrictModel):
    """The model's answer text."""

    text: str


class SearchRequest(_FrozenStrictModel):
    """A search of the vector store by vector, by words, or both fused.

    ``dense`` needs ``embedding``, ``lexical`` needs ``text``, and ``hybrid``
    needs both and ``rrf_k``, the rank constant of reciprocal rank fusion.
    """

    embedding: list[float] | None = None
    text: str | None = None
    top_k: int = Field(ge=1)
    retrieval_mode: RetrievalMode
    rrf_k: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _has_what_its_mode_needs(self) -> Self:
        needed = {
            "dense": ("embedding",),
            "lexical": ("text",),
            "hybrid": ("embedding", "text", "rrf_k"),
        }[self.retrieval_mode]
        missing = [field for field in needed if getattr(self, field) is None]
        if missing:
            raise ValueError(
                f"retrieval_mode={self.retrieval_mode!r} needs {', '.join(missing)} "
                f"(expected a request with {', '.join(needed)} set)"
            )
        return self


class SearchAnswer(_FrozenStrictModel):
    """The passages found, best first, each with its ``retrieval_score``."""

    passages: list[ScoredPassage]
