"""The request and answer shapes of the standard formats, one per component.

Embedding speaks OpenAI's embeddings format, rerank Cohere's rerank format and
generation OpenAI's chat format. These classes are the project's own contract:
the endpoint adapters build and read them, the model server answers in them and
the load test sends them. The format names and the ``/v3`` routes are in
``domain/model_api.py``.

Example:
    >>> CohereRerankRequest(model="r", query="q", documents=["a"]).to_wire()
    {'model': 'r', 'query': 'q', 'documents': ['a']}
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class BaseApiRequest(BaseModel):
    """A request body of some API format; ``to_wire`` is what goes on the wire.

    Example:
        >>> CohereRerankRequest(model="r", query="q", documents=[]).to_wire()
        {'model': 'r', 'query': 'q', 'documents': []}
    """

    def to_wire(self) -> dict[str, object]:
        """Return the JSON body, with unset optional fields left out, never null.

        Returns:
            The body as a dict, ready for ``json=``.

        Example:
            >>> CohereRerankRequest(model="r", query="q", documents=[]).to_wire()
            {'model': 'r', 'query': 'q', 'documents': []}
        """
        return self.model_dump(exclude_none=True)


# --- OpenAI embeddings: ``{model, input}`` in, ``data[].embedding`` out -----


class OpenAIEmbeddingsRequest(BaseApiRequest):
    """``{model, input}``; ``input`` is always a list, even for one text."""

    model: str
    input: list[str]

    @field_validator("input", mode="before")
    @classmethod
    def _one_text_as_a_list(cls, value: object) -> object:
        # The format also allows one bare string; our model server must accept
        # it as OVMS does, so a received request reads it as a one-text list.
        return [value] if isinstance(value, str) else value


class OpenAIEmbeddingRow(BaseModel):
    """One vector; ``index`` is the position of its text in the request ``input``."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class OpenAIEmbeddingsResponse(BaseModel):
    """The answer; readers ignore fields they do not use, such as ``usage``."""

    object: str = "list"
    model: str = ""
    data: list[OpenAIEmbeddingRow]

    @model_validator(mode="after")
    def _one_row_per_index(self) -> OpenAIEmbeddingsResponse:
        # A missing or repeated index would silently pair a vector with the
        # wrong text once the rows are sorted.
        indexes = sorted(row.index for row in self.data)
        if indexes != list(range(len(indexes))):
            raise ValueError(
                f"row indexes {indexes} (expected 0..{len(indexes) - 1}, once each)"
            )
        return self


# --- Cohere rerank: ``{model, query, documents, top_n}`` in, scores out -----


class CohereRerankRequest(BaseApiRequest):
    """Score ``documents`` against ``query``; the server keeps ``top_n``."""

    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


class CohereRerankResult(BaseModel):
    """One score; ``index`` is the position of its document in the request."""

    index: int
    relevance_score: float


class CohereRerankResponse(BaseModel):
    """The answer: at most ``top_n`` results; ``id`` and ``meta`` are ignored."""

    results: list[CohereRerankResult]


# --- OpenAI chat: messages in, ``choices[0]`` out ---------------------------


class OpenAIChatRequest(BaseApiRequest):
    """``{model, messages, temperature}`` of ``POST /v1/chat/completions``."""

    model: str
    messages: list[dict[str, str]]
    temperature: float


class OpenAIChatMessage(BaseModel):
    """The answer's message; ``content`` is null for, say, a tool call."""

    content: str | None = None


class OpenAIChatChoice(BaseModel):
    """One completion choice."""

    message: OpenAIChatMessage


class OpenAIChatResponse(BaseModel):
    """The answer; an empty ``choices`` list carries no answer, so it is malformed."""

    choices: list[OpenAIChatChoice] = Field(min_length=1)
