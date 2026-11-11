"""OpenAI's embeddings API format: ``{model, input}`` in, ``data[].embedding`` out.

OVMS ``/v3/embeddings``, vLLM, LiteLLM and OpenAI itself all speak it. Rows may
come back in any ``index`` order; they are put back in request order.

Example:
    >>> api_format = OpenAIEmbeddingsFormat()
    >>> api_format.request_body("bge-small", ["a"])
    {'model': 'bge-small', 'input': ['a']}
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, field_validator, model_validator

from .api_format_bases import BaseApiRequest, BaseEmbeddingApiFormat


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


class OpenAIEmbeddingsFormat(BaseEmbeddingApiFormat):
    """The embeddings shape of OpenAI's ``POST /v1/embeddings``.

    Example:
        >>> body = {"data": [{"index": 0, "embedding": [1.0]}]}
        >>> OpenAIEmbeddingsFormat().vectors(body)
        [[1.0]]
    """

    def request_body(self, model: str, texts: Sequence[str]) -> dict[str, object]:
        """Build ``{model, input}`` with ``input`` a list, even for one text.

        Args:
            model: The model name sent with the request.
            texts: The texts to embed, in order.

        Returns:
            The JSON body.
        """
        return OpenAIEmbeddingsRequest(model=model, input=list(texts)).to_wire()

    def vectors(self, body: object) -> list[list[float]]:
        """Read ``data[].embedding``, re-ordered by ``index``.

        Args:
            body: The decoded JSON answer.

        Returns:
            One vector per row, in request order.

        Raises:
            pydantic.ValidationError: If ``body`` is not an embeddings answer, or
                its row indexes are not 0..n-1 once each.
        """
        response = OpenAIEmbeddingsResponse.model_validate(body)
        rows = sorted(response.data, key=lambda row: row.index)
        return [row.embedding for row in rows]
