"""Pure conversion of a fetched corpus file's text into documents or questions.

Blank lines are skipped; every other line must be a JSON object of the expected
shape, or a ``CorpusFormatError`` names the file and the line.

Example:
    >>> reference_documents("References/V1/AMF.jsonl", '{"text": "Un sinistre."}')[0].id
    'AMF:00001'
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from ...domain.corpus import SourceDocument
from ...domain.passages import MetadataValue


class CorpusFormatError(ValueError):
    """A line of a corpus file that is not what its format promises.

    Args:
        source: The file, as the corpus names it.
        lineno: The 1-based line number in that file.
        detail: What is wrong with the line.

    Example:
        >>> str(CorpusFormatError("QnA/V1/q.jsonl", 3, "no question"))
        'QnA/V1/q.jsonl: line 3: no question'
    """

    def __init__(self, source: str, lineno: int, detail: str) -> None:
        super().__init__(f"{source}: line {lineno}: {detail}")
        self.source, self.lineno = source, lineno


class _ReferenceLine(BaseModel):
    # A reference passage; other keys a future release adds are ignored.
    model_config = ConfigDict(extra="ignore")
    text: str
    url: str | None = None


LineT = TypeVar("LineT", bound=BaseModel)


class _QuestionLine(BaseModel):
    # A question with its evaluation material, of which only the question is kept.
    model_config = ConfigDict(extra="ignore")
    question: str


def reference_documents(source: str, text: str) -> list[SourceDocument]:
    """Convert a reference file: one document per non-blank line.

    Args:
        source: The file's path in the corpus; its stem is each document's title
            and ``source`` metadata.
        text: The file's content, JSON lines ``{"text", "url"?}``.

    Returns:
        The documents in file order, ids ``<stem>:<line number, 5 digits>``,
        metadata ``{"source": stem}`` plus ``url`` when the line has one.

    Example:
        >>> reference_documents("References/V1/GAA.jsonl", '{"text": "t"}')[0].title
        'GAA'
    """
    stem = PurePosixPath(source).stem
    return [
        _document(stem, lineno, line.text, line.url)
        for lineno, line in _parsed_lines(source, text, _ReferenceLine)
    ]


def corpus_questions(source: str, text: str) -> list[str]:
    """Convert a questions file: the ``question`` of each non-blank line.

    Args:
        source: The file's path in the corpus, for error messages.
        text: The file's content, JSON lines with a ``question`` key.

    Returns:
        The questions in file order.

    Example:
        >>> corpus_questions("QnA/V1/q.jsonl", '{"question": "Qui paie ?"}')
        ['Qui paie ?']
    """
    return [line.question for _, line in _parsed_lines(source, text, _QuestionLine)]


def _document(stem: str, lineno: int, text: str, url: str | None) -> SourceDocument:
    metadata: dict[str, MetadataValue] = {"source": stem}
    if url:  # many lines have no url, some a null one
        metadata["url"] = url
    # Five digits keep the ids of a file sorted in line order (about 1,200 lines).
    return SourceDocument(
        id=f"{stem}:{lineno:05d}", title=stem, text=text, metadata=metadata
    )


def _parsed_lines(
    source: str, text: str, shape: type[LineT]
) -> Iterator[tuple[int, LineT]]:
    # Each non-blank line validated as ``shape``, with its 1-based line number.
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield lineno, shape.model_validate(json.loads(line))
        except ValueError as exc:  # json.JSONDecodeError and ValidationError
            detail = f"{_describe(exc)}: {line[:80]!r}"
            raise CorpusFormatError(source, lineno, detail) from exc


def _describe(exc: ValueError) -> str:
    # One line: a ValidationError's own text spans several and names our model.
    if not isinstance(exc, ValidationError):
        return f"{type(exc).__name__}: {exc}"
    return "; ".join(
        f"{'.'.join(map(str, error['loc'])) or '<line>'}: {error['msg']}"
        for error in exc.errors()
    )
