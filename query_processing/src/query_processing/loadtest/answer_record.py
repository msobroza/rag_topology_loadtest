"""The answer record: what a target answered, without what changes from run to run.

``qp-record-answers`` writes ``<prefix>_answers.json``: for each request it
sent, in order, the endpoint, the body and the status, and from a 200's JSON
object the ``mode``, the ``metadata_filter`` applied, the ``answer`` and each
passage with its ``id``, the SHA-256 of its ``text``, its ``metadata`` and its
two scores. It also holds each topology alias and run label the responses
reported, counted as the target record counts them. It keeps none of
``timings_ms``, ``tokens``, ``request_id``, ``run_label`` or a passage's text:
they change from one run to the next, or say nothing a hash does not.
A response that is not a 200 with a JSON object is recorded with its status
alone, and a 200 whose object holds a key of the answer in another shape is
refused (``MalformedAnswerError``). ``qp-compare-answers`` reads the record
back through the same class, so a file of another shape fails with a message
naming the file, the field and the shape expected.

Example:
    >>> answer = recorded_answer("/retrieve", {"question": "q"}, 503, None)
    >>> AnswerRecord(seen=[], answers=[answer]).answers[0].passages is None
    True
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from .target_record import TargetPair, TargetSighting, sightings_of

ANSWER_RECORD_SUFFIX = "_answers.json"  # after the --out prefix
ANSWER_RECORD_SHAPE = (
    '{"seen": [{"topology": str | null, "run_label": str, "responses": int >= 0}], '
    '"answers": [{"endpoint": str, "body": object, "status": int, '
    '"mode"?: str, "metadata_filter"?: list, "answer"?: str | null, '
    '"passages"?: [{"id": str, "text_sha256": str, "metadata": object, '
    '"retrieval_score": float | null, "rerank_score": float | null}]}]}'
)
# What a record reads of a 200's body, named when a body is not of it.
ANSWER_BODY_SHAPE = (
    '{"mode"?: str, "metadata_filter"?: list, "answer"?: str | null, '
    '"passages"?: [{"id": str, "text"?: str, "metadata"?: object, '
    '"retrieval_score"?: float | null, "rerank_score"?: float | null}]}'
)
OK_STATUS = HTTPStatus.OK


class RecordedPassage(BaseModel):
    """One passage of an answer: its id, its text's hash, its metadata, its scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text_sha256: str
    metadata: dict[str, object]
    retrieval_score: float | None
    rerank_score: float | None


class RecordedAnswer(BaseModel):
    """What one request was and what the target answered to it.

    The fields after ``status`` are None for a response that was not a 200
    with a JSON object.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str
    body: dict[str, object]
    status: int
    mode: str | None = None
    metadata_filter: list[object] | None = None
    answer: str | None = None
    passages: list[RecordedPassage] | None = None


class AnswerRecord(BaseModel):
    """What one target answered to each request, in the order they were sent."""

    model_config = ConfigDict(extra="forbid")

    seen: list[TargetSighting]
    answers: list[RecordedAnswer]


def recorded_answer(
    endpoint: str, body: dict[str, object], status: int, payload: object
) -> RecordedAnswer:
    """Return what a response holds that another run must hold too.

    Args:
        endpoint: The route the request went to, ``/retrieve`` or ``/query``.
        body: The body sent.
        status: The response's status.
        payload: The response's JSON, or None when it was not JSON.

    Returns:
        The answer; its status alone unless a 200 brought a JSON object.

    Raises:
        MalformedAnswerError: If a key of a 200's JSON object is not of the
            answer's shape; the message names the endpoint, the field and the
            shape expected.

    Example:
        >>> recorded_answer("/query", {"question": "q"}, 200, {"mode": "query",
        ...     "answer": "a", "passages": [], "metadata_filter": []}).answer
        'a'
    """
    if status != OK_STATUS or not isinstance(payload, dict):
        return RecordedAnswer(endpoint=endpoint, body=body, status=status)
    read = _answer_body(endpoint, payload)
    return RecordedAnswer(
        endpoint=endpoint,
        body=body,
        status=status,
        mode=read.mode,
        metadata_filter=read.metadata_filter,
        answer=read.answer,
        passages=[_recorded_passage(passage) for passage in read.passages],
    )


class _PassageBody(BaseModel):
    # What a record reads of one passage of a 200's body; the rest is left out.
    model_config = ConfigDict(extra="ignore")

    id: str
    text: str = ""
    metadata: dict[str, object] = {}
    retrieval_score: float | None = None
    rerank_score: float | None = None


class _AnswerBody(BaseModel):
    # What a record reads of a 200's body. Every key is optional, so a body
    # holding none is recorded empty; a key of another type is refused.
    model_config = ConfigDict(extra="ignore")

    mode: str | None = None
    metadata_filter: list[object] | None = None
    answer: str | None = None
    passages: list[_PassageBody] = []


class MalformedAnswerError(ValueError):
    """A 200's JSON object is not an answer of the workflow app's shape.

    Example:
        >>> raise MalformedAnswerError("/retrieve: passages.0.id: Field required")
    """


def _answer_body(endpoint: str, payload: dict[str, object]) -> _AnswerBody:
    try:
        return _AnswerBody.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"])
        raise MalformedAnswerError(
            f"{endpoint}: {field}: {first['msg']}; expected the workflow app's "
            f"answer {ANSWER_BODY_SHAPE}"
        ) from exc


def _recorded_passage(passage: _PassageBody) -> RecordedPassage:
    # The text stands as its hash: equal texts are equal hashes, and a record
    # of a private corpus does not hold the corpus.
    return RecordedPassage(
        id=passage.id,
        text_sha256=hashlib.sha256(passage.text.encode("utf-8")).hexdigest(),
        metadata=passage.metadata,
        retrieval_score=passage.retrieval_score,
        rerank_score=passage.rerank_score,
    )


def answer_record(
    answers: list[RecordedAnswer], counts: Mapping[TargetPair, int]
) -> AnswerRecord:
    """Return the record of a run: its answers, and the pairs seen, most first.

    Args:
        answers: The answers, in the order the requests were sent.
        counts: Responses per ``(topology alias, run label)`` they reported.

    Returns:
        The record.

    Example:
        >>> answer_record([], {(None, "fake"): 2}).seen[0].responses
        2
    """
    return AnswerRecord(seen=sightings_of(counts), answers=answers)


def write_answer_record(prefix: Path, record: AnswerRecord) -> Path:
    """Write ``<prefix>_answers.json`` and return its path.

    Args:
        prefix: The ``--out`` prefix of ``qp-record-answers``.
        record: The whole record: it is written once, when every answer is in.

    Returns:
        The file written.

    Example:
        >>> write_answer_record(Path("results/monolith"), record).name
        'monolith_answers.json'
    """
    path = Path(f"{prefix}{ANSWER_RECORD_SUFFIX}")
    path.write_text(json.dumps(record.model_dump(), indent=2), encoding="utf-8")
    return path


def read_answer_record(prefix: Path) -> AnswerRecord:
    """Read ``<prefix>_answers.json``.

    Args:
        prefix: The ``--out`` prefix the record was written under.

    Returns:
        The record.

    Raises:
        ValueError: If the file is missing or not an answer record; the
            message names the file, the field at fault and the shape expected.

    Example:
        >>> len(read_answer_record(Path("results/monolith")).answers)
        100
    """
    path = Path(f"{prefix}{ANSWER_RECORD_SUFFIX}")
    try:
        return AnswerRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path}: {exc}; expected an answer record") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "the document"
        raise ValueError(
            f"{path}: {field}: {first['msg']}; expected an answer record "
            f"{ANSWER_RECORD_SHAPE}"
        ) from exc
