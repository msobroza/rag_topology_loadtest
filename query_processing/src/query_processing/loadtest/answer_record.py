"""The answer record: what a target answered, without what changes from run to run.

``qp-record-answers`` writes ``<prefix>_answers.json``: for each request it
sent, in order, the endpoint, the body and the status, and from a 200's JSON
object the ``mode``, the ``metadata_filter`` applied, the ``answer`` and, under
``records``, each vector-store record with the SHA-256 of its ``text``, its
``metadata`` holding its ``id``, and its two scores. It also holds each
topology alias and run label the responses reported, counted as the target
record counts them. It keeps none of ``timings_ms``, ``tokens``,
``request_id``, ``run_label`` or a vector-store record's text: they change
from one run to the next, or say nothing a hash does not.
A response that is not a 200 with a JSON object is recorded with its status
alone, and a 200 whose object holds a key of the answer in another shape is
refused (``MalformedAnswerError``). ``qp-compare-answers`` reads the record
back through the same class, so a file of another shape fails with a message
naming the file, the field and the shape expected.

The workflow app answered in an old shape, and the pinned answer records and
the registry's are of the old form: the vector-store records under
``passages``, each with its ``id`` beside its metadata, the handbook's window
index under the metadata key ``chunk``. ``answer_in_the_new_shape`` is where
the two meet: a response's body and a record's file are read through it, so
the form held and written is the new one whichever was read. The old form is
read with no end date.

Example:
    >>> answer = recorded_answer("/retrieve", {"question": "q"}, 503, None)
    >>> AnswerRecord(seen=[], answers=[answer]).answers[0].records is None
    True
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

from .target_record import TargetPair, TargetSighting, sightings_of

ANSWER_RECORD_SUFFIX = "_answers.json"  # after the --out prefix
_OLD_FORM = (
    '; or of the old form, "passages" in place of "records", each '
    '{"id": str, "metadata": object, ...} with its id beside its metadata, a '
    'whole number under the metadata key "chunk" read as "sequence"'
)
ANSWER_RECORD_SHAPE = (
    '{"seen": [{"topology": str | null, "run_label": str, "responses": int >= 0}], '
    '"answers": [{"endpoint": str, "body": object, "status": int, '
    '"mode"?: str, "metadata_filter"?: list, "answer"?: str | null, '
    '"records"?: [{"text_sha256": str, "metadata": {"id": str, ...}, '
    '"retrieval_score": float | null, "rerank_score": float | null}]}]}' + _OLD_FORM
)
# What a record reads of a 200's body, named when a body is not of it.
ANSWER_BODY_SHAPE = (
    '{"mode"?: str, "metadata_filter"?: list, "answer"?: str | null, '
    '"records"?: [{"text"?: str, "metadata": {"id": str, ...}, '
    '"retrieval_score"?: float | null, "rerank_score"?: float | null}]}; or of '
    'the old shape, "passages"?: [{"id": str, "text"?: str, "metadata"?: object, '
    '...}] in place of "records", a whole number under the metadata key '
    '"chunk" read as "sequence"'
)
OK_STATUS = HTTPStatus.OK
_OLD_LIST_KEY, _LIST_KEY = "passages", "records"
# doclang keeps "chunk" for the chunk itself; the handbook's window index,
# under "chunk" in an index ingested before, is doclang's "sequence".
_OLD_SEQUENCE_KEY, _SEQUENCE_KEY = "chunk", "sequence"


def answer_in_the_new_shape(answer: Mapping[str, object]) -> dict[str, object]:
    """Return an answer of either shape in the new one; the input is not changed.

    The list under ``passages`` is read under ``records`` (unless ``records``
    is there too), each entry's ``id`` is read into its metadata, and a whole
    number under the metadata key ``chunk`` is read as ``sequence``, in either
    shape. A text under ``chunk`` is a chunk's own payload and stays where it
    is, and a boolean is no whole number. What is of neither shape is left as
    it is, for the validation to refuse.

    Args:
        answer: A 200's JSON object, or one answer of an answer record's file.

    Returns:
        A new dict of the new shape.

    Example:
        >>> answer_in_the_new_shape({"passages": [{"id": "a", "metadata": {}}]})
        {'records': [{'metadata': {'id': 'a'}}]}
    """
    new = dict(answer)
    if _OLD_LIST_KEY in new and _LIST_KEY not in new:
        new[_LIST_KEY] = new.pop(_OLD_LIST_KEY)
    entries = new.get(_LIST_KEY)
    if isinstance(entries, list):
        new[_LIST_KEY] = [_entry_in_the_new_shape(entry) for entry in entries]
    return new


def _entry_in_the_new_shape(entry: object) -> object:
    if not isinstance(entry, Mapping):
        return entry
    new = dict(entry)
    metadata = new.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return new
    moved = _metadata_in_the_new_shape(metadata)
    # The id beside the metadata is the one the store holds the record under,
    # which the new shape's metadata holds: it wins.
    if "id" in new:
        moved["id"] = new.pop("id")
    new["metadata"] = moved
    return new


def _metadata_in_the_new_shape(metadata: Mapping[str, object]) -> dict[str, object]:
    new = dict(metadata)
    index = new.get(_OLD_SEQUENCE_KEY)
    whole = isinstance(index, int) and not isinstance(index, bool)
    if whole and _SEQUENCE_KEY not in new:
        new[_SEQUENCE_KEY] = new.pop(_OLD_SEQUENCE_KEY)
    return new


def _holding_its_id(metadata: dict[str, object]) -> dict[str, object]:
    # A vector-store record has no id field: its id is metadata["id"].
    if isinstance(metadata.get("id"), str):
        return metadata
    raise ValueError(
        f"the metadata {metadata!r} holds no id; expected a str under 'id'"
    )


MetadataWithItsId = Annotated[dict[str, object], AfterValidator(_holding_its_id)]


class RecordedVectorStoreRecord(BaseModel):
    """One vector-store record of an answer: its text's hash, its metadata, its scores.

    Its id is ``metadata["id"]``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text_sha256: str
    metadata: MetadataWithItsId
    retrieval_score: float | None
    rerank_score: float | None

    @property
    def record_id(self) -> str:
        """The id the store holds the record under, ``metadata["id"]``."""
        return str(self.metadata["id"])


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
    records: list[RecordedVectorStoreRecord] | None = None


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
        payload: The response's JSON, or None when it was not JSON; of either
            shape (``answer_in_the_new_shape``).

    Returns:
        The answer, of the new form; its status alone unless a 200 brought a
        JSON object.

    Raises:
        MalformedAnswerError: If a key of a 200's JSON object is not of the
            answer's shape; the message names the endpoint, the field and the
            shape expected.

    Example:
        >>> recorded_answer("/query", {"question": "q"}, 200, {"mode": "query",
        ...     "answer": "a", "records": [], "metadata_filter": []}).answer
        'a'
    """
    if status != OK_STATUS or not isinstance(payload, dict):
        return RecordedAnswer(endpoint=endpoint, body=body, status=status)
    read = _answer_body(endpoint, answer_in_the_new_shape(payload))
    return RecordedAnswer(
        endpoint=endpoint,
        body=body,
        status=status,
        mode=read.mode,
        metadata_filter=read.metadata_filter,
        answer=read.answer,
        records=[_recorded(entry) for entry in read.records],
    )


class _RecordBody(BaseModel):
    # What a record reads of one vector-store record of a 200's body; the rest
    # is left out.
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    metadata: MetadataWithItsId
    retrieval_score: float | None = None
    rerank_score: float | None = None


class _AnswerBody(BaseModel):
    # What a record reads of a 200's body, in the new shape. Every key is
    # optional, so a body holding none is recorded empty; a key of another
    # type is refused.
    model_config = ConfigDict(extra="ignore")

    mode: str | None = None
    metadata_filter: list[object] | None = None
    answer: str | None = None
    records: list[_RecordBody] = []


class MalformedAnswerError(ValueError):
    """A 200's JSON object is not an answer of the workflow app's shape.

    Example:
        >>> raise MalformedAnswerError("/retrieve: records.0.metadata: ...")
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


def _recorded(entry: _RecordBody) -> RecordedVectorStoreRecord:
    # The text stands as its hash: equal texts are equal hashes, and a record
    # of a private corpus does not hold the corpus.
    return RecordedVectorStoreRecord(
        text_sha256=hashlib.sha256(entry.text.encode("utf-8")).hexdigest(),
        metadata=entry.metadata,
        retrieval_score=entry.retrieval_score,
        rerank_score=entry.rerank_score,
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
    """Write ``<prefix>_answers.json``, of the new form, and return its path.

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
    """Read ``<prefix>_answers.json``, of either form, into the new one.

    Args:
        prefix: The ``--out`` prefix the record was written under.

    Returns:
        The record; each answer read through ``answer_in_the_new_shape``.

    Raises:
        ValueError: If the file is missing or not an answer record; the
            message names the file, the field at fault and the shape expected.

    Example:
        >>> len(read_answer_record(Path("results/monolith")).answers)
        100
    """
    path = Path(f"{prefix}{ANSWER_RECORD_SUFFIX}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return AnswerRecord.model_validate(_record_in_the_new_form(document))
    except OSError as exc:
        raise ValueError(f"{path}: {exc}; expected an answer record") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "the document"
        raise ValueError(
            f"{path}: {field}: {first['msg']}; expected an answer record "
            f"{ANSWER_RECORD_SHAPE}"
        ) from exc
    except ValueError as exc:  # not JSON; a ValidationError is caught above
        raise ValueError(
            f"{path}: the document: {exc}; expected an answer record "
            f"{ANSWER_RECORD_SHAPE}"
        ) from exc


def _record_in_the_new_form(document: object) -> object:
    # What is not an object holding a list of answers is left to the
    # validation, which names the field at fault.
    if not isinstance(document, dict) or not isinstance(document.get("answers"), list):
        return document
    answers = [
        answer_in_the_new_shape(answer) if isinstance(answer, dict) else answer
        for answer in document["answers"]
    ]
    return {**document, "answers": answers}
