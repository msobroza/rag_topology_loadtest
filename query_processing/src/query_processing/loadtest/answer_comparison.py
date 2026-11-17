"""What differs between two answer records of the same requests.

Two records compare only when they hold the same requests: the same number,
in the same order, each to the same endpoint with the same body; otherwise
they are refused, naming the first request that differs. Then each answer of
a run is held against the baseline's:

- exactly: the status, the mode, the metadata filter applied, the passages'
  ids in order, their text hashes and their metadata;
- within ``score_tolerance``: the retrieval and the rerank scores, an absent
  score equal only to an absent one;
- the answer's text only when ``compare_answer_text`` says so: an LLM behind
  an API does not repeat itself.

Example:
    >>> require_same_requests(("monolith", base), ("split-all", run))
    >>> answer_differences(base, run, AnswerTolerance(1e-4, False))
    []
"""

from __future__ import annotations

from dataclasses import dataclass

from .answer_record import AnswerRecord, RecordedAnswer, RecordedPassage

# The fields compared exactly, in the order a difference is reported.
_EXACT_FIELDS = ("status", "mode", "metadata_filter")
_EXACT_PASSAGE_FIELDS = ("text_sha256", "metadata")
_SCORE_FIELDS = ("retrieval_score", "rerank_score")


@dataclass(frozen=True)
class AnswerTolerance:
    """What two answers may differ by and still be the same.

    Example:
        >>> AnswerTolerance(score_tolerance=1e-4, compare_answer_text=False)
    """

    score_tolerance: float
    compare_answer_text: bool


@dataclass(frozen=True)
class AnswerDifference:
    """One field of one answer that differs from the baseline's.

    ``request`` is the answer's place in the record, 1 for the first.
    """

    request: int
    endpoint: str
    question: str
    field: str
    baseline: object
    run: object


class RequestListMismatchError(ValueError):
    """Two records of different requests; the message names the first that differs.

    Example:
        >>> raise RequestListMismatchError("split-all: request 3 ...")
    """


def require_same_requests(
    baseline: tuple[str, AnswerRecord], run: tuple[str, AnswerRecord]
) -> None:
    """Refuse two records whose requests differ: number, order, endpoint or body.

    Args:
        baseline: The baseline's name and record.
        run: The run's name and record.

    Raises:
        RequestListMismatchError: Naming both runs and the first request that
            differs, or the two numbers of requests.

    Example:
        >>> require_same_requests(("monolith", base), ("split-all", base))
    """
    (base_name, base), (run_name, record) = baseline, run
    # Not strict: a shorter record is held request by request first, so the
    # first request that differs is named before the two numbers are.
    pairs = zip(base.answers, record.answers, strict=False)
    for number, (ours, theirs) in enumerate(pairs, 1):
        if (ours.endpoint, ours.body) != (theirs.endpoint, theirs.body):
            raise RequestListMismatchError(
                f"{run_name}: request {number} is {theirs.endpoint} {theirs.body}, "
                f"{base_name}'s is {ours.endpoint} {ours.body}; expected the same "
                "requests in the same order"
            )
    if len(base.answers) != len(record.answers):
        raise RequestListMismatchError(
            f"{run_name} holds {len(record.answers)} requests, {base_name} "
            f"{len(base.answers)}; expected the same requests in the same order"
        )


def answer_differences(
    baseline: AnswerRecord, run: AnswerRecord, tolerance: AnswerTolerance
) -> list[AnswerDifference]:
    """Return every field of ``run``'s answers that differs from ``baseline``'s.

    Args:
        baseline: The record every run is held against.
        run: A record of the same requests (``require_same_requests``).
        tolerance: What two answers may differ by.

    Returns:
        The differences, by request and then in the order the fields are held.

    Example:
        >>> answer_differences(base, base, AnswerTolerance(1e-4, False))
        []
    """
    found: list[AnswerDifference] = []
    for number, (ours, theirs) in enumerate(
        zip(baseline.answers, run.answers, strict=True), 1
    ):
        question = str(ours.body.get("question", ""))
        found += [
            AnswerDifference(number, ours.endpoint, question, field, base, value)
            for field, base, value in _differing_fields(ours, theirs, tolerance)
        ]
    return found


def different_answers(differences: list[AnswerDifference]) -> int:
    """Return how many answers ``differences`` names, each counted once.

    Args:
        differences: What ``answer_differences`` found for one run.

    Returns:
        The number of distinct requests among them.

    Example:
        >>> different_answers([])
        0
    """
    return len({difference.request for difference in differences})


Differing = list[tuple[str, object, object]]  # (field, baseline's value, run's)


def _differing_fields(
    ours: RecordedAnswer, theirs: RecordedAnswer, tolerance: AnswerTolerance
) -> Differing:
    fields = list(_EXACT_FIELDS) + (["answer"] if tolerance.compare_answer_text else [])
    found: Differing = [
        (name, getattr(ours, name), getattr(theirs, name))
        for name in fields
        if getattr(ours, name) != getattr(theirs, name)
    ]
    return found + _differing_passages(ours.passages, theirs.passages, tolerance)


def _differing_passages(
    ours: list[RecordedPassage] | None,
    theirs: list[RecordedPassage] | None,
    tolerance: AnswerTolerance,
) -> Differing:
    # Passages are held one by one only when the ids agree in order: a passage
    # out of order or changed is the list of ids, said once.
    ids, their_ids = _ids(ours), _ids(theirs)
    if ids != their_ids:
        return [("passages.id", ids, their_ids)]
    found: Differing = []
    for index, (mine, other) in enumerate(zip(ours or [], theirs or [], strict=True)):
        found += [
            (f"passages[{index}].{name}", base, value)
            for name, base, value in _differing_passage_fields(mine, other, tolerance)
        ]
    return found


def _ids(passages: list[RecordedPassage] | None) -> list[str] | None:
    return None if passages is None else [passage.id for passage in passages]


def _differing_passage_fields(
    ours: RecordedPassage, theirs: RecordedPassage, tolerance: AnswerTolerance
) -> Differing:
    exact: Differing = [
        (name, getattr(ours, name), getattr(theirs, name))
        for name in _EXACT_PASSAGE_FIELDS
        if getattr(ours, name) != getattr(theirs, name)
    ]
    scores: Differing = [
        (name, getattr(ours, name), getattr(theirs, name))
        for name in _SCORE_FIELDS
        if not _scores_equal(getattr(ours, name), getattr(theirs, name), tolerance)
    ]
    return exact + scores


def _scores_equal(
    ours: float | None, theirs: float | None, tolerance: AnswerTolerance
) -> bool:
    if ours is None or theirs is None:
        return ours is theirs  # an absent score equals only an absent one
    return abs(ours - theirs) <= tolerance.score_tolerance
