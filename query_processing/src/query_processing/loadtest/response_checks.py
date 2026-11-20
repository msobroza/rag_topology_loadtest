"""What a response becomes in the load test's statistics: a failure, or STAGE rows.

Only an HTTP 200 whose body is a JSON object succeeds. A success's
``timings_ms`` becomes one STAGE row per stage it names, which ``qp-compare``
reports beside the request rows. On every endpoint, each stage of its
``tokens`` that has a STAGE row and whole numbers under ``prompt`` and
``completion`` becomes a ``SpentTokens``, named as its STAGE row is
(``/query:generate``, ``/retrieve:rewrite``), with that row's time; the run adds
them up per stage in its target record. The stages are read from the response,
so the client measures any workflow without knowing its graph.

Example:
    >>> verdict = judge_response(plan.path, resp.status_code, resp.text or "")
    >>> verdict.stage_rows
    (('/query:embed', 4.0),)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, TypeGuard

from query_processing_core.query_response import TOTAL_TIMING_KEY

# The request type of STAGE rows, used both when firing them and by qp-compare.
STAGE_REQUEST_TYPE = "STAGE"


@dataclass(frozen=True)
class SpentTokens:
    """The tokens one stage of one response spent, and how long that stage took.

    Attributes:
        prompt: The tokens the LLM was sent in the stage.
        completion: The tokens the LLM wrote in the stage.
        stage_ms: The stage's time in ``timings_ms``, which the completion took.

    Example:
        >>> SpentTokens(829, 128, 7474.0).completion
        128
    """

    prompt: int
    completion: int
    stage_ms: float


@dataclass(frozen=True)
class ResponseVerdict:
    """What one response becomes in the statistics: a failure, or STAGE rows.

    The STAGE rows are ``(name, ms)`` pairs to fire; ``tokens`` holds a
    ``(STAGE row name, SpentTokens)`` pair per stage that reported usable
    counts, none for a failure.
    """

    failure: str | None
    stage_rows: tuple[tuple[str, float], ...] = ()
    tokens: tuple[tuple[str, SpentTokens], ...] = ()


def judge_response(endpoint: str, status: int, body_text: str) -> ResponseVerdict:
    """Decide how a response is recorded; only a 200 with a JSON object succeeds.

    Args:
        endpoint: The path requested; it prefixes each STAGE row name.
        status: The HTTP status of the response.
        body_text: The response body; failure messages quote its first 200
            characters.

    Returns:
        A failure message, or no failure and one STAGE row per key of
        ``timings_ms`` timed above zero, in the response's order, the total
        excluded; none for a stage whose time is no number, and none at all
        when ``timings_ms`` is missing or not an object. A success's
        tokens are one pair per stage of ``tokens`` that has a STAGE row and
        whole numbers under ``prompt`` and ``completion``, in the order of
        ``tokens``; a stage without them is left out alone.

    Example:
        >>> judge_response("/query", 200, '{"timings_ms": {"embed": 4}}').stage_rows
        (('/query:embed', 4.0),)
    """
    if status != HTTPStatus.OK:
        return ResponseVerdict(failure=f"HTTP {status}: {body_text[:200]}")
    try:
        body = json.loads(body_text)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        message = f"HTTP 200 but the body is not a JSON object: {body_text[:200]}"
        return ResponseVerdict(failure=message)
    times = {_row_name(endpoint, stage): ms for stage, ms in _stage_times(body).items()}
    tokens = _spent_tokens(endpoint, body.get("tokens"), times)
    return ResponseVerdict(failure=None, stage_rows=tuple(times.items()), tokens=tokens)


def _row_name(endpoint: str, stage: str) -> str:
    # A stage's STAGE row, and the key its tokens are counted under.
    return f"{endpoint}:{stage}"


# The time of each stage that gets a STAGE row: each key of ``timings_ms`` timed
# above zero, in the response's order, the total excluded; a stage that did not
# run (generate on /retrieve) is 0 and gets none, as does a stage whose time is
# no number. Empty when ``timings_ms`` is missing or not an object.
def _stage_times(body: Mapping[str, Any]) -> dict[str, float]:
    timings_ms = body.get("timings_ms")
    if not isinstance(timings_ms, dict):
        return {}
    times = {str(stage): _stage_ms(ms) for stage, ms in timings_ms.items()}
    return {
        stage: ms for stage, ms in times.items() if stage != TOTAL_TIMING_KEY and ms > 0
    }


def _stage_ms(value: object) -> float:
    # Each stage's time is read on its own, so one unusable time leaves the
    # other stages their rows and tokens. ``True`` is a number in Python, but
    # never a time.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


# Each stage of ``tokens`` with usable counts and a STAGE row, under that row's
# name and with its time; one stage's unusable counts or time leave the others
# counted, since each stage is reported on its own and no sum across stages is
# made.
def _spent_tokens(
    endpoint: str, tokens: object, row_times: Mapping[str, float]
) -> tuple[tuple[str, SpentTokens], ...]:
    if not isinstance(tokens, dict):
        return ()
    spent: list[tuple[str, SpentTokens]] = []
    for stage, counts in tokens.items():
        row = _row_name(endpoint, stage)
        if row in row_times and _is_stage_counts(counts):
            prompt, completion = counts["prompt"], counts["completion"]
            spent.append((row, SpentTokens(prompt, completion, row_times[row])))
    return tuple(spent)


def _is_stage_counts(value: object) -> TypeGuard[Mapping[str, int]]:
    return isinstance(value, dict) and all(
        _token_count(value.get(key)) for key in ("prompt", "completion")
    )


def _token_count(value: object) -> TypeGuard[int]:
    # ``True`` is an int in Python, but never a token count.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
