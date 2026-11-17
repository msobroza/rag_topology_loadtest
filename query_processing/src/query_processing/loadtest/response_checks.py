"""What a response becomes in the load test's statistics: a failure, or STAGE rows.

Only an HTTP 200 whose body is a JSON object succeeds. A success's
``timings_ms`` becomes one STAGE row per stage it names, which ``qp-compare``
reports beside the request rows. On ``/query``, the endpoint that answers, its
``tokens``, summed over every stage the map holds, with the time of those same
stages, become the answer's ``AnswerTokens``, which the run adds up in its
target record. The stages are
read from the response, so the client measures any workflow without knowing its
graph.

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

from ..domain.passages import TOTAL_TIMING_KEY

# The request type of STAGE rows, used both when firing them and by qp-compare.
STAGE_REQUEST_TYPE = "STAGE"
# The endpoint whose responses are answers, the only ones whose tokens count: a
# /retrieve's rewrite tokens would lower every per-answer figure of qp-compare,
# which also reads this endpoint's row for the run's seconds.
ANSWER_ENDPOINT = "/query"


@dataclass(frozen=True)
class AnswerTokens:
    """The token counts of one answer, and how long the stages that spent them took.

    Attributes:
        prompt: The tokens the LLM was sent, over every stage that reported them.
        completion: The tokens the LLM wrote, over the same stages.
        llm_ms: The time of those same stages, which the completion took.

    Example:
        >>> AnswerTokens(829, 128, 7474.0).completion
        128
    """

    prompt: int
    completion: int
    llm_ms: float


@dataclass(frozen=True)
class ResponseVerdict:
    """What one response becomes in the statistics: a failure, or STAGE rows.

    The STAGE rows are ``(name, ms)`` pairs to fire; ``tokens`` is the answer's,
    or None when the response reported no usable counts.
    """

    failure: str | None
    stage_rows: tuple[tuple[str, float], ...] = ()
    tokens: AnswerTokens | None = None


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
        excluded; none when ``timings_ms`` is missing or malformed. A success
        has tokens only on ``ANSWER_ENDPOINT``, when ``tokens`` holds at least
        one stage, each with
        whole numbers under ``prompt`` and ``completion`` and a time in
        ``timings_ms``, and those times add up to more than 0 ms.

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
    rows = _stage_rows(endpoint, body)
    tokens = _answer_tokens(body) if endpoint == ANSWER_ENDPOINT else None
    return ResponseVerdict(failure=None, stage_rows=rows, tokens=tokens)


# One row per key of ``timings_ms`` timed above zero, in the response's order,
# the total excluded: a stage that did not run (generate on /retrieve) is 0 and
# gets no row. None when ``timings_ms`` is missing or malformed.
def _stage_rows(
    endpoint: str, body: Mapping[str, Any]
) -> tuple[tuple[str, float], ...]:
    try:
        timings_ms = {str(k): float(v) for k, v in body["timings_ms"].items()}
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    return tuple(
        (f"{endpoint}:{stage}", ms)
        for stage, ms in timings_ms.items()
        if stage != TOTAL_TIMING_KEY and ms > 0
    )


# The answer's tokens summed over every stage of ``tokens``, with the time of
# those stages; None when the map is empty, when any stage's counts or time are
# unusable (a partial sum would misstate the answer), or when the times add up
# to 0, which would divide a tokens/s figure by zero.
def _answer_tokens(body: Mapping[str, object]) -> AnswerTokens | None:
    tokens, timings_ms = body.get("tokens"), body.get("timings_ms")
    if not tokens or not isinstance(tokens, dict) or not isinstance(timings_ms, dict):
        return None
    prompt, completion, llm_ms = 0, 0, 0.0
    for stage, counts in tokens.items():
        ms = timings_ms.get(stage)
        if not _is_stage_counts(counts) or not _is_time(ms):
            return None
        prompt += counts["prompt"]
        completion += counts["completion"]
        llm_ms += ms
    return AnswerTokens(prompt, completion, llm_ms) if llm_ms > 0 else None


def _is_stage_counts(value: object) -> TypeGuard[Mapping[str, int]]:
    return isinstance(value, dict) and all(
        _token_count(value.get(key)) for key in ("prompt", "completion")
    )


def _token_count(value: object) -> TypeGuard[int]:
    # ``True`` is an int in Python, but never a token count.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_time(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value >= 0
