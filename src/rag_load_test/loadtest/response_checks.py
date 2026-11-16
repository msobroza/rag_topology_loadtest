"""What a response becomes in the load test's statistics: a failure, or STAGE rows.

Only an HTTP 200 whose body is a JSON object succeeds. A success's
``timings_ms`` becomes one STAGE row per stage, which ``rag-compare`` reports
beside the request rows; its ``tokens``, with the generate stage's time, become
the answer's ``AnswerTokens``, which the run adds up in its target record.

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

from ..domain.passages import STAGES

# The request type of STAGE rows, used both when firing them and by rag-compare.
STAGE_REQUEST_TYPE = "STAGE"


@dataclass(frozen=True)
class AnswerTokens:
    """The token counts of one answer, and how long the generate stage took.

    Attributes:
        prompt: The tokens of the conversation the LLM was sent.
        completion: The tokens of the answer.
        generate_ms: The generate stage's time, which the completion took.

    Example:
        >>> AnswerTokens(829, 128, 7474.0).completion
        128
    """

    prompt: int
    completion: int
    generate_ms: float


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
        A failure message, or no failure and one STAGE row per stage timed above
        zero, in ``STAGES`` order (``total`` excluded); none when ``timings_ms``
        is missing or malformed. A success has tokens only when ``tokens`` holds
        whole numbers under ``prompt`` and ``completion`` and the generate stage
        took more than 0 ms.

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
    return ResponseVerdict(failure=None, stage_rows=rows, tokens=_answer_tokens(body))


# One row per stage timed above zero, in ``STAGES`` order (``total`` excluded);
# none when ``timings_ms`` is missing or malformed.
def _stage_rows(
    endpoint: str, body: Mapping[str, Any]
) -> tuple[tuple[str, float], ...]:
    try:
        timings_ms = {str(k): float(v) for k, v in body["timings_ms"].items()}
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    return tuple(
        (f"{endpoint}:{s}", timings_ms[s]) for s in STAGES if timings_ms.get(s, 0) > 0
    )


# The answer's tokens, or None unless both counts are whole numbers and the
# generate stage took time: a zero time would divide a tokens/s figure by zero.
def _answer_tokens(body: Mapping[str, Any]) -> AnswerTokens | None:
    tokens, timings_ms = body.get("tokens"), body.get("timings_ms")
    if not isinstance(tokens, dict) or not isinstance(timings_ms, dict):
        return None
    prompt, completion = tokens.get("prompt"), tokens.get("completion")
    generate_ms = timings_ms.get("generate")
    if _token_count(prompt) and _token_count(completion) and _took_time(generate_ms):
        return AnswerTokens(prompt, completion, float(generate_ms))
    return None


def _token_count(value: object) -> TypeGuard[int]:
    # ``True`` is an int in Python, but never a token count.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _took_time(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value > 0
