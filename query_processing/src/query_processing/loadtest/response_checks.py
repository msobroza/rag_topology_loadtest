"""What a response becomes in the load test's statistics: a failure, or STAGE rows.

Only an HTTP 200 whose body is a JSON object succeeds. A success's
``timings_ms`` becomes one STAGE row per stage it names, which ``qp-compare``
reports beside the request rows. The stages are read from the response, so the
client measures any workflow without knowing its graph.

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
from typing import Any

from ..domain.passages import TOTAL_TIMING_KEY

# The request type of STAGE rows, used both when firing them and by qp-compare.
STAGE_REQUEST_TYPE = "STAGE"


@dataclass(frozen=True)
class ResponseVerdict:
    """What one response becomes in the statistics: a failure, or STAGE rows.

    The STAGE rows are ``(name, ms)`` pairs to fire.
    """

    failure: str | None
    stage_rows: tuple[tuple[str, float], ...] = ()


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
        excluded; none when ``timings_ms`` is missing or malformed.

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
    return ResponseVerdict(failure=None, stage_rows=_stage_rows(endpoint, body))


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
