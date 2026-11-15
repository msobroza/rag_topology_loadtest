"""How patiently an adapter calls another service.

``HttpResources`` builds the default once from ``RAG_HTTP_TIMEOUT_S``,
``RAG_HTTP_RETRIES`` and ``RAG_HTTP_BACKOFF_S``. A component on ``served`` or
``api`` is called under its own: the values of its ``http`` block, such as
``RAG_COMPONENTS__LLM__HTTP__TIMEOUT_S``, and the default's for each it leaves
out.

Example:
    >>> policy = HttpPolicy(timeout_s=30.0, retries=2, backoff_s=0.2)
    >>> policy.backoff_before(2)  # the wait before the second retry
    0.4
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HttpPolicy:
    """How patiently an adapter calls another service."""

    timeout_s: float
    retries: int  # extra attempts after the first
    backoff_s: float  # wait before the first retry; doubles each time

    def backoff_before(self, attempt: int) -> float:
        """Return the seconds to wait before ``attempt``.

        Args:
            attempt: 1 is the first retry; 0 is the first attempt, which never
                waits.

        Returns:
            ``backoff_s`` doubled for each retry after the first; 0.0 for attempt 0.

        Example:
            >>> HttpPolicy(30.0, 2, 0.2).backoff_before(1)
            0.2
        """
        return self.backoff_s * 2 ** (attempt - 1) if attempt else 0.0
