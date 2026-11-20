"""How the load-test client calls its target: the API key a request carries, a POST.

A request carries the API key of the key source the ``key`` block names
(``Authorization: Bearer <key>``), asked of the source for each request, so a
run token that expires is asked for again; without a block it carries
``DOMINO_API_KEY`` in ``X-Domino-Api-Key``, and without either nothing. A 401
tells the source which key was refused, so the next request carries a new one.
This is the one module of the load-test client that imports, of the
composition, the module that opens the source (``opened_target_key``), and the
key source's port, to name it; Locust's file and ``qp-record-answers`` take a
``TargetKey``, and the Locust file the error a source that gives no key raises.

``qp-record-answers`` posts with ``TargetClient``: a request the target never
answered (no connection, no answer within the timeout) is sent again,
``retries`` times at most, after a wait of ``backoff_s`` that doubles before
each next retry; then the call fails with ``TargetUnreachableError``. A 401
with a key source is sent again within the same retries, once the source is
told; any other response, and a 401 once the retries ran out, is returned as
it is: the record holds its status. The wait is injected, so a test does not
wait.

Example:
    >>> with opened_target_key(settings) as key:
    ...     poster = TargetClient(client, "http://host:8888", key, policy)
    ...     poster.post_json("/retrieve", {"question": "q"}).status_code
    200
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

import httpx

from ..adapters.key_source.base_key_source import BaseKeySource
from ..composition.standalone_key_source import (
    StandaloneKeyNeeds,
    opened_standalone_key_source,
)

# What ``TargetKey.headers`` raises when its source gives no key, named here for
# the Locust file, which imports the load-test client alone.
from ..domain.errors import RagDependencyError as RagDependencyError
from .loadtest_settings import LoadTestSettings

_BEARER = "Bearer "


class TargetKey:
    """The API key each request to the target carries, and the 401 it is told of.

    Its methods do not await and are safe under Locust's gevent: the key
    source's locks are the threading module's, which gevent patches.

    Args:
        source: The key source of the ``key`` block; None without a block.
        domino_api_key: ``DOMINO_API_KEY``, sent when there is no source.

    Example:
        >>> TargetKey(None, "dk").headers()
        {'X-Domino-Api-Key': 'dk'}
    """

    def __init__(self, source: BaseKeySource | None, domino_api_key: str = "") -> None:
        self._source, self._domino_api_key = source, domino_api_key

    def headers(self) -> dict[str, str]:
        """Return the headers of one request: the key its source gives now.

        Returns:
            ``Authorization: Bearer <key>`` with a source, else
            ``X-Domino-Api-Key`` when ``DOMINO_API_KEY`` is set, else none.

        Raises:
            RagDependencyError: If the source could not be asked, once the
                load test's retries ran out.

        Example:
            >>> key.headers()
            {'Authorization': 'Bearer eyJ...'}
        """
        if self._source is not None:
            return {"Authorization": _BEARER + str(self._source.key(may_ask=True))}
        if self._domino_api_key:
            return {"X-Domino-Api-Key": self._domino_api_key}
        return {}

    def refused(self, headers: Mapping[str, str]) -> bool:
        """Tell the source that a request carrying ``headers`` got a 401.

        Args:
            headers: What the refused request carried.

        Returns:
            Whether a source was told, so that sending again may carry a new key.

        Example:
            >>> key.refused({"Authorization": "Bearer eyJ..."})
            True
        """
        carried = headers.get("Authorization", "")
        if self._source is None or not carried.startswith(_BEARER):
            return False
        self._source.refused(carried.removeprefix(_BEARER))
        return True


@contextmanager
def opened_target_key(
    settings: LoadTestSettings, transport: httpx.BaseTransport | None = None
) -> Generator[TargetKey, None, None]:
    """Open the key the settings name, and close its source when the block ends.

    Asking the ``domino`` source is bounded by the load test's own wait for
    one answer, retries and backoff, since it reads no ``http`` block.

    Args:
        settings: The load test's settings: its ``key`` and ``domino`` blocks,
            ``DOMINO_API_KEY``, and its timeout, retries and backoff.
        transport: The ``domino`` source's; None for the network.

    Yields:
        The key each request carries.

    Raises:
        ValueError: A key block start-up would refuse, naming its variable.
        MissingKeyError: If an ``env`` source's variable is unset or empty.

    Example:
        >>> with opened_target_key(LoadTestSettings()) as key:
        ...     key.headers()
        {}
    """
    if settings.key is None:
        yield TargetKey(None, settings.domino_api_key)
        return
    with opened_standalone_key_source(
        settings.key, _needs_of(settings), transport
    ) as source:
        yield TargetKey(source)


def _needs_of(settings: LoadTestSettings) -> StandaloneKeyNeeds:
    env_file = LoadTestSettings.model_config.get("env_file")
    return StandaloneKeyNeeds(
        LoadTestSettings.variable_of("key"),
        token_url=settings.domino.token_url,
        token_ttl_s=settings.domino.token_ttl_s,
        timeout_s=settings.request_timeout_s,
        retries=settings.retries,
        backoff_s=settings.backoff_s,
        env_file=Path(env_file) if isinstance(env_file, str | Path) else None,
    )


@dataclass(frozen=True)
class RetryPolicy:
    """How long one attempt waits, how many more are made, the first wait between.

    Example:
        >>> RetryPolicy(timeout_s=60.0, retries=2, backoff_s=0.2).wait_before(2)
        0.4
    """

    timeout_s: float
    retries: int  # extra attempts after the first
    backoff_s: float  # before the first retry; doubles each time

    def wait_before(self, retry: int) -> float:
        """Return the seconds to wait before a retry.

        Args:
            retry: Which retry, 1 for the first.

        Returns:
            ``backoff_s``, doubled for each retry after the first.
        """
        return self.backoff_s * 2 ** (retry - 1)


class TargetUnreachableError(RuntimeError):
    """The target answered none of a request's attempts; the message says where.

    Example:
        >>> raise TargetUnreachableError("http://host/retrieve: no answer ...")
    """


class TargetClient:
    """Posts JSON bodies to one target under one retry policy.

    Args:
        client: What does the HTTP, with its own timeout; a test hands it an
            app in process.
        host: The target's base URL; a trailing slash is dropped.
        key: The API key each request carries, told of a 401.
        policy: The retries and the backoff between them.
        sleep: The wait before a retry.

    Example:
        >>> TargetClient(client, "http://host", key, policy).post_json("/query", {})
    """

    def __init__(
        self,
        client: httpx.Client,
        host: str,
        key: TargetKey,
        policy: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client, self._host = client, host.rstrip("/")
        self._key, self._policy, self._sleep = key, policy, sleep

    def post_json(self, path: str, body: Mapping[str, object]) -> httpx.Response:
        """POST ``body`` to ``path`` on the target and return its response.

        Args:
            path: The route, such as ``/retrieve``.
            body: The JSON body.

        Returns:
            The first response the target gave that is not a 401 its key
            source was told of, or the last response once the retries ran out.

        Raises:
            TargetUnreachableError: If no attempt got an answer; the message
                names the URL, the attempts and the last error.
        """
        url = f"{self._host}{path}"
        error: httpx.TransportError | None = None
        last: httpx.Response | None = None
        for attempt in range(self._policy.retries + 1):
            if attempt:
                self._sleep(self._policy.wait_before(attempt))
            headers = self._key.headers()  # asked each attempt: a refusal renews it
            try:
                last = self._send(url, body, headers)
            except httpx.TransportError as exc:
                error = exc
                continue
            if not self._key_refused(last, headers):
                return last
        if last is not None:
            return last
        raise TargetUnreachableError(
            f"{url}: no answer after {self._policy.retries + 1} attempt(s) of "
            f"{self._policy.timeout_s} s each; last error: {error!r}"
        )

    def _key_refused(
        self, response: httpx.Response, headers: Mapping[str, str]
    ) -> bool:
        # A 401 whose key source was told: sending again may carry a new key.
        refused = response.status_code == HTTPStatus.UNAUTHORIZED
        return refused and self._key.refused(headers)

    def _send(
        self, url: str, body: Mapping[str, object], headers: Mapping[str, str]
    ) -> httpx.Response:
        # The timeout is the client's own: the command opens it with the
        # policy's, and a client handed in by a test answers in process.
        return self._client.post(url, json=dict(body), headers=dict(headers))
