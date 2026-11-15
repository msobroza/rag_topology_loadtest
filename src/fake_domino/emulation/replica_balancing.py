"""Which of an App's replicas the fake's App proxy forwards a request to.

Documented by Domino (6.3, "Deploy Apps to production"): with session affinity
off, requests spread across an App's pods. Not documented: by which rule, so
both rules Kubernetes itself uses can be chosen (``ReplicaBalancing``):

- ``RoundRobinBalancer``: each request goes to the next replica in turn, as an
  ingress controller does for each request by default;
- ``ConnectionRandomBalancer``: each caller connection gets one replica at
  random and keeps it, as a Service does through kube-proxy (a random pod per
  new TCP connection).

Session affinity is not emulated: ``useSessionAffinity`` is stored and echoed.

Emulation only: a replica is one machine of the App, ``http://<machine>:<port>``.

Example:
    >>> replica_balancer(ReplicaBalancing.ROUND_ROBIN).pick("app-1", upstreams, None)
    'http://10.10.0.21:8888'
"""

from __future__ import annotations

import abc
import itertools
import random
import secrets
from collections import defaultdict
from collections.abc import Sequence

from .proxy_settings import ReplicaBalancing


class BaseReplicaBalancer(abc.ABC):
    """Picks the replica of an App that answers one request.

    Example:
        >>> balancer.pick("app-1", ("http://10.10.0.21:8888",), ("10.10.0.5", 40000))
        'http://10.10.0.21:8888'
    """

    @abc.abstractmethod
    def pick(
        self, app_id: str, replicas: Sequence[str], connection: tuple[str, int] | None
    ) -> str:
        """Pick the replica that answers a request for ``app_id``.

        Args:
            app_id: The App, e.g. ``"app-1"``.
            replicas: Its replicas' base URLs; never empty.
            connection: The caller's host and port, or None if unknown.

        Returns:
            One of ``replicas``.
        """


class RoundRobinBalancer(BaseReplicaBalancer):
    """Sends each request for an App to its next replica in turn.

    Example:
        >>> RoundRobinBalancer().pick("app-1", ("http://a", "http://b"), None)
        'http://a'
    """

    def __init__(self) -> None:
        # Each App keeps its own turn; next() on a count is atomic under the GIL.
        self._turns: defaultdict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )

    def pick(
        self, app_id: str, replicas: Sequence[str], connection: tuple[str, int] | None
    ) -> str:
        """Pick ``app_id``'s next replica; the connection plays no part.

        Args:
            app_id: The App.
            replicas: Its replicas' base URLs; never empty.
            connection: Ignored.

        Returns:
            The replica after the one its previous request went to.
        """
        del connection  # the port's signature; a turn is per App, not per caller
        return replicas[next(self._turns[app_id]) % len(replicas)]


class ConnectionRandomBalancer(BaseReplicaBalancer):
    """Gives each caller connection one replica at random, kept for its requests.

    The draw is seeded by the balancer's token, the App and the connection, so
    the same connection always draws the same replica without a table of
    connections to keep, and two balancers (two processes) draw differently.

    Args:
        token: Seeds every draw; None takes a fresh random one (tests fix it).

    Example:
        >>> ConnectionRandomBalancer().pick("app-1", upstreams, ("10.10.0.5", 40000))
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token if token is not None else secrets.token_hex(16)
        self._unpinned = random.Random()  # for a request with no connection

    def pick(
        self, app_id: str, replicas: Sequence[str], connection: tuple[str, int] | None
    ) -> str:
        """Pick the replica ``connection`` keeps for ``app_id``.

        Args:
            app_id: The App.
            replicas: Its replicas' base URLs; never empty.
            connection: The caller's host and port; None draws for this
                request alone.

        Returns:
            The same replica for every request of one connection.
        """
        if connection is None:
            return self._unpinned.choice(replicas)
        host, port = connection
        seed = f"{self._token}|{app_id}|{host}|{port}"
        return random.Random(seed).choice(replicas)


def replica_balancer(rule: ReplicaBalancing) -> BaseReplicaBalancer:
    """Build the balancer that follows ``rule``.

    Args:
        rule: How requests are spread over an App's replicas.

    Returns:
        A new balancer.

    Example:
        >>> replica_balancer(ReplicaBalancing.CONNECTION_RANDOM)
    """
    if rule is ReplicaBalancing.CONNECTION_RANDOM:
        return ConnectionRandomBalancer()
    return RoundRobinBalancer()
