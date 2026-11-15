"""The rules that spread an App's requests over its replicas."""

from __future__ import annotations

import pytest

from fake_domino.emulation.proxy_settings import ReplicaBalancing
from fake_domino.emulation.replica_balancing import (
    BaseReplicaBalancer,
    ConnectionRandomBalancer,
    RoundRobinBalancer,
    replica_balancer,
)

REPLICAS = (
    "http://10.10.0.21:8888",
    "http://10.10.0.22:8888",
    "http://10.10.0.23:8888",
)
CONNECTIONS = [("10.10.0.5", port) for port in range(40000, 40300)]


def test_round_robin_takes_each_replica_in_turn_per_app() -> None:
    balancer = RoundRobinBalancer()
    first = [balancer.pick("app-1", REPLICAS, None) for _ in range(2)]
    other = balancer.pick("app-2", REPLICAS, ("10.10.0.5", 40000))
    rest = [balancer.pick("app-1", REPLICAS, None) for _ in range(2)]
    assert [*first, *rest] == [*REPLICAS, REPLICAS[0]]
    assert other == REPLICAS[0]


def test_connection_random_keeps_a_connection_on_one_replica() -> None:
    balancer = ConnectionRandomBalancer(token="fixed")
    picks = {balancer.pick("app-1", REPLICAS, ("10.10.0.5", 40000)) for _ in range(20)}
    assert len(picks) == 1


def test_connection_random_uses_every_replica_over_many_connections() -> None:
    balancer = ConnectionRandomBalancer(token="fixed")
    picks = {balancer.pick("app-1", REPLICAS, c) for c in CONNECTIONS}
    assert picks == set(REPLICAS)


def test_connection_random_draws_differ_between_balancers_not_within_one() -> None:
    def drawn(balancer: ConnectionRandomBalancer) -> list[str]:
        return [balancer.pick("app-1", REPLICAS, c) for c in CONNECTIONS]

    assert drawn(ConnectionRandomBalancer("a")) == drawn(ConnectionRandomBalancer("a"))
    assert drawn(ConnectionRandomBalancer("a")) != drawn(ConnectionRandomBalancer("b"))
    assert drawn(ConnectionRandomBalancer()) != drawn(ConnectionRandomBalancer())


def test_connection_random_without_a_connection_draws_per_request() -> None:
    balancer = ConnectionRandomBalancer(token="fixed")
    picks = {balancer.pick("app-1", REPLICAS, None) for _ in range(200)}
    assert picks == set(REPLICAS)


@pytest.mark.parametrize(
    ("rule", "kind"),
    [
        (ReplicaBalancing.ROUND_ROBIN, RoundRobinBalancer),
        (ReplicaBalancing.CONNECTION_RANDOM, ConnectionRandomBalancer),
    ],
)
def test_each_rule_has_its_balancer(
    rule: ReplicaBalancing, kind: type[BaseReplicaBalancer]
) -> None:
    assert isinstance(replica_balancer(rule), kind)
