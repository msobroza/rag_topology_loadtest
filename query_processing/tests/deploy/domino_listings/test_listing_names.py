"""Tests of the names given on the command line: Domino's order, an unknown name refused."""

from __future__ import annotations

import pytest

from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.domino_listings.listing_names import only_named
from query_processing.deploy.domino_lookups.environments import DominoEnvironment
from query_processing.deploy.domino_lookups.hardware_tiers import HardwareTier

TIERS = [
    HardwareTier("small-k8s", "cpu-8", 8, "32 GiB", 0, default=True),
    HardwareTier("large-k8s", "cpu-16", 16, "64 GiB", 0),
    HardwareTier("gpu-k8s", "gpu-1", 8, "32 GiB", 1),
]


def environment(environment_id: str, name: str) -> DominoEnvironment:
    return DominoEnvironment(environment_id, name, None, None, None)


def test_no_name_keeps_every_item() -> None:
    assert only_named(TIERS, [], "hardware tier") == TIERS


def test_only_named_keeps_dominos_order_and_refuses_an_unknown_name() -> None:
    named = only_named(TIERS, ["cpu-16", "cpu-8"], "hardware tier")
    assert [tier.name for tier in named] == ["cpu-8", "cpu-16"]
    with pytest.raises(DominoDeployError) as refused:
        only_named(TIERS, ["cpu-61"], "hardware tier")
    assert str(refused.value) == (
        "no hardware tier of Domino is named 'cpu-61'; it has: cpu-8, cpu-16, gpu-1"
    )


def test_two_items_of_one_name_are_both_kept() -> None:
    two = [environment("env-1", "qp-rag"), environment("env-2", "qp-rag")]
    assert only_named([*two, environment("env-3", "other")], ["qp-rag"], "x") == two
