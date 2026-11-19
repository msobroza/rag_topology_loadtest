"""Tests of the hardware tiers: read from the fake Domino, and one found by its name."""

from __future__ import annotations

import pytest

from domino_deploy_helpers import WHERE, http_on
from fake_domino.stand_ins import fake_tier
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.domino_lookups.hardware_tiers import (
    HARDWARE_TIERS_PATH,
    HardwareTier,
    list_hardware_tiers,
    tier_named,
)

SMALL = fake_tier("small-k8s", "cpu-8", 8, 32, is_default=True)
LARGE = fake_tier("large-k8s", "cpu-16", 16, 64)


def refusal_of(tiers: list[HardwareTier], name: str) -> str:
    with pytest.raises(DominoDeployError) as refusal:
        tier_named(tiers, name, WHERE)
    return str(refusal.value)


def test_the_tiers_are_read_with_their_size_and_every_field() -> None:
    gpu_tier = fake_tier("gpu-k8s", "gpu-1", 0.5, 2, gpus=2)
    small, gpu = list_hardware_tiers(http_on(FakeDominoApps(tiers=[SMALL, gpu_tier])))
    assert small == HardwareTier("small-k8s", "cpu-8", 8.0, "32 GiB", 0, default=True)
    assert gpu.size() == "0.5 cores, 2 GiB, 2 GPUs"
    assert small.fields["centsPerMinute"] == 0.0  # a field no attribute holds


def test_one_gpu_is_said_in_the_singular() -> None:
    (gpu,) = list_hardware_tiers(
        http_on(FakeDominoApps(tiers=[fake_tier("g", "gpu-1", 8, 32, gpus=1)]))
    )
    assert gpu.size() == "8 cores, 32 GiB, 1 GPU"


def test_a_tier_on_the_second_page_is_read() -> None:
    apps = FakeDominoApps(
        tiers=[fake_tier(f"tier-{n}", f"cpu-{n}", n, n) for n in range(1, 12)]
    )
    assert len(list_hardware_tiers(http_on(apps))) == 11
    assert apps.received == [("GET", HARDWARE_TIERS_PATH)] * 2


def test_a_tier_is_found_by_its_exact_name() -> None:
    tiers = list_hardware_tiers(http_on(FakeDominoApps(tiers=[SMALL, LARGE])))
    assert tier_named(tiers, "cpu-8", WHERE).id == "small-k8s"


@pytest.mark.parametrize("name", ["cpu-61", "CPU-8", "cpu-8 "])
def test_a_name_no_tier_has_is_refused_with_the_tiers_domino_has(name: str) -> None:
    tiers = list_hardware_tiers(http_on(FakeDominoApps(tiers=[SMALL, LARGE])))
    assert refusal_of(tiers, name) == (
        f"hardware_tier {name!r} of {WHERE} names no hardware tier of Domino, which has: "
        "cpu-8 (id small-k8s, 8 cores, 32 GiB); cpu-16 (id large-k8s, 16 cores, 64 GiB)"
    )


def test_no_tier_at_all_is_said() -> None:
    assert refusal_of([], "cpu-8") == (
        f"hardware_tier 'cpu-8' of {WHERE} names no hardware tier of Domino, "
        "which has none"
    )


def test_a_name_two_tiers_share_is_refused_with_their_ids() -> None:
    twins = [LARGE, fake_tier("large-k8s-2", "cpu-16", 16, 64)]
    tiers = list_hardware_tiers(http_on(FakeDominoApps(tiers=twins)))
    assert refusal_of(tiers, "cpu-16") == (
        f"hardware_tier 'cpu-16' of {WHERE} is the name of 2 hardware tiers of "
        "Domino (ids large-k8s, large-k8s-2): write the one meant as hardware_tier_id"
    )


def test_a_forbidden_list_names_the_permission_and_what_its_caller_needs() -> None:
    http = http_on(FakeDominoApps.from_behaviour("forbidden"))
    with pytest.raises(DominoDeployError) as refusal:
        list_hardware_tiers(http, needs="the caller's advice")
    assert str(refusal.value) == (
        "Domino API GET /api/hardwaretiers/v1/hardwaretiers returned HTTP 403: "
        "Forbidden by the fake Domino (behaviour: forbidden); the caller's advice"
    )
