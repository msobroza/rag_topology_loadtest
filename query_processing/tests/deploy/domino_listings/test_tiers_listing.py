"""Tests of the hardware tiers shown: the table with its default, and the details."""

from __future__ import annotations

import yaml

from query_processing.deploy.domino_listings.tiers_listing import (
    tiers_details,
    tiers_table,
)
from query_processing.deploy.domino_lookups.hardware_tiers import HardwareTier

TIERS = [
    HardwareTier("small-k8s", "cpu-8", 8, "32 GiB", 0, default=True),
    HardwareTier("large-k8s", "cpu-16", 16, "64 GiB", 0),
    HardwareTier("gpu-k8s", "gpu-1", 8, "32 GiB", 1),
]


def test_the_tiers_table_aligns_its_columns_and_marks_the_default() -> None:
    assert tiers_table(TIERS) == (
        "NAME    CORES  MEMORY  GPUS  DEFAULT  ID\n"
        "cpu-8       8  32 GiB     -  yes      small-k8s\n"
        "cpu-16     16  64 GiB     -           large-k8s\n"
        "gpu-1       8  32 GiB     1           gpu-k8s\n"
    )


def test_the_details_give_every_field_domino_gave_the_name_first() -> None:
    odd = HardwareTier(
        "x",
        "odd",
        1,
        "1 GiB",
        0,
        fields={"id": "x", "name": "odd", "aFieldOfAnotherVersion": [1, 2]},
    )
    [shown] = yaml.safe_load(tiers_details([odd]))
    assert list(shown) == ["name", "id", "aFieldOfAnotherVersion"]
    assert shown["aFieldOfAnotherVersion"] == [1, 2]
