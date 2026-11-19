"""Domino's hardware tiers shown: as an aligned table, or with every field as YAML.

Example:
    >>> print(tiers_table(list_hardware_tiers(http)), end="")
    NAME    CORES  MEMORY  GPUS  DEFAULT  ID
    cpu-8       8  32 GiB     -  yes      small-k8s
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domino_lookups.hardware_tiers import HardwareTier
from .aligned_table import aligned_table
from .listing_names import details_yaml

_HEADERS = ("NAME", "CORES", "MEMORY", "GPUS", "DEFAULT", "ID")
_RIGHT = (1, 3)  # CORES and GPUS, numbers


def tiers_table(tiers: Sequence[HardwareTier]) -> str:
    """Lay out the tiers in Domino's order, the default one marked ``yes``.

    Args:
        tiers: The tiers (see ``list_hardware_tiers``).

    Returns:
        The table, a ``-`` under ``GPUS`` for a tier with none.

    Example:
        >>> tier = HardwareTier("large-k8s", "cpu-16", 16, "64 GiB", 0)
        >>> tiers_table([tier]).splitlines()[1]
        'cpu-16     16  64 GiB     -           large-k8s'
    """
    rows = [
        [
            tier.name,
            f"{tier.cores:g}",
            tier.memory,
            str(tier.gpus) if tier.gpus else "-",
            "yes" if tier.default else "",
            tier.id,
        ]
        for tier in tiers
    ]
    return aligned_table(_HEADERS, rows, right=_RIGHT)


def tiers_details(tiers: Sequence[HardwareTier]) -> str:
    """Write every field Domino gave for each tier, as YAML, the name first.

    Args:
        tiers: The tiers (see ``list_hardware_tiers``).

    Returns:
        A YAML list, in Domino's order.

    Example:
        >>> print(tiers_details(tiers).splitlines()[0])
        - name: cpu-16
    """
    return details_yaml([tier.fields for tier in tiers])
