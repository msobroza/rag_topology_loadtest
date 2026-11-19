"""Domino's hardware tiers: every one read, and one found by its exact name.

The list is ``GET /api/hardwaretiers/v1/hardwaretiers`` (``ViewHardwareTiers``),
whose answer holds ``hardwareTiers`` and ``metadata.totalCount``.

Example:
    >>> tier_named(list_hardware_tiers(http), "cpu-16", "App rag-reranker").id
    'large-k8s'
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import httpx

from ..apps_client import DominoDeployError, json_mapping
from .paged_list import read_pages

# Written here as well as in the fake Domino, which the product may not import.
HARDWARE_TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"
# What a 401 or 403 on the list says after Domino's answer: a listing asks for
# the permission, a deploy can also do without the list.
_PERMISSION = "listing the hardware tiers needs the permission ViewHardwareTiers"
_TIERS_NEEDED_TO_LIST = f"{_PERMISSION}: ask for it"
TIERS_NEEDED_TO_DEPLOY = (
    f"{_PERMISSION}: ask for it, or write the tier's id as hardware_tier_id, "
    "which needs no such call"
)


@dataclass(frozen=True)
class HardwareTier:
    """The tier fields a deploy reads, and the tier as Domino gave it in ``fields``."""

    id: str
    name: str
    cores: float
    memory: str  # e.g. "64 GiB"
    gpus: int
    default: bool = False
    fields: Mapping[str, object] = field(default_factory=dict, compare=False)

    def size(self) -> str:
        """Say the tier's size: cores, memory, and its GPUs when it has any.

        Returns:
            E.g. ``"16 cores, 64 GiB"`` or ``"0.5 cores, 2 GiB, 2 GPUs"``.

        Example:
            >>> HardwareTier("large-k8s", "cpu-16", 16.0, "64 GiB", 1).size()
            '16 cores, 64 GiB, 1 GPU'
        """
        size = f"{self.cores:g} cores, {self.memory}"
        if not self.gpus:
            return size
        return f"{size}, {self.gpus} GPU{'' if self.gpus == 1 else 's'}"


def list_hardware_tiers(
    http: httpx.Client, needs: str = _TIERS_NEEDED_TO_LIST
) -> list[HardwareTier]:
    """Read every hardware tier of Domino, in Domino's order.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).
        needs: What a 401 or 403 says after Domino's answer: the caller's advice
            (``TIERS_NEEDED_TO_DEPLOY`` for a deploy).

    Returns:
        The tiers.

    Raises:
        DominoDeployError: If the list cannot be read; on 401 or 403 the message
            names the permission ``ViewHardwareTiers``, then ``needs``.

    Example:
        >>> [tier.name for tier in list_hardware_tiers(http)]
        ['cpu-8', 'cpu-16']
    """
    listed = read_pages(http, HARDWARE_TIERS_PATH, "hardwareTiers", needs=needs)
    return [_tier_from_json(tier) for tier in listed]


def tier_named(tiers: Sequence[HardwareTier], name: str, where: str) -> HardwareTier:
    """Find the one tier whose name is ``name``, capitals and spaces included.

    Args:
        tiers: Domino's tiers (see ``list_hardware_tiers``).
        name: The value of ``hardware_tier``.
        where: The App and the blocks that set it, for the message.

    Returns:
        The tier.

    Raises:
        DominoDeployError: If no tier has the name (the message lists every tier),
            or several have it (the message gives their ids).

    Example:
        >>> tier_named(tiers, "cpu-16", "App rag-reranker").id
        'large-k8s'
    """
    found = [tier for tier in tiers if tier.name == name]
    if len(found) == 1:
        return found[0]
    if found:
        ids = ", ".join(tier.id for tier in found)
        raise DominoDeployError(
            f"hardware_tier {name!r} of {where} is the name of {len(found)} hardware "
            f"tiers of Domino (ids {ids}): write the one meant as hardware_tier_id"
        )
    listed = "; ".join(f"{t.name} (id {t.id}, {t.size()})" for t in tiers)
    raise DominoDeployError(
        f"hardware_tier {name!r} of {where} names no hardware tier of Domino, "
        f"which has{': ' + listed if listed else ' none'}"
    )


def _tier_from_json(tier: Mapping[str, object]) -> HardwareTier:
    resources = json_mapping(tier.get("resources"))
    memory = json_mapping(resources.get("memory"))
    gpus = json_mapping(tier.get("gpuConfiguration")).get("numberOfGpus")
    return HardwareTier(
        id=str(tier.get("id", "")),
        name=str(tier.get("name", "")),
        cores=_number(resources.get("cores")),
        memory=f"{_number(memory.get('value')):g} {memory.get('units', '')}".strip(),
        gpus=int(_number(gpus)),
        default=json_mapping(tier.get("flags")).get("isDefault") is True,
        fields=dict(tier),
    )


def _number(value: object) -> float:
    # Read leniently, like every field of an answer: a missing number is 0.
    return float(value) if isinstance(value, int | float) else 0.0
