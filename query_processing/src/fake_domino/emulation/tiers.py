"""The hardware tiers routes as the emulation answers them: the placement file.

Served in front of the core's routes (which list the state's tiers and describe
any id at a fixed size) when a placement file is loaded: the list gives every
tier of the file in one page, a tier reports its machines' real cores and
memory, ``isDefault`` on the default tier, and an unknown id is 404.

Example:
    >>> create_fake_domino(apps, routers=[_placed_tiers_router])
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request

from ..api_shapes import HardwareTierEnvelopeV1, PaginatedHardwareTierEnvelopeV1
from ..stand_ins import (
    HARDWARE_TIERS_PATH,
    fake_tier,
    paginated_metadata,
    tier_envelope,
)
from .placement import AppPlacer
from .placement_settings import HardwareTierPool

_placed_tiers_router = APIRouter(
    prefix=HARDWARE_TIERS_PATH, tags=["Hardware tiers (Domino 6.2 API)"]
)


@_placed_tiers_router.get("")
def list_placed_hardware_tiers(request: Request) -> PaginatedHardwareTierEnvelopeV1:
    """List every hardware tier of the placement file, in one page.

    Args:
        request: The incoming request, whose app holds the fake's state.

    Returns:
        Each tier with its id as its name, as ``tier_envelope`` describes it.
    """
    tiers = _placed_tiers(request)
    described = [
        fake_tier(t.id, t.id, t.cores, t.memory_gib, is_default=t.default)
        for t in tiers
    ]
    return PaginatedHardwareTierEnvelopeV1(
        hardwareTiers=described,
        metadata=paginated_metadata(0, len(tiers), len(tiers)),
    )


@_placed_tiers_router.get("/{hardwareTierId}")
def get_placed_hardware_tier(
    request: Request, hardwareTierId: str  # noqa: N803 - Domino's path name
) -> HardwareTierEnvelopeV1:
    """Get one hardware tier of the placement file.

    Args:
        request: The incoming request, whose app holds the fake's state.
        hardwareTierId: The tier's id, also used as its name.

    Returns:
        The tier's cores and memory in GiB, ``isDefault`` on the default tier.

    Raises:
        HTTPException: 404 if no tier of the placement file has that id.
    """
    tiers = _placed_tiers(request)
    tier = next((tier for tier in tiers if tier.id == hardwareTierId), None)
    if tier is None:
        known = [tier.id for tier in tiers]
        raise HTTPException(
            HTTPStatus.NOT_FOUND, f"no hardware tier {hardwareTierId!r}; known: {known}"
        )
    return tier_envelope(tier.id, tier.cores, tier.memory_gib, is_default=tier.default)


def _placed_tiers(request: Request) -> tuple[HardwareTierPool, ...]:
    scheduler = request.app.state.fake_domino_apps.scheduler
    return scheduler.settings.tiers if isinstance(scheduler, AppPlacer) else ()
