"""The hardware tiers route as the emulation answers it: the placement file's tiers.

Served in front of the core's route (which describes any id at a fixed size)
when a placement file is loaded: a tier reports its machines' real cores and
memory, ``isDefault`` on the default tier, and an unknown id is 404.

Example:
    >>> create_fake_domino(apps, routers=[_placed_tiers_router])
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request

from ..api_shapes import HardwareTierEnvelopeV1
from ..stand_ins import HARDWARE_TIERS_PATH, tier_envelope
from .placement import AppPlacer

_placed_tiers_router = APIRouter(
    prefix=HARDWARE_TIERS_PATH, tags=["Hardware tiers (Domino 6.2 API)"]
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
    scheduler = request.app.state.fake_domino_apps.scheduler
    tiers = scheduler.settings.tiers if isinstance(scheduler, AppPlacer) else ()
    tier = next((tier for tier in tiers if tier.id == hardwareTierId), None)
    if tier is None:
        known = [tier.id for tier in tiers]
        raise HTTPException(
            HTTPStatus.NOT_FOUND, f"no hardware tier {hardwareTierId!r}; known: {known}"
        )
    return tier_envelope(tier.id, tier.cores, tier.memory_gib, is_default=tier.default)
