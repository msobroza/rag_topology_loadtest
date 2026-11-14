"""What the fake serves besides the Apps API: hardware tiers and two stand-ins.

``/api/hardwaretiers/v1/hardwaretiers/{id}`` (any id is a tier of a fixed size),
``/access-token`` (the token service of a Domino run, documented to answer the
raw token as plain text) and an App instance's ``/readyz`` on its legacy URL
(``readyz_stand_in``, no credential checked), so ``rag-deploy status`` has
something to read without Domino.

Example:
    >>> fake.include_router(_tiers_router)  # as create_fake_domino does
"""

from __future__ import annotations

import shlex
from http import HTTPStatus

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from .api_shapes import HardwareTierEnvelopeV1
from .apps_routes import FakeState
from .instances import FakeVersion

HARDWARE_TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"
LEGACY_INSTANCE_PATH = "/{owner}/{project}/r/notebookSession/{instanceId}"
FAKE_RUN_TOKEN = "fake-run-token"
# Every hardware tier the fake describes has this size.
FAKE_TIER_CORES = 4.0
FAKE_TIER_MEMORY_GIB = 16.0

_tiers_router = APIRouter(
    prefix=HARDWARE_TIERS_PATH, tags=["Hardware tiers (Domino 6.2 API)"]
)
_stand_ins_router = APIRouter(tags=["Stand-ins (not Domino's Apps API)"])


@_tiers_router.get("/{hardwareTierId}")
def get_hardware_tier(hardwareTierId: str) -> HardwareTierEnvelopeV1:
    """Get one hardware tier; any id is a tier with ``FAKE_TIER_CORES`` cores.

    Args:
        hardwareTierId: The tier's id, also used as its name.

    Returns:
        The tier, with 16 GiB of memory and every flag false.
    """
    return tier_envelope(
        hardwareTierId, FAKE_TIER_CORES, FAKE_TIER_MEMORY_GIB, is_default=False
    )


def tier_envelope(
    tier_id: str, cores: float, memory_gib: float, *, is_default: bool
) -> HardwareTierEnvelopeV1:
    """Describe a hardware tier in the 6.2 API's shape.

    Args:
        tier_id: The tier's id, also used as its name.
        cores: Its CPU cores.
        memory_gib: Its memory, in GiB.
        is_default: Its ``isDefault`` flag; every other flag is false.

    Returns:
        The response body carrying the tier.

    Example:
        >>> tier_envelope("cpu-8", 8, 32, is_default=True).hardwareTier.id
        'cpu-8'
    """
    memory = {"value": memory_gib, "units": "GiB"}
    resources = {"cores": cores, "memory": memory}
    flags = dict.fromkeys(_TIER_FLAGS, False) | {"isDefault": is_default}
    tier = {"id": tier_id, "name": tier_id, "flags": flags}
    return HardwareTierEnvelopeV1.model_validate(
        {"hardwareTier": tier | _TIER_DEFAULTS | {"resources": resources | _SHARED}}
        | {"metadata": {"notices": [], "requestId": "fake-request"}}
    )


_TIER_FLAGS = ("isDefault", "isVisible", "isGlobal", "isArchived", "isDataAnalystTier")
_SHARED = {"allowSharedMemoryToExceedDefault": False}
_TIER_DEFAULTS: dict[str, object] = {
    "centsPerMinute": 0.0,
    "creationTime": "2026-01-01T00:00:00Z",
    "updateTime": "2026-01-01T00:00:00Z",
    "podCustomization": {},
    "metadata": {},
}


@_stand_ins_router.get("/access-token", response_class=PlainTextResponse)
def access_token() -> str:
    """Return a fake run token, standing in for the token service of a Domino run.

    Returns:
        ``FAKE_RUN_TOKEN``, as plain text.
    """
    return FAKE_RUN_TOKEN


@_stand_ins_router.get(f"{LEGACY_INSTANCE_PATH}/readyz")
def app_readyz(apps: FakeState, instanceId: str) -> JSONResponse:
    """Answer ``/readyz`` as an App instance would (no credential is checked).

    Args:
        apps: The fake's state.
        instanceId: The instance that answers, e.g. ``"app-1-v1-i"``.

    Returns:
        What ``readyz_stand_in`` answers for the instance's version.

    Raises:
        HTTPException: 404 if no version has that instance.
    """
    owned = apps.instance_owner(instanceId)
    if owned is None:
        raise HTTPException(HTTPStatus.NOT_FOUND, f"no App instance {instanceId}")
    return readyz_stand_in(owned[1])


def readyz_stand_in(version: FakeVersion) -> JSONResponse:
    """Answer ``/readyz`` as the App would, run headers from its entry script.

    The ``X-RAG-Topology`` and ``X-RAG-Run-Label`` headers carry the values the
    version exports as ``RAG_TOPOLOGY`` and ``RAG_RUN_LABEL``.

    Args:
        version: The App version whose instance answers.

    Returns:
        A ready status with one passing check, and the run headers.

    Example:
        >>> readyz_stand_in(apps.apps["app-1"].versions[-1])
    """
    exported = _exported(version.request.entryScript or "")
    headers = {key: exported[name] for key, name in _RUN_HEADERS if name in exported}
    return JSONResponse({"status": "ready", "checks": [_FAKE_CHECK]}, headers=headers)


_RUN_HEADERS = [
    ("X-RAG-Topology", "RAG_TOPOLOGY"),
    ("X-RAG-Run-Label", "RAG_RUN_LABEL"),
]
_FAKE_CHECK = {"name": "fake_domino", "ok": True, "detail": "answered by the fake"}


def _exported(script: str) -> dict[str, str]:
    # The last export of a name wins, as in the shell.
    lines = [shlex.split(line) for line in script.splitlines()]
    pairs = [line[1].partition("=") for line in lines if line[:1] == ["export"]]
    return {name: value for name, _, value in pairs}
