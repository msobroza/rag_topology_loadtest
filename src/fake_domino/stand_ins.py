"""What the fake serves besides the Apps API: hardware tiers and two stand-ins.

``/api/hardwaretiers/v1/hardwaretiers/{id}`` (any id is a tier), ``/access-token``
(the token service of a Domino run) and an App instance's ``/readyz``, so
``rag-deploy status`` has something to read without Domino.

Example:
    >>> fake.include_router(_tiers_router)  # as create_fake_domino does
"""

from __future__ import annotations

import shlex

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from .api_shapes import HardwareTierEnvelopeV1
from .apps_routes import FakeState

HARDWARE_TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"
FAKE_TIER_CORES = 4.0  # every hardware tier the fake describes has this many cores
FAKE_RUN_TOKEN = "fake-run-token"

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
    memory = {"value": 16, "units": "GiB"}
    resources = {"cores": FAKE_TIER_CORES, "memory": memory}
    flags = dict.fromkeys(_TIER_FLAGS, False)
    tier = {"id": hardwareTierId, "name": hardwareTierId, "flags": flags}
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


@_stand_ins_router.get("/{owner}/{project}/r/notebookSession/{instanceId}/readyz")
def app_readyz(apps: FakeState, instanceId: str) -> JSONResponse:
    """Answer as an App instance would, run headers from its entry script.

    The ``X-RAG-Topology`` and ``X-RAG-Run-Label`` headers carry the values the
    instance's version exports as ``RAG_TOPOLOGY`` and ``RAG_RUN_LABEL``.

    Args:
        apps: The fake's state.
        instanceId: The instance that answers.

    Returns:
        A ready status with one passing check, and the run headers.

    Raises:
        HTTPException: 404 if no version has that instance.
    """
    version = apps.version_of_instance(instanceId)
    if version is None:
        raise HTTPException(404, f"no App instance {instanceId}")
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
