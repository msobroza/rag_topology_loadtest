"""What the fake serves besides the Apps API: tiers, environments and stand-ins.

The hardware tiers: their list (``GET /api/hardwaretiers/v1/hardwaretiers``, the
tiers of the state, by pages) and one by its id (any id is a tier of a fixed
size). The environments: their list (``/api/environments/beta/environments``)
and one by its id (``/api/environments/v1/environments/{environmentId}``, with
its three revisions), both from the state. And three stand-ins that are not
Domino's documented API: every revision of an environment (``REVISIONS_ROUTE``,
an internal route whose shape alone is published), ``/access-token`` (the token
service of a Domino run, documented to answer the raw token as plain text) and
an App instance's ``/readyz`` on its legacy URL (``readyz_stand_in``, no
credential checked), so ``qp-deploy status`` has something to read without Domino.

Example:
    >>> fake.include_router(_tiers_router)  # as create_fake_domino does
"""

from __future__ import annotations

import math
import shlex
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from .api_shapes import (
    EnvironmentEnvelopeV1,
    EnvironmentOwnerV1,
    EnvironmentRevisionV1,
    EnvironmentV1,
    HardwareTierEnvelopeV1,
    HardwareTierV1,
    MetadataV1,
    PaginatedEnvironmentEnvelopeV1,
    PaginatedHardwareTierEnvelopeV1,
    PaginatedMetadataV1,
    PaginatedRevisionData,
    PaginatedRevisionInfo,
    RevisionSummary,
)
from .apps_routes import FakeState
from .instances import FakeVersion
from .state import FAKE_OWNER, FakeDominoApps, FakeEnvironment

HARDWARE_TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"
ENVIRONMENTS_PATH = "/api/environments/v1/environments"
ENVIRONMENTS_LIST_PATH = "/api/environments/beta/environments"
REVISIONS_ROUTE = "/v4/environments/{environmentId}/paginatedRevisionData"
LEGACY_INSTANCE_PATH = "/{owner}/{project}/r/notebookSession/{instanceId}"
FAKE_RUN_TOKEN = "fake-run-token"
# Every hardware tier the fake describes by its id has this size.
FAKE_TIER_CORES = 4.0
FAKE_TIER_MEMORY_GIB = 16.0

_tiers_router = APIRouter(
    prefix=HARDWARE_TIERS_PATH, tags=["Hardware tiers (Domino 6.2 API)"]
)
_environments_router = APIRouter(tags=["Environments (Domino 6.2 API)"])
_stand_ins_router = APIRouter(tags=["Stand-ins (not Domino's Apps API)"])


# The lists' default page sizes are the reference's: 10 for the tiers and the
# environments, 5 for the internal route of revisions (Domino's own pages).
@_tiers_router.get("")
def list_hardware_tiers(
    apps: FakeState, offset: int = 0, limit: int = 10
) -> PaginatedHardwareTierEnvelopeV1:
    """List the hardware tiers of the fake's state, by pages.

    Args:
        apps: The fake's state.
        offset: Tiers skipped before the page starts.
        limit: Most tiers in the page.

    Returns:
        The page, with the number of tiers in ``metadata.totalCount``.
    """
    return PaginatedHardwareTierEnvelopeV1(
        hardwareTiers=apps.tiers[offset : offset + limit],
        metadata=paginated_metadata(offset, limit, len(apps.tiers)),
    )


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
    """Describe a hardware tier in the 6.2 API's shape, its id as its name.

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
    tier = fake_tier(tier_id, tier_id, cores, memory_gib, is_default=is_default)
    return HardwareTierEnvelopeV1(hardwareTier=tier, metadata=_metadata())


def fake_tier(
    tier_id: str,
    name: str,
    cores: float,
    memory_gib: float,
    *,
    gpus: int = 0,
    is_default: bool = False,
) -> HardwareTierV1:
    """Build a hardware tier in the 6.2 API's shape, e.g. for a test's state.

    Args:
        tier_id: The tier's id.
        name: The tier's name.
        cores: Its CPU cores.
        memory_gib: Its memory, in GiB.
        gpus: Its GPUs; 0 gives no ``gpuConfiguration``.
        is_default: Its ``isDefault`` flag; every other flag is false.

    Returns:
        The tier.

    Example:
        >>> fake_tier("gpu-k8s", "gpu-1", 8, 32, gpus=1).gpuConfiguration.numberOfGpus
        1
    """
    memory = {"value": memory_gib, "units": "GiB"}
    resources = {"cores": cores, "memory": memory} | _SHARED
    flags = dict.fromkeys(_TIER_FLAGS, False) | {"isDefault": is_default}
    gpu = {"numberOfGpus": gpus, "gpuKey": _GPU_KEY} if gpus else None
    tier = {"id": tier_id, "name": name, "flags": flags, "resources": resources}
    return HardwareTierV1.model_validate(
        tier | _TIER_DEFAULTS | {"gpuConfiguration": gpu}
    )


def paginated_metadata(offset: int, limit: int, total: int) -> PaginatedMetadataV1:
    """Describe one page of a list in the 6.2 API's shape.

    Args:
        offset: Items skipped before the page starts.
        limit: Most items in the page.
        total: Items in the whole list.

    Returns:
        The page's metadata, with no notice.

    Example:
        >>> paginated_metadata(0, 10, 11).totalCount
        11
    """
    return PaginatedMetadataV1(
        notices=[], requestId=_REQUEST_ID, limit=limit, offset=offset, totalCount=total
    )


def _metadata() -> MetadataV1:
    return MetadataV1(notices=[], requestId=_REQUEST_ID)


_REQUEST_ID = "fake-request"
_GPU_KEY = "nvidia.com/gpu"
_TIER_FLAGS = ("isDefault", "isVisible", "isGlobal", "isArchived", "isDataAnalystTier")
_SHARED = {"allowSharedMemoryToExceedDefault": False}
_TIER_DEFAULTS: dict[str, object] = {
    "centsPerMinute": 0.0,
    "creationTime": "2026-01-01T00:00:00Z",
    "updateTime": "2026-01-01T00:00:00Z",
    "podCustomization": {},
    "metadata": {},
}


# --- Environments ---


@_environments_router.get(ENVIRONMENTS_LIST_PATH)
def list_environments(
    apps: FakeState, offset: int = 0, limit: int = 10
) -> PaginatedEnvironmentEnvelopeV1:
    """List the environments of the fake's state, by pages.

    Args:
        apps: The fake's state.
        offset: Environments skipped before the page starts.
        limit: Most environments in the page.

    Returns:
        The page, with the number of environments in ``metadata.totalCount``.
    """
    every = [fake.environment for fake in apps.environments.values()]
    return PaginatedEnvironmentEnvelopeV1(
        environments=every[offset : offset + limit],
        metadata=paginated_metadata(offset, limit, len(every)),
    )


@_environments_router.get(f"{ENVIRONMENTS_PATH}/{{environmentId}}")
def get_environment(apps: FakeState, environmentId: str) -> EnvironmentEnvelopeV1:
    """Get one environment of the fake's state, with its three revisions.

    Args:
        apps: The fake's state.
        environmentId: The environment's id.

    Returns:
        The environment: its latest, selected (active) and restricted revision.

    Raises:
        HTTPException: 404 if the state has no environment with that id.
    """
    environment = _environment_of(apps, environmentId).environment
    return EnvironmentEnvelopeV1(environment=environment, metadata=_metadata())


def _environment_of(apps: FakeDominoApps, environment_id: str) -> FakeEnvironment:
    if environment_id not in apps.environments:
        known = list(apps.environments)
        raise HTTPException(
            HTTPStatus.NOT_FOUND, f"no environment {environment_id!r}; known: {known}"
        )
    return apps.environments[environment_id]


def fake_environment(
    environment_id: str,
    *revisions: tuple[int, str, str],
    name: str = "",
    active: int | None = None,
    restricted: int | None = None,
) -> FakeEnvironment:
    """Build an environment with every one of its revisions, e.g. for a test's state.

    Args:
        environment_id: The environment's id.
        *revisions: Each ``(number, id, built)``, ``built`` an ISO date in UTC
            such as ``"2026-11-12T09:41:00Z"``; the latest is the highest number.
        name: The environment's name; empty takes its id.
        active: The number of the active (selected) revision; None takes the
            latest.
        restricted: The number of the restricted revision; None gives none.

    Returns:
        The environment, private and owned by ``FAKE_OWNER``, each revision
        ``succeeded``, and its revisions newest first.

    Raises:
        ValueError: If ``active`` or ``restricted`` is no revision's number.

    Example:
        >>> fake_environment("env-1", (7, "rev-7", "2026-11-12T09:41:00Z"),
        ...                  (6, "rev-6", "2026-11-03T16:02:00Z"), active=6)
    """
    newest_first = sorted(revisions, key=lambda revision: revision[0], reverse=True)
    numbers = [number for number, _, _ in newest_first]
    latest = numbers[0] if numbers else None
    active = latest if active is None else active
    _check_numbers(environment_id, numbers, active=active, restricted=restricted)
    named = {n: _named_revision(n, rid) for n, rid, _ in newest_first}
    shown = {
        "latestRevision": _revision_numbered(named, latest),
        "selectedRevision": _revision_numbered(named, active),
        "restrictedRevision": _revision_numbered(named, restricted),
    }
    identity: dict[str, object] = {"id": environment_id, "name": name or environment_id}
    environment = EnvironmentV1.model_validate(identity | _ENVIRONMENT_DEFAULTS | shown)
    summaries = [_summary(n, rid, built, active) for n, rid, built in newest_first]
    return FakeEnvironment(environment, tuple(summaries))


def _check_numbers(
    environment_id: str, numbers: list[int], **asked: int | None
) -> None:
    for role, number in asked.items():
        if number is not None and number not in numbers:
            raise ValueError(
                f"environment {environment_id!r}: {role} revision {number} is none "
                f"of its revision numbers {numbers}"
            )


def _named_revision(number: int, revision_id: str) -> EnvironmentRevisionV1:
    return EnvironmentRevisionV1(
        id=revision_id, number=number, availableTools=[], status="succeeded"
    )


def _revision_numbered(
    named: dict[int, EnvironmentRevisionV1], number: int | None
) -> EnvironmentRevisionV1 | None:
    return None if number is None else named[number]


def _summary(
    number: int, revision_id: str, built: str, active: int | None
) -> RevisionSummary:
    return RevisionSummary(
        id=revision_id,
        number=number,
        created=_epoch_ms(built),
        active=number == active,
    )


def _epoch_ms(built: str) -> int:
    # The internal route's ``created`` is taken to count milliseconds (design).
    moment = datetime.fromisoformat(built)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return round(moment.timestamp() * 1000)


_ENVIRONMENT_DEFAULTS: dict[str, object] = {
    "visibility": "private",
    "rebuildOnBaseChanges": "never",
    "supportedClusters": [],
    "archived": False,
    "owner": EnvironmentOwnerV1(
        id=FAKE_OWNER, username=FAKE_OWNER, ownerType="Individual"
    ),
}


# --- Stand-ins: not Domino's documented API ---


@_stand_ins_router.get(REVISIONS_ROUTE)
def list_revisions(
    apps: FakeState,
    environmentId: str,
    page: Annotated[int, Query(ge=0)] = 0,
    pageSize: Annotated[int, Query(ge=1)] = 5,
) -> PaginatedRevisionData:
    """List every revision of an environment, newest first, standing in for Domino.

    The route is Domino's internal one; its shape alone is published.

    Args:
        apps: The fake's state; ``revisions_route_status`` other than 200
            makes the route answer that status.
        environmentId: The environment's id.
        page: The page, counted from 0.
        pageSize: Most revisions in the page.

    Returns:
        The page of revisions, with the number of pages.

    Raises:
        HTTPException: The status a test set, or 404 if the state has no
            environment with that id.
    """
    if apps.revisions_route_status != HTTPStatus.OK:
        status = apps.revisions_route_status
        raise HTTPException(status, f"the test set this route to answer {status}")
    revisions = _environment_of(apps, environmentId).revisions
    start = page * pageSize
    info = PaginatedRevisionInfo(
        totalPages=math.ceil(len(revisions) / pageSize),
        currentPage=page,
        pageSize=pageSize,
    )
    return PaginatedRevisionData(
        revisions=list(revisions[start : start + pageSize]), pageInfo=info
    )


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
    version exports as ``QP_TOPOLOGY_ALIAS`` and ``QP_RUN_LABEL``; a header
    whose variable is not exported is left out, as the workflow app does.

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
    ("X-RAG-Topology", "QP_TOPOLOGY_ALIAS"),
    ("X-RAG-Run-Label", "QP_RUN_LABEL"),
]
_FAKE_CHECK = {"name": "fake_domino", "ok": True, "detail": "answered by the fake"}


def _exported(script: str) -> dict[str, str]:
    # The last export of a name wins, as in the shell.
    lines = [shlex.split(line) for line in script.splitlines()]
    pairs = [line[1].partition("=") for line in lines if line[:1] == ["export"]]
    return {name: value for name, _, value in pairs}
