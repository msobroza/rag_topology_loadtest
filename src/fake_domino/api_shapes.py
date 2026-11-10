"""The Domino 6.2 beta Apps API shapes, transcribed from its Platform API reference.

Names, fields and required flags are as documented there. The fake Domino serves
them, so its Swagger page shows the contract ``rag-deploy`` is written against.
They are a transcription, not a recording of a live Domino: the client in
``rag_load_test/deploy/apps_client.py`` reads responses leniently and does not
validate against them.

Example:
    >>> AppCreationRequest(name="rag-workflow", projectId="p1",
    ...                    visibility="AUTHENTICATED",
    ...                    version=AppVersionCreationRequest())
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Visibility = Literal["AUTHENTICATED", "GRANT_BASED", "GRANT_BASED_STRICT", "PUBLIC"]
ConfigurationType = Literal["STANDARD", "AISYSTEM"]


# --- The Domino 6.2 reference's shapes (names and required fields as documented) ---


class _Request(BaseModel):
    # Unknown fields are refused so a misspelt field in rag-deploy shows up here.
    model_config = ConfigDict(extra="forbid")


class GitRef(_Request):
    """A git reference an App version is built from."""

    type: Literal["head", "commitId", "branches", "tags", "custom"]
    value: str | None = None


class AppVersionTag(_Request):
    """A key and value tagged on an App version."""

    key: str
    value: str


class AppAutoscalingSpecification(_Request):
    """How an App version scales its replicas."""

    enabled: bool
    minReplicas: int | None = None
    maxReplicas: int | None = None
    targetCpuAvgUtilizationPct: int | None = None
    targetMemoryAvgUtilizationPct: int | None = None
    scaleUpStabilizationWindowSeconds: int | None = None
    scaleDownStabilizationWindowSeconds: int | None = None
    useSessionAffinity: bool | None = None


class AppAccessStatus(_Request):
    """One user's access to an App."""

    userId: str
    status: Literal["ALLOWED", "DENIED", "PENDING"]


class AppVersionCreationRequest(_Request):
    """The version part of a request that creates an App."""

    autoscalingSpecification: AppAutoscalingSpecification | None = None
    environmentId: str | None = None
    environmentRevisionId: str | None = None
    hardwareTierId: str | None = None
    gitRef: GitRef | None = None
    dfsCommitId: str | None = None
    bundleId: str | None = None
    entryScript: str | None = None
    vanityUrl: str | None = None
    externalVolumeMountIds: list[str] | None = None
    netAppVolumeIds: list[str] | None = None
    tags: list[AppVersionTag] | None = None
    extendedIdentityPropagationToAppsEnabled: bool | None = None


class AppCreationRequest(_Request):
    """The request body that creates an App."""

    name: str
    projectId: str
    visibility: Visibility
    version: AppVersionCreationRequest
    accessStatuses: list[AppAccessStatus] | None = None
    entryPoint: str | None = None
    description: str | None = None
    discoverable: bool | None = None
    configurationType: ConfigurationType | None = None
    mountDatasets: bool | None = None
    mountNetAppVolumes: bool | None = None
    renderIFrame: bool | None = None
    sendNotifications: bool | None = None


class AppInstanceSummaryResponse(BaseModel):
    """A running or stopped instance of an App version."""

    id: str
    createdAt: float
    status: str
    describeUrl: str | None = None


class AppVersionResponse(BaseModel):
    """A version of an App, with its current instance if any."""

    id: str
    createdAt: float
    updatedAt: float
    environmentId: str
    environmentRevisionId: str
    hardwareTierId: str
    externalVolumeMountIds: list[str]
    netAppVolumeIds: list[str]
    tags: list[AppVersionTag]
    extendedIdentityPropagationToAppsEnabled: bool
    autoscalingSpecification: AppAutoscalingSpecification | None = None
    gitRef: GitRef | None = None
    dfsCommitId: str | None = None
    bundleId: str | None = None
    entryScript: str | None = None
    vanityUrl: str | None = None
    currentInstance: AppInstanceSummaryResponse | None = None


class AppProjectResponse(BaseModel):
    """The project an App belongs to."""

    id: str
    name: str
    ownerId: str
    ownerUsername: str
    projectType: Literal["Analytic", "Dataset", "Template"]


class AppUserResponse(BaseModel):
    """The user who published an App."""

    id: str
    name: str


class AppResponse(BaseModel):
    """An App, with its project and current version."""

    id: str
    name: str
    entryPoint: str
    url: str
    visibility: Visibility
    configurationType: ConfigurationType
    discoverable: bool
    mountDatasets: bool
    mountNetAppVolumes: bool
    renderIFrame: bool
    views: int
    project: AppProjectResponse
    accessStatuses: list[AppAccessStatus]
    vanityUrl: str | None = None
    description: str | None = None
    updatedAt: float | None = None
    publisher: AppUserResponse | None = None
    currentVersion: AppVersionResponse | None = None


class PaginatedMetadataV1(BaseModel):
    """The paging metadata of a list response."""

    notices: list[str]
    requestId: str
    limit: int
    offset: int
    totalCount: int | None = None


class ListAppsResponse(BaseModel):
    """One page of Apps."""

    items: list[AppResponse]
    metadata: PaginatedMetadataV1


class ListAppInstancesResponse(BaseModel):
    """One page of an App version's instances."""

    items: list[AppInstanceSummaryResponse]
    metadata: PaginatedMetadataV1


# --- Hardware tiers (GET /api/hardwaretiers/v1/hardwaretiers/{hardwareTierId}) ---


class MetadataV1(BaseModel):
    """The metadata of a single-object response."""

    notices: list[str]
    requestId: str


class InformationV1(BaseModel):
    """An amount of memory with its unit."""

    value: float
    units: Literal["MB", "MiB", "GB", "GiB"]


class HardwareTierResourcesV1(BaseModel):
    """The CPU and memory a hardware tier gives a run."""

    cores: float
    memory: InformationV1
    allowSharedMemoryToExceedDefault: bool
    coresLimit: float | None = None
    memoryLimit: InformationV1 | None = None


class HardwareTierFlagsV1(BaseModel):
    """The flags that say where a hardware tier is used."""

    isDefault: bool
    isVisible: bool
    isGlobal: bool
    isArchived: bool
    isDataAnalystTier: bool
    isModelApiTier: bool | None = None
    isDefaultForModelApi: bool | None = None


class HardwareTierPodCustomizationV1(BaseModel):
    """Extra Kubernetes requests and limits of a hardware tier."""

    additionalRequests: dict[str, str] | None = None
    additionalLimits: dict[str, str] | None = None


class HardwareTierV1(BaseModel):
    """A hardware tier: the required fields; the reference lists more optional ones."""

    id: str
    name: str
    resources: HardwareTierResourcesV1
    flags: HardwareTierFlagsV1
    centsPerMinute: float
    creationTime: str
    updateTime: str
    podCustomization: HardwareTierPodCustomizationV1
    metadata: dict[str, str]


class HardwareTierEnvelopeV1(BaseModel):
    """The response body that carries one hardware tier."""

    hardwareTier: HardwareTierV1
    metadata: MetadataV1


class FailureEnvelopeV1(BaseModel):
    """The response body of a failed request."""

    errors: list[str]
    requestId: str
