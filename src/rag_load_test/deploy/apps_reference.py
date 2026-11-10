"""The Domino 6.2 beta Apps API shapes, transcribed from its Platform API reference.

Names, fields and required flags are as documented there. The fake Domino serves
them, so its Swagger page shows the contract ``rag-deploy`` is written against.
They are a transcription, not a recording of a live Domino: the client in
``apps_client.py`` reads responses leniently and does not validate against them.

Example::

    AppCreationRequest(name="rag-workflow", projectId="p1",
                       visibility="AUTHENTICATED", version=AppVersionCreationRequest())
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
    type: Literal["head", "commitId", "branches", "tags", "custom"]
    value: str | None = None


class AppVersionTag(_Request):
    key: str
    value: str


class AppAutoscalingSpecification(_Request):
    enabled: bool
    minReplicas: int | None = None
    maxReplicas: int | None = None
    targetCpuAvgUtilizationPct: int | None = None
    targetMemoryAvgUtilizationPct: int | None = None
    scaleUpStabilizationWindowSeconds: int | None = None
    scaleDownStabilizationWindowSeconds: int | None = None
    useSessionAffinity: bool | None = None


class AppAccessStatus(_Request):
    userId: str
    status: Literal["ALLOWED", "DENIED", "PENDING"]


class AppVersionCreationRequest(_Request):
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
    id: str
    createdAt: float
    status: str
    describeUrl: str | None = None


class AppVersionResponse(BaseModel):
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
    id: str
    name: str
    ownerId: str
    ownerUsername: str
    projectType: Literal["Analytic", "Dataset", "Template"]


class AppUserResponse(BaseModel):
    id: str
    name: str


class AppResponse(BaseModel):
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
    notices: list[str]
    requestId: str
    limit: int
    offset: int
    totalCount: int | None = None


class ListAppsResponse(BaseModel):
    items: list[AppResponse]
    metadata: PaginatedMetadataV1


class ListAppInstancesResponse(BaseModel):
    items: list[AppInstanceSummaryResponse]
    metadata: PaginatedMetadataV1


# --- Hardware tiers (GET /api/hardwaretiers/v1/hardwaretiers/{hardwareTierId}) ---


class MetadataV1(BaseModel):
    notices: list[str]
    requestId: str


class InformationV1(BaseModel):
    value: float
    units: Literal["MB", "MiB", "GB", "GiB"]


class HardwareTierResourcesV1(BaseModel):
    cores: float
    memory: InformationV1
    allowSharedMemoryToExceedDefault: bool
    coresLimit: float | None = None
    memoryLimit: InformationV1 | None = None


class HardwareTierFlagsV1(BaseModel):
    isDefault: bool
    isVisible: bool
    isGlobal: bool
    isArchived: bool
    isDataAnalystTier: bool
    isModelApiTier: bool | None = None
    isDefaultForModelApi: bool | None = None


class HardwareTierPodCustomizationV1(BaseModel):
    additionalRequests: dict[str, str] | None = None
    additionalLimits: dict[str, str] | None = None


class HardwareTierV1(BaseModel):
    """The required fields; the reference lists more optional ones."""

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
    hardwareTier: HardwareTierV1
    metadata: MetadataV1


class FailureEnvelopeV1(BaseModel):
    errors: list[str]
    requestId: str
