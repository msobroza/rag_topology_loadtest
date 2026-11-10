"""The Domino 6.2 beta Apps API shapes, transcribed from its Platform API reference.

Names, fields and required flags are as documented there. The fake Domino serves
them, so its Swagger page shows the contract ``rag-deploy`` is written against.
They are a transcription, not a recording of a live Domino: the client in
``domino_apps.py`` reads responses leniently and does not validate against them.

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


class Tag(_Request):
    key: str
    value: str


class AppVersionCreationRequest(_Request):
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
    tags: list[Tag] | None = None


class AppCreationRequest(_Request):
    name: str
    projectId: str
    visibility: Visibility
    version: AppVersionCreationRequest
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
    tags: list[Tag]
    gitRef: GitRef | None = None
    dfsCommitId: str | None = None
    bundleId: str | None = None
    entryScript: str | None = None
    vanityUrl: str | None = None
    currentInstance: AppInstanceSummaryResponse | None = None


class ProjectSummary(BaseModel):
    id: str
    name: str
    ownerId: str
    ownerUsername: str
    projectType: Literal["Analytic", "Dataset", "Template"]


class Publisher(BaseModel):
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
    project: ProjectSummary
    vanityUrl: str | None = None
    description: str | None = None
    updatedAt: float | None = None
    publisher: Publisher | None = None
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


class FailureEnvelopeV1(BaseModel):
    errors: list[str]
    requestId: str
