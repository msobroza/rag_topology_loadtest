"""Names a framework uses but this code never reads, so vulture reports them unused.

``make deadcode`` scans this file with the code (``[tool.vulture]`` in
pyproject.toml): each bare name below counts as a use. It is never imported.
Names that only a decorator registers (routes, validators, fixtures, Locust
tasks) are in ``ignore_decorators`` instead.
"""

# Locust: finds the User subclasses of the locustfile.
EmbeddingsUser
RerankUser

# pydantic-settings: calls this classmethod hook to order the settings sources.
settings_customise_sources

# httpx: a BaseTransport subclass answers through handle_request (fake_domino).
handle_request

# elasticsearch-py: async_bulk calls client.bulk by name (FakeAsyncElasticsearch
# in tests/adapters/vector_store/test_elasticsearch_store.py).
bulk

# pydantic: Route.utterances, read from routes.yaml for a use case's router to
# learn its routes from; the toolkit's fake router reads none.
utterances

# LangChain: calls a callback handler's on_chain_start by name (ChainNameRecorder
# in tests/helpers/workflow_fakes.py).
on_chain_start

# pydantic: RecordedPassage.text_sha256, written to an answer record and compared
# by name (loadtest/answer_comparison.py's _EXACT_PASSAGE_FIELDS).
text_sha256

# pydantic + FastAPI: fields of the Domino API bodies in fake_domino/api_shapes.py
# (Apps, hardware tiers, environments and their revisions), read only when a body
# is validated or serialised to JSON.
accessStatuses
additionalLimits
additionalRequests
allowSharedMemoryToExceedDefault
archived
availableTools
bundleId
centsPerMinute
coresLimit
createdAt
creationTime
currentInstance
currentPage
currentVersion
describeUrl
dfsCommitId
discoverable
entryPoint
externalVolumeMountIds
gitRef
gpuConfiguration
gpuKey
hardwareTier
hardwareTiers
isArchived
isDataAnalystTier
isDefault
isDefaultForModelApi
isGlobal
isModelApiTier
isVisible
latestRevision
maxReplicas
memoryLimit
mountDatasets
mountNetAppVolumes
netAppVolumeIds
notices
numberOfGpus
ownerId
ownerType
ownerUsername
pageInfo
podCustomization
projectType
publisher
rebuildOnBaseChanges
renderIFrame
requestId
restrictedRevision
scaleDownStabilizationWindowSeconds
scaleUpStabilizationWindowSeconds
selectedRevision
sendNotifications
supportedClusters
targetCpuAvgUtilizationPct
targetMemoryAvgUtilizationPct
totalCount
totalPages
units
updateTime
updatedAt
useSessionAffinity
userId
vanityUrl
views
visibility
