"""Names a framework uses but this code never reads, so vulture reports them unused.

``make deadcode`` scans this file with the code (``[tool.vulture]`` in
pyproject.toml): each bare name below counts as a use. It is never imported.
Names that only a decorator registers (routes, validators, fixtures, Locust
tasks) are in ``ignore_decorators`` instead.
"""

# Locust: finds the User subclasses of the locustfile and reads their wait_time.
EmbeddingsUser
RerankUser
wait_time

# pydantic-settings: calls this classmethod hook to order the settings sources.
settings_customise_sources

# httpx: a BaseTransport subclass answers through handle_request (fake_domino).
handle_request

# elasticsearch-py: async_bulk calls client.bulk by name (FakeAsyncElasticsearch
# in tests/adapters/test_elasticsearch_store.py).
bulk

# LangGraph: a RagState key, written and read by the nodes as a string key.
query_embedding

# pydantic: the wire field OpenAIEmbeddingsRequest.input, sent and read as JSON
# (the model server reads it as body.input; vulture may not tie the two).
input

# pydantic + FastAPI: fields of the Domino Apps API bodies in fake_domino/api_shapes.py,
# read only when a body is validated or serialised to JSON.
accessStatuses
additionalLimits
additionalRequests
allowSharedMemoryToExceedDefault
bundleId
centsPerMinute
coresLimit
createdAt
creationTime
currentInstance
currentVersion
describeUrl
dfsCommitId
discoverable
entryPoint
environmentRevisionId
extendedIdentityPropagationToAppsEnabled
externalVolumeMountIds
gitRef
hardwareTier
isArchived
isDataAnalystTier
isDefault
isDefaultForModelApi
isGlobal
isModelApiTier
isVisible
maxReplicas
memoryLimit
minReplicas
mountDatasets
mountNetAppVolumes
netAppVolumeIds
notices
ownerId
ownerUsername
podCustomization
projectType
publisher
renderIFrame
requestId
scaleDownStabilizationWindowSeconds
scaleUpStabilizationWindowSeconds
sendNotifications
targetCpuAvgUtilizationPct
targetMemoryAvgUtilizationPct
totalCount
units
updateTime
updatedAt
useSessionAffinity
userId
vanityUrl
views
visibility
