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

# pydantic: the named fields of OpenVinoCpuSettings, read by their names in
# config/openvino_cpu_settings.py's _PROPERTY_OF_FIELD, which pairs each with
# the OpenVINO property it writes.
performance_hint
performance_hint_num_requests
hyper_threading
cpu_pinning
cpu_reservation
inference_precision_hint
execution_mode_hint

# pydantic: the flag fields of OvmsSettings, read by their names through
# config/apps/ovms_flags.py's SERVER_FLAGS and LLM_FLAGS, which pair each with
# the ovms flag it writes.
cache_size_gb
prefix_caching
kv_cache_precision
model_distribution_policy
rest_workers
metrics
log_level
cache_dir

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
