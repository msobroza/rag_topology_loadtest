"""Names a framework uses but this code never reads, so vulture reports them unused.

``make deadcode`` scans this file with the code (``[tool.vulture]`` in
pyproject.toml): each bare name below counts as a use. It is never imported.
Names that only a decorator registers (routes, validators, fixtures, Locust
tasks) are in ``ignore_decorators`` instead.
"""

# Locust: finds the User subclasses of the locustfile.
EmbeddingsUser
RerankUser

# pytest: calls this conftest hook by name (tests/conftest.py adds --pin-answers).
pytest_addoption

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

# Textual: reads an App's BINDINGS and calls its compose, the action a binding
# names (action_<name>) and a message's handler (on_<widget>_<message>, on_mount)
# by name, and reads the display and show_root fields a widget is given
# (OpeningList in deploy/domino_listings/opening_list.py).
BINDINGS
action_filter
action_unfilter
compose
on_input_changed
on_input_submitted
on_mount
on_tree_node_collapsed
on_tree_node_expanded
display
show_root

# pydantic: RecordedVectorStoreRecord.text_sha256, written to an answer record and
# compared by name (loadtest/answer_comparison.py's _EXACT_RECORD_FIELDS).
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

# pydantic: the OpenMP fields of ProcessEnvironmentSettings, read by their names
# through config/apps/process_environment_settings.py's OPENMP_VARIABLES, each
# the upper-cased field.
omp_proc_bind
kmp_affinity

# pydantic: the flag and CPU fields of VllmSettings, read by their names through
# config/apps/vllm_settings.py's VLLM_FLAGS and CPU_VARIABLES, which pair each
# with the vLLM flag or variable it writes.
max_model_len
optimization_level
max_num_batched_tokens
performance_mode
kv_cache_dtype
dtype
tensor_parallel_size
stream_interval
cpu_omp_threads_bind
cpu_reserved_cpus

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
