# Server settings for throughput and latency, as each server's documentation gives them

What the documentation of OpenVINO Model Server, vLLM, and FastAPI with Uvicorn and
the in-process model libraries says to set to raise throughput or to lower latency
on a CPU, for an embedder, a reranker and an LLM. It was read on 19 November 2026,
from the files at the tag of the version this project runs:

- OVMS 2026.4.1 (`query_processing/ovms/versions.env`), and OpenVINO's own CPU
  pages, which OVMS hands its `plugin_config` to;
- vLLM 0.32.0, the CPU image `vllm/vllm-openai-cpu:v0.32.0-x86_64` of the sitting
  of 16 November 2026;
- FastAPI 0.162.2, Uvicorn 0.54.0, Starlette 1.10.0, AnyIO 4.16.0,
  Sentence Transformers 6.2.0, PyTorch 2.15.0 and httpx 0.28.1, the versions
  installed in the repository's `.venv`.

This page reports what the documentation says, not measurements. What the sittings
measured is in `docs/model-servers.md` and `experiments/`. "The docs do not say"
means that no page read ties the setting to throughput or latency. Each claim names
its source. A few were checked again against the source code: the defaults of OVMS's
LLM options, its embeddings and rerank request queue, vLLM's CPU batch defaults and
its reading of `VLLM_CPU_KVCACHE_SPACE`, Uvicorn's refusal of workers for an app
passed as an object, and AnyIO's thread limit.

## In short

"Here today" says what this project does with the setting, and the variable that
sets it, since the settings of 19 November 2026 (the README's "Variables").

### OVMS, embedder and reranker

| Setting | For throughput | For latency | Here today |
|---|---|---|---|
| `NUM_STREAMS` | More streams under concurrency: fewer than the requests in flight, never more than the cores | 1 | Calibrated (`QP_OVMS_NUM_STREAMS`); its candidates stop at the cores, the CPUs divided by `QP_LOCAL_MODELS__THREADS_PER_CORE` (2) |
| `INFERENCE_NUM_THREADS` | Every core | The cores of one socket | Calibrated (`QP_OVMS_THREADS`) |
| `PERFORMANCE_HINT` | `THROUGHPUT`, which picks the streams itself; never with `NUM_STREAMS` | `LATENCY`: one stream | Not set (`QP_OVMS_OPENVINO__PERFORMANCE_HINT`); set, it replaces `NUM_STREAMS`, and the streams are not calibrated |
| `ENABLE_HYPER_THREADING` | By platform under `THROUGHPUT` | Off under `LATENCY`: one request does not use both threads of a core | On by default (`QP_OVMS_OPENVINO__HYPER_THREADING`, `QP_LOCAL_MODELS__OPENVINO__HYPER_THREADING`) |
| `ENABLE_CPU_PINNING` | Off when several workloads share the machine | Same | Not set (`QP_OVMS_OPENVINO__CPU_PINNING`); OVMS turns it off by itself under a container CPU quota |
| `INFERENCE_PRECISION_HINT` | `bf16` on CPUs with AMX (4th-generation Xeon and later): "significantly faster" | Same | Not set (`QP_OVMS_OPENVINO__INFERENCE_PRECISION_HINT`, `QP_LOCAL_MODELS__OPENVINO__INFERENCE_PRECISION_HINT`) |
| Weight format | `int8`, the export helper's default | Same | `fp32` by default (`QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT`) |
| Tokens of a request (texts × length) | Fewer, shorter: one request is one inference, padded to its longest text | Same | 20 vector-store records (`QP_GRAPH__STAGES__RETRIEVE__TOP_K`), never split |
| Graph pool (`# OVMS_GRAPH_QUEUE_MAX_SIZE`) | Saves building a graph for each request | Same | As exported, `AUTO` (`QP_OVMS_GRAPH_QUEUE_MAX_SIZE`) |
| `nireq` | Has no effect: these graphs size their queue from OpenVINO's optimal request count | — | — |
| `metrics_enable`, `log_level` | Metrics cost a share of CPU on light models; keep `INFO` | Same | Defaults (`QP_OVMS_METRICS`, `QP_OVMS_LOG_LEVEL`) |

### OVMS, LLM

| Setting | For throughput | For latency | Here today |
|---|---|---|---|
| `max_num_batched_tokens` | 256, the default, with many clients and short contexts | 4096, 8192 or the context length with few clients and long prompts: sooner first token | 8192 by default (`QP_OVMS_MAX_BATCHED_TOKENS`) |
| `dynamic_split_fuse` | On, the default | Off at very low concurrency or very short prompts, with `max_num_batched_tokens` at the context length | Off by default (`QP_OVMS_DYNAMIC_SPLIT_FUSE`) |
| `cache_size` (KV cache) | 0, dynamic: avoids preemption, may take all the memory | Same | Pull-mode default (`QP_OVMS_CACHE_SIZE_GB`) |
| `enable_prefix_caching` | Less work when prompts repeat | Much sooner first token on a repeated prefix | Pull-mode default (`QP_OVMS_PREFIX_CACHING`; the docs disagree on the default) |
| `KV_CACHE_PRECISION` | Lower (`u8`, `u4`): more requests fit | Lower "can also improve" it | Pull-mode default (`QP_OVMS_KV_CACHE_PRECISION`; the docs disagree on the default) |
| `max_num_seqs` | The docs do not say | Keeps the speed of the requests already running | Pull-mode default (`QP_OVMS_MAX_NUM_SEQS`) |
| NUMA | One OVMS per NUMA node, behind a balancer | — | One App per machine; no variable |
| `rest_workers` | Raise it for very many clients | Lower it to cap connections and avoid overload | Default (`QP_OVMS_REST_WORKERS`) |

### vLLM on CPU

| Setting | For throughput | For latency | Here today |
|---|---|---|---|
| `VLLM_CPU_OMP_THREADS_BIND` | Check it first: one thread per physical core, each rank on one NUMA node | The docs do not say | vLLM's own, `auto` (`QP_VLLM_CPU_OMP_THREADS_BIND`) |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | Keep 1 or 2 cores for the API server | The docs do not say | vLLM's own, 1 (`QP_VLLM_CPU_RESERVED_CPUS`) |
| `--max-num-seqs` | Larger (default 128 under `vllm serve` on CPU) | Smaller | vLLM's own (`QP_VLLM_MAX_NUM_SEQS`) |
| `--max-num-batched-tokens` | Default 2048 on CPU; it mostly decides the first token | Smaller: steadier tokens after the first; larger: sooner first token | vLLM's own (`QP_VLLM_MAX_NUM_BATCHED_TOKENS`) |
| `--performance-mode throughput` | Doubles the two above when they are left unset | — | Not set (`QP_VLLM_PERFORMANCE_MODE`) |
| `--enable-chunked-prefill` | On by default; helps both | Same | vLLM's own, on; no variable of its own (`QP_VLLM_ARGS`) |
| `--enable-prefix-caching` | When requests share a prefix | Prefill only, never decode | vLLM's own (`QP_VLLM_PREFIX_CACHING`) |
| `--kv-cache-memory-bytes`, `--gpu-memory-utilization` | More KV cache: more requests at once, less preemption. By default the worker reserves 0.92 of the NUMA node's memory | Less preemption | 8 GiB (`QP_VLLM_KV_CACHE_GIB`), what the one run asked |
| `--kv-cache-dtype fp8` | More tokens fit | The docs do not say | Not set (`QP_VLLM_KV_CACHE_DTYPE`) |
| `--dtype bfloat16` | Advised when float16 misbehaves on CPU | Same | Not set (`QP_VLLM_DTYPE`) |
| Weights | AWQ, GPTQ or INT8 W8A8 on x86 | — | The model's published weights, which the component's `source_id` names |
| `-O0` or `--enforce-eager` | Slower steady state | Fastest start: no compiling | vLLM's own level, 2 (`QP_VLLM_OPTIMIZATION_LEVEL`; 0 starts with no compilation) |
| `-tp` | One rank per NUMA node | More TP adds synchronisation | vLLM's own, 1 (`QP_VLLM_TENSOR_PARALLEL_SIZE`) |
| `--stream-interval` | Larger "may increase" it | 1: smoother streaming | vLLM's own, 1 (`QP_VLLM_STREAM_INTERVAL`) |
| Docker `--cap-add SYS_NICE`, seccomp, `--shm-size` | Without them NUMA binding silently does not apply | Same | The container's, which Domino sets; no variable: ask an administrator |

### FastAPI model server and its in-process model

| Setting | For throughput | For latency | Here today |
|---|---|---|---|
| Uvicorn `workers` | More processes use more cores; each loads its own copy of the model. On a cluster the docs prefer one process per container and more replicas | The docs do not say | One process (`QP_UVICORN_WORKERS`, 1); above 1 the app goes to Uvicorn as an import string, and the workers divide the CPUs |
| Model threads (intra-op) | Fewer per request avoids the cost of parallel work and oversubscription | More per request lowers its latency | Calibrated (`QP_COMPONENTS__<NAME>__LOCAL__THREADS`) |
| Batch size | The best one per backend, found by measuring | May differ from the throughput's best | Calibrated, reranker only (`QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE`) |
| Backend (Sentence Transformers) | Cross-encoders on CPU: OpenVINO `int8` if a small loss is acceptable, else OpenVINO, else ONNX `O3` | Same; ONNX is stronger at small batches | OpenVINO, or `torch` (`QP_LOCAL_MODELS__RUNTIME`) |
| `fp16`, `bf16` on CPU | Avoid: about 0.17 times the speed of `fp32` for cross-encoders | Same | Not used; no variable |
| Memory allocator, OpenMP pinning, `numactl` | jemalloc or TCMalloc, threads pinned, one instance per socket | The docs do not say | Not set (`QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY`, `QP_PROCESS_OMP_PROC_BIND`); no `numactl` |
| `limit_concurrency` | Sheds load with an immediate 503; to queue it, a proxy in front or more workers | Same | Not set (`QP_UVICORN_LIMIT_CONCURRENCY`) |
| `backlog` | Matters only while the event loop is blocked | Same | Default, 2048 (`QP_UVICORN_BACKLOG`) |

The model calls run on the project's own bounded thread pool
(`infrastructure/local_models/model_pool.py`), not on AnyIO's pool of 40, so the
limit on that pool does not bound them.

### Workflow app, which waits on the others

| Setting | For throughput | For latency | Here today |
|---|---|---|---|
| One shared client | Fewer handshakes, less CPU | No handshake on each request | One shared `AsyncClient`, created on the first request (`composition/http_resources.py`) |
| `httpx.Limits` | 100 connections, 20 kept alive, 5 s idle; a full pool waits up to the `pool` timeout | The docs do not say | Defaults (`QP_HTTP__MAX_CONNECTIONS`, `QP_HTTP__MAX_KEEPALIVE_CONNECTIONS`, `QP_HTTP__KEEPALIVE_EXPIRY_S`) |
| `http2=True` | Worth trying for highly concurrent requests; needs `h2` and a server that speaks HTTP/2 (Uvicorn's own is experimental) | The docs do not say | Off (`QP_HTTP__HTTP2`) |
| `loop`, `http` | uvloop and httptools, picked by themselves when installed | Same | Installed with `uvicorn[standard]` |

## Corrections to `docs/model-servers.md`

That page was written from OVMS 2026.3.1. Its table "What OVMS's own material says
matters for a reranker" still holds at 2026.4.1, but for two points, corrected
there on 19 November 2026:

- OVMS adjusts threads, streams and pinning to its container only when it detects
  Docker or Kubernetes on Linux, and it sets a stream count only under the
  `THROUGHPUT` hint. A CPU quota turns pinning off (`src/ov_utils.cpp`).
- `nireq` never reaches an embeddings or a rerank graph. Their queue of inference
  requests is OpenVINO's optimal number for the compiled model
  (`src/sidepacket_servable.cpp`).

The advice of one stream a model with cores reserved, when several models share a
server, rests on a maintainer's answer in issue 3136, not on a page of the
documentation; that page now says so. Its sentence that the stream candidates go
above the cores is corrected too: they stop at the cores.

The detail of each server follows: every setting the documentation ties to speed,
where it is set, its default, and its source.

## OpenVINO Model Server 2026.4.1

Scope: what OVMS's own docs and source at tag `v2026.4.1`, and OpenVINO Runtime's docs, say
about raising throughput or lowering latency on an Intel x86 CPU, for an embedding model, a
reranker (cross-encoder) and an LLM with continuous batching. Paths below are files of
`github.com/openvinotoolkit/model_server` at tag `v2026.4.1` unless a URL is given. "The docs
do not say" means no page read here ties the setting to throughput or latency.

Version note on the OpenVINO pages: docs.openvino.ai serves one rolling `2026` tree (the
version selector offers 2026, 2025, 2024, 2023.3, nightly); no 2026.4-specific page exists.
OVMS's own links point at that `2026` tree. Where a fact came from OpenVINO's source, it is
`src/inference/include/openvino/runtime/properties.hpp` of `github.com/openvinotoolkit/openvino`
at tag `2026.4.1` (the release that matches by number; the OVMS Makefile at the tag takes the
OpenVINO version from variables it does not set, so the match is by number, not proven).

### 1. Server-level settings (all graph types)

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `rest_workers` | CLI `--rest_workers` | Linux: the smaller of the CPU count OVMS detects and (open-files limit − 15) / 7; minimum 2, max 10 000 (source). Docs: "based on the number of CPUs" | Raise it for stress benchmarks with very many clients. The LLM tokenizer/detokenizer get `min(rest_workers, cores)` streams (source) | Lower it to stop server overload; excess clients wait for a connection. For LLMs it caps simultaneous connections | `docs/performance_tuning.md` "Text generation using LLM calculator…"; `docs/parameters.md` "Server configuration options"; `demos/continuous_batching/long_context/README.md` "Concurrency Control"; `src/config.cpp` (`getDefaultRestWorkers`, `restWorkers`); `src/llm/language_model/continuous_batching/servable_initializer.cpp` (tokenizer streams) |
| `grpc_workers` | CLI | 1 (1 to core count; Windows only 1) | Higher may raise transfer rate under heavy load and many parallel connections; 1 suffices in most cases | The docs do not say | `docs/performance_tuning.md` "Tuning Model Server configuration parameters"; `docs/parameters.md`; `src/cli_parser.cpp` |
| `grpc_max_threads` | CLI | 8 × detected cores (source) | The docs do not say | The docs do not say | `docs/parameters.md`; `src/config.cpp` (`DEFAULT_GRPC_MAX_THREADS`) |
| `nireq` | CLI `--nireq`, `config.json` per model | 0 = OpenVINO's `OPTIMAL_NUMBER_OF_INFER_REQUESTS` | A queue size, not parallelism; keep it ≥ `NUM_STREAMS`, raise toward the expected parallel clients under high load | The docs do not say | `docs/performance_tuning.md` "Tuning Model Server configuration parameters"; `docs/parameters.md`; `src/modelinstance.cpp` (`getNumOfParallelInferRequests`). **Applies to classic models only**: embeddings and rerank graphs size their request queue from `OPTIMAL_NUMBER_OF_INFER_REQUESTS` of the compiled model and read no `nireq` (`src/sidepacket_servable.cpp`) |
| Graph pool `# OVMS_GRAPH_QUEUE_MAX_SIZE` | Comment line in `graph.pbtxt`; env `OVMS_GRAPH_QUEUE_OFF=1` turns all pools off | Off when the line is absent; `export_model.py` and `--pull --task` write `AUTO` = hardware threads (16 if unknown) | Removes per-request graph creation and teardown cost (docs call it "especially beneficial" when calculator set-up is costly) | Same mechanism; the docs do not separate the two | `docs/mediapipe.md` "Graph Pool (Pre-initialized Graph Queue)"; `demos/common/export_models/export_model.py` (templates) |
| Container CPU allotment | `docker --cpuset-cpus` (not `--cpus`); Kubernetes CPU manager | — | Pin each instance to its own cores when several share a host | The docs do not say | `docs/performance_tuning.md` "Scalability" |
| Automatic CPU adaptation | None: OVMS sets it when no value is given | Only when OVMS detects Docker or Kubernetes on Linux (source): `ENABLE_CPU_PINNING` = false if a CPU quota is set, else true; `INFERENCE_NUM_THREADS` = detected cores (min of affinity and quota) when ≤ physical cores; with `PERFORMANCE_HINT: THROUGHPUT` and no `NUM_STREAMS`, streams = min(detected cores, physical cores) | The docs do not say | The docs do not say | `docs/performance_tuning.md` "Scalability" (note); `src/ov_utils.cpp` (`applyDefaultCpuProperties`); `src/systeminfo.cpp`. Applied to classic models, embeddings/rerank (`src/sidepacket_servable.cpp`) and LLM (`…/continuous_batching/servable_initializer.cpp`) |
| `log_level` | CLI | `INFO` | Keep `INFO` for very high throughput; `DEBUG` logs heavily | The docs do not say | `docs/performance_tuning.md` "Tuning Model Server configuration parameters" |
| `metrics_enable` | CLI, `config.json` | off | Negligible cost for average models; noticeable share of CPU for very fast, light models | Same | `docs/performance_tuning.md`; `docs/metrics.md` "Performance considerations" |
| `file_system_poll_wait_seconds` | CLI | 1 | The docs tie it to storage API cost, not speed; 0 disables | The docs do not say | `docs/performance_tuning.md` |
| Client address | Client URL; or server `--rest_bind_address`/`--grpc_bind_address` with IPv6 | Binds all IPv4 | — | Use `http://127.0.0.1` rather than `localhost`, which may try IPv6 first and add latency | `docs/performance_tuning.md` "Network Configuration for Optimal Performance" |
| CPU frequency | OS (`cpupower`) | OS-managed | — | Lock to base frequency for reproducible latency (power saving and Turbo add variability) | `docs/performance_tuning.md` "CPU Power Management Settings" |
| `cache_dir` (model cache) | CLI `--cache_dir`; `export_model.py --ov_cache_dir` writes `CACHE_DIR` into `plugin_config` | Enabled if `--cache_dir` is set or `/opt/cache` exists | — | Shortens model load and first-inference latency; helps only for repeated input shapes | `docs/parameters.md`; `docs/model_cache.md`; CPU device page "Model Caching" |
| `idle_unload_timeout_seconds` | CLI | 0 (off) | — | The first request after an unload pays the reload latency; pair with `cache_dir` | `docs/parameters.md` "Idle servable unload" (new at 2026.4.1) |
| `target_device` | CLI, `config.json`, graph | 2026.4.1: auto-detect, dGPU > iGPU > CPU (NPU never chosen) | — | — | `docs/parameters.md` "Automatic target device detection". On a CPU-only host it lands on CPU |

### 2. OpenVINO CPU properties and graph options for embeddings and rerank graphs

Embeddings and rerank graphs carry `plugin_config` as a JSON string in `node_options`
(`EmbeddingsCalculatorOVOptions.plugin_config`, `RerankCalculatorOVOptions.plugin_config`, default
`""`), handed to OpenVINO at compile time. Neither calculator proto has a stream, thread or
batch field of its own (`src/embeddings/embeddings_calculator_ov.proto`,
`src/rerank/rerank_calculator_ov.proto`). Unlike classic models, OVMS does not add a default
`PERFORMANCE_HINT: LATENCY` to these graphs (`src/modelinstance.cpp` `prepareDefaultPluginConfig`
is for classic models; `src/sidepacket_servable.cpp` only applies the automatic CPU adaptation).

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `NUM_STREAMS` | `export_model.py embeddings_ov/rerank_ov --num_streams`; `ovms --pull --task embeddings/rerank --num_streams`; or by hand in the graph's `plugin_config` | 1 (export script, pull mode). OpenVINO without hint: bare minimum, usually 1 | Raise under many concurrent requests, but below the average number in flight and never above the core count (example: 24 streams for ~50 clients on 48 cores). Particularly efficient for models that cannot use all cores, or CPUs with many cores. Use at least 2 on two-socket systems | 1 for a single client. OpenVINO: streams raise the latency of each request | `docs/performance_tuning.md` "Adjusting the number of streams…"; `docs/parameters.md` "Embeddings", "Rerank"; `demos/common/export_models/export_model.py`; OpenVINO "Advanced Throughput Options" "OpenVINO Streams", "Number of Streams Considerations" |
| `PERFORMANCE_HINT` | Graph `plugin_config` | OpenVINO: `LATENCY` | `THROUGHPUT`: OpenVINO chooses streams from the model's memory pressure (1–5 threads a stream); may load slower and use more memory | `LATENCY`: one stream, threads = cores of one socket | `docs/performance_tuning.md` "Performance Hints"; OpenVINO "High-level Performance Hints"; "Performance Hints and Thread Scheduling" |
| `PERFORMANCE_HINT_NUM_REQUESTS` | `plugin_config` | unset | Caps the streams the THROUGHPUT hint creates to the parallelism the application really has | The docs do not say | OpenVINO "High-level Performance Hints" "Additional (Optional) Hints from the App"; `properties.hpp` (`num_requests`) |
| `INFERENCE_NUM_THREADS` | `plugin_config` | LATENCY: cores of one socket. THROUGHPUT: all cores. In a container OVMS sets it to the detected CPUs (section 1) | OpenVINO: THROUGHPUT streams = threads ÷ threads-per-stream | Limits the logical processors used; OVMS lists it as worth tuning under container CPU limits | `docs/performance_tuning.md` "Scalability"; OpenVINO "Performance Hints and Thread Scheduling" |
| `ENABLE_CPU_PINNING` | `plugin_config` | OpenVINO on Linux: on (except when P- and E-cores mix). OVMS in a container with a CPU quota: off | The docs do not say | The docs do not say. Both docs: turn it off when several workloads run in parallel on the machine | `docs/performance_tuning.md` "Disabling CPU pinning"; OpenVINO "Performance Hints and Thread Scheduling" |
| `ENABLE_CPU_RESERVATION` | `plugin_config` | false | The docs do not say. Maintainer (collaborator) in issue #3136: without it several models may land on the same cores, lowering performance | The docs do not say | `properties.hpp` (`enable_cpu_reservation`); issue #3136 |
| `ENABLE_HYPER_THREADING` | `plugin_config` | LATENCY: no. THROUGHPUT: yes or no by platform | The docs do not say | OpenVINO does not put both logical processors of a core into one stream for a single request, hence off under LATENCY | OpenVINO "Performance Hints and Thread Scheduling" |
| `SCHEDULING_CORE_TYPE` | `plugin_config` | By model precision and P/E-core ratio | No effect on a CPU with one core type (all Xeon) | Same | OpenVINO "Performance Hints and Thread Scheduling" |
| `INFERENCE_PRECISION_HINT` / `EXECUTION_MODE_HINT` | `plugin_config` | `EXECUTION_MODE_HINT: PERFORMANCE`: bf16/f16 where the CPU supports it (AMX on 4th-gen Xeon and newer) | Half precision enables AMX, "significantly faster" than AVX-512/AVX2, and smaller memory | Same | OpenVINO CPU device page "Floating-Point Data Types Specifics", "Execution Mode Hint"; "Precision Control"; `docs/performance_tuning.md` "Analyzing accuracy issues" (forcing `f32`) |
| Weight format | `export_model.py --weight-format`; pull mode `--weight-format` | `int8` | Demos say lower precision (fp16/int8/int4) reduces memory and improves performance; every CPU example uses `int8` | Same | `demos/embeddings/README.md` (note on `--weight-format`); `demos/rerank/README.md`; `export_model.py`; `docs/parameters.md` "Pull Mode Options for optimum-cli mode" |
| Request size (inputs × tokens) | Client | — | One request is one inference: the embeddings calculator tokenizes the whole input list into one batch, the reranker builds one batch of all chunks padded to the longest document (source). Demos: rerank 20 texts × 500 tokens = 3.6 req/s, × 75 tokens = 12.7 req/s; embeddings 32 × 510 tokens = 38.7k tokens/s | Embeddings demo: 1 input of 5 tokens at 10 req/s = 12.75 ms mean | `src/embeddings/embeddings_calculator_ov.cc`; `src/rerank/rerank_calculator_ov.cc`; `demos/rerank/README.md` "Performance benchmarking"; `demos/embeddings/README.md` "Benchmarking feature extraction" |
| `max_allowed_chunks` (rerank) | Graph option; pull `--max_allowed_chunks` | 10 000 | The docs tie it to limits, not speed | The docs do not say | `src/rerank/rerank_calculator_ov.proto`; `docs/parameters.md` "Rerank"; `docs/model_server_rest_api_rerank.md` |
| `truncate`, `max_length` (embeddings) | Graph options; export `--truncate`, `--max_length` | false; model context | The docs do not say for CPU | For NPU only: set `max_length` as low as possible to speed loading and inference | `src/embeddings/embeddings_calculator_ov.proto`; `demos/embeddings/README.md` NPU notes |
| Static shape | Model reshape | Dynamic | OpenVINO: some runtime optimisations work better with shapes known in advance; reshape to static if the input shape does not change | Same | OpenVINO CPU device page "Dynamic Shapes" |
| Denormals, sparse weights | `plugin_config` (`CPU_DENORMALS_OPTIMIZATION`, `CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE`) | off; rate 1 (off) | Sparse weights: helps memory-bound (throughput) cases on int8 models with AMX, may degrade. Denormals: faster if accuracy holds | The docs do not say | OpenVINO CPU device page "Denormals Optimization", "Sparse weights decompression" |

### 3. LLM graph options (continuous batching, `LLMCalculatorOptions`)

Set in `graph.pbtxt` `node_options` (`src/llm/llm_calculator.proto`), by `export_model.py
text_generation`, or by `ovms --pull/--configure --task text_generation` flags
(`docs/parameters.md` "Text generation"). The LLM's `plugin_config` also goes to OpenVINO and
carries pipeline extras (`docs/llm/reference.md` "OpenVINO runtime settings").

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `max_num_batched_tokens` | graph; export/pull `--max_num_batched_tokens` | 256 (proto) | 256 is efficient for short contexts and high concurrency | Raising it to 4096, 8192 or the context length can improve first-token latency, mainly with one client and long prompts; trade-off: interactive use under higher concurrency | `src/llm/llm_calculator.proto`; `docs/performance_tuning.md`; `docs/llm/reference.md` "Scheduling settings"; `demos/continuous_batching/long_context/README.md` |
| `max_num_seqs` | graph; export/pull | 256 | The docs do not say | Can hold a level of generation speed for requests already in the pipeline; must not exceed `max_num_batched_tokens`. Also a per-model concurrency cap | `docs/llm/reference.md` "Scheduling settings"; long_context "Concurrency Control" |
| `dynamic_split_fuse` | graph; export `--disable_dynamic_split_fuse`; pull `--dynamic_split_fuse` | true | On by default to boost throughput by splitting tokens into even chunks | With very low concurrency or very short prompts, turning it off may help; then set `max_num_batched_tokens` to the context length | `docs/llm/reference.md` "Scheduling settings" |
| `cache_size` (KV cache, GB) | graph; export/pull `--cache_size` | 0 = dynamic | Dynamic reserves enough to avoid preemption, "most efficient" for performance, but can take all RAM. Static: safer memory bound, better prompt reuse under steady load; keep under 100% use, as preemption recomputes the cache and hurts performance | Same | `docs/llm/reference.md` "Caching settings"; `docs/performance_tuning.md` |
| `enable_prefix_caching` | graph; export/pull `--enable_prefix_caching` | **Conflicting**: proto and `docs/llm/reference.md` false; `docs/parameters.md`, `export_model.py`, `docs/performance_tuning.md`, long_context guide true. Graphs from the export script or pull mode set true | Lowers resource use when prompts repeat | Reuses KV blocks of repeated prompt prefixes; much faster first token (long_context table: 50k tokens, TTFT 88 s → 0.9 s, iGPU) | `src/llm/llm_calculator.proto`; `docs/llm/reference.md` "Caching settings"; `docs/parameters.md`; `export_model.py`; long_context README |
| `KV_CACHE_PRECISION` | LLM `plugin_config`; export/pull `--kv_cache_precision` | **Conflicting**: `docs/parameters.md` and export script: `u8` or empty (default), only `u8` accepted by the export script; long_context: default `u8`, and elsewhere "default matches model precision"; it recommends `u4`. OpenVINO's source says only "hint… for kv cache compression" | Lower precision raises throughput and cuts memory; frees memory for more parallel requests | Lower precision "can also improve latency" (u4 vs u8 at 50k tokens: 945 vs 985 ms TTFT, iGPU) | `docs/performance_tuning.md`; `docs/llm/reference.md` "OpenVINO runtime settings"; long_context "KV Cache Precision"; `properties.hpp` (`kv_cache_precision`; also `KEY_CACHE_PRECISION`, `VALUE_CACHE_PRECISION`, `KEY_CACHE_GROUP_SIZE`) |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE` | LLM `plugin_config` | The docs do not say for CPU; 0 disables | OpenVINO: dynamic quantization gains from int8 compute; smaller groups give better accuracy and worse performance | Same | `docs/llm/reference.md` "OpenVINO runtime settings" (example value 32); `properties.hpp` (`dynamic_quantization_group_size`); "Precision Control" (off in ACCURACY mode) |
| `cache_interval_multiplier` | graph; pull `--cache_interval_multiplier` | Adaptive (perf guide); long_context also says "default value 8" | Linear-attention models only. Lower values improve prefix-caching efficiency, cost memory; 8 is the minimum, "optimize for performance" with short inputs; 64 for medium context. Range 8–256 | Same | `docs/performance_tuning.md`; long_context README; `llm_calculator.proto` field 25 |
| `cache_eviction_config` | graph | off | The docs do not say | Can help the latency of long generations at the cost of accuracy | `docs/llm/reference.md` "Caching settings" |
| `sparse_attention_config` | graph | off | The docs do not say | The docs do not say | `docs/llm/reference.md` "Scheduling settings" |
| `prompt_lookup` | LLM `plugin_config` `{"prompt_lookup": true}`; export `--prompt_lookup_decoding` | off | The docs do not say | The long_context demo credits it for better second-token latency; with it, turn off `dynamic_split_fuse` and set `max_num_batched_tokens` above the context length | `docs/llm/reference.md` "Additional settings in plugin_config"; long_context "Deployment" |
| Weight format | `export_model.py --weight-format` | `int8` | Lower precision via quantization improves throughput and memory | The docs do not say separately | `docs/performance_tuning.md`; `export_model.py` |
| NUMA binding / instances | Docker `--cpuset-cpus` per NUMA node, one OVMS per node behind a load balancer | One process | Text generation on Xeon is most efficient bound to one NUMA node; scaling demo multiplies throughput with 6 instances on 6 nodes. Benchmark with and without virtual NUMA | The docs do not say | `docs/performance_tuning.md`; `demos/continuous_batching/scaling/README.md` "Scaling on dual CPU sockets" |
| `MODEL_DISTRIBUTION_POLICY` | LLM `plugin_config`; pull `--model_distribution_policy` | empty | `TENSOR_PARALLEL` spreads one tensor over sockets processed in parallel; the docs do not tie it to throughput or latency | The docs do not say | `docs/parameters.md` "Text generation"; `properties.hpp` (`model_distribution_policy`) |
| `NUM_STREAMS`, `PERFORMANCE_HINT` for the LLM | LLM `plugin_config` | Not set by the export script | The docs do not say for continuous batching. The tokenizer and detokenizer always run under THROUGHPUT with `min(rest_workers, cores)` streams (source) | The docs do not say | `export_model.py`; `…/continuous_batching/servable_initializer.cpp` |
| `best_of_limit`, `max_tokens_limit` | graph | 20; unset | The docs tie them to memory over-consumption by invalid requests, not speed | — | `docs/llm/reference.md` "OpenVINO runtime settings" |

Note: no metrics for text generation are exposed at `/metrics` beyond `ovms_current_graphs`
and request/response counts; the KV cache use and scheduled requests appear in the server log
(`docs/llm/reference.md` "Limitations", "Caching settings").

### 4. Trade-offs and conditions the docs state

- Streams versus clients and cores: streams fewer than the average requests in flight (else the
  server is under-used), never more than the cores; size to the average parallel slack, not
  the peak (`docs/performance_tuning.md`; OpenVINO "Number of Streams Considerations").
- Streams cost: each stream raises single-request latency, duplicates intermediate buffers
  (memory) and lengthens compile time; weights are shared, so prefer streams over several copies
  of a model (OpenVINO "Advanced Throughput Options"). A collaborator in issue #3136 says the
  same: one model with more streams, not duplicated models; about 2 streams for 4 clients.
- Hint versus `NUM_STREAMS`: OVMS says not to use them together (`docs/performance_tuning.md`);
  OpenVINO allows overriding a hint with low-level values at some loss of portability
  ("Combining the Hints and Individual Low-Level Settings"). OpenVINO recommends hints over
  low-level settings and benchmarking with `-hint none` before hand-tuning.
- `nireq` is a queue, not parallelism; ≥ `NUM_STREAMS` (perf guide; issue #3136). Embeddings and
  rerank graphs ignore it (source).
- Dynamic shapes on CPU: use streams, not batching (OpenVINO "Advanced Throughput Options").
- Containers: whole cores (`--cpuset-cpus`) rather than CPU time (`--cpus`); OVMS adapts
  threads, streams and pinning to what it detects, but per the source only inside Docker or
  Kubernetes on Linux. A CPU quota turns pinning off. Collaborators in issue #2721 traced slow
  inference to a Kubernetes CPU limit; a later comment there by a CONTRIBUTOR (not clearly a
  maintainer) adds that the model ran with one infer request, so requests also queued.
- Pinning: off helps when several workloads share the machine (both docs).
- Multi-socket: with LATENCY, OpenVINO uses one socket; running only one stream on several
  sockets can raise cross-NUMA latency (CPU device page "Multi-stream Execution"). On multi-socket
  CPUs, one request per NUMA node can be served at single-request latency ("Optimizing for
  Latency"). For LLMs, bind to one NUMA node and scale out per node.
- KV cache: dynamic avoids preemption but may consume all RAM; static bounds memory but
  preempts when full; prefix caching benefits grow with `cache_size`.
- `max_num_batched_tokens`: larger → better TTFT at low concurrency, worse interactivity under
  high concurrency.
- Precision: bf16 on CPU can lose LLM accuracy; forcing f32 on AMX machines can cut batch
  throughput by more than 60% ("Precision Control"). `DYNAMIC_QUANTIZATION_GROUP_SIZE`: smaller
  groups, more accuracy, less speed.
- Measuring: the guide's method is OpenVINO `benchmark_app` first (`-dump_config`,
  `-inference_only=false`), then the OVMS benchmark client locally and remotely, then network
  bandwidth (`docs/performance_tuning.md` "Analyzing performance issues"). OpenVINO's own way
  to raise throughput: fix a tail-latency bound, then add streams until it is met or throughput
  stops growing.

### Pages read

OVMS at tag `v2026.4.1` (raw at `https://raw.githubusercontent.com/openvinotoolkit/model_server/v2026.4.1/<path>`):

- `docs/performance_tuning.md`, `docs/parameters.md`, `docs/llm/reference.md`,
  `docs/model_server_rest_api_rerank.md`, `docs/model_server_rest_api_embeddings.md`,
  `docs/mediapipe.md`, `docs/metrics.md`, `docs/model_cache.md`; `docs/deploying_server_docker.md`
  and `docs/deploying_server_kubernetes.md` searched for CPU and speed terms (none found)
- `demos/rerank/README.md`, `demos/embeddings/README.md`, `demos/continuous_batching/README.md`,
  `demos/continuous_batching/scaling/README.md`, `demos/continuous_batching/long_context/README.md`
- `demos/common/export_models/export_model.py`
- `src/embeddings/embeddings_calculator_ov.proto`, `src/rerank/rerank_calculator_ov.proto`,
  `src/llm/llm_calculator.proto`, `src/embeddings/embeddings_calculator_ov.cc`,
  `src/rerank/rerank_calculator_ov.cc`, `src/sidepacket_servable.cpp`, `src/ov_utils.cpp`,
  `src/systeminfo.cpp`, `src/config.cpp`, `src/cli_parser.cpp`, `src/modelinstance.cpp`,
  `src/llm/language_model/continuous_batching/servable_initializer.cpp`
- For the comparison: the same docs, demos, export script, rerank proto and `src/ov_utils.cpp`
  at tag `v2026.3.1`.

OpenVINO:

- https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-throughput.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-throughput/advanced_throughput_options.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-memory-usage.html (no speed content beyond compile memory)
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html
- https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html
- `src/inference/include/openvino/runtime/properties.hpp` of `github.com/openvinotoolkit/openvino` at tag `2026.4.1`
- Not reachable: `https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__cpp__prop__api.html` (404); the Python `openvino.properties.hint` page rendered without its tables.

GitHub issues of `openvinotoolkit/model_server`, comments by users GitHub marks COLLABORATOR
(maintainer) or CONTRIBUTOR, as of 19 November 2026: #2721, #3136, #3248, #3310. A search for
newer maintainer answers on embeddings streams, `max_num_batched_tokens` and `rest_workers`
found none.

## vLLM 0.32.0 on CPU

Tag checked: `v0.32.0` exists (`gh api repos/vllm-project/vllm/git/ref/tags/v0.32.0`, commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`). Every source below is a file read at that tag,
written as a path; the URL prefix is `https://github.com/vllm-project/vllm/blob/v0.32.0/`.
Short names used in the Source column:

- **cpu.md**: `docs/getting_started/installation/cpu.md`
- **cpu.x86**: `docs/getting_started/installation/cpu.x86.inc.md`
- **opt.md**: `docs/configuration/optimization.md`
- **levels.md**: `docs/design/optimization_levels.md`
- **mem.md**: `docs/configuration/conserving_memory.md`
- **tc.md**: `docs/design/torch_compile.md`
- **apc.md**: `docs/features/automatic_prefix_caching.md`
- **platform**: `vllm/platforms/cpu.py`, `CpuPlatform.check_and_update_config`
- **worker**: `vllm/v1/worker/cpu_worker.py`
- **args**: `vllm/engine/arg_utils.py`
- **envs**: `vllm/envs.py`

"The docs do not say" means that the docs and the docstrings at the tag give no throughput or latency effect for that setting. Rows marked **GPU only** do nothing on the CPU backend, or the docs tie them to GPUs.

### 1. Tables

#### CPU environment variables

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | env (GiB) | Unset. The docs say `0`, meaning no override. In envs.py the variable is `None` when unset, and the comment there still says "4 GB" | A larger value lets more requests run in parallel and allows longer contexts | The docs do not say | cpu.md "Related runtime environment variables" and "How to decide `VLLM_CPU_KVCACHE_SPACE`"; envs (lines 885-891); platform (it sets `kv_cache_memory_bytes = value × GiB` whenever the variable is set and not empty) |
| `VLLM_CPU_OMP_THREADS_BIND` | env | `auto` | The docs say to check that binding takes effect first (with `htop`) before any other tuning. `auto` puts each OpenMP thread on its own physical core, keeps each rank on one NUMA node, and leaves the reserved CPUs to other vLLM parts | The docs do not say | cpu.md "How to decide `VLLM_CPU_OMP_THREADS_BIND`" and "How to do performance tuning"; envs |
| ↳ with hyper-threading | same | n/a | Bind one logical CPU per physical core: the docs' example on 16 logical / 8 physical cores is `0-7` or `8-15` | The docs do not say | cpu.md, same section |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | env; works only when binding is `auto` | Unset. On x86, `local_world_size` CPUs are reserved (one with TP=1), plus one per rank when KV transfer is on | The docs recommend reserving 1 to 2 cores for the serving framework to avoid oversubscription, and 1 CPU for the front-end when `world_size == 1` | The docs do not say | cpu.md "Related runtime environment variables" and "How to launch a vLLM service on CPU?" |
| `CPU_VISIBLE_MEMORY_NODES` | env; works only when binding is `auto` | All nodes | Masks NUMA nodes and changes their binding order. The docs give no throughput effect | The docs do not say | cpu.md "Related runtime environment variables"; worker (binds memory to the node of the first allowed core) |
| `VLLM_CPU_ATTN_SPLIT_KV` | env | `1` | Turns on split-KV in CPU attention. The docs give no effect | The docs do not say | envs ("whether to enable attention spilt KV"); `vllm/v1/attention/backends/cpu_attn.py` (`enable_kv_split=`) |
| `VLLM_CPU_INT4_W4A8` | env | `1` | Uses SGLang INT4 W4A8 kernels for AWQ. The docs give no effect | The docs do not say | envs (lines 913-915) |
| `VLLM_ZENTORCH_WEIGHT_PREPACK` | env (**AMD Zen only**, does nothing on Intel) | `1` | Prepacks weights at load, which removes a layout conversion on every inference | The docs do not say | cpu.md "AMD Zen optimizations" |
| `VLLM_CPU_SGL_KERNEL` | env | **Not defined in envs.py at v0.32.0** | The CPU benchmark configs still set it to `1`, but no code at this tag reads it | n/a | `.buildkite/performance-benchmarks/tests/serving-tests-cpu*.json`; envs (no match) |
| `LD_PRELOAD` (tcmalloc and OpenMP) | env, set by vLLM itself | The platform adds PyTorch's `libgomp` and the bundled `libtcmalloc`, unless `LD_PRELOAD` already names them | The source comment says tcmalloc reduces memory-allocation overhead and that without the preloaded `libgomp` only one core is used under thread binding (issue #27369). The wheel docs say to preload TCMalloc and Intel OpenMP | The docs do not say | platform (lines 442-510); cpu.x86 "set `LD_PRELOAD`" warnings |
| `VLLM_USE_FASTOKENS` | env | `0` (needs the `fastokens` package, which the image may not ship) | Faster tokenizing and detokenizing. The largest gains are on tokenizer-bound loads (long shared prefixes, bursts of short prompts). There is no gain when model execution is the bottleneck | Same | opt.md "fastokens Backend" |

#### Scheduler and batching

| Setting | Where set | Default (CPU, `vllm serve`) | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `--max-num-seqs` | CLI / engine arg | `128 × world_size` (256 × world_size offline) | A larger batch usually gives higher throughput. The docs say this setting mostly affects output-token performance | A smaller batch gives lower latency. The docs say to tune from the default to balance the two | cpu.md "How to do performance tuning for vLLM CPU?"; args (`if current_platform.is_cpu()` block, about line 2974) |
| `--max-num-batched-tokens` | CLI / engine arg | `2048 × world_size` (4096 × world_size offline) | Larger values: opt.md recommends more than 8192 for best throughput, but ties that advice to "smaller models on large GPUs" | cpu.md says this setting mostly affects first-token performance. opt.md: smaller values (for example 2048) give better ITL, larger values give better TTFT | cpu.md, same section; opt.md "Performance Tuning with Chunked Prefill"; args |
| `--enable-chunked-prefill` | CLI / engine arg | On when the model supports it, on CPU too (only RISC-V and MLA models turn it off) | opt.md says it helps both throughput and latency: decodes go first, then prefills fill the remaining budget, and long prefills are split | Better ITL, because decodes are prioritised | opt.md "Chunked Prefill"; args `_set_default_chunked_prefill_and_prefix_caching_args`; platform (MLA) |
| ↳ when chunked prefill is off | — | — | `max_num_batched_tokens` must be greater than `max_model_len`, otherwise start-up may crash | — | opt.md "Chunked Prefill" warning |
| `--performance-mode` | CLI (`balanced` / `interactivity` / `throughput`) | `balanced` | `throughput` doubles `max_num_batched_tokens` and `max_num_seqs` when you did not set them yourself. Per the source this happens on CPU too | `interactivity` only changes CUDA-graph capture sizes: **GPU only**, because the CPU platform empties `cudagraph_capture_sizes` | `vllm/config/vllm.py` (`performance_mode` docstring, about line 433, and line 2467); args (line 3130); platform |
| `--stream-interval` | CLI / engine arg | `1` | A larger value (for example 10) cuts host overhead and "may increase throughput" | `1` makes streaming smoother, because each token is sent at once | `vllm/config/scheduler.py` (`stream_interval` docstring) |
| `--async-scheduling` | CLI / engine arg | **Forced off on CPU** | Not available on CPU: the platform sets it to `False` ("not required on CPU") | — | platform (line 331); `vllm/config/scheduler.py` |
| `--max-num-queued-reqs` / `--max-num-queued-tokens` | CLI / engine arg | `None` (no limit) | Not a throughput setting | Rejects requests with HTTP 503 beyond the limit. The queued-tokens limit is described as a TTFT QoS mechanism (target TTFT × prefill throughput) | `vllm/config/scheduler.py` docstrings |
| `-dp` / `--data-parallel-size`, `-tp`, `-pp` | CLI | 1 | DP scales throughput rather than model size. On CPU the docs recommend TP equal to the number of NUMA nodes (TP=6 is not supported), and DP, TP and PP together when there are enough sockets. Each NUMA node becomes one TP/PP rank | opt.md: more TP can add synchronisation overhead, more PP can add latency | cpu.md "How to find benchmark configuration examples…" and "How to do performance tuning…"; opt.md "Parallelism Strategies" and "Preemption" |
| `--api-server-count` | CLI | `1` (equal to DP) | Helps when input processing in the API server is the bottleneck and the machine has spare CPU (online serving only) | The docs do not say | opt.md "Parallel Processing" |
| `--numa-bind`, `--numa-bind-nodes`, `--numa-bind-cpus` | CLI | off | **GPU only**: the docs say these do not configure CPU worker affinity. Use the `VLLM_CPU_*` variables instead | — | opt.md "NUMA Binding…" and "CPU Backend Thread Affinity" |
| Recipe-based tuning (`--hardware xeon6`) | image entrypoint `tools/recipes/serve_with_recipe.sh` | off | Recipes give validated settings per model for Xeon 6. Runtime tuning sets `max-num-seqs` to the expected concurrency and warns that a lower value queues requests and badly hurts TTFT | Same | cpu.x86 "Serve with vLLM Recipes"; `tools/recipes/RUNTIME_TUNING.md` "`max-num-seqs`", "`max-num-batched-tokens`" |

#### Memory and KV cache

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `--kv-cache-memory-bytes` | CLI / engine arg | `None` | Fixes the KV cache size and ignores the memory-utilisation fraction. cpu.md calls `VLLM_CPU_KVCACHE_SPACE` legacy and points here. opt.md: a cautious value caps concurrency and therefore throughput | The docs do not say | `vllm/config/cache.py` docstring; cpu.md "How to decide `VLLM_CPU_KVCACHE_SPACE`"; opt.md "Faster Startup" |
| `--gpu-memory-utilization` (alias `--device-memory-utilization`) | CLI / engine arg | `0.92` | On CPU it is the fraction of the **NUMA node's total memory** the worker reserves. When no KV size is given, the KV cache gets that fraction minus the process's RSS after warm-up. More KV space means more concurrency (opt.md "Preemption") | Preemption and recompute hurt end-to-end latency, so more KV space means less preemption | worker (lines 73-94 and `determine_available_memory`); `vllm/config/cache.py`; opt.md "Preemption" |
| `--max-model-len` | CLI / engine arg | From the model config | A shorter context reduces memory use | The docs do not say | mem.md "Context length and batch size" |
| `--block-size` | CLI / engine arg | `128` on CPU (16 for MLA models) | The docs say to use multiples of 32. They give no effect on throughput or latency | The docs do not say | cpu.md "How to do performance tuning…"; platform (lines 274-299); `cpu_attn.py` (`MultipleOf(32)`) |
| `--enable-prefix-caching` | CLI / engine arg | On for generative models | Higher throughput when requests share a prefix (a long document, a multi-turn chat). No gain on long answers or when prefixes are not shared | Lower latency, but only for prefill. Decode time does not change | apc.md "Example workloads" and "Limits" |
| `--kv-cache-dtype` | CLI / engine arg | `auto` (the model's dtype) | FP8 KV cache: more tokens fit in memory, so throughput improves and longer contexts fit. The CPU attention backend lists `fp8`, `fp8_e4m3` and `fp8_e5m2` as supported | The docs do not say | `docs/features/quantization/quantized_kvcache.md` intro; `cpu_attn.py` `supported_kv_cache_dtypes` |
| `--dtype` | CLI / engine arg | The model's default dtype | Use `bfloat16` if there is any performance or accuracy problem, because float16 support in torch on CPU is unstable | Same | cpu.md FAQ "Which `dtype` should be used?" |
| `--quantization` (weights) | from the checkpoint, or the CLI | n/a | On x86 CPU the docs list AWQ, GPTQ and compressed-tensors INT8 W8A8. Marlin, FP8 W8A8, bitsandbytes and GGUF are marked unsupported on x86 CPU. AutoRound (`inc.md`) says W4A16 and W8A16 are supported "on Intel platforms" without naming the CPU | The docs do not say | cpu.md "Which quantization configs does vLLM CPU support?"; `docs/features/quantization/README.md` table; `docs/features/quantization/inc.md` |
| Docker `--shm-size`, `--cap-add SYS_NICE`, `--security-opt seccomp=unconfined` | `docker run` | Docker defaults | Without them, NUMA memory binding and page migration may not take effect, and performance "can be suboptimal". The launch example uses `--shm-size=4g` | Same | cpu.md "Why do I see `get_mempolicy`…"; cpu.x86 "Launching the OpenAI server" |

#### Start-up and compilation

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `-O0` … `-O3` / `--optimization-level` | CLI | `-O2` | `-O2` is recommended for production. On CPU, any level above `-O0` turns into `DYNAMO_TRACE_ONCE` with the Inductor backend; CUDA graphs are dropped | `-O0`: no compilation, so the fastest start-up and the lowest performance. `-O1` is named for faster start-up, but on CPU it reaches the same Inductor path. The docs do not say for CPU what `-O1` saves | levels.md; opt.md "Optimization Levels"; platform (lines 353-377); `vllm/config/vllm.py` (lines 1752-1756) |
| `--enforce-eager` | CLI / engine arg | `False` | Steady-state decode is slower (opt.md, GPU wording) | Skips compilation, so it gives the fastest start-up. Equivalent to `-cc.mode=none -cc.cudagraph_mode=none`. On CPU, with eager mode **and** an explicit KV size (`--kv-cache-memory-bytes` or `VLLM_CPU_KVCACHE_SPACE`), the worker also skips its warm-up pass | opt.md "Faster Startup"; `vllm/config/model.py` `enforce_eager`; worker `_should_warm_up_model` |
| Compile cache (`VLLM_CACHE_ROOT`) | env | `~/.cache/vllm` | n/a | Shortens repeat starts: `torch_compile_cache` can be copied between machines or built into the image. Any change to the model, config, `VLLM_*` variables or torch build invalidates it. The docs do not say whether this applies to the CPU's `DYNAMO_TRACE_ONCE` path | opt.md "Faster Startup"; tc.md "Compilation Cache" |
| `VLLM_DISABLE_COMPILE_CACHE` | env | `0` | n/a | Setting it to `1` turns the cache off (for debugging) | tc.md; envs |
| `VLLM_USE_AOT_COMPILE` / `VLLM_FORCE_AOT_LOAD` | env | AOT is on when torch ≥ 2.10 and the cache is on; FORCE is `0` | n/a | AOT compiles during warm-up and reuses the result. FORCE fails at start-up instead of recompiling silently when the cache misses | envs (lines 377-386, 787-799); opt.md "Faster Startup" |
| `VLLM_ENABLE_STARTUP_PLAN` | env | `0` | n/a | Saves the memory-profiling result under `VLLM_CACHE_ROOT/startup_plan/` and skips profiling on later boots that match. The docs do not say whether it works on CPU | envs (lines 1988-1997) |
| `--kv-cache-memory-bytes` (as a start-up setting) | CLI | `None` | A cautious value caps throughput | Skips memory profiling (opt.md, GPU wording). On CPU it also enables the warm-up skip under eager mode (see `--enforce-eager`) | opt.md "Faster Startup"; worker |
| CUDA graphs (`-cc.cudagraph_mode`, capture sizes) | CLI | — | **GPU only**: the CPU platform empties `cudagraph_capture_sizes` | — | platform (line 353) |
| Inductor autotuning | compilation config | off | The docs recommend trying it for maximum performance on specific shapes. It takes seconds to minutes. Their example is Triton GEMMs on GPU; the docs do not say whether it applies to CPU | — | tc.md (around line 246) |

#### Pooling models (embed, score / rerank)

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| CPU support | — | — | The CPU attention backend supports decoder, encoder, encoder-only and encoder-decoder attention, so BERT-style embedders and cross-encoders can run. The CPU benchmark suite has an embedding config (`jinaai/jina-embeddings-v3`, bf16, `VLLM_CPU_KVCACHE_SPACE=40`, concurrency 32/64/128) | n/a | `cpu_attn.py` `supports_attn_type`; cpu.md "How to find benchmark configuration examples…" (`serving-tests-cpu-embed.json`) |
| General caveat | — | — | The docs say pooling support is mainly for convenience and is **not guaranteed** to be faster than HF Transformers or Sentence Transformers | Same | `docs/models/pooling_models/README.md` note at top |
| Chunked prefill and prefix caching | automatic | **Off** for encoder-only pooling models, and off for decoder pooling models with MEAN or CLS pooling. On for decoder models with LAST pooling | Enabling them by hand logs a warning that the engine may crash or give wrong outputs. With chunked prefill off, `max_num_batched_tokens` is raised to at least `max_model_len` when you did not set it | The docs do not say | `vllm/config/model.py` `is_chunked_prefill_supported` / `is_prefix_caching_supported`; args (lines 2994-3035, 3136-3142) |
| `--runner pooling`, `--convert embed|classify` | CLI | `auto` detection | Selects the pooling runner. The docs give no effect on speed | The docs do not say | pooling README "Model Runner" and "Model Conversion" |
| `--pooler-config` | CLI | From the model or Sentence Transformers metadata | Sets the pooling type and `use_activation` (normalisation). The docs give no effect on speed | The docs do not say | pooling README "Pooler Configuration" |
| Matryoshka `dimensions` | request, plus `--hf-overrides` | Off unless the model declares it | Trades performance against cost (output size). The docs do not say more | The docs do not say | `docs/models/pooling_models/embed.md` "Matryoshka Embeddings" |
| Rerank endpoints | served routes | `/score`, `/v1/score`, `/rerank`, `/v1/rerank`, `/v2/rerank` | A cross-encoder needs `num_labels == 1` to get these endpoints | n/a | `docs/models/pooling_models/scoring.md` "Summary" |
| Scheduler and CPU settings above | same | same | The docs give no separate tuning advice for pooling on CPU. The benchmark config uses the same CPU variables and only `dtype` | The docs do not say | `serving-tests-cpu-embed.json` |

### 2. Trade-offs and conditions the docs state

- **Batch size.** A larger batch usually raises throughput and a smaller one lowers latency. The docs say to tune `--max-num-seqs` (output tokens) and `--max-num-batched-tokens` (first token) starting from the CPU defaults (cpu.md, performance tuning).
- **Chunked prefill budget.** A small `max_num_batched_tokens` gives better ITL and a large one gives better TTFT. The advice to go above 8192 for throughput is for small models on large GPUs (opt.md).
- **KV cache size.** More KV memory gives more concurrency and longer contexts. Weights plus KV cache must fit in **one NUMA node**, otherwise the TP worker is killed with exit code 9 (cpu.md). A cautious fixed KV size caps throughput (opt.md).
- **The default memory reservation.** On CPU the worker reserves `0.92 ×` the node's total memory and gives the rest after its own RSS to the KV cache (worker; `vllm/config/cache.py`). Setting `--kv-cache-memory-bytes` or `--gpu-memory-utilization` is the documented way to bound that reservation. The docs do not explain the 20 GB measured on the 16 November sitting.
- **Preemption.** When KV space runs short, requests are recomputed, which hurts end-to-end latency. The fixes the docs list: more memory, fewer `max_num_seqs` or batched tokens, more TP or PP. More TP adds synchronisation; more PP adds latency (opt.md).
- **Threads.** Use one OpenMP thread per physical core and keep each rank on one NUMA node. Reserve 1 to 2 cores for the API server and engine; contention hurts throughput and latency because the engine core runs a busy loop (cpu.md; opt.md "CPU Resources for GPU Deployments", written for GPU deployments).
- **Start-up against speed.** `-O0` or `--enforce-eager` start fastest at the cost of steady-state performance, and `-O2` is the production default (levels.md, opt.md). On CPU, compilation is Inductor with `DYNAMO_TRACE_ONCE` (platform). A source comment says CPU compilation "consume[s] too much time, even with torch compile cache", which is why CI runs eager.
- **Prefix caching** helps only prefill and only when prefixes are shared. It never speeds up decode (apc.md).
- **Not on CPU:** async scheduling and Dual-Batch Overlap are forced off; CUDA graphs and `performance-mode=interactivity` have no effect; `--numa-bind*` is GPU only (platform; opt.md).
- **Docker:** NUMA syscalls need `SYS_NICE` and a seccomp exception, otherwise NUMA optimisations silently do not apply (cpu.md).
- **Doc and code disagree on `VLLM_CPU_KVCACHE_SPACE`.** The docs say the default `0` means no override. In the code, any non-empty value, `0` included, sets `kv_cache_memory_bytes`; leave the variable unset for automatic sizing (platform, line 324).

### 3. Pages read (raw files at tag v0.32.0)

- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/getting_started/installation/cpu.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/getting_started/installation/cpu.x86.inc.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/configuration/optimization.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/design/optimization_levels.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/configuration/conserving_memory.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/configuration/engine_args.md (generated from the config classes, so those were read directly)
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/configuration/env_vars.md (includes `vllm/envs.py`)
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/design/torch_compile.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/features/automatic_prefix_caching.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/features/quantization/README.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/features/quantization/inc.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/features/quantization/quantized_kvcache.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/models/hardware_supported_models/cpu.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/models/pooling_models/README.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/models/pooling_models/embed.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/models/pooling_models/scoring.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/tools/recipes/RUNTIME_TUNING.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/.buildkite/performance-benchmarks/README.md
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/.buildkite/performance-benchmarks/tests/serving-tests-cpu.json
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json
- Source: `vllm/envs.py`, `vllm/platforms/cpu.py`, `vllm/v1/worker/cpu_worker.py`, `vllm/v1/attention/backends/cpu_attn.py`, `vllm/engine/arg_utils.py`, `vllm/config/{cache,scheduler,model,vllm,compilation}.py`, `vllm/model_executor/layers/quantization/inc/inc.py`, all at the same raw prefix.

## FastAPI, Uvicorn and the in-process model

Versions read: fastapi 0.162.2, uvicorn 0.54.0, starlette 1.10.0, anyio 4.16.0,
sentence-transformers v6.2.0, torch v2.15.0 (docstrings) and v2.8.0 (threading
note, see below), pytorch/tutorials `main` (commit 9e73898, 2025-07-09), httpx
0.28.1 (the version installed in the repository's `.venv`; uvloop 0.23.0 and
httptools 0.9.0 are installed there too, h2 is not). Every source is a raw file
at that git tag; paths are given as `repo@tag:path` plus the section heading.

Tags used in the columns: **CPU** = the app that runs the embedder or the
cross-encoder in process; **I/O** = the workflow app that waits on the model
servers; **both** = applies to either.

### (1) Settings

#### FastAPI and Starlette (and AnyIO under them)

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `async def` vs `def` path operation (both) | the endpoint's declaration | n/a | A `def` endpoint runs in an external thread pool that is awaited, so it does not block the server. The docs advise `async def` for trivial compute-only endpoints unless they do blocking I/O, since in FastAPI the thread pool hop costs more than the ~100 ns gain a `def` brings elsewhere. They give no throughput figure. | the docs do not say beyond the above | `fastapi/fastapi@0.162.2:docs/en/docs/async.md`, "In a hurry?" and "Very Technical Details > Path operation functions" |
| `def` dependencies (both) | the dependency's declaration | n/a | A `def` dependency also runs in the thread pool; the docs do not say what that does to throughput. | the docs do not say | same file, "Dependencies", "Sub-dependencies" |
| CPU-bound work in one process (CPU) | deployment | one process | The docs class ML as "CPU bound" work: concurrency does not help it, parallelism (several processes) does. They point to Deployment for how. | the docs do not say | same file, "Is concurrency better than parallelism?", "Concurrency + Parallelism: Web + Machine Learning" |
| Thread pool size: AnyIO default `CapacityLimiter` (both; CPU when its endpoint is `def`) | in code, once the loop runs: `anyio.to_thread.current_default_thread_limiter().total_tokens = N` | **40** tokens | Raise it to run more threads at once; the 40 are shared with FastAPI's `def` dependencies, `FileResponse`, `UploadFile`, sync background tasks. The docs warn that more threads "may have a performance and memory impact". | the docs do not say | `encode/starlette@1.10.0:docs/threadpool.md`, "Concurrency Limitations"; `agronholm/anyio@4.16.0:docs/threads.rst`, "Adjusting the default maximum worker thread count"; default in `agronholm/anyio@4.16.0:src/anyio/_backends/_asyncio.py` line 3257 `CapacityLimiter(40)` |
| `run_in_threadpool(func, ...)` (both) | Starlette helper, called by FastAPI for `def` endpoints and dependencies | uses the default limiter above | It is `anyio.to_thread.run_sync` with no `limiter` argument, so it shares the 40 tokens. A call to `anyio.to_thread.run_sync(..., limiter=CapacityLimiter(n))` gets its own limit. The docs do not tie this to throughput or latency. | the docs do not say | `encode/starlette@1.10.0:starlette/concurrency.py` (`run_in_threadpool`); `fastapi/fastapi@0.162.2:fastapi/routing.py` lines 102, 139; `agronholm/anyio@4.16.0:src/anyio/to_thread.py` (`limiter` parameter) |
| AnyIO limiter vs asyncio's executor (both) | n/a | n/a | AnyIO's default limiter does not affect asyncio's default thread pool executor (`loop.run_in_executor`). | n/a | `agronholm/anyio@4.16.0:docs/threads.rst`, note under "Adjusting the default maximum worker thread count" |
| Thread pool size in FastAPI's own docs | n/a | n/a | FastAPI's docs (all 157 English pages at the tag) say the pool is "external" but give no size and no way to change it; the size is only in Starlette's and AnyIO's docs. | n/a | `fastapi/fastapi@0.162.2:docs/en/docs/` (searched) |

#### Uvicorn

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `workers` (CPU mainly) | `uvicorn.run(..., workers=N)`, `--workers`, `UVICORN_WORKERS`, or `$WEB_CONCURRENCY` | `$WEB_CONCURRENCY` if set, else 1 | FastAPI: several worker processes use multiple cores and serve "more requests". Uvicorn: a process manager for multiple processes; it uses `spawn`, not pre-fork, restarts dead or stuck children, and takes SIGHUP/SIGTTIN/SIGTTOU. | the docs do not say | `fastapi/fastapi@0.162.2:docs/en/docs/deployment/server-workers.md`, "Multiple Workers", "Recap"; `fastapi/fastapi@0.162.2:docs/en/docs/deployment/concepts.md`, "Multiple Processes - Workers"; `encode/uvicorn@0.54.0:docs/settings.md`, "Production"; `encode/uvicorn@0.54.0:docs/deployment/index.md`, "Using a process manager"; default in `uvicorn/config.py` lines 274, 358-359 |
| app as import string (both, needed for `workers>1`) | first argument of `uvicorn.run` | the project passes the app object | `uvicorn.run` logs "You must pass the application as an import string to enable 'reload' or 'workers'" and exits when `workers > 1` and the app is not a string. Because `Config` reads `$WEB_CONCURRENCY` when `workers` is not passed, the same check applies when that variable is above 1. The docs also ask for `uvicorn.run` under `if __name__ == '__main__'` with `workers`. | n/a | `encode/uvicorn@0.54.0:uvicorn/main.py` lines 614-618; `uvicorn/config.py` lines 358-359; `docs/settings.md`, "Configuration Methods" note |
| `limit_concurrency` (both) | `limit_concurrency=N`, `--limit-concurrency` | `None` | An application-level gate: past N open connections *or* in-flight tasks, a request gets an immediate 503; it is not queued. Its use is to keep memory use known "even under over-resourced loads". Idle keep-alive connections count against it; N admits at most N-1 other requests; 1 refuses everything. To queue excess load the docs say to put a reverse proxy in front or add workers. | the docs do not say (a refusal is immediate) | `encode/uvicorn@0.54.0:docs/settings.md`, "Resource Limits"; `docs/server-behavior.md`, "Resource Limits" and "Concurrency and backlog" |
| `backlog` (both) | `backlog=N`, `--backlog` | 2048 | The OS accept queue passed to `listen()`, separate from `limit_concurrency`. Uvicorn accepts eagerly, so it matters only when connections arrive faster than they are accepted, "for example ... while the event loop is blocked". When full, clients see a connect failure or timeout, not a 503. "Relevant for heavy incoming traffic." | the docs do not say | `encode/uvicorn@0.54.0:docs/settings.md`, "Resource Limits"; `docs/server-behavior.md`, "Concurrency and backlog" |
| `limit_max_requests` (+ `limit_max_requests_jitter`) (both) | `limit_max_requests=N`, `--limit-max-requests`; jitter `--limit-max-requests-jitter` | `None`; jitter 0 | Restarts a process after N requests, with a process manager, to stop memory leaks. Jitter staggers worker restarts so they do not all restart at once. Not tied to throughput or latency. | the docs do not say | `encode/uvicorn@0.54.0:docs/settings.md`, "Resource Limits"; `docs/server-behavior.md`, "Resource Limits" |
| `timeout_keep_alive` (both) | `timeout_keep_alive=S`, `--timeout-keep-alive` | 5 s | Closes a keep-alive connection that receives no new data in S seconds. Not tied to throughput or latency. | the docs do not say | `encode/uvicorn@0.54.0:docs/settings.md`, "Timeouts"; `docs/server-behavior.md`, "Timeouts"; default `uvicorn/config.py` line 234 |
| `loop` (I/O mainly) | `loop="uvloop"`, `--loop` | `"auto"`: uvloop if installed, else asyncio | uvloop "provides greater performance" (no figure); not on Windows or PyPy. FastAPI calls uvloop "the big concurrency performance boost" that comes with `uvicorn[standard]`. `zuvloop` (CPython 3.14+, separate install) and `rloop` (experimental, not for production) also exist. | the docs do not say | `encode/uvicorn@0.54.0:docs/settings.md`, "Implementation"; `docs/concepts/event-loop.md`; `fastapi/fastapi@0.162.2:docs/en/docs/deployment/manually.md`, tip under the `uvicorn[standard]` install |
| `http` (both) | `http="httptools"`, `--http` | `"auto"`: httptools if installed, else h11 | httptools "provides greater performance" than h11 (no figure); not on PyPy. `zttp` is experimental and not for production. | the docs do not say | `encode/uvicorn@0.54.0:docs/settings.md`, "Implementation"; `docs/concepts/http-protocols.md` |
| `http2` (server side) | `--http2` | False | Needs `--http zttp`, which the docs call experimental, not for production. | the docs do not say | `encode/uvicorn@0.54.0:docs/settings.md`, "Implementation"; `docs/concepts/http-protocols.md`, "zttp" |
| `access_log` | `access_log=False`, `--no-access-log` | True | Turns off the access log only. The docs do not say it changes throughput or latency. | the docs do not say | `encode/uvicorn@0.54.0:docs/settings.md`, "Logging" |
| Gunicorn with `UvicornWorker` | `gunicorn -k ...` | n/a | The docs list it for production; `uvicorn.workers` is deprecated in favour of the `uvicorn-worker` package; some options such as `--limit-concurrency` are "not yet supported" under Gunicorn. | the docs do not say | `encode/uvicorn@0.54.0:docs/deployment/index.md`, "Gunicorn" |
| Nginx in front (both) | deployment | none | Recommended for resilience: it buffers slow requests, which leaves the app servers "free from load". The docs recommend a UNIX socket (`--uds`) between Nginx and the process manager. | the docs do not say | `encode/uvicorn@0.54.0:docs/deployment/index.md`, "Running behind Nginx" (lines 142-146) |

#### Sentence Transformers (CPU)

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `backend` | `SentenceTransformer(..., backend=...)` / `CrossEncoder(..., backend=...)` | `"torch"` | Three backends with their own speed-ups: ONNX ("flexible and efficient model accelerator"), OpenVINO (fast CPU inference, "mainly for Intel Hardware"). In the benchmarks, OpenVINO INT8 was fastest on a local i7-13700K for embedders, but llama.cpp led on a cloud `cpu-upgrade` instance. The docs ask you to test on your deployment hardware. | The benchmarks measure throughput with tuned batches; "the best configuration for single-request latency may differ". | `UKPLab/sentence-transformers@v6.2.0:docs/sentence_transformer/usage/efficiency.rst`, intro, "OpenVINO", "Benchmarks" (CPU tab and the note after the llama.cpp charts) |
| Backend choice on CPU (decision chart) | as above | n/a | Embedders: openvino-qint8 if a small accuracy loss is acceptable, else openvino on an Intel CPU, else onnx. Cross-encoders: openvino-qint8 if minor quality loss is acceptable, else openvino on Intel, else onnx-O3. Both say "your mileage may vary". | the docs do not say | same file, "Recommendations"; `docs/cross_encoder/usage/efficiency.rst`, "Recommendations" |
| Cross-encoder backends (detail) | as above | n/a | Averaging hides patterns: "ONNX seems to perform stronger at low batch sizes"; ONNX and OpenVINO can even be slightly slower than PyTorch. | the docs do not say beyond "low batch sizes" | `UKPLab/sentence-transformers@v6.2.0:docs/cross_encoder/usage/efficiency.rst`, "Benchmarks" (paragraph after the figures) |
| `model_kwargs={"torch_dtype": "float16"/"bfloat16"}` | constructor | fp32 | Presented as a **GPU** speed-up. For cross-encoders on CPU, half precision "must be avoided altogether": torch-fp16 and torch-bf16 run at about 0.17x of fp32. | the docs do not say | `docs/sentence_transformer/usage/efficiency.rst`, "PyTorch" (float16, bfloat16 tabs); `docs/cross_encoder/usage/efficiency.rst`, "Benchmarks" |
| `model.compile(...)` (torch.compile) | after construction | off | The gain depends on model and hardware and grows with model size; small models can gain nothing or slow down slightly, as tokenization and Python overhead dominate. "Always measure". `mode="reduce-overhead"` (CUDA graphs) targets batch-size-1 inference on GPU; compilation is lazy, so warm up before serving. | `reduce-overhead` is named for batch-size-1 inference, on GPU only | `docs/sentence_transformer/usage/efficiency.rst`, "PyTorch", torch.compile tab |
| ONNX `model_kwargs`: `provider`, `file_name`, `export`, `session_options` | constructor, `backend="onnx"` | strongest provider; `model.onnx` | `session_options` takes an `onnxruntime.SessionOptions`, "for example, adjust `intra_op_num_threads`" to tune CPU inference. The benchmark set ONNX Runtime's intra-op threads explicitly, with inter-op threads set to 1. | the docs do not say | `docs/sentence_transformer/usage/efficiency.rst`, "ONNX" and "Benchmarks" |
| ONNX optimisation `export_optimized_onnx_model(..., optimization_config="O1".."O4")` | one-off export, then `model_kwargs={"file_name": ...}` | none | Speed-ups "on CPUs and GPUs alike"; O4 is fp16. | the docs do not say | same file, "Optimizing ONNX Models" |
| ONNX int8 `export_dynamic_quantized_onnx_model(..., "arm64"/"avx2"/"avx512"/"avx512_vnni")` | one-off export | none | "Faster inference on CPUs"; the four configs gave about the same speed-up on the author's CPU. No calibration set is needed. | the docs do not say | same file, "Quantizing ONNX Models" |
| OpenVINO `model_kwargs`: `file_name`, `export`, and `ov_config` | constructor, `backend="openvino"` | `openvino_model.xml`; `ov_config={}` | All `model_kwargs` go to Optimum Intel's `OVBaseModel.from_pretrained()`. The docs name `file_name` and `export` only. The source also takes `ov_config` as a dict or a path to a JSON file and passes it on (empty dict when absent); this is how OpenVINO's own CPU properties reach the model. The ST docs say nothing on what they do. | the docs do not say | `docs/sentence_transformer/usage/efficiency.rst`, "OpenVINO"; `UKPLab/sentence-transformers@v6.2.0:sentence_transformers/backend/load.py` lines 146-166 |
| OpenVINO int8 `export_static_quantized_openvino_model(...)` | one-off export with a calibration set (default `glue`/`sst2`) | none | Int8 "to speed up inference"; best in the local CPU benchmark (see `backend`). | the docs do not say | same file, "Quantizing OpenVINO Models" |
| `batch_size` in `encode` / `predict` / `rank` | per call | 32 | The benchmarks use each backend's best batch size; for cross-encoders each backend was measured at growing batch sizes until throughput fell or memory ran out. The docs give no rule for picking it. | The benchmarks are tuned for throughput and may differ for single-request latency (see `backend`). | `sentence_transformers/sentence_transformer/model.py` line 758 and docstring line 802; `sentence_transformers/cross_encoder/model.py` lines 525, 720; `docs/cross_encoder/usage/efficiency.rst`, "Benchmarks" (Batch sizes) |
| `start_multi_process_pool(target_devices=[...])` / `pool=` | in code | not used; when `target_devices` is None on a machine with no CUDA/NPU, 4 CPU processes | The docstring recommends it for predicting on multiple GPUs or CPUs, with one process per GPU. Not tied to a latency figure. | the docs do not say | `sentence_transformers/base/model.py` `start_multi_process_pool` (line 1807 on) |

#### PyTorch on CPU

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| `torch.set_num_threads(n)` (intra-op) | in code, before any eager, JIT or autograd work | number of CPU cores (one row covers inter- and intra-op in the table) | Parallel op implementations add overhead per request, which can lower overall throughput; avoid oversubscription, and with a large application thread pool consider `set_num_threads(1)`. | "In latency critical applications" raise intra-op threads to process each request as fast as possible. | `pytorch/pytorch@v2.8.0:docs/source/notes/cpu_threading_torchscript_inference.rst`, "Runtime API" and "Tuning the number of threads"; docstring `pytorch/pytorch@v2.15.0:torch/_torch_docs.py` line 10186 |
| `torch.set_num_interop_threads(n)` | in code, once, before any inter-op parallel work | number of CPU cores | Its pool serves `fork`-ed tasks (TorchScript/JIT); same oversubscription warning. | the docs do not say | same v2.8.0 note, "Runtime API"; docstring `torch/_torch_docs.py` line 10199 |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | environment, before start-up | not stated | The tutorial calls `OMP_NUM_THREADS` "the easiest switch" to speed up computation. `torch.set_num_threads` takes precedence over both variables, and `MKL_NUM_THREADS` over `OMP_NUM_THREADS`. With OpenMP, two application or inter-op threads may use different OpenMP pools, so a multi-threaded app can end up with very many threads. | the docs do not say | `pytorch/tutorials@main:recipes_source/recipes/tuning_guide.py`, "Utilize OpenMP"; v2.8.0 note, "Runtime API" and warning |
| Core pinning: `OMP_PROC_BIND=CLOSE`, `OMP_SCHEDULE=STATIC`, `GOMP_CPU_AFFINITY="N-M"` (GNU OpenMP); `KMP_AFFINITY=granularity=fine,compact,1,0`, `KMP_BLOCKTIME=1` (Intel `libiomp`, via `LD_PRELOAD`) | environment | GNU `libgomp` | Affinity changes communication, cache-line invalidation and page-thrashing overhead, and pinning keeps threads from moving between CPUs. `libiomp` "sometimes" beats `libgomp` on Intel; `KMP_BLOCKTIME` 0 or 1 is good "in most cases". | the docs do not say | `tuning_guide.py`, "Utilize OpenMP", "Intel OpenMP Runtime Library (libiomp)" |
| NUMA: `numactl --cpunodebind=N --membind=N` | launch command | none | Avoid crossing NUMA nodes; "inference can be run with multiple instances, each instance runs on one socket, to raise throughput". | the docs do not say | `tuning_guide.py`, "Utilize Non-Uniform Memory Access (NUMA) Controls" |
| Memory allocator: jemalloc or TCMalloc via `LD_PRELOAD` | environment | system `malloc` | "Can get better performance" for deep learning workloads by reusing memory. | the docs do not say | `tuning_guide.py`, "Switch Memory allocator" |
| `torch.no_grad()` for inference | in code | grad on | Speeds up execution and lowers memory. | the docs do not say | `tuning_guide.py`, "Disable gradient calculation for validation or inference" |
| `torch.compile` (operator fusion) | in code | off | Fuses pointwise ops into fewer kernels. | the docs do not say | `tuning_guide.py`, "Fuse operations" |

Note on the PyTorch threading source: the page "CPU threading and TorchScript
inference" held the thread table and the throughput-latency advice up to
v2.8.0. From v2.9.0 on (and at v2.15.0, as `.md`) it is a stub with only a
TorchScript deprecation warning. torch 2.15.0 keeps only the one-line
docstrings, and the docstrings give no default.

#### httpx client (I/O, the workflow app)

| Setting | Where set | Default | For throughput | For latency | Source |
|---|---|---|---|---|---|
| One shared `Client`/`AsyncClient` | app code | n/a | Pooling reuses TCP connections, giving lower CPU use, fewer round-trips and less network congestion. Do not create clients in a "hot loop"; keep one scoped or global client. | "Reduced latency across requests (no handshaking)". | `encode/httpx@0.28.1:docs/advanced/clients.md`, "Why use a Client?"; `docs/async.md`, warning on connection pooling |
| `limits=httpx.Limits(max_connections=...)` | client constructor | 100 | The maximum number of connections, or `None` for no limit. The docs do not tie it to throughput. When the pool is full, a request waits up to the `pool` timeout and then raises `PoolTimeout`. | the docs do not say | `encode/httpx@0.28.1:docs/advanced/resource-limits.md`; `docs/advanced/timeouts.md`, "Fine tuning the configuration" (pool); `httpx/_config.py` line 247 |
| `Limits(max_keepalive_connections=...)` | client constructor | 20 | The number of keep-alive connections kept, or `None` for always. | the docs do not say | same files |
| `Limits(keepalive_expiry=...)` | client constructor | 5.0 s | Time limit on idle keep-alive connections, or `None`. | the docs do not say | `docs/advanced/resource-limits.md`; `httpx/_config.py` line 178 |
| `timeout=httpx.Timeout(t, connect=, read=, write=, pool=)` | client or request | 5 s for each phase | Not tied to throughput; `pool` is how long to wait for a free connection, linked to `limits`. | the docs do not say | `encode/httpx@0.28.1:docs/advanced/timeouts.md`; `httpx/_config.py` line 246 |
| `http2=True` (needs `httpx[http2]`, i.e. `h2`, not installed) | client constructor | off | Off by default because HTTP/1.1 is the "more robust option". HTTP/2 multiplexes many concurrent requests on one TCP connection and compresses headers, with "potential performance benefits". Consider it for "highly concurrent requests", mostly with async clients. The server must support HTTP/2, or the client falls back to 1.1. | the docs do not say | `encode/httpx@0.28.1:docs/http2.md`, intro and "Enabling HTTP/2" |

### (2) Trade-offs and conditions the docs state

- **Workers and memory (CPU).** Processes do not share memory, so each worker
  loads its own copy of the model: a 1 GB model with 4 workers takes 4 GB.
  Loading more than the machine holds "will cause problems". Source:
  `fastapi@0.162.2:docs/en/docs/deployment/concepts.md`, "Memory per Process"
  and "Server Memory"; `docker.md`, "Memory" (with ML models, check memory use
  and adjust containers per machine).
- **Workers vs replicas.** On Kubernetes or a similar cluster, replicate at the
  cluster level: run one Uvicorn process per container and many containers.
  Workers inside a container add "unnecessary complexity". Multi-worker
  containers suit a simple app on a single server, or Docker Compose on one
  server. Source: `fastapi@0.162.2:docs/en/docs/deployment/docker.md`,
  "Replication - Number of Processes", "One Process per Container",
  "Containers with Multiple Processes and Special Cases";
  `server-workers.md`, note at the top.
- **Workers need an import string.** With today's `uvicorn.run(app, ...)`,
  `workers > 1` (or `$WEB_CONCURRENCY > 1`) makes Uvicorn exit at start-up.
  Source: `uvicorn@0.54.0:uvicorn/main.py` lines 614-618 and
  `uvicorn/config.py` lines 358-359.
- **`limit_concurrency` sheds load instead of queuing it.** Excess requests get
  an immediate 503, and idle keep-alive connections count toward the limit. To
  queue load, put a reverse proxy in front or add workers. A blocked event loop
  is when the `backlog` fills. Source: `uvicorn@0.54.0:docs/server-behavior.md`,
  "Concurrency and backlog".
- **Thread pool tokens are shared.** All of these draw on the same 40 tokens:
  `def` endpoints, `def` dependencies, file responses and uploads, and sync
  background tasks. Raising the count "may have a performance and memory
  impact". Source: `starlette@1.10.0:docs/threadpool.md`.
- **Threads vs throughput vs latency (PyTorch).** More intra-op threads per
  request lower latency, but parallel-op overhead can lower total throughput.
  Avoid oversubscription: with a big application thread pool, consider one
  intra-op thread. Under OpenMP, different application threads may each get
  their own OpenMP pool. Source: `pytorch@v2.8.0` threading note, "Tuning the
  number of threads" and its warning. (This bears on the CPU app when 40
  Starlette threads can each call the model.) The docs do not say how Starlette
  threads and torch threads interact; that remark is the only link they give.
- **Several instances, one socket each, raise throughput** on multi-socket
  machines. Source: `tutorials:recipes_source/recipes/tuning_guide.py`, NUMA.
- **Backend results depend on hardware.** CPU rankings differed between the
  author's desktop and a cloud instance. Benchmarks are for tuned-batch
  throughput, not single-request latency. Test speed and task quality on
  representative inputs. Source: `sentence-transformers@v6.2.0` efficiency
  pages, "Benchmarks" and the "Recommendations" notes.
- **Quantization costs a little accuracy.** int8 is offered "if a small
  accuracy loss is acceptable" (embedders) or if "minor performance
  degradations" are acceptable (cross-encoders). Half precision on CPU slows
  cross-encoders to about 0.17x. Source: the same two pages.
- **Export once and save.** Exporting to ONNX or OpenVINO happens at load time
  when the repository has no exported file; the docs "heavily recommend"
  saving the export so it is not redone on every run. Source: the same pages,
  the "ONNX" and "OpenVINO" tips.
- **HTTP/2 on the client** is opt-in and less battle-tested than HTTP/1.1. It
  only applies if the server speaks HTTP/2; Uvicorn's own HTTP/2 needs the
  experimental `zttp`. Source: `httpx@0.28.1:docs/http2.md`;
  `uvicorn@0.54.0:docs/settings.md`, "Implementation".

### (3) Pages read

Raw files fetched at their tags (`https://raw.githubusercontent.com/<repo>/<tag>/<path>`):

- fastapi/fastapi@0.162.2: `docs/en/docs/async.md`, `docs/en/docs/deployment/concepts.md`, `docs/en/docs/deployment/server-workers.md`, `docs/en/docs/deployment/docker.md`, `docs/en/docs/deployment/manually.md`, `fastapi/routing.py`; all 157 `docs/en/docs/**/*.md` searched for thread-pool and server settings
- encode/uvicorn@0.54.0: `docs/settings.md`, `docs/server-behavior.md`, `docs/concepts/event-loop.md`, `docs/concepts/http-protocols.md`, `docs/deployment/index.md`, `uvicorn/config.py`, `uvicorn/main.py`
- encode/starlette@1.10.0: `docs/threadpool.md`, `starlette/concurrency.py`
- agronholm/anyio@4.16.0: `docs/threads.rst`, `src/anyio/to_thread.py`, `src/anyio/_backends/_asyncio.py`
- UKPLab/sentence-transformers@v6.2.0: `docs/sentence_transformer/usage/efficiency.rst`, `docs/cross_encoder/usage/efficiency.rst`, `sentence_transformers/backend/load.py`, `sentence_transformers/base/model.py`, `sentence_transformers/sentence_transformer/model.py`, `sentence_transformers/cross_encoder/model.py`
- pytorch/tutorials@main (last commit on the file 9e73898, 2025-07-09): `recipes_source/recipes/tuning_guide.py` (rendered at https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- pytorch/pytorch@v2.8.0: `docs/source/notes/cpu_threading_torchscript_inference.rst` (the last full version); @v2.9.0 and @v2.15.0: the same page as a stub; @v2.15.0: `torch/_torch_docs.py`
- encode/httpx@0.28.1: `docs/advanced/resource-limits.md`, `docs/advanced/timeouts.md`, `docs/advanced/clients.md`, `docs/http2.md`, `docs/async.md`, `httpx/_config.py`
