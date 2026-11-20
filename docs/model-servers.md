# The model servers: OpenVINO Model Server, the FastAPI model server and vLLM

A model server serves the reranker or the embedder to the workflow app, OVMS and
vLLM an LLM too, and the FastAPI model server a component of a registered type on
its call route (`POST /call/<component type>`). There are three model-server kinds.
Two serve on the /v3 routes: OpenVINO Model Server (OVMS), a product, and
`qp-model-server`, this project's own server built on FastAPI. Both run the same
engine, OpenVINO, on the same model. This page says what the documentation of OVMS
and of FastAPI says about serving, how each is set up here, what the two sittings
of 15 November 2026 measured of the reranker, and what one sitting of 16 November
2026 measured of text generation. The third kind, vLLM, came after the sittings on
the reranker. It serves the same API formats on its own /v1 routes. Its section
below says what its documentation says and how it is set up here; nothing has been
measured on it here but the first look at text generation. The settings are named
as they are today; the sittings ran them under the names of that day.

Read for this page: OVMS's documentation at 2026.3.1, the version pinned in
`query_processing/ovms/versions.env` when the page was written (the pin is 2026.4.1
now; `docs/server-settings-for-throughput-and-latency.md` read it again at that tag
on 19 November 2026, and its two corrections are made below), FastAPI's pages on
concurrency and deployment as they stood on 15 November 2026, and vLLM's
documentation and its release 0.32.0 on 16 and 17 November 2026. The pages are
listed at the end.

## What their documentation says

They are different kinds of thing. OVMS is a finished inference server. FastAPI is a
web framework: it knows nothing about models, and the serving is code written on top
of it.

| | OVMS | FastAPI |
|---|---|---|
| What it is | A C++ inference server that exposes models over network APIs | A Python framework for building web APIs |
| APIs | Built in: OpenAI-compatible for generation and embeddings, Cohere-compatible rerank, KServe for classic models; REST and gRPC | The routes the application defines; request validation and an OpenAPI description come with it |
| Several requests at once, in one process | Execution streams split the CPUs so that several requests run in parallel; a queue of inference requests (`nireq`) sits in front | One process serves many clients concurrently, which helps work that waits on I/O; for CPU-bound work such as machine learning the documentation points to parallelism with several processes |
| Using more cores | More streams in the same process; it adjusts threads, streams and pinning to the CPUs of its container (when it detects Docker or Kubernetes: see row 4 below) | More worker processes, or on Kubernetes one process per container and more containers; each process loads its own copy of the model |
| Batching | Continuous batching for text generation; batch size and dynamic shapes for classic models | None |
| Model management | Model versions, and configuration changes applied at run time | None: a change is a new deployment |
| Metrics | A Prometheus endpoint: requests, inference time, time waiting in the queue, streams | None |
| Tuning | `NUM_STREAMS`, a performance hint, the thread count, CPU pinning, `nireq` | The number of workers; the rest is the application's code |

OVMS's guide says when streams pay: they are "particularly efficient for models which
cannot effectively consume all CPU cores", or for CPUs with many cores. It also says
that `NUM_STREAMS` and the performance hint must not be set together, and that the
stream count should not exceed the cores.

## How each is set up here

| | OVMS | FastAPI model server |
|---|---|---|
| Weights | Its own export, at `QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT` | The in-process export, at the same setting |
| Requests at once | Its streams (`QP_OVMS_NUM_STREAMS`) | One under `openvino`: a loaded model holds one inference request. `QP_LOCAL_MODELS__WORKERS` under `torch` |
| Calibration | Streams and threads, for `QP_CALIBRATION__USERS` requests in flight | Threads and batch size, on one request |
| Batch size of a rerank request | The whole request, 20 vector-store records: its rerank endpoint has no such setting | The reranker's `local.batch_size` (`QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE`), calibrated |
| More load | More streams, or more replicas | More replicas |
| Stream candidates | 1, 2, 4, ... up to the cores, the CPUs divided by `QP_LOCAL_MODELS__THREADS_PER_CORE` (2); none under a performance hint | None (see "Requests at once") |
| OpenVINO's CPU properties | `QP_OVMS_OPENVINO__*`, written into the graph's `plugin_config`: `ENABLE_HYPER_THREADING` by default, a performance hint in place of `NUM_STREAMS` when set | The same block, `QP_LOCAL_MODELS__OPENVINO__*`, at load: `ENABLE_HYPER_THREADING` by default |
| Graph options | `QP_OVMS_GRAPH_QUEUE_MAX_SIZE`, an embedder's `QP_OVMS_EMBEDDINGS_TRUNCATE` and `QP_OVMS_EMBEDDINGS_MAX_LENGTH`; as exported by default | None |
| Server | OVMS's flags, each left out at OVMS's own value: `QP_OVMS_REST_WORKERS`, `QP_OVMS_METRICS`, `QP_OVMS_LOG_LEVEL`, `QP_OVMS_CACHE_DIR`, and `QP_OVMS_ARGS` for any other | Uvicorn at its own values: `QP_UVICORN_WORKERS` (1), `QP_UVICORN_LIMIT_CONCURRENCY` (none), `QP_UVICORN_BACKLOG`, `QP_UVICORN_TIMEOUT_KEEP_ALIVE_S`; several workers divide the CPUs |
| Process | Not set here: OVMS is the App's process | `QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY`, `QP_PROCESS_OMP_PROC_BIND`, `QP_PROCESS_KMP_AFFINITY`, by one restart; none by default |
| An LLM | `--dynamic_split_fuse false --max_num_batched_tokens 8192` by default (`QP_OVMS_DYNAMIC_SPLIT_FUSE`, `QP_OVMS_MAX_BATCHED_TOKENS`), and the other options of pull mode, each left out at OVMS's own | Serves none |

Compare the two on the same weight format: see "Experiment 2" in the README.

## vLLM, the third kind

Nothing in the measurements below is about vLLM but the first look at text
generation. This section says what its documentation says and how it is set up
here, so that a sitting can measure it.

### What its documentation says

| | vLLM |
|---|---|
| APIs | OpenAI's embeddings format on `/v1/embeddings`, Cohere's rerank format on `/rerank`, `/v1/rerank` and `/v2/rerank`, OpenAI's chat format on `/v1/chat/completions`: the three formats the workflow already speaks |
| Embedders and rerankers | It calls them pooling models, and supports them mainly for convenience, with no promise of being faster than the same model under Sentence Transformers. It finds by itself that a model is an embedder or a reranker |
| Models on a CPU | Its list of models validated on a CPU (dated 28 September 2026) holds text and multimodal language models only. Its rerankers include BERT, RoBERTa, XLM-RoBERTa, ModernBERT and Qwen3 (`tomaarsen/Qwen3-Reranker-0.6B-seq-cls` is named); its embedders include XLM-RoBERTa |
| The CPU | x86 with AVX-512 recommended; Python 3.11 to 3.14; release wheels for x86 since 0.17.0. Installed from a wheel, it asks for Intel OpenMP to be preloaded |
| Batching | `--max-num-batched-tokens` (2048 by default on a CPU) and `--max-num-seqs` (128 by default on a CPU) bound what it serves at once |
| Using the cores | `VLLM_CPU_OMP_THREADS_BIND` binds threads to cores by default (`auto`; `nobind` leaves the count to `OMP_NUM_THREADS`). `VLLM_CPU_NUM_OF_RESERVED_CPU` keeps CPUs out of inference, one by default for one server; it advises keeping one or two cores for the serving framework |
| Memory | Left unset, the CPU worker reserves 0.92 of the NUMA node's memory for its cache, the host's and not the container's. `--kv-cache-memory-bytes` sets the cache; `VLLM_CPU_KVCACHE_SPACE`, the GiB kept for it, is the legacy setting it replaces. In Docker its page runs with 4 GB of shared memory and two rights for memory placement (`--cap-add SYS_NICE`, a seccomp profile); without those rights it works, possibly more slowly |
| The start | `--optimization-level` is 2 by default; 0 starts with no compilation |
| Behind a proxy | `--root-path` is the path prefix when a proxy routes to it by path |

Two of these bear on the RAG's models. Its reranker is a CamemBERT
(`CamembertForSequenceClassification`), which vLLM's list of rerankers does not
hold. Its embedder is an XLM-RoBERTa, an architecture on the list, but the model
itself is not named.

### How it is set up here

| | vLLM |
|---|---|
| Weights | The model's published weights, which its component's `source_id` names. It reads no weight format, so it is compared with the two others on `fp32` |
| The models | Any embedder, reranker or LLM of the workflow file. For the RAG's LLM vLLM loads the published weights, where OVMS pulls OpenVINO's conversion (`deploy.example.yaml` names both) |
| Version | 0.32.0, the `vllm` dependency group of the toolkit's `pyproject.toml`, in a virtualenv of its own: it pins transformers 5.16 to 5.20 and torch 2.14, and the toolkit's `optimum-intel` holds transformers below 5.6. Its one CPU wheel for x86 needs glibc 2.39 (Ubuntu 24.04) |
| Options | Flat `QP_VLLM_*` settings of the App, one per flag (the README's "Variables"), each left out at vLLM's own value, and a pass-through, `QP_VLLM_ARGS`; a flag given both ways is refused. `QP_VLLM_KV_CACHE_GIB` is 8 by default, the one value that differs from vLLM's own, since vLLM sizes the cache from the host's memory. `QP_VLLM_CPU_OMP_THREADS_BIND` and `QP_VLLM_CPU_RESERVED_CPUS` export vLLM's CPU variables |
| Calibration | None |
| Requests at once, more load | vLLM's own batching, or more replicas |
| Docker's rights and shared memory | The container's, which Domino sets: a Domino tier gives 64 MB of shared memory unless an administrator raises it. Ask one before a sitting |

The README's section "vLLM" has the command line, the settings and the start on
Domino.

### What was measured on it

Only the first look at text generation of 16 November 2026, told below under
"Which to use": vLLM 0.32.0 from its CPU image, by hand, on one machine of
16 vCPUs. It took 17 minutes to start, because it compiles the model, and a second
start was no shorter. It held 20 GB, with 8 GB asked for its cache. Not run: a vLLM
App through its launch script, vLLM installed by `uv` from the group, vLLM serving
the RAG's embedder or reranker, vLLM answering under Domino's path prefix, a vLLM
LLM with no compilation or with the cache asked by its flag, and a Domino
environment built from vLLM's image.

## What was measured

Two sittings on the Google Cloud emulation, 15 November 2026: one reranker instance of
2 vCPUs in the morning, then of 8 and of 16 vCPUs (4 and 8 cores, the same CPU model).
Runs last 2 minutes, those of the larger reranker 3 ("A larger reranker did not turn
the result"). "Alone" means Locust straight at the reranker, with the load
test's uniform texts (303 to 380 characters). "End to end" means `/retrieve` through
the proxy and two workflow replicas, with the corpus's vector-store records (texts of
137 to 816 characters).

### Same weights, same throughput

One reranker instance alone, requests a second at 4 users, both servers calibrated on
8 and 16 vCPUs:

| Instance | Weights | FastAPI | OVMS |
|---|---|---|---|
| 2 vCPUs | `fp32` | 0.24 | 0.24 |
| 2 vCPUs | `int8` | 0.36 | 0.38 |
| 8 vCPUs | `fp32` | 1.00 | 0.97 |
| 8 vCPUs | `int8` | 1.55 | 1.57 |
| 16 vCPUs | `fp32` | 1.81 | 2.07 |
| 16 vCPUs | `int8` | 3.10 | 3.05 |

The work is the same model computation, 0.3 to 4 seconds a request. The
server around it costs milliseconds. The one gap, `fp32` on 16 vCPUs, is the batch
size: these texts are uniform, where OVMS's one batch of 20 is efficient, and the
FastAPI model server's calibration had chosen batches of 5 on texts of mixed length.

### Streams gained 2% at most

One request already uses the cores almost fully. Seconds for one request, from the
FastAPI model server's calibration:

| Threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 8-vCPU machine | 4.93 | 2.51 | 1.34 | 1.24 | |
| 16-vCPU machine | | | 1.34 | 0.70 | 0.68 |

So several requests on a few cores each take about as long as the same requests one
after the other on every core. OVMS, seconds until every request in flight is answered,
corpus-like texts:

| Streams | 8 vCPUs, 4 in flight | 8 vCPUs, 16 in flight | 16 vCPUs, 4 in flight | 16 vCPUs, 16 in flight |
|---|---|---|---|---|
| 1 | 7.6 | 31.8 | 3.8 | 14.8 |
| 2 | 7.6 | 30.8 | 3.8 | 14.5 |
| 4 | 8.9 | 38.4 | 5.7 | 14.4 |
| 8 | not timed | 45.3 | 11.6 | 18.2 |
| 16 | | | 14.8 | 29.4 |

On 8 vCPUs calibration kept 2 streams for both loads. On 16 vCPUs it kept 1 stream for
4 in flight and 4 streams for 16. Under load 4 streams served 2.11 requests a second
against 2.07 on one stream, and a lone request then took 0.99 seconds instead of 0.48.
Turning CPU pinning off, or setting the throughput hint alone, changed nothing on
8 vCPUs. Candidates this close swap places between runs: one timed round does not
separate them.

### On texts of mixed length the batch size decides

A batch is padded to its longest text. Seconds for one request of 20 corpus-like
vector-store records, by batch size, from the FastAPI model server's calibration:

| Batch size | 20 | 10 | 5 | 2 | 1 |
|---|---|---|---|---|---|
| 8 vCPUs | 1.93 | 1.47 | 1.36 | 1.24 | 1.33 |
| 16 vCPUs | 0.98 | 0.72 | 0.68 | 0.73 | 0.78 |

OVMS scores the request as one batch of 20. End to end, `fp32`, requests a second:

| Reranker | Users | FastAPI, calibrated | OVMS, calibrated |
|---|---|---|---|
| 8 vCPUs | 4 | 0.93, p50 3.6 s, no failure | 0.73, p50 4.8 s, no failure |
| 8 vCPUs | 16 | 0.93, p50 16 s, no failure | 0.64, p50 23 s, 31 of 76 failed |
| 8 vCPUs | 32 | 0.99, 84 of 117 failed | 0.96, 84 of 113 failed |
| 16 vCPUs | 4 | 1.70, p50 1.8 s, no failure | 1.24, p50 2.5 s, no failure |
| 16 vCPUs | 16 | 1.71, p50 8.7 s, no failure | 1.52, p50 7.7 s, no failure |
| 16 vCPUs | 32 | 1.71, p50 18 s, no failure | 1.40, 47 of 163 failed |

A failure is a request that waited past the proxy's 30-second limit. OVMS ran 2 streams
on 8 vCPUs and 4 on 16.

### OVMS does not serve in arrival order

Alone, `fp32`, median and 95th percentile:

| Instance | Users | FastAPI | OVMS, 1 stream | OVMS, more streams |
|---|---|---|---|---|
| 8 vCPUs | 4 | 3.4 s and 3.8 s | 3.3 s and 3.7 s | 2.7 s and 6.5 s |
| 8 vCPUs | 16 | 15 s and 16 s | 13 s and 27 s | 12 s and 37 s |
| 8 vCPUs | 32 | 31 s and 32 s | not run | 17 s and 60 s, 9 failed |
| 16 vCPUs | 16 | 8.2 s and 8.6 s | 7.1 s and 16 s | 5.4 s and 11 s |
| 16 vCPUs | 32 | 17 s and 18 s | 11 s and 31 s, 1 failed | 12 s and 26 s |

The FastAPI model server queues requests in order, so they all wait about the same.
Under OVMS some wait far longer than others, with one stream too, and the tail varies
between runs: a second run of one stream at 16 users on 16 vCPUs gave 5.1 s and 10 s.
OVMS's guide gives
its REST worker count as the setting against overload. With 4 REST workers on 4 streams
the tail grew: 3.6 s and 15 s at 16 users, and 4 requests failed at 32.

### `torch` workers did not replace streams

The FastAPI model server under `torch` runs several calls at once (`QP_LOCAL_MODELS__WORKERS`,
4 here). Alone, `fp32`, requests a second: 0.90 at 4 users and 0.98 at 16 on 8 vCPUs,
against 1.00 and 1.01 under `openvino`; 1.61 and 1.94 on 16 vCPUs, against 1.81 and 1.80.

### A larger reranker did not turn the result

The same comparison on Qwen3 with 0.6 B parameters, about five times the small
reranker's size: the FastAPI model server on the project's own `fp32` export of
`tomaarsen/Qwen3-Reranker-0.6B-seq-cls` and on OpenVINO's own conversion,
`OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov`, and OVMS on that conversion. Every
request carries the prompt format the two model cards give. One reranker alone, 3-minute
runs, requests a second:

| CPUs | Texts, users | FastAPI, own export | FastAPI, OpenVINO's files | OVMS, OpenVINO's files |
|---|---|---|---|---|
| 8 | uniform, 1 | 0.15, p50 6.3 s | 0.15, p50 6.0 s | 0.08, p50 12 s |
| 8 | uniform, 4 | 0.16 | 0.17 | 0.16 |
| 8 | mixed, 4 | 0.13 | 0.14 | 0.11 |
| 16 | uniform, 1 | not run | 0.27, p50 3.1 s | 0.15, p50 6.0 s |
| 16 | uniform, 4 | not run | 0.33 | 0.33 |
| 16 | mixed, 4 | not run | 0.27 | 0.21 |

- The conversion changes nothing: the project's export and OpenVINO's files serve alike.
- Under load on uniform texts the two servers serve alike. On texts of mixed
  length the FastAPI model server leads, through its batch size again: 4.5 s a request
  in batches of 2 against 6.8 s as one batch of 20, on 8 CPUs.
- Streams gain more here than on the small reranker, and cost the lone user. For
  4 requests in flight on 8 CPUs, 4 streams took 25.9 s against 29.7 s on one; a lone
  request then takes 12 s on OVMS against 6 s on the FastAPI model server.
- OVMS does not return the model cards' reference scores. For the cards' own example
  the FastAPI path returns 0.0427, 0.9992, 0.4064 and 0.9718, as the cards print; OVMS,
  on the same model file, returns 0.0655, 0.9956, 0.4441 and 0.9497. The order is the
  same. The likely cause, not tested: OVMS's source puts its own start, end and
  separator tokens around the query and the text, which the cards' code does not.

## What OVMS's own material says matters for a reranker

OVMS has no tuning guide for rerankers. Its rerank demo, its general tuning guide, its
export helper and its maintainers' answers in issues name the settings below, set here
in the order of their effect in the sittings above. The seconds are those of 4 requests
in flight on 8 vCPUs.

| | Setting | What the material says | Measured here |
|---|---|---|---|
| 1 | The tokens of a request: its texts times their length | The demo's own benchmark, 20 texts a request: 3.6 requests a second on texts of 500 tokens, 12.7 on texts of 75. A text longer than the model's context is cut into chunks, each one more row of the batch. `top_n` shortens the response, not the work | The batch is padded to its longest text: "On texts of mixed length the batch size decides" |
| 2 | The weight format | Every example exports with `--weight-format int8`, which is also the export helper's default | About 1.5 times the requests a second, on both servers |
| 3 | `NUM_STREAMS` | The one speed option the export helper has for rerank, 1 by default. The guide: 1 for a single client, more under concurrency, fewer than the requests in flight, never more than the cores. A maintainer: about 2 streams for 4 clients, and more streams raise the throughput of most models at the cost of latency | Calibration kept 1 or 2 streams for 4 requests in flight. Gain: 2% at most, 13% at most on Qwen3; a lone request then takes twice as long |
| 4 | `INFERENCE_NUM_THREADS` against the container's CPUs | The guide: give a container whole cores (Kubernetes' CPU manager, or `--cpuset-cpus`), not a share of CPU time. OVMS reads the CPUs of its container and adjusts threads and pinning only when it detects Docker or Kubernetes on Linux, and sets a stream count only under the `THROUGHPUT` hint; a CPU quota turns pinning off (`src/ov_utils.cpp`, 2026.4.1; this sentence corrected on 19 November 2026). One issue traces slow inference under a Kubernetes CPU limit to more threads than the limit allows | Not measurable on the emulation, where an App has its machine to itself. The App pins the thread count in OVMS's configuration to the CPUs it counts for itself |
| 5 | `ENABLE_CPU_PINNING`, `ENABLE_CPU_RESERVATION` | Pinning is on by default, and is turned off only where several workloads share the machine. Several models in one server: one stream each, with cores reserved and pinned for each (a maintainer's answer in issue 3136, not a page of the documentation) | Pinning off: 7.9 s against 7.6 s on 2 streams |
| 6 | `PERFORMANCE_HINT` | In place of `NUM_STREAMS`, never with it | The throughput hint alone: 7.7 s against 7.6 s |
| 7 | `rest_workers`, `nireq` | The defaults are enough in most cases. `nireq` never reaches an embeddings or a rerank graph: their queue of inference requests is OpenVINO's optimal number for the compiled model (`src/sidepacket_servable.cpp`, 2026.4.1; corrected on 19 November 2026, where this row said it is at least the stream count) | 4 REST workers on 4 streams: no gain |

What follows for this project:

- The two settings the material treats as tunable on a CPU, streams and threads, are
  the two that OVMS's calibration searches. Its stream candidates stop at the cores,
  the CPUs divided by `QP_LOCAL_MODELS__THREADS_PER_CORE` (2 by default), as the
  guide advises; until 19 November 2026 they went up to the CPU count, and the
  candidates above the cores were never kept.
- The largest effects lie outside the server: how many vector-store records a request carries, how
  long they are, and the weight format.
- The rerank endpoint has no batch size. OVMS's source builds one batch from every
  text of a request and scores it in one inference, so a smaller batch is a request
  with fewer vector-store records. Splitting a request on the client side was not tried.
- Not done: reading OVMS's metrics (`QP_OVMS_METRICS=true` turns them on), where the time waiting for an inference request
  (`ovms_wait_for_infer_req_time_us`) set against the inference time
  (`ovms_inference_time_us`) would tell queueing from computing in "OVMS does not
  serve in arrival order"; and the guide's method of settling the plugin configuration
  with OpenVINO's benchmark tool first.

Outside OVMS, Sentence Transformers' page on speeding up cross-encoders compares
backends and `int8` on ranking quality as well as speed, and advises testing each
backend on one's own model and data.

## Which to use

A reading of the results above, not a measurement.

- **The FastAPI model server** for these rerankers on machines of up to 8 cores. It
  serves as much as OVMS on the same weights, more on texts of mixed length, and
  under overload it loses fewer requests. It scales with the cores of one instance
  (0.28, 1.0 and 1.8 requests a second on 2, 8 and 16 vCPUs) and with replicas: end to
  end, four replicas of 2 vCPUs served 0.87 requests a second where one instance of
  8 vCPUs served 0.94, and a lone request took 4.8 s on the replicas against 1.3 s on
  the one instance (the sitting on scale out against scale up, in `experiments/`).
- **OVMS** when its own features are needed: model versions, configuration changes
  at run time, metrics, gRPC, or batching for text generation. Its streams are worth
  trying again where one request cannot use every core.
- **For text generation, a first look only.** One sitting by hand on 16 vCPUs
  (16 November 2026, in `experiments/`) ran Qwen3.5 with 2 B parameters and 4-bit
  weights on OVMS 2026.4.1, on PyTorch behind a stand-in FastAPI server and on vLLM.
  OVMS answered one user soonest (7.9 s against 13 s and 19 s). At 4 users its batching
  gained only once it read each prompt in one pass (`--dynamic_split_fuse false
  --max_num_batched_tokens 8192`): 20.4 tokens a second against 11.6 as it ships. An
  `ovms` App now serves an LLM with those two flags by default
  (`QP_OVMS_DYNAMIC_SPLIT_FUSE`, `QP_OVMS_MAX_BATCHED_TOKENS`; the README's "An LLM
  on OVMS"), measured at 1 and 4 users only; what of it is not verified: its
  "A served LLM". A `vllm` App serves one too, and has not run yet. The FastAPI
  model server serves no LLM.
- **vLLM** for the embedder or the reranker: no reading yet. Nothing has been
  measured, its own documentation promises no speed for these models, and the RAG's
  reranker is not on its list. It is there to be compared, on `fp32` against the two
  others.

## Limits

- Each figure is one run: of 2 minutes for the small reranker, of 3 minutes for the
  Qwen3 reranker and for text generation. The stream timings are one or two rounds each.
- Two models at 20 vector-store records a request: `crossencoder-camembert-base-mmarcoFR` for
  every figure but the last section's, and Qwen3 with 0.6 B parameters there, whose
  runs hold 14 to 60 requests each.
- Machines of 2, 8 and 16 vCPUs. Nothing here says what happens with more cores, or
  with several sockets.
- The text-generation figures are one sitting on one machine, with runs of 11 to 30
  requests and a stand-in server for PyTorch.
- Ranking quality on `int8` weights was not checked.
- Everything here but text generation is about the reranker. OVMS could not serve the default embedder: it
  crashes while loading it (the check of 16 November 2026 in `experiments/`).

## Where the runs are

Every run behind these figures, with its settings and its statistics, is registered in
`experiments/` (the sittings of 15 November 2026, and that of 16 November 2026 for
text generation).

## Pages read

- OVMS 2026.3.1: `README.md`, `docs/features.md`, `docs/performance_tuning.md`,
  `docs/mediapipe.md`, `docs/dynamic_input.md`, `docs/starting_server.md`,
  `docs/metrics.md`, `docs/model_server_rest_api_rerank.md`, `demos/rerank/README.md`,
  `demos/embeddings/README.md`, `demos/common/export_models/export_model.py`,
  `src/rerank/rerank_calculator_ov.proto`, `src/rerank/rerank_calculator_ov.cc` and
  `src/rerank/rerank_utils.cpp` of `github.com/openvinotoolkit/model_server` at tag
  `v2026.3.1`.
- Issues 2721, 3136, 3248 and 3310 of the same repository, as they stood on
  15 November 2026.
- Sentence Transformers, "Speeding up Inference" for cross-encoders, on `sbert.net`.
- The model cards of `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` and
  `OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov` on Hugging Face, for the prompt format
  and the OVMS command.
- FastAPI: "Concurrency and async / await", "Deployments Concepts" and "Server
  Workers - Uvicorn with Workers" of `fastapi.tiangolo.com`.
- vLLM, on 16 and 17 November 2026: on `docs.vllm.ai`, the installation on CPU, the
  supported models on CPU, the pooling models, the scoring and the embedding usages,
  and the `vllm serve` reference at 0.32.0; the files of the release `v0.32.0` of
  `github.com/vllm-project/vllm`, and its `docker/Dockerfile.cpu` at that tag.
- The `config.json` of the RAG's embedder and reranker on Hugging Face, for their
  architectures.
- Domino's 6.3 documentation (its 6.2 one is no longer published): external images,
  the environment catalogue, Apps, hardware tiers and central configuration.
- OVMS 2026.4.1, OpenVINO's CPU pages, vLLM 0.32.0, Uvicorn, Sentence Transformers
  and httpx, as `docs/server-settings-for-throughput-and-latency.md` lists them, for
  the two corrections and the settings of "How each is set up here".
