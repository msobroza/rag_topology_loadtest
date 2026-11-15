# The two model servers: OpenVINO Model Server and the FastAPI model server

A model server serves the reranker or the embedder on the /v3 routes. There are two:
OpenVINO Model Server (OVMS), a product, and `rag-model-server`, this project's own
server built on FastAPI. Both run the same engine, OpenVINO, on the same model. This
page says what the documentation of OVMS and of FastAPI says about serving, how each
is set up here, and what the two sittings of 15 November 2026 measured.

Read for this page: OVMS's documentation at the version pinned in `ovms/versions.env`
(2026.3.1), and FastAPI's pages on concurrency and deployment as they stood on
15 November 2026. The pages are listed at the end.

## What their documentation says

They are different kinds of thing. OVMS is a finished inference server. FastAPI is a
web framework: it knows nothing about models, and the serving is code written on top
of it.

| | OVMS | FastAPI |
|---|---|---|
| What it is | A C++ inference server that exposes models over network APIs | A Python framework for building web APIs |
| APIs | Built in: OpenAI-compatible for generation and embeddings, Cohere-compatible rerank, KServe for classic models; REST and gRPC | The routes the application defines; request validation and an OpenAPI description come with it |
| Several requests at once, in one process | Execution streams split the CPUs so that several requests run in parallel; a queue of inference requests (`nireq`) sits in front | One process serves many clients concurrently, which helps work that waits on I/O; for CPU-bound work such as machine learning the documentation points to parallelism with several processes |
| Using more cores | More streams in the same process; it adjusts threads, streams and pinning to the CPUs of its container | More worker processes, or on Kubernetes one process per container and more containers; each process loads its own copy of the model |
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
| Weights | Its own export, at `RAG_OPENVINO_WEIGHT_FORMAT` | The in-process export, at the same setting |
| Requests at once | Its streams (`RAG_OVMS_NUM_STREAMS`) | One under `openvino`: a loaded model holds one inference request. `RAG_MODEL_WORKERS` under `torch` |
| Calibration | Streams and threads, for `RAG_CALIBRATION_USERS` requests in flight | Threads and batch size, on one request |
| Batch size of a rerank request | The whole request, 20 passages: its rerank endpoint has no such setting | `RAG_RERANK_BATCH_SIZE`, calibrated |
| More load | More streams, or more replicas | More replicas |

Compare the two on the same weight format: see "Experiment 2" in the README.

## What was measured

Two sittings on the Google Cloud emulation, 15 November 2026. Runs last 2 minutes.
"Alone" means Locust straight at one reranker instance, with the load test's uniform
passages (303 to 380 characters). "End to end" means `/retrieve` through the proxy
and two workflow replicas, with corpus passages (137 to 816 characters).

### Same weights, same throughput

One reranker instance alone, requests a second at 4 users:

| Instance | Weights | FastAPI | OVMS |
|---|---|---|---|
| 2 vCPUs | `fp32` | 0.24 | 0.24 |
| 2 vCPUs | `int8` | 0.36 | 0.38 |
| 8 vCPUs | `fp32` | 1.00, calibrated | 0.97, calibrated |

The work is the same model computation, 1 to 4 seconds a request. The server around
it costs milliseconds.

### Streams gained nothing on 8 vCPUs

The 8-vCPU machine has 4 cores with two threads each. One request already uses them
almost fully. Seconds for one request, from the FastAPI model server's calibration:

| Threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Seconds | 4.93 | 2.51 | 1.34 | 1.24 |

So four requests on one core each take about as long as four requests one after the
other on four cores. OVMS, requests in flight served at once, corpus-like passages:

| Streams, on 8 threads | 4 in flight | 16 in flight |
|---|---|---|
| 1 | 7.6 s | 31.8 s |
| 2 | 7.6 s | 30.8 s |
| 4 | 8.9 s | 38.4 s |
| 8 | not timed | 45.3 s |

Calibration kept 2 streams for both loads. Turning CPU pinning off, or setting the
throughput hint alone, changed nothing. Why four streams are slower was not
established.

### On passages of mixed length the batch size decides

A batch is padded to its longest passage. Seconds for one request of 20 corpus-like
passages on 8 threads, by batch size, from the FastAPI model server's calibration:

| Batch size | 20 | 10 | 5 | 2 | 1 |
|---|---|---|---|---|---|
| Seconds | 1.93 | 1.47 | 1.36 | 1.24 | 1.33 |

OVMS scores the request as one batch of 20. End to end, reranker on 8 vCPUs, `fp32`:

| Users | FastAPI, calibrated | OVMS, 2 streams |
|---|---|---|
| 4 | 0.93 requests a second, p50 3.6 s, no failure | 0.73, p50 4.8 s, no failure |
| 16 | 0.93, p50 16 s, no failure | 0.64, p50 23 s, 31 of 76 failed |
| 32 | 0.99, 84 of 117 failed | 0.96, 84 of 113 failed |

A failure is a request that waited past the proxy's 30-second limit.

### OVMS does not serve in arrival order

Alone on 8 vCPUs, `fp32`, median and 95th percentile:

| Users | FastAPI | OVMS, 2 streams | OVMS, 1 stream |
|---|---|---|---|
| 4 | 3.4 s and 3.8 s | 2.7 s and 6.5 s | 3.3 s and 3.7 s |
| 16 | 15 s and 16 s | 12 s and 37 s | 13 s and 27 s |
| 32 | 31 s and 32 s, no failure | 17 s and 60 s, 9 failed | not run |

The FastAPI model server queues requests in order, so they all wait about the same.
Under OVMS some wait far longer than others, with one stream too.

## Which to use

A reading of the results above, not a measurement.

- **The FastAPI model server** for this reranker on machines of up to 4 cores. It
  serves as much as OVMS on uniform passages and more on passages of mixed length,
  and under overload it loses fewer requests. It scales with replicas.
- **OVMS** when its own features are needed: model versions, configuration changes
  at run time, metrics, gRPC, or batching for text generation. Its streams are worth
  trying again where one request cannot use every core.

## Limits

- Each figure is one 2-minute run. The stream timings are one or two rounds each.
- One model, `crossencoder-camembert-base-mmarcoFR`, at 20 passages a request.
- Machines of 2 and 8 vCPUs. Nothing here says what happens with more cores.
- Ranking quality on `int8` weights was not checked.

## Pages read

- OVMS 2026.3.1: `README.md`, `docs/features.md`, `docs/performance_tuning.md`,
  `docs/mediapipe.md`, `docs/dynamic_input.md`, `docs/starting_server.md`,
  `docs/metrics.md`, `docs/model_server_rest_api_rerank.md`, `demos/rerank/README.md`
  of `github.com/openvinotoolkit/model_server` at tag `v2026.3.1`.
- FastAPI: "Concurrency and async / await", "Deployments Concepts" and "Server
  Workers - Uvicorn with Workers" of `fastapi.tiangolo.com`.
