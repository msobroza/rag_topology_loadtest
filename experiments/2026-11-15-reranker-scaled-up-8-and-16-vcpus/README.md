# The reranker scaled up to 8, then 16 vCPUs: OVMS against FastAPI

Sitting of 15 November 2026, 14:00 to 19:44 UTC. Registered on 15 November 2026 from
the notes, the phase scripts and the statistics files of the session that ran it. Its
last group compares the two servers on a larger reranker, Qwen3 with 0.6 B parameters.

## Question

Does scaling the reranker up, one instance on 8 then on 16 vCPUs, favour OVMS over the
FastAPI model server? OVMS's streams can run several requests at once; the FastAPI model
server runs one at a time under `openvino`. Everything but the reranker keeps the
morning's hardware.

## Hardware

| Role | Machine type | vCPUs | Count |
|---|---|---|---|
| Workspace (emulated Domino, Locust), on demand | `n2-standard-4` | 4 | 1 |
| Elasticsearch, Spot | `n2-standard-2` | 2 | 1 |
| Workflow replicas, hardware tier `cpu-2` | `n2-highmem-2` (1 core) | 2 | 2 |
| Reranker or monolith, phase A, hardware tier `cpu-8` | `n2-standard-8` (4 cores, 31 GB) | 8 | 1 |
| Reranker or monolith, phase B, hardware tier `cpu-16` | `n2-standard-16` (8 cores, 62 GB) | 16 | 1 |

Every App machine has the same CPU, an Intel Xeon 2.80 GHz with AVX-512 VNNI. The
16-vCPU machine replaced the 8-vCPU one at 15:58 UTC: the project's quota of 32 vCPUs
does not hold both. In `split-reranker` the two workflow replicas, with the embedder in
their process, sit on the 2-vCPU machines and the reranker on the big machine. In
`monolith` one workflow replica runs on the big machine and the 2-vCPU machines are
idle.

## Settings common to the sitting

- Code: branch `claude/ovms-stream-calibration`, at `01670af` in phase A (pull requests
  16, 17 and 18). In phase B the big machine and the Workspace ran `f42b13e`, which fixes
  the calibration defect found in phase A; the workflow machines kept `01670af`.
- Reranker `crossencoder-camembert-base-mmarcoFR`, 20 passages a request. Corpus:
  French, 1,592 passages, restored from the snapshot.
- Calibration at start for every parameter the settings leave at 0. OVMS is calibrated
  for `RAG_CALIBRATION_USERS` requests in flight, 4 by default.
- Run names: `<v8|v16>-alone-<variant>-u<users>` are Locust straight at the reranker,
  with the load test's uniform passages (303 to 380 characters).
  `<v8|v16>-e2e-<variant>-u<users>` are `/retrieve` through the proxy and the two
  workflow replicas, with corpus passages (137 to 816 characters).
- 2 minutes a run, spawn rate 5 users a second. The proxy gives up after 30 s; every
  end-to-end failure is that limit.

## Experiments

Figures are the route's row: requests a second, then p50 and p95. No failure unless said.

### Phase A, 8 vCPUs

| Id | What varies | Users | Headline |
|---|---|---|---|
| `v8-alone-fastapi-fp32` | FastAPI `fp32`, calibrated (8 threads, batch 2) | 1 to 32 | 1.00 to 1.03 req/s from 4 users; p50 3.4 s, p95 3.8 s at 4 |
| `v8-e2e-fastapi-fp32` | the same, end to end | 4, 16, 32 | 0.93 req/s at 4 and 16; 84 of 117 failed at 32 |
| `v8-alone-ovms-fp32-cal4` | OVMS `fp32`, calibrated for 4 (2 streams) | 1 to 32 | 0.96 to 1.03 req/s from 4 users; p50 2.7 s, p95 6.5 s at 4; 9 of 119 failed at 32 |
| `v8-probe-ovms-plugin-configs` | Hand-written OVMS configs, 4 in flight | 4 in flight | 0.53 req/s on 1 or 2 streams, 0.45 on 4 |
| `v8-ovms-fp32-cal16-at-start` | OVMS calibrated for 16 at start | none | void: the App failed (a defect) |
| `v8-ovms-fp32-cal16-on-demand` | `rag-calibrate` for 16 in flight | 16 in flight | 2 streams, 8 threads: 30.8 s for 16 at once |
| `v8-e2e-ovms-fp32-2streams` | OVMS `fp32`, 2 streams pinned, end to end | 4, 16, 32 | 0.73 req/s at 4; 31 of 76 failed at 16 |
| `v8-alone-ovms-fp32-1stream` | OVMS `fp32`, 1 stream | 4, 16 | 1.02 req/s; p50 13 s, p95 27 s at 16 |
| `v8-alone-fastapi-torch` | FastAPI under `torch`, 4 workers | 4, 16 | 0.90 and 0.98 req/s |
| `v8-alone-fastapi-int8` | FastAPI `int8`, calibrated | 4, 16 | 1.55 and 1.56 req/s |
| `v8-alone-ovms-int8-cal4` | OVMS `int8`, calibrated for 4 | 4, 16 | 1.57 req/s; p95 22 s at 16 |
| `v8-e2e-monolith-openvino` | `monolith` on the big machine, calibrated | 1, 4, 16, 32 | 0.85 req/s at 4; 89 of 118 failed at 32 |
| `v8-e2e-monolith-torch` | `monolith` under `torch` | 4, 16 | 0.77 and 0.81 req/s |
| `v8-rag-calibrate-on-demand` | `rag-calibrate` under its default budget | n/a | 2 streams for OVMS; 8 threads, batch 2 in process |

### Phase B, 16 vCPUs

| Id | What varies | Users | Headline |
|---|---|---|---|
| `v16-alone-fastapi-fp32` | FastAPI `fp32`, calibrated (16 threads, batch 5) | 1 to 32 | 1.79 to 1.81 req/s from 4 users; p50 1.6 s, p95 2.0 s at 4 |
| `v16-e2e-fastapi-fp32` | the same, end to end | 4, 16, 32 | 1.70 to 1.71 req/s, no failure |
| `v16-alone-ovms-fp32-cal4` | OVMS `fp32`, calibrated for 4 (1 stream) | 1 to 32 | 2.06 to 2.07 req/s from 4 users; p95 16 s at 16; 1 of 247 failed at 32 |
| `v16-ovms-fp32-cal16-default-budget` | OVMS calibrated for 16 at start, default budget | none | the App started; search cut short, 2 streams kept |
| `v16-ovms-fp32-cal16-on-demand` | `rag-calibrate` for 16 in flight | 16 in flight | 4 streams, 16 threads: 14.4 s for 16 at once |
| `v16-alone-ovms-fp32-cal16` | OVMS `fp32`, 4 streams pinned | 1, 16, 32 | 2.11 req/s at 16 and 32, no failure |
| `v16-e2e-ovms-fp32-cal16` | the same, end to end | 4, 16, 32 | 1.24 req/s at 4, 1.52 at 16; 47 of 163 failed at 32 |
| `v16-alone-ovms-fp32-1stream` | OVMS `fp32`, 1 stream | 4, 16 | 2.06 and 2.07 req/s; p95 10 s at 16 |
| `v16-alone-fastapi-torch` | FastAPI under `torch`, 4 workers | 4, 16 | 1.61 and 1.94 req/s |
| `v16-alone-fastapi-int8` | FastAPI `int8`, calibrated | 4, 16 | 3.10 and 3.11 req/s |
| `v16-alone-ovms-int8-cal4` | OVMS `int8`, calibrated for 4 (2 streams) | 4, 16 | 3.06 and 3.20 req/s; p95 9.8 s at 16 |
| `v16-e2e-monolith-openvino` | `monolith` on the big machine, calibrated | 1, 4, 16, 32 | 1.53 to 1.55 req/s from 4 users, no failure |
| `v16-e2e-monolith-torch` | `monolith` under `torch` | 4, 16 | 1.29 and 1.46 req/s |
| `v16-rag-calibrate-on-demand` | `rag-calibrate` under its default budget | n/a | 4 streams (`fp32`) and 2 (`int8`) for OVMS; 16 threads, batch 5 in process |
| `v16-alone-ovms-fp32-4streams-4workers` | OVMS by hand, `--rest_workers 4` | 16, 32 | 2.08 req/s, p95 15 s at 16; 4 of 250 failed at 32 |
| `v16-alone-ovms-fp32-1streams-2workers` | OVMS by hand, `--rest_workers 2` | 16, 32 | void: OVMS was not serving |

Settings in full: [experiments-phase-a-8-vcpus.yaml](experiments-phase-a-8-vcpus.yaml)
and [experiments-phase-b-16-vcpus.yaml](experiments-phase-b-16-vcpus.yaml). Every row
of every run: [results.csv](results.csv).

### Qwen3 reranker, 0.6 B parameters

The big machine of phase B, first with 8 CPUs online (4 cores with both threads), then
with its 16. One reranker alone; 3 minutes a run; 20 passages a request; the prompt
format of the two model cards on every request. Cell a is the FastAPI model server on
the project's own `fp32` export of `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`; cells b and
c are the FastAPI model server and OVMS on OpenVINO's conversion,
`OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov`. Every cell is calibrated. "uniform" are
the load test's passages (303 to 380 characters), "mixed" the calibration sample's
(150 to 800). Settings: `experiments-qwen3-reranker.yaml`.

| Id | What varies | Users | Headline |
|---|---|---|---|
| `q-model-checks` | Each way of loading the model, on the cards' example | none | FastAPI path returns the cards' scores; OVMS returns other values in the same order |
| `q8-fastapi-own-export-uniform` | Cell a, 8 CPUs (8 threads, batch 2) | 1, 4 | 0.15 req/s at 1 user, p50 6.3 s; 0.16 at 4, p50 25 s |
| `q8-fastapi-own-export-mixed` | Cell a, mixed passages | 4 | 0.13 req/s, p50 30 s |
| `q8-fastapi-openvino-ir-uniform` | Cell b, 8 CPUs (8 threads, batch 2) | 1, 4 | 0.15 req/s at 1 user, p50 6.0 s; 0.17 at 4, p50 23 s |
| `q8-fastapi-openvino-ir-mixed` | Cell b, mixed passages | 4 | 0.14 req/s, p50 27 s |
| `q8-ovms-openvino-ir-uniform` | Cell c, 8 CPUs (4 streams) | 1, 4 | 0.08 req/s at 1 user, p50 12 s; 0.16 at 4, p50 22 s, p95 39 s |
| `q8-ovms-openvino-ir-mixed` | Cell c, mixed passages | 4 | 0.11 req/s, p50 35 s |
| `q16-fastapi-openvino-ir-uniform` | Cell b, 16 CPUs (16 threads, batch 2) | 1, 4, 8 | 0.27 req/s at 1 user, p50 3.1 s; 0.33 at 4 and 8 |
| `q16-fastapi-openvino-ir-mixed` | Cell b, 16 CPUs, mixed passages | 4 | 0.27 req/s, p50 14 s |
| `q16-ovms-openvino-ir-uniform` | Cell c, 16 CPUs (4 streams) | 1, 4, 8 | 0.15 req/s at 1 user, p50 6.0 s; 0.33 at 4, 0.34 at 8 with p95 36 s |
| `q16-ovms-openvino-ir-mixed` | Cell c, 16 CPUs, mixed passages | 4 | 0.21 req/s, p50 16 s, p95 29 s |

## What it showed

- On 8 vCPUs both servers served about 1 request a second alone on `fp32`. One request
  already uses the 4 cores: the FastAPI calibration timed 4.93 s on 1 thread and 1.34 s
  on 4. Streams had nothing to win; they changed who waits.
- On passages of mixed length the batch size decides. FastAPI's batches of 2 took 1.24 s
  a request where one batch of 20 took 1.93 s. OVMS scores the request as one batch. End
  to end on 8 vCPUs, FastAPI served 0.93 requests a second at 16 users with no failure;
  OVMS on 2 streams served 0.64 and lost 31 of 76.
- OVMS does not serve in arrival order. On 8 vCPUs at 16 users its p95 was 37 s on 2
  streams and 27 s on 1 stream, against 16 s for FastAPI.
- On 16 vCPUs OVMS led alone: 2.07 against 1.80 requests a second on `fp32`. End to end
  FastAPI led: 1.71 requests a second with no failure at 32 users, against 1.40 and 47
  failures for OVMS on 4 streams.
- On `int8` the two served alike: 1.55 to 1.57 requests a second on 8 vCPUs, 3.06 to
  3.20 on 16.
- A smaller REST worker count did not shorten OVMS's tail; it lengthened it.
- Calibration for 16 in flight at start failed the App in phase A (the calibration used
  the workflow's 30 s timeout). With the fix, in phase B, the App started and the search
  was cut short by the budget.
- The larger reranker did not turn the result. On Qwen3 the conversion changed nothing
  (cells a and b served alike), the two servers served alike under load on uniform
  passages (0.16 to 0.17 requests a second on 8 CPUs, 0.33 on 16), and the FastAPI model
  server led on passages of mixed length (0.14 against 0.11 on 8 CPUs, 0.27 against 0.21
  on 16).
- For Qwen3 streams gained more than for the small reranker: 4 requests in flight took
  25.9 s on 4 streams against 29.7 s on one, on 8 CPUs. They cost the lone user: 12 s a
  request on OVMS against 6 s on the FastAPI model server.
- OVMS does not return the model cards' reference scores for their own example, on the
  same model file on which the FastAPI path does.

## Limits

- One 2-minute run per point. The calibration and probe timings are one or two rounds
  each; candidates within a few percent swap places from run to run.
- Two models at 20 passages a request. One reranker machine of 8, then 16 vCPUs; the
  Qwen3 runs on 8 CPUs are the 16-vCPU machine with half its CPUs offline.
- The Qwen3 runs hold 14 to 60 requests each.
- Ranking quality on `int8` weights was not checked.
- Two runs are void and stay in `results.csv` for the record:
  `v16-alone-ovms-fp32-1streams-2workers-u16` and `-u32`. OVMS was not serving: the
  previous OVMS still held the port when it was started, a fault of the driving script.
- Phase B's machines did not all run the same commit (see "Settings").
