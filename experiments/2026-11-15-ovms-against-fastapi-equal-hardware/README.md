# OVMS against the FastAPI model server, on equal hardware

Sitting of 15 November 2026, 03:17 to 06:26 UTC. Registered on 15 November 2026 from
the notes, the phase scripts and the statistics files of the session that ran it.

## Question

Does OpenVINO Model Server (OVMS) serve the reranker better than the FastAPI model
server? On the same machines, does a topology serve `/retrieve` better than another?
Smaller questions on the side: the `torch` runtime, the balancing rule, calibration, and
OVMS's own throughput settings.

## Hardware

Six machines. Every App replica has a machine of 2 vCPUs to itself.

| Role | Machine type | vCPUs | Count | Spot |
|---|---|---|---|---|
| Workspace (emulated Domino, Locust) | `n2-standard-4` | 4 | 1 | no |
| Elasticsearch | `n2-standard-2` | 2 | 1 | yes |
| App machines, hardware tier `cpu-2` | `n2-highmem-2` (1 core, 2 hyper-threads, 16 GB) | 2 | 4 | yes |

The owner fixed four App replicas for every layout, written workflow / embedder /
reranker: `monolith` 4/0/0, `split-reranker` 2/0/2, `split-all` 1/1/2. Both sides of a
comparison use the same layout; a reranker alone is measured on 2/0/2. CPU: Intel Xeon
2.80 GHz with AVX-512 VNNI.

## Settings common to the sitting

- Code: commit `ec600e6` on every machine. From phase R8 on, the App machines ran branch
  `claude/local-model-calibration` at `8573501` (pull request 14, calibration).
- Reranker `crossencoder-camembert-base-mmarcoFR`, embedder
  `snowflake-arctic-embed-l-v2.0`. 20 candidates reranked, 10 kept.
- Weights: the FastAPI model server's export is `fp32` (442.5 MB); OVMS's own export was
  `int8` (110.6 MB), until the hand-made `fp32` export of `m8-ovms-fp32-rerank`.
- Corpus: French, 1,223 documents and 1,592 passages, restored from the snapshot.
- Workflow timeout 30 s with 2 retries; the emulated Domino's proxy gives up after 30 s.
  Balancing rule `round-robin` unless said.
- "Alone": Locust straight at one reranker machine, the load test's 20 synthetic
  passages (303 to 380 characters). "End to end": `/retrieve` through the proxy, corpus
  passages (137 to 816 characters).
- 2 minutes a run, spawn rate 5 users a second.

## Experiments

Figures are the route's row (requests a second; median p50 and 95th percentile p95).
No failure unless said.

| Id | What varies | Users | Headline |
|---|---|---|---|
| `m0-fastapi-rerank-in-split-all` | FastAPI `fp32` alone, on the 1/1/2 layout | 1, 2, 4, 8 | 0.22 to 0.25 req/s; p50 4.1 s at 1 user |
| `m1-fastapi-rerank` | FastAPI `fp32` alone, defaults | 1, 2, 4, 8 | 0.21 to 0.24 req/s; p50 4.2 s at 1 user |
| `m1-ovms-rerank-first-attempt` | OVMS alone, no target | 1 | void |
| `m1-ovms-rerank` | OVMS `int8` alone, defaults | 1, 2, 4, 8 | 0.29 at 1 user, 0.37 to 0.38 at 2 to 8; p50 5.1 s and p95 21 s at 4 |
| `m2-split-reranker` | `split-reranker` 2/0/2, FastAPI | 1, 2, 4, 8 | 0.15, 0.30, 0.35, 0.34 req/s |
| `m3-split-reranker-ovms` | `split-reranker` 2/0/2, OVMS `int8` rerankers | 4, 8 | 0.52 at 4; 0.40 at 8 with 8 of 47 failed |
| `m2-split-all` | `split-all` 1/1/2, FastAPI | 1, 2, 4, 8 | 0.14, 0.30, 0.35, 0.36 req/s |
| `m6-split-all-query` | `/query` with generation | 2 | not run: the generation key was refused |
| `m3-split-all-ovms` | `split-all` 1/1/2, OVMS `int8` rerankers | 4, 8 | 0.52 and 0.53 req/s |
| `m2-monolith` | `monolith` 4/0/0 | 1, 2, 4, 8 | 0.15, 0.24, 0.56, 0.65 req/s |
| `m5-torch-rerank` | FastAPI `fp32` alone, `torch` runtime | 1, 4, 8 | 0.22, 0.26, 0.26 req/s |
| `m4-split-all-connection-random` | `split-all`, balancing rule `connection-random` | 4, 8 | 0.35 at 4; 0.31 at 8 with 18 of 36 failed |
| `m-side-sweep-rerank-threads-batch` | Threads and batch size of one rerank, side process | 1 at a time | best 0.340 req/s (2 hyper-threads, batch 2) against 0.182 (the App's setting, batch 20), corpus payload |
| `m7-calibrated-fastapi-rerank` | FastAPI `fp32` alone, calibrated | 1, 2, 4, 8 | 0.24 at 1 user, 0.28 at 2 to 8 |
| `m7-split-reranker-calibrated` | `split-reranker` 2/0/2, calibrated | 1, 2, 4, 8 | 0.21, 0.43, 0.52, 0.51 req/s |
| `m8-fastapi-int8-rerank` | FastAPI alone on OVMS's `int8` file, 1 thread, one batch | 1, 2, 4, 8 | 0.31 at 1 user, 0.36 to 0.37 at 2 to 8 |
| `m8-ovms-fp32-rerank` | OVMS alone on an `fp32` export | 1, 2, 4, 8 | 0.21 at 1 user, 0.24 at 2 to 8; 2 of 28 failed at 8 |
| `m8-fastapi-int8-calibrated-rerank` | FastAPI alone on the `int8` file, calibrated | 1, 4 | 0.34 and 0.42 req/s |
| `m9-ovms-plugin-configs` | OVMS settings, first attempt | 1, 4 | void |
| `m10-ovms-streams2-rerank` | OVMS `int8` alone, `NUM_STREAMS 2` | 1, 4 | 0.31 and 0.38 req/s |
| `m10-ovms-throughput-rerank` | OVMS `int8` alone, throughput hint | 1, 4 | 0.31 and 0.41 req/s |
| `m10-ovms-threads2-rerank` | OVMS `int8` alone, 1 stream on both hyper-threads | 1, 4 | 0.32 and 0.395 req/s |

Settings in full: [experiments-defaults.yaml](experiments-defaults.yaml) (phases Q1 to
R6, defaults) and
[experiments-calibration-and-weights.yaml](experiments-calibration-and-weights.yaml)
(phases R8 to R11). Every row of every run: [results.csv](results.csv).

## What it showed

- On equal weights the two servers served the same: 0.24 requests a second each on
  `fp32`, 0.36 to 0.38 on `int8`. OVMS's lead on defaults (38 to 65%) was its 8-bit
  weights.
- OVMS does not serve in arrival order under load. At 4 users its p95 was 2 to 4 times
  its p50, and it lost requests at 8 users. The FastAPI model server serves in turn.
- On one core, OVMS's own settings barely helped: 2 streams nothing, the throughput hint
  +8% at 4 users.
- Calibration (batch size 1 or 2 in place of one batch of 20, both hyper-threads used)
  was worth 17 to 20% alone on uniform passages and 45 to 50% end to end on corpus
  passages. Calibrated FastAPI on `fp32` matched OVMS on `int8` end to end at 4 users and
  lost no request at 8.
- On equal hardware `monolith` served nearly twice the split layouts at 8 users (0.65
  against 0.34 to 0.36 requests a second): it has four rerankers.
- `connection-random` lost half the requests at 8 users where `round-robin` lost none.
- Every end-to-end failure is the proxy's 30-second limit on `/retrieve`.

## Limits

- One 2-minute run per point, a few dozen requests each.
- Ranking quality on `int8` weights was not checked.
- The `fp32` OVMS export and the `int8` folder of the FastAPI runs were made by hand;
  neither was a supported setting then.
- OVMS's unevenness varied from run to run with identical settings.
- `monolith` at 2 users (0.24 requests a second, p95 13 s) is unexplained.
- Machines are whole VMs, not pods with CPU limits. The 14 November sittings ran on
  8-vCPU machines: their figures do not compare with these.
