# Scale out against scale up: 8 reranker vCPUs as four machines or as one

Sitting of 15 November 2026, 22:40 UTC, to 01:30 UTC the next day. Registered on
16 November 2026 from the phase scripts, the App logs and the statistics files of the
session that ran it.

## Question

With the same 8 reranker vCPUs, the same code and the same settings, do four machines of
2 vCPUs (scale out) serve what one machine of 8 vCPUs (scale up) serves? And how long
does a user wait on each, from 1 user to saturation? Asked of the FastAPI model server,
of OVMS and of the monolith.

The earlier sittings could not answer: they never measured the same total of vCPUs both
ways on the same code.

## Hardware

| Role | Machine type | vCPUs | Count |
|---|---|---|---|
| Workspace (emulated Domino, Locust), on demand | `n2-standard-4` | 4 | 1 |
| Elasticsearch, Spot | `n2-standard-2` | 2 | 1 |
| Small App machines, hardware tier `cpu-2`, Spot | `n2-highmem-2` (1 core, 15 GB) | 2 | 6 |
| Big App machine, hardware tier `cpu-8`, Spot | `n2-standard-8` (4 cores, 31 GB) | 8 | 1 |

Every App machine reports the same CPU, an Intel Xeon 2.80 GHz, stepping 7, with AVX-512
VNNI. Written workflow / reranker, the layouts are:

| Topology | Scale out | Scale up |
|---|---|---|
| `split-reranker` | 2 small / 4 small | 2 small / 1 big |
| `monolith` | 4 small, both models in each | 1 big, both models in it |

In `split-reranker` the two workflow replicas, with the embedder in their process, are
the same in both layouts. The machines a layout does not use stay idle.

## Settings common to the sitting

- Code: `main` at `8fe562f` on every machine.
- Reranker `crossencoder-camembert-base-mmarcoFR`, 20 passages a request, `fp32` weights,
  the `openvino` runtime. Corpus: French, 1,592 passages, restored from the snapshot.
- Calibration at start for every parameter the settings leave at 0. OVMS is calibrated
  for 4 requests in flight, the default, with `RAG_CALIBRATION_BUDGET_S=300` so that its
  whole search runs.
- The balancing rule of the emulated Domino's proxy is `round-robin`, its default.
- Before every layout the emulated Domino was restarted and every App stopped.
- Run names: `s8-<out|up>-split-<server>-u<users>` and `s8-<out|up>-mono-u<users>` are
  `/retrieve` through the proxy, with corpus passages (137 to 816 characters).
  `s8-<out|up>-rerank-<server>-u<users>` are `/v3/rerank` through the proxy at the
  reranker App, with the load test's uniform passages (303 to 380 characters). `-r2` is
  a second run of the same point.
- 2 minutes a run, spawn rate 5 users a second. The proxy gives up after 30 s; every
  failure is that limit.

## Experiments

Figures are the route's row: requests a second, then p50 and p95. No failure unless said.

### The FastAPI model server

| Id | Layout | Users | Headline |
|---|---|---|---|
| `s8-out-split-fastapi` | 4 rerankers of 2 vCPUs, end to end | 1, 2, 4, 8, 16 | p50 4.7 to 4.8 s up to 4 users; 0.87 req/s at 8 (p50 8.4 s, p95 14 s), 0.88 at 16 (18 s, 24 s) |
| `s8-out-split-fastapi-r2` | the same, second run | 8 | 0.85 req/s; p50 9.1 s, p95 13 s |
| `s8-out-rerank-fastapi` | the same, reranker App only | 1, 4, 16 | p50 4.1 s at 1 user; 0.84 req/s at 4; 0.96 at 16 (17 s, 21 s) |
| `s8-up-split-fastapi` | 1 reranker of 8 vCPUs, end to end | 1, 2, 4, 8, 16 | p50 1.3 s at 1 user, 3.5 s at 4; 0.94 req/s at 8 (7.9 s, 8.6 s) and at 16 (16 s, 17 s) |
| `s8-up-split-fastapi-r2` | the same, second run | 8 | 0.95 req/s; p50 7.8 s, p95 8.5 s |
| `s8-up-rerank-fastapi` | the same, reranker App only | 1, 4, 16 | p50 0.94 s at 1 user; 1.08 req/s at 4 and at 16 (14 s, 15 s) |

### OVMS

| Id | Layout | Users | Headline |
|---|---|---|---|
| `s8-out-split-ovms` | 4 OVMS of 2 vCPUs (1 stream each), end to end | 1, 2, 4, 8, 16 | p50 6.1 to 6.2 s up to 4 users; 0.65 req/s at 8 (11 s, 21 s); 27 of 73 failed at 16 |
| `s8-out-rerank-ovms` | the same, reranker App only | 1, 4, 16 | p50 4.4 s at 1 user; 0.79 req/s at 4; 0.93 at 16 with 18 of 110 failed |
| `s8-up-split-ovms` | 1 OVMS of 8 vCPUs (4 streams), end to end | 1, 2, 4, 8, 16 | p50 3.0 s at 1 user, 4.9 s at 4; 0.76 req/s at 8 (8.6 s, 18 s); 21 of 84 failed at 16 |
| `s8-up-rerank-ovms` | the same, reranker App only | 1, 4, 16 | p50 1.9 s at 1 user; 0.92 req/s at 4; 1.05 at 16 with 2 of 124 failed |

### The monolith

| Id | Layout | Users | Headline |
|---|---|---|---|
| `s8-out-mono` | 4 replicas of 2 vCPUs | 1, 2, 4, 8, 16 | p50 4.9 s up to 4 users; 0.81 req/s at 8 (9.3 s, 14 s), 0.82 at 16 (20 s, 26 s) |
| `s8-out-mono-r2` | the same, second run | 8 | 0.81 req/s; p50 9.9 s, p95 14 s |
| `s8-up-mono` | 1 replica of 8 vCPUs | 1, 2, 4, 8, 16 | p50 1.1 s at 1 user, 4.0 s at 4; 0.89 req/s at 8 (8.2 s, 9.1 s), 0.93 at 16 (16 s, 17 s) |
| `s8-up-mono-r2` | the same, second run | 8 | 0.90 req/s; p50 8.2 s, p95 8.9 s |

## What it showed

- At saturation the two ways serve about the same, the big machine a little more. With
  the FastAPI model server, end to end at 8 and 16 users: 0.85 to 0.88 requests a second
  on four machines of 2 vCPUs, 0.94 to 0.95 on one of 8. On the reranker App alone at
  16 users: 0.96 against 1.08.
- The small machines were not equally fast. At start, three of the four reranker
  machines timed one request at 8.4 s as one batch of 20 and the fourth at 6.8 s, on
  the same machine type and the same CPU. The monolith's four machines showed the same
  split, 8.1 to 8.3 s against 6.7 s. That difference is as large as the gap between
  the two ways of scaling.
- Below saturation the big machine answers far sooner. One user waits 1.3 s on it and
  4.8 s on the small machines, whose wait stays at 4.7 to 4.8 s up to 4 users: each
  user then has a machine to itself, and none is faster than a small machine allows.
  At 4 users the big machine serves 0.97 requests a second against 0.76.
- At saturation the small machines' waits are less even. At 8 users their p95 was 13 to
  14 s against 8.5 to 8.6 s on the big machine; at 16 users, 24 s against 17 s.
- OVMS showed the same order, further apart: 0.65 requests a second at 8 users on four
  machines against 0.76 on one, and 6.2 s against 3.0 s for one user. Both OVMS layouts
  lost requests at 16 users, 27 of 73 and 21 of 84, where neither FastAPI layout lost
  any.
- On the big machine OVMS's calibration kept 4 streams on timings within 2% of each
  other (7.24, 7.35 and 7.20 s for 1, 2 and 4 streams). One user then waits 1.9 s on
  the reranker App, against 0.94 s with the FastAPI model server.
- The two servers compared as in the earlier sittings. On uniform passages they serve
  alike (0.96 and 0.93 requests a second on four machines, 1.08 and 1.05 on one). End
  to end on corpus passages the FastAPI model server leads (0.87 against 0.65, and 0.94
  against 0.76 at 8 users).
- The monolith gave the same picture: 0.81 to 0.82 requests a second on four replicas
  against 0.89 to 0.93 on one, and 4.9 s against 1.1 s for one user. In it the embed
  stage waits behind reranks once the replicas are full: 3.0 s at the median at 8 users
  on four replicas, 3.8 s on one.
- The second runs at 8 users agree with the first within 3%: 0.87 and 0.85, 0.94 and
  0.95, 0.81 and 0.81, 0.89 and 0.90 requests a second.

## Limits

- One sitting, one 2-minute run per point, and a second run at 8 users only.
- The machines are Spot. One small machine was taken back 5 minutes after it was
  created, before any run, and created again; the Workspace's startup script was then
  run again.
- Which small machines the cloud gives decides part of the result: three of the four
  were about a quarter slower than the fourth.
- 8 vCPUs in total and nothing larger; `fp32` weights; one balancing rule; 16 users at
  most.
- The reranker-only runs use uniform passages and the end-to-end runs corpus passages:
  their figures do not compare with each other.
- Machines are whole VMs, not pods with CPU limits.
