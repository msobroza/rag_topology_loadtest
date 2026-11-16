# The cache bucket: a sitting from scratch, then one restored from the bucket

Two sittings on 14 November 2026, 23:06 to 23:45 UTC. Registered on 15 November 2026
from the notes and the statistics files of the session that ran them.

## Question

Pull request 11 added the cache bucket: the exported models and the index snapshot
outlive the machines. Does a sitting on fresh machines, restored from the bucket, start
faster and measure what a sitting built from scratch measures?

## Hardware

Each sitting created five machines and deleted them at its end:

| Role | Machine type | vCPUs | Count | Spot |
|---|---|---|---|---|
| Workspace (emulated Domino, Locust) | `n2-standard-4` | 4 | 1 | no |
| Elasticsearch | `n2-standard-4` | 4 | 1 | yes |
| App machines, hardware tier `cpu-8` | `n2-standard-8` | 8 | 3 | yes |

`split-all`: the workflow, the embedder and the reranker, one replica each, one machine
each.

## Settings common to the sitting

- Code: branch `claude/emulation-cache-and-spot` at commit `17ec043`.
- FastAPI model servers, runtime `openvino`. Run label `split-all-fastapi`.
- `RAG_TOP_K_RERANK` 5, the default at that commit. 20 candidates per rerank.
- Corpus: French, 1,223 documents and 1,592 passages.
- One run each: `/retrieve` only, 5 users, 2 minutes, through the emulated Domino's proxy.

## Experiments

| Id | What varies | Users | Headline |
|---|---|---|---|
| `cache-sitting1` | Bucket empty: the corpus ingested and the models exported during the sitting | 5 | 87 requests, 0 failures, 0.73 req/s, p50 6.2 s, p95 6.8 s |
| `cache-sitting2` | Fresh machines, models and index restored from the bucket | 5 | 87 requests, 0 failures, 0.73 req/s, p50 6.3 s, p95 6.7 s |

Settings and timings in full: [experiments.yaml](experiments.yaml). Every row of every
run: [results.csv](results.csv).

## What it showed

- The two sittings measured the same: 87 requests and 0.73 requests a second each.
- From the end of `apply` to a workflow ready to measure took 23 minutes from scratch and
  3 minutes 18 seconds from the bucket. Building the index took 20 minutes 15 seconds;
  restoring it took 3 seconds.
- On fresh machines no App exported a model at start.
- The `/retrieve` figure at 5 users is the clean one that the 14 November baseline lacked
  (its 5-user run followed a 20-user run).

## Limits

- One run per sitting.
- Not exercised: a Spot machine taken back, `spot = false`, the OVMS models, `FORCE=1`.
