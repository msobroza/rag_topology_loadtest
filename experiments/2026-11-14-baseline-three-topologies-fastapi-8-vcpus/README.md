# Three topologies with FastAPI model servers, 8-vCPU machines

Sitting of 14 November 2026, 17:01 to 17:40 UTC. Registered on 15 November 2026 from
the notes and the statistics files of the session that ran it.

## Question

Which topology serves `/retrieve` best when the model servers are FastAPI model servers?
And how much does each model server serve alone?

## Hardware

Five on-demand machines, the first sitting of the emulation:

| Role | Machine type | vCPUs | Count |
|---|---|---|---|
| Workspace (emulated Domino, Locust) | `n2-standard-4` | 4 | 1 |
| Elasticsearch | `n2-standard-4` | 4 | 1 |
| App machines, hardware tier `cpu-8` | `n2-standard-8` | 8 | 3 |

Each App has one replica on a machine of its own. The three topologies were deployed in
turn on the same three machines: `split-all` (workflow, embedder, reranker), then
`monolith` (the workflow alone), then `split-reranker` (workflow and reranker).

## Settings common to the sitting

- Code: commit `4f729d9` on every machine.
- Models: embedder `Snowflake/snowflake-arctic-embed-l-v2.0`, reranker
  `antoinelouis/crossencoder-camembert-base-mmarcoFR`, runtime `openvino`.
- Every App logged the same thread budget: 8 CPUs from the hardware tier, 1 worker,
  8 threads for each model.
- Corpus: French, 1,223 documents and 1,592 passages, ingested during the sitting.
- `RAG_TOP_K_RETRIEVE` 20, `RAG_TOP_K_RERANK` 5 (the defaults at that commit): the
  reranker scores 20 passages a request.
- `/retrieve` runs go through the emulated Domino's proxy with
  `RAG_LOADTEST_QUERY_WEIGHT=0`. Model-server runs go straight at the machine.
- 2 minutes a run, spawn rate 5 users a second.

## Experiments

Figures are the route's row of each run (requests a second, median and 95th percentile).
No run had a failure.

| Id | What varies | Users | Headline |
|---|---|---|---|
| `pr8-rerank` | FastAPI reranker alone | 20 | 0.95 req/s, p50 20 s, p95 21 s |
| `pr8-embed` | FastAPI embedder alone | 20 | 18.0 req/s, p50 0.50 s, p95 0.65 s |
| `pr8-split-all` | `split-all`, `/retrieve` | 5, 10, 15, 20 | 0.55 to 0.70 req/s; p50 14 s at 10 users, 28 s at 20 |
| `pr8-monolith` | `monolith`, `/retrieve` | 10 | 0.65 req/s, p50 15 s, p95 15 s |
| `pr8-split-reranker` | `split-reranker`, `/retrieve` | 10 | 0.68 req/s, p50 14 s, p95 15 s |

Settings in full: [experiments.yaml](experiments.yaml). Every row of every run:
[results.csv](results.csv).

## What it showed

- The reranker is the limit in every topology. Each served about 0.7 `/retrieve` a
  second, whatever the topology or the number of users.
- In `split-all` at 10 users the rerank stage took 14 s at the median, the embed stage
  67 ms and the retrieve stage 12 ms.
- In `monolith` both models share the workflow's process: at 10 users the embed stage
  took 7.0 s at the median, against 61 to 69 ms in the split topologies, where the
  reranker runs on another machine.
- No topology was measurably better than another.

## Limits

- One 2-minute run per point, a few dozen requests each.
- The `split-all` run at 5 users is distorted by reranks left over from the run at 20
  users before it. The clean figure is in the cache sitting of the same evening.
- The model-server runs ran while the corpus was still being embedded on the Workspace,
  the machine Locust runs on.
- Each machine is a whole VM, not a pod with a CPU limit.
- These machines are not the size of the 15 November sittings: the figures do not
  compare with theirs.
