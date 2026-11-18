# The answers of `main` before the toolkit and of the toolkit, on real models

Sitting of 17 November 2026: a first attempt from 13:01 to about 17:00 UTC, then a second
from 23:24 to 00:16 UTC the next day, when the last machines were deleted. Registered on
18 November from the sitting's notes and its answer records. It holds no Locust run: every
entry is a measurement, `results.csv` is empty, and the answer records are in `records/`.

## Question

Pull request 29 replaced the layout `main` had with the toolkit and its RAG use case. Does
the toolkit, deployed on real models, answer `/retrieve` as the code before it did, and
does a topology change an answer? Each deployment was sent the same 50 questions once
with `qp-record-answers`, and the records were compared with `qp-compare-answers`: the
passages' ids in order, their text hashes and metadata exactly, their two scores within
a tolerance.

## Hardware

| Role | Machine type | vCPUs | Count |
|---|---|---|---|
| Workspace (emulated Domino, indexing, recorder), on demand | `n2-standard-4`, 16 GiB | 4 | 1 |
| Elasticsearch, Spot | `n2-standard-2` | 2 | 1 |
| Pool machine, hardware tier `cpu-2`, Spot | `n2-highmem-2`, 16 GiB | 2 | 4 |

The same six machines for each phase, created anew each time. The four pool machines hold
4 workflow replicas (monolith), 2 workflow and 2 reranker replicas (split-reranker), or 1
workflow, 1 embedder and 2 reranker replicas (split-all).

## Settings common to the sitting

- Code before: `main` at `dcb6e9b`, the layout from before the toolkit. Code after: `main`
  at `0488bda`, the toolkit and the RAG use case. Each on every machine of its phase.
- The corpus: the synthetic corpus the code generates, 200 documents and 200 passages,
  indexed by `make ingest` with the real embedder. Nothing was fetched.
- The requests: the 50 synthetic questions of the load test, each sent once to
  `/retrieve`, one after the other. `/query` was left out (`QP_LOADTEST_QUERY_WEIGHT=0`):
  the project's LLM key does not work.
- Embedder `Snowflake/snowflake-arctic-embed-l-v2.0`, reranker
  `antoinelouis/crossencoder-camembert-base-mmarcoFR`, both on the OpenVINO runtime with
  32-bit weights, read from the dataset folder of each machine. No model was downloaded:
  the Workspaces' Hugging Face cache stayed empty.
- FastAPI model servers. Dense retrieval, 20 candidates, 10 passages kept.
- On the toolkit only, the router and its embedder on `fake`: the dataset folder holds no
  export of the router's embedder, and a plain `/retrieve` never calls the router.
- The recorder: `qp-record-answers` of `0488bda`, through the emulated Domino's proxy.

## Experiments

| Id | What was done | Result |
|---|---|---|
| `ans-main-monolith`, `ans-main-split-reranker`, `ans-main-split-all` | The code before, placed from scratch under each topology on one index, and recorded | 50 answers each, all 200, 10 passages each; placed in 108, 104 and 104 s |
| `ans-toolkit-monolith`, `ans-toolkit-split-reranker`, `ans-toolkit-split-all` | The toolkit, the same on its own machines and its own index | 50 answers each, all 200, 10 passages each; placed in 116, 118 and 110 s |
| `ans-toolkit-split-all-reindexed` | The toolkit's corpus indexed a second time, the layout left in place, recorded again | 50 answers, all 200 |
| `ans-main-monolith-first-attempt` | The first attempt's one record: the code before, as a monolith, on that attempt's index | 50 answers, all 200 |
| `ans-compare-main-topologies` | The three records of the code before | Identical, every score equal to the last digit |
| `ans-compare-toolkit-topologies` | The three records of the toolkit | Identical |
| `ans-compare-main-against-toolkit` | The two monoliths of the second attempt, each on its own index | The same passages in the same order on 50 of 50; rerank scores equal; retrieval scores differ on 4 of 109 documents, by 0.00046 at most: 7 answers over a tolerance of 0.0001, none over 0.001 |
| `ans-compare-index-builds` | One record per index build, four builds, pair by pair | The first attempt's build by the code before and the toolkit's second build are equal to the last digit; every other pair differs on 4 or 31 documents, by 0.00065 at most |

## What it showed

- A topology does not change an answer, on either code: the monolith, split-reranker and
  split-all give the same passages with the same scores.
- The toolkit answers what the code before it answered. On two index builds, made eleven
  hours apart on different machines, the two codes are equal to the last digit of every
  score: the same passages, the same order, the same retrieval and rerank scores.
- An index build is not reproducible to four decimals. The same code, indexing the same
  corpus twice, differs from itself on 31 of the 109 documents that appear, by up to
  0.00065, and one answer of 50 changes its order. That is more than the two codes differ
  on the same evening (4 documents, 0.00046).
- So retrieval scores compare at `0.0001`, the default of
  `QP_LOADTEST_SCORE_TOLERANCE`, only between deployments that read the same index.
  Across two index builds, `0.001` held here.
- The rerank scores were equal to the last digit in every pair of records.
- The toolkit's Apps were published through the emulated Domino onto pool machines for
  the first time, and each layout started at its first deploy.

## Limits

- 50 questions, each sent once; no load.
- `/retrieve` only. No LLM answered, and the router was on `fake` on the toolkit: this
  says nothing of a rewrite, of a routed message or of an answer's text.
- What makes one index build differ from the next is not established. The two builds
  that are equal are the two that indexed fastest (92 s and 87 s, against 111 s and
  99 s); nothing was measured that would say why.
- The reranker served by OVMS was not recorded. The cached OVMS export holds 8-bit
  weights; the code before could give that weight format to the reranker's App alone,
  and the toolkit refuses it, since the weight format is a value of the workflow file
  that every App shares.
- The first attempt was driven step by step from a laptop that went offline after one
  record. Its machines ran until they deleted themselves, four hours after they were
  created, for one record; the second attempt ran each phase as one script on the
  Workspace, in 22 and 28 minutes.
- The first attempt's record was taken from the laptop through a tunnel, the others on
  the Workspace, with the same command.

## Reading the records again

From this folder, with the toolkit installed:

```bash
qp-compare-answers records/main-monolith records/main-split-reranker records/main-split-all
```

```bash
qp-compare-answers records/main-monolith-first-attempt records/toolkit-split-all-reindexed
```

```bash
QP_LOADTEST_SCORE_TOLERANCE=0.001 qp-compare-answers records/main-monolith records/toolkit-monolith
```
