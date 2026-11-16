# The router's pipeline trained on Solon, and the app on its real backends

Sitting of 16 November 2026, 19:31 to 20:42 UTC, when the three machines were deleted. Registered the same day from the
sitting's notes. It holds no Locust run: every entry is a measurement taken once, by
hand, and `results.csv` is empty.

## Question

The RAG's router, on its `local` backend, is a kNN pipeline trained on the vectors its
embedder, `OrdalieTech/Solon-embeddings-large-0.1`, gives the utterances of its two
routes. Does `make train-router` train it on a machine that has the model, how well does it route,
and does the workflow app start on its real backends with the file it writes? The
pipeline file and its manifest are the ones committed in `query_processing_rag_usecase/`.

## Hardware

| Role | Machine type | vCPUs | Count |
|---|---|---|---|
| Workspace (emulated Domino), on demand | `n2-standard-4`, 16 GiB | 4 | 1 |
| Elasticsearch 9.0.0, Spot | `n2-standard-2` | 2 | 1 |
| Pool machine, hardware tier `cpu-2`, Spot, idle | `n2-standard-2` | 2 | 1 |

Everything ran on the Workspace. The pool machine stayed idle: no App was published, and
the Terraform refuses a pool of zero machines.

## Settings common to the sitting

- Code: branch `claude/use-case-components` at `c58e0f5` on every machine. It was the
  first time this branch's layout, the toolkit and the use case, booted on the machines:
  the Workspace's start-up script was done about 3 minutes after the machine was created.
- The RAG use case's `workflow.yaml`, `routes.yaml` and `prompts.yaml` as committed at
  `c58e0f5`.
- Router on `local`: the kNN router, `n_neighbors: 3`, `metric: cosine`. Its embedder:
  Solon, query prefix `"query : "`, on `local`, `local_models.runtime: openvino`.
  scikit-learn 1.9.1, joblib 1.6.0.
- The index: the cached snapshot of the corpus `quebec-insurance` (analyzer `french`,
  retrieval embedder `Snowflake/snowflake-arctic-embed-l-v2.0`), restored: 1592 passages
  in `rag_passages`.

## Experiments

| Id | What was done | Result |
|---|---|---|
| `t-train-router` | `make train-router` | 1 min 49 s, Solon's download and conversion included; peak memory 11.3 GiB; leave-one-out accuracy 0.8125 (13 of 16) |
| `t-train-router-again` | The same command a second time | The same pipeline file and manifest, byte for byte |
| `t-solon-vectors` | Solon's vectors through the app's embedder | 1024 numbers, length 1.0000 for each of 28 texts; mean cosine 0.338 within a route, 0.238 across routes |
| `t-corpus-questions` | The corpus's 82 questions through the trained router | 81 `self_contained`, 1 `leans_on_history` |
| `t-probe-messages` | Twelve messages written for the check, not of the corpus | 8 of 8 that lean on a history routed `leans_on_history`, 4 of 4 that stand alone routed `self_contained` |
| `t-app-start-real-backends` | `qp-serve` on every component's own backend, then one message on `/retrieve` | `/readyz` 200 after about 125 s; the message took 3220 ms |
| `t-three-paths-llm-fake` | The same app with the LLM on `fake`, one request on `/query` per conversation path | 2796 ms (first message), 2319 ms (leans on its history), 2737 ms (stands alone) |
| `t-real-llm-requests` | The app started again with every component on its real backend, then three requests on `/query`: a first message, a message that leans on its history, a follow-up in a conversation that names a source | Each answered HTTP 503 `llm_unavailable`, whose detail names the endpoint and its own answer, HTTP 401 "Incorrect API key provided"; no LLM answered |

## What it showed

- The pipeline trains in under two minutes on a machine of 4 vCPUs and 16 GiB, and the
  same inputs give the same file: the file and its manifest are reproducible.
- The three utterances that leave-one-out misses are all of `leans_on_history` and each
  names a subject that utterances of the other route also name: "Et pour un véhicule
  loué ?", "Et si c'est l'autre conducteur qui est responsable ?" and "Quel est le
  montant dans ce cas ?". With sixteen utterances, one utterance is 6.25 points.
- The router sends the corpus's stand-alone questions past the rewrite (81 of 82; the
  one sent to it is "Qu'est-ce qu'un avenant?") and the follow-ups of the check to it
  (8 of 8, 6 of them new to the router).
- Start-up accepts the committed pipeline file against its manifest: the workflow app
  starts on its real backends, its four readiness checks passed (`router_embedder`,
  `embedder`, `vector_store`, `reranker`), with 5 GiB of 15 used by the three models.
- The routing stages add about 0.15 to 0.3 s to a message that has a conversational
  history: `embed_for_route` 294 ms and `route` 14 ms on one request, 133 ms and 3 ms on
  the other.
- An LLM endpoint that refuses the key gives the client a 503 that names the endpoint,
  on the first message as on the routed ones, and not a 500. The app's log holds one
  `rag_request` line with status 503 for each request, and no traceback.

## Limits

- One run of each measurement; the timings are single requests, not a load test.
- The LLM endpoint was called three times and refused the key each time: the value of the
  Project environment variable for the LLM's key was not a bare key. No LLM answered and
  nothing was generated, so the rewrite on a real LLM, with the schema narrowed to the
  store's fields, is still not checked by this sitting.
- No App was published through the emulated Domino: the app ran on the Workspace itself.
- The Project environment variable that holds the LLM's key still had the name the code
  before the toolkit reads; this branch reads `QP_COMPONENTS__LLM__API__API_KEY`. It was
  mapped in the shell for the app's start.
- The corpus's questions file is never committed.
