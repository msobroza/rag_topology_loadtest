# The embedder on a real OVMS: a check that found a crash

Sitting of 16 November 2026, 06:39 to 07:14 UTC. Registered the same day from the check's
scripts and logs. It holds no Locust run: every entry is a measurement, and `results.csv`
is empty.

## Question

Pull request 24 makes `RAG_EMBEDDING_NORMALIZE` govern an embedder served by OpenVINO
Model Server (OVMS): the value is written into the embedder's graph at export and when
an OVMS App starts. Does a real OVMS follow it? The pull request's tests use a stand-in
`ovms`.

## Hardware

| Role | Machine type | vCPUs | Count |
|---|---|---|---|
| Workspace (emulated Domino), on demand | `n2-standard-4` | 4 | 1 |
| Elasticsearch, Spot | `n2-standard-2` | 2 | 1 |
| Embedder App, hardware tier `cpu-2`, Spot | `n2-highmem-2` (1 core) | 2 | 1 |

## Settings common to the sitting

- Code: branch `claude/ovms-embedding-normalize` at `480123c` on every machine.
- Embedder `Snowflake/snowflake-arctic-embed-l-v2.0`, the default. OVMS 2026.3.1.
- The embedder App alone, published with `deploy-embedder`; no workflow, no load.
- A vector's length is its Euclidean norm.

## Experiments

| Id | What was done | Result |
|---|---|---|
| `n-fastapi-embedder-both-values` | The FastAPI model server's embedder, the setting unset then true | Length 1.000000 both times, the same vectors |
| `n-ovms-embedder-app-start` | An OVMS embedder App: the cached export with the setting unset, then true; then a new export | The App failed three times: `ovms` exited with code -11 |
| `n-ovms-embedder-graph-variants` | `ovms` alone on nine variants of the embedder's graph, on two exports | Every variant crashed, the export as written included |
| `n-ovms-tested-model-normalize` | A model of OVMS's tested list, with `normalize_embeddings` true, false, true | Length 1.0, 16.18, 1.0 |

## What it showed

- OVMS does not start on the default embedder. It crashes with a segmentation fault
  while it loads the model's graph, on the export exactly as its own exporter wrote it,
  with 8-bit and with 32-bit weights. The serving folder, the plugin config and the
  normalisation value play no part. An OVMS embedder App had never been started on real
  machines before: in every earlier sitting the embedder was in-process or behind a
  FastAPI model server.
- OVMS itself serves embeddings and follows `normalize_embeddings`: on
  `OpenVINO/bge-base-en-v1.5-int8-ov` the returned vector has length 1.0 with `true` and
  16.18 with `false`.
- The in-process embedder returns unit-length vectors whatever the setting says. The
  default model's own pipeline ends with a normalisation step (`modules.json` lists
  Transformer, Pooling and Normalize), so `RAG_EMBEDDING_NORMALIZE` changes nothing for
  it.
- The pull request does what it says to the files: with the setting unset the served
  graph and a new export's graph hold `normalize_embeddings: false`. But for a model
  whose pipeline normalises, that is the wrong value: OVMS would return vectors that
  are not unit length where the in-process embedder returns unit-length ones. As
  designed, the pull request would make the two servers disagree, not agree.

## Limits

- Why OVMS crashes on this model was not established. The exported model looks ordinary:
  inputs `input_ids` and `attention_mask`, one output `last_hidden_state` of width 1024.
- One embedder model on OVMS, and one tested model for the comparison.
- One App machine of 2 vCPUs; whole VMs, not pods.
