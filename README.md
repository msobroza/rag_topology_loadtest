# rag_load_test — one RAG workflow, three Domino topologies, one Locust suite

A self-contained project that load tests **the same text RAG workflow**
(sentence-transformers embedder → Elasticsearch → cross-encoder reranker →
OpenAI, orchestrated by LangGraph, served by FastAPI) under three deployment
topologies on Domino Data Lab 6.2 and compares them:

| Topology | Domino apps | What runs where |
| --- | --- | --- |
| `monolith` | 1 | embedder + reranker + workflow + OpenAI client in one process |
| `split-reranker` | 2 | **A** OpenVINO Model Server (OVMS) serving the reranker; **B** workflow + embedder |
| `split-all` | 3 | **A** OVMS reranker; **C** OVMS embedder; **B** workflow calling A and C over HTTP |

The workflow code is identical everywhere; only the transport behind the
`EmbedderPort` and `RerankerPort` changes (in-process model vs. HTTP call to
OVMS), selected by `RAG_TOPOLOGY` alone, and every topology reads the same
Elasticsearch index.

To run it on Domino, follow [Step by step: first test run on Domino](#step-by-step-first-test-run-on-domino).

## Workflow

```mermaid
flowchart LR
    L[Locust] -->|POST /query, /retrieve| API[FastAPI · rag-serve]
    API --> E["embed<br/>(EmbedderPort: in-process or OVMS /v3/embeddings)"]
    E --> R["retrieve<br/>(VectorStorePort: Elasticsearch kNN, top 20)"]
    R --> K["rerank<br/>(RerankerPort: in-process or OVMS /v3/rerank, top 5)"]
    K -->|mode = query| G["generate<br/>(ChatModelPort: OpenAI)"]
    K -->|mode = retrieve| Z([response + timings_ms])
    G --> Z
```

Each node is timed; responses carry `timings_ms{embed,retrieve,rerank,generate,total}`
and Locust turns those into per-stage statistics.

## Code layout

```text
src/rag_load_test/
  __init__.py       package docstring, __version__
  models.py         Passage, ScoredPassage, RagResult, the four async ports, RagDependencies, RagDependencyError
  settings.py       RagSettings: the 17 RAG_* variables (pydantic-settings, .env aware)
  topology.py       the topology rule: backends per topology, contradictions, default run label, Domino rules
  adapters.py       SentenceTransformerEmbedder, CrossEncoderReranker, NoReranker, ElasticsearchVectorStore, OpenAIChatModel
  ovms.py           BearerToken (static or Domino run token), OvmsEmbedder, OvmsReranker (httpx, timeout + retries)
  fakes.py          FakeEmbedder, FakeReranker, InMemoryVectorStore, FakeChatModel (the `fake` backends)
  rag_ports.py      RagPorts: settings in, ready ports out; owns the thread pool, HTTP client, bearer token and their shutdown
  workflow.py       RagWorkflow (LangGraph graph with timed nodes)
  api.py            FastAPI app (/query /retrieve /healthz /readyz), Domino tracing, JSON logs, rag-serve
  corpus.py         synthetic handbook corpus, JSONL loader, chunking, rag-ingest
  loadtest.py       locust-free helpers (auth headers, stage events, questions) + rag-compare
  model_server.py   FastAPI model server with the OVMS /v3 contract (rag-model-server)
  served_models.py  ServedModel (source id + alias) from the settings defaults, rag-served-models
locustfile.py       RagUser (FastHttpUser) + opt-in StepLoadShape; `locust` finds it in this directory
docker-compose.yml  Elasticsearch 9.0.0 (:9200) + OVMS reranker (:8001) + OVMS embedder (:8002)
ovms/               versions.env (single OVMS version pin), export_models.sh, models/ (exported, git-ignored)
domino/             app.sh (one entry point, role from RAG_ROLE), Dockerfile (targets `rag` and `ovms`)
tests/              pytest: no model weights, no network, no Elasticsearch
```

## Quickstart (monolith on a laptop)

```bash
make install-dev              # pip install -e ".[dev]" (locust, pytest, ruff, black, mypy)
cp .env.example .env          # set OPENAI_API_KEY, or RAG_LLM_BACKEND=fake to skip the LLM
make infra-up                 # docker compose: Elasticsearch on :9200 (the OVMS containers need `make ovms-export` first)
make ingest                   # rag-ingest --synthetic 200 -> index rag_passages (downloads BAAI/bge-small-en-v1.5 once)
make serve                    # rag-serve on http://0.0.0.0:8888
```

```bash
curl -s -X POST localhost:8888/query -H 'Content-Type: application/json' \
  -d '{"question":"How many paid vacation days do employees accrue each year at Northwind Analytics?"}'
curl -s -X POST localhost:8888/retrieve -H 'Content-Type: application/json' \
  -d '{"question":"Do I need a receipt for an expense claim?"}'
curl -s localhost:8888/readyz
```

`make ingest` embeds with the topology's embedder backend and upserts by
passage id (idempotent); `rag-ingest --corpus docs.jsonl` loads a real corpus, one
`{"id","title","text","metadata"}` object per line. `make infra-down` stops the containers.

No Elasticsearch, model weights or OpenAI key at hand? Every backend has a `fake`:

```bash
RAG_EMBEDDER_BACKEND=fake RAG_RERANKER_BACKEND=fake RAG_LLM_BACKEND=fake RAG_VECTOR_STORE_BACKEND=fake make serve
```

serves the same routes from an in-memory store seeded with the 200 synthetic
documents at start-up (nothing to ingest: `rag-ingest` refuses the fake store).

## Split topologies on a laptop

```bash
make ovms-export              # once, after make install: export both models (int8) into ovms/models, needs HF Hub access
make infra-up                 # the OVMS containers now serve the reranker on :8001 and the embedder on :8002

# split-reranker: reranker over HTTP, embedder in-process
RAG_TOPOLOGY=split-reranker make serve

# split-all: reranker and embedder over HTTP
RAG_TOPOLOGY=split-all make serve
```

The topology sets the backends; no `RAG_*_BACKEND` variable is needed. The
defaults `RAG_OVMS_RERANK_URL=http://localhost:8001` and
`RAG_OVMS_EMBEDDINGS_URL=http://localhost:8002` match the compose ports; `RAG_OVMS_AUTH`
stays empty locally. `RAG_RERANKER_BACKEND=none` skips reranking in any topology
(`rerank ≈ 0 ms`); a backend that contradicts the topology (say
`RAG_EMBEDDER_BACKEND=ovms` with `RAG_TOPOLOGY=monolith`) stops `rag-serve` at
start-up with the fix in the message.

## Load test and comparison

One headless Locust run per topology, then a comparison table:

```bash
make loadtest RUN_NAME=monolith HOST=http://localhost:8888
make loadtest RUN_NAME=split-reranker HOST=http://localhost:8888
make loadtest RUN_NAME=split-all HOST=http://localhost:8888
make compare                          # results/comparison.md
```

`make loadtest` runs `locust --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME)`
(defaults `USERS=20 SPAWN_RATE=5 RUN_TIME=2m`) with the `locustfile.py` of this
directory and writes `results/<RUN_NAME>_stats.csv`, `_failures.csv` and
`.html`. `HOST` is the service base URL without a trailing slash (on Domino: the
App URL).

| Knob | Effect |
| --- | --- |
| `USERS`, `SPAWN_RATE`, `RUN_TIME` | Makefile variables passed to `locust -u -r -t` |
| `RAG_LOADTEST_QUERY_WEIGHT` / `RAG_LOADTEST_RETRIEVE_WEIGHT` | task weights, default `1` / `3` |
| `RAG_LOADTEST_SHAPE=step` | step load 5 → 10 → 20 → 40 users, 60 s each (set `RUN_TIME=4m` or more) |
| `RAG_LOADTEST_BEARER_TOKEN` | sent as `Authorization: Bearer ...` (Domino run token) |
| `DOMINO_API_KEY` | sent as `X-Domino-Api-Key` (from outside Domino) |

Every successful response's `timings_ms` is re-emitted as Locust `STAGE` rows
(`/query:embed`, `/query:retrieve`, `/query:rerank`, `/query:generate`, same for
`/retrieve`), so p50/p95/p99 per stage land in the same CSV as the endpoint
totals. `make compare` feeds every `results/*_stats.csv` prefix (sorted, so
`monolith` is the baseline) to `rag-compare`, which writes one markdown table
per endpoint and per stage: requests, failure %, RPS, p50, p95, p99 and Δp95
versus the first run. Pick another baseline by calling it directly:
`rag-compare results/split-all results/monolith --out results/comparison.md`.

## Experiment 2: serving the models with FastAPI versus OVMS

`rag-model-server` serves one model per process behind the same `/v3` contract
as OpenVINO Model Server (`POST /v3/embeddings`, `POST /v3/rerank`,
`GET /v3/models/<alias>`), using the in-process sentence-transformers adapters.
The workflow app cannot tell the two servers apart, so the comparison needs no
code change: point `RAG_OVMS_RERANK_URL` / `RAG_OVMS_EMBEDDINGS_URL` at one or
the other.

Direct comparison (Locust straight at the model servers, same payloads as the
workflow sends: 20 candidate passages per rerank, one question per embedding):

```bash
make infra-up                                   # OVMS reranker :8001, embedder :8002
make model-server SERVE=reranker PORT=8011      # FastAPI reranker (downloads the weights once)
make model-server SERVE=embedder PORT=8012      # FastAPI embedder, in a second shell

make loadtest-models RUN_NAME=rerank-ovms    HOST=http://localhost:8001 TARGET=rerank
make loadtest-models RUN_NAME=rerank-fastapi HOST=http://localhost:8011 TARGET=rerank
make loadtest-models RUN_NAME=embed-ovms     HOST=http://localhost:8002 TARGET=embeddings
make loadtest-models RUN_NAME=embed-fastapi  HOST=http://localhost:8012 TARGET=embeddings
rag-compare results/rerank-ovms results/rerank-fastapi --out results/rerank.md
rag-compare results/embed-ovms results/embed-fastapi --out results/embeddings.md
```

End-to-end comparison (the split topologies with either server behind them):

```bash
RAG_TOPOLOGY=split-reranker RAG_RUN_LABEL=split-reranker-fastapi RAG_OVMS_RERANK_URL=http://localhost:8011 make serve
make loadtest RUN_NAME=split-reranker-fastapi HOST=http://localhost:8888
```

On Domino, `RAG_ROLE=fastapi-reranker` / `fastapi-embedder` publish the FastAPI
servers with the `rag` environment (no OVMS binary needed); the workflow app's
`RAG_OVMS_*_URL` then points at those App URLs exactly as for the OVMS apps.
`RAG_LOADTEST_TARGET` selects the Locust user class: `rag` (default,
`/query` + `/retrieve`), `rerank` or `embeddings`.

## HTTP API

| Route | Body → result |
| --- | --- |
| `POST /query` | `{question}` → `{request_id, topology, run_label, question, mode, answer, passages[], timings_ms{embed,retrieve,rerank,generate,total}}` |
| `POST /retrieve` | same request → same response with `answer: null` (`generate` skipped, `generate = 0`) |
| `GET /healthz` | `{"status":"ok"}` |
| `GET /readyz` | `{"status":"ready"\|"not_ready","checks":[{name,ok,detail}]}` for `vector_store` (index count > 0), `embedder`, `reranker`; `503` when a check fails |

`X-RAG-Topology` (the topology), `X-RAG-Run-Label` (the run label) and
`X-Request-ID` are set on every response (an incoming `X-Request-ID` is
propagated). Errors: dependency failure → `503`
`{"error":"<component>_unavailable","detail","target"}`, upstream timeout →
`504` `{"error":"<component>_timeout",...}`, validation → `422`. `root_path`
follows `DOMINO_RUN_HOST_PATH`, so the app answers with or without Domino's
prefix on the wire and `/docs` works behind the proxy.
`rag-serve --host 0.0.0.0 --port 8888` are the defaults (host and port are
flags, not settings); `rag-serve --domino` (what `domino/app.sh` runs) also
requires `RAG_OVMS_*_URL` for every port on a model server and defaults
`RAG_OVMS_AUTH=domino`. A settings mistake ends `rag-serve` with a one-line
usage error, not a traceback.

## Configuration (`RAG_*` environment variables or `.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAG_TOPOLOGY` | `monolith` | `monolith` \| `split-reranker` \| `split-all`; decides the embedder and reranker backends (`local`/`local`, `local`/`ovms`, `ovms`/`ovms`); echoed as `topology` and `X-RAG-Topology` |
| `RAG_RUN_LABEL` | the default run label | names the measured configuration, echoed as `run_label` and `X-RAG-Run-Label`; default: the topology plus `+<port>=<backend>` for each backend off its default (e.g. `monolith+reranker=none`); letters, digits and `._+=:-`, 1 to 64 characters |
| `RAG_EMBEDDER_BACKEND` | from the topology | only `fake` (or the topology's own value) may override it |
| `RAG_EMBEDDER_MODEL` | `BAAI/bge-small-en-v1.5` | HF id for `local` |
| `RAG_RERANKER_BACKEND` | from the topology | only `fake` or `none` (or the topology's own value) may override it |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | HF id for `local` |
| `RAG_OVMS_EMBEDDINGS_URL` | `http://localhost:8002` | base URL (App URL on Domino); the adapter appends `/v3/embeddings` |
| `RAG_OVMS_EMBEDDINGS_MODEL` | `bge-small-en-v1.5` | servable name in the OVMS config |
| `RAG_OVMS_RERANK_URL` | `http://localhost:8001` | base URL; the adapter appends `/v3/rerank` |
| `RAG_OVMS_RERANK_MODEL` | `bge-reranker-base` | servable name in the OVMS config |
| `RAG_OVMS_AUTH` | `` | empty = no auth; `domino` = the run's token from `http://localhost:8899/access-token`; anything else = a static bearer token |
| `RAG_LLM_BACKEND` | `openai` | `openai` \| `fake` (echoes the question, no network) |
| `RAG_OPENAI_MODEL` | `gpt-4o-mini` | chat model id |
| `RAG_VECTOR_STORE_BACKEND` | `elasticsearch` | `elasticsearch` \| `fake` (in-memory, seeded with the synthetic corpus at start-up) |
| `RAG_ES_URL` | `http://localhost:9200` | `http://user:password@host:9200` for basic auth |
| `RAG_ES_API_KEY` | `` | Elasticsearch API key (alternative to basic auth) |
| `RAG_DOMINO_TRACING` | `false` | wrap the workflow run with Domino `add_tracing` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | | read by the OpenAI SDK; point `OPENAI_BASE_URL` at a Domino AI Gateway to swap providers |
| `RAG_LOADTEST_BEARER_TOKEN`, `DOMINO_API_KEY` | | Locust side: auth header, see the load-test table |
| `RAG_LOADTEST_QUERY_WEIGHT`, `RAG_LOADTEST_RETRIEVE_WEIGHT`, `RAG_LOADTEST_SHAPE` | `1`, `3`, unset | Locust side: task weights and the step shape |

Those 17 `RAG_*` fields are the whole of `settings.py`. Retrieval sizes
(`TOP_K_RETRIEVE=20`, `TOP_K_RERANK=5`), the model thread pool (`4`), the OVMS
timeout (`30 s`) and retries (`2`, backoff 0.2 s then 0.4 s), the Domino token
cache (`240 s`) and the index name are constants, so experiments only vary the
topology. An invalid backend or topology name, a backend that contradicts the
topology, or a bad run label fails at start-up with the accepted values listed.

## Elasticsearch

* The store is LangChain's `AsyncElasticsearchStore` over index `rag_passages`,
  created on the first `rag-ingest` with a dense-vector kNN mapping; documents
  hold `text`, `metadata` and the vector, and the `_id` is the passage id
  (`<doc id>:<chunk>`), so re-ingesting overwrites in place.
* Auth: basic credentials inside `RAG_ES_URL` (`https://user:password@host:9200`)
  or `RAG_ES_API_KEY`. Any Elasticsearch 8 or 9 cluster works; the compose file
  runs a single 9.0.0 node with security disabled, for development only.
* Failures surface as `503 vector_store_unavailable`; `/readyz` reports
  `not_ready` while the index is missing or empty.
* On Domino every workflow app points at one shared cluster (`RAG_ES_URL` in the
  project's environment variables), so the corpus is ingested once. All
  topologies embed with `bge-small-en-v1.5` (in-process or its int8 OVMS
  export), so one index serves every run.

## OpenVINO Model Server

`ovms/versions.env` is the single pin of the OpenVINO Model Server version,
sourced by `export_models.sh`, read by `domino/Dockerfile` and passed to docker
compose with `--env-file`:

| Key | Value |
| --- | --- |
| `OVMS_VERSION` / `OVMS_IMAGE` | `2026.3.1` / `openvino/model_server:2026.3.1` |
| `OVMS_BINARY_URL` | `ovms_ubuntu22_2026.3.1_python_off.tar.gz` from the GitHub release `v2026.3.1` (Domino `ovms` image) |

The served models (source id and alias) are pinned once, by the `RagSettings`
defaults in `settings.py` (`RAG_RERANKER_MODEL` / `RAG_OVMS_RERANK_MODEL` =
`BAAI/bge-reranker-base` / `bge-reranker-base`, `RAG_EMBEDDER_MODEL` /
`RAG_OVMS_EMBEDDINGS_MODEL` = `BAAI/bge-small-en-v1.5` / `bge-small-en-v1.5`).
`rag-served-models` prints them, `RAG_*` overrides included, as shell
assignments (`RERANKER_SOURCE_MODEL=...`) for `export_models.sh`; the `rag`
Domino image reads the source ids from the installed package.

**Export** (`make ovms-export` = `./ovms/export_models.sh [MODELS_DIR]`, default
`ovms/models`; run `make install` first so `rag-served-models` is on `PATH`, or
the script stops and says so): takes the served models from `rag-served-models`,
downloads `export_model.py` and its `requirements.txt` from the OVMS repository at tag `v${OVMS_VERSION}` into `<MODELS_DIR>/.export/` (cached),
`pip install`s them into `$PYTHON` (set `PYTHON=/path/to/venv/bin/python` for a
separate environment), then runs `rerank_ov` and `embeddings_ov --pooling CLS`
with `--weight-format int8`. The result is `config_reranker.json`,
`config_embedder.json` and one directory per model; the configs reference the
model directories relatively, so the tree can be mounted anywhere.

**Compose** (`make infra-up`) mounts `ovms/models` read-only at `/workspace` and
starts the reranker with `--rest_port 8001 --config_path /workspace/config_reranker.json`,
the embedder on `8002` with `config_embedder.json`. Verify:

```bash
curl -s localhost:8001/v2/health/ready && echo reranker ready
curl -s localhost:8001/v3/rerank -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-base","query":"vacation days","documents":["Employees get 25 vacation days.","Expense reports are due monthly."],"top_n":2}'
curl -s localhost:8002/v3/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"bge-small-en-v1.5","input":["vacation policy"]}'
```

**Contracts** used by `ovms.py`:

| Adapter | Request | Response |
| --- | --- | --- |
| `OvmsReranker` → `POST {RAG_OVMS_RERANK_URL}/v3/rerank` | `{"model","query","documents":[...],"top_n"}` | `{"results":[{"index","relevance_score"}]}` |
| `OvmsEmbedder` → `POST {RAG_OVMS_EMBEDDINGS_URL}/v3/embeddings` | `{"model","input":[...]}` (batches of 64) | `{"data":[{"index","embedding":[...]}]}` |

Both re-order results by `index`, time out after 30 s and retry twice on
transport errors, 5xx and 401 (the Domino token is refetched first); other 4xx
fail immediately. Readiness probes `GET /v3/models/{model}`.

**Bumping OVMS**: change `ovms/versions.env` only, re-run `make ovms-export`
(the exporter is fetched from the new tag), rebuild the `ovms` Domino image and
refill the models Dataset. **Changing a served model**: change its default in
`settings.py` only, then re-run `make ovms-export`, rebuild the `rag` Domino
image (it pre-downloads the source ids) and refill the models Dataset.

## Deploying on Domino 6.2

Every app is published with the same entry point, `domino/app.sh`,
which `cd`s into the project root, reads its role from `RAG_ROLE` and listens on
`0.0.0.0:8888` (a Domino requirement). Only environment variables differ between
topologies, so the Locust numbers stay comparable.

| Topology | App | `RAG_ROLE` | Environment variables | Image |
| --- | --- | --- | --- | --- |
| `monolith` | one app: workflow + embedder + reranker | `monolith` (default) | `OPENAI_API_KEY`, `RAG_ES_URL` (+ `RAG_ES_API_KEY`) | `rag` |
| `split-reranker` | **A** OVMS reranker | `ovms-reranker` | `RAG_OVMS_MODELS_DIR` when the Dataset is not at `/mnt/data/ovms-models`; `RAG_OVMS_PREFIX_PROXY=1` if the prefix reaches OVMS | `ovms` |
| | **B** workflow + embedder | `workflow` | `RAG_OVMS_RERANK_URL=<URL of A>`, `OPENAI_API_KEY`, `RAG_ES_URL` | `rag` |
| `split-all` | **A** OVMS reranker | `ovms-reranker` | as above | `ovms` |
| | **C** OVMS embedder | `ovms-embedder` | as above | `ovms` |
| | **B** workflow | `workflow` | `RAG_TOPOLOGY=split-all`, `RAG_OVMS_RERANK_URL=<URL of A>`, `RAG_OVMS_EMBEDDINGS_URL=<URL of C>`, `OPENAI_API_KEY`, `RAG_ES_URL` | `rag` |
| FastAPI variants | **A'** / **C'** the same models behind `rag-model-server` | `fastapi-reranker` / `fastapi-embedder` | none (weights from `RAG_RERANKER_MODEL` / `RAG_EMBEDDER_MODEL`); point B's `RAG_OVMS_*_URL` at these apps instead | `rag` |

Both workflow app roles run `rag-serve --domino`. A role only defaults
`RAG_TOPOLOGY` (`monolith` for `monolith`, `split-reranker` for `workflow`) and
refuses a topology it does not serve (`monolith` serves only `monolith`;
`workflow` serves `split-reranker` and `split-all`). Python then resolves the
backends from the topology, requires `RAG_OVMS_RERANK_URL` /
`RAG_OVMS_EMBEDDINGS_URL` for each port on a model server and defaults
`RAG_OVMS_AUTH=domino`. `RAG_RERANKER_BACKEND=none` and `fake` backends pass
through untouched. The run label ends up in every response, so set a distinct
`RAG_RUN_LABEL` when testing several hardware tiers or model servers.

**Environments.** `domino/Dockerfile` has two targets built from the project
root (commands in its header): `rag` installs this package with the
`[loadtest,domino]` extras on `quay.io/domino/compute-environment-images:ubuntu22-py3.10-r4.4-domino6.2-standard`,
pre-downloads both HF models and sets `HF_HUB_OFFLINE=1`; `ovms` unpacks the
pinned binary tarball under `/opt/ovms`. Push both images and create two Domino
Environments from them (custom base image), or paste one stage's instructions
into the Dockerfile Instructions field. Workspaces that ingest or run Locust use
the `rag` environment.

**Models Dataset.** The OVMS roles read `config_reranker.json` /
`config_embedder.json` from `RAG_OVMS_MODELS_DIR` (default `/mnt/data/ovms-models`)
and exit with a clear message when the file is missing. Create a Dataset named
`ovms-models` (Data → Datasets; git-based projects mount it at
`/mnt/data/ovms-models`, DFS projects at `/domino/datasets/local/ovms-models`, so
set `RAG_OVMS_MODELS_DIR` there) and fill it from a Workspace with
`./ovms/export_models.sh /mnt/data/ovms-models`, or upload the contents of a
local `ovms/models/`. Apps inherit the project's Dataset mounts.

**Elasticsearch and corpus.** Put `RAG_ES_URL` (and `RAG_ES_API_KEY`) of a cluster the
apps can reach in Project → Settings → Environment variables, then run `make ingest` once from a Workspace.

**Publishing.** For each app (Apps → Publish): entry point
`domino/app.sh`, the environment from the table, a CPU tier with
at least 4 cores (the int8 OVMS models and the in-process models are CPU-bound;
8 cores for the monolith is a fair baseline) and the variables above (project
variables are inherited by every app, so use one project per app when two apps
need different values). Publish the OVMS apps first, copy their URLs from the
Apps view, then publish the workflow app. Share the OVMS apps with the owner of
the workflow app's run and the workflow app with whoever runs Locust ("Anyone
with an account" is simplest). Check `GET <App URL>/readyz` before every run.

**App-to-app auth.** Domino apps only accept authenticated requests. Inside a
run a short-lived bearer token is served at `http://localhost:8899/access-token`;
with `RAG_OVMS_AUTH=domino` the workflow app fetches it, caches it for 4 minutes,
refreshes it on `401` and sends it as `Authorization: Bearer` on every OVMS
call. Set `RAG_OVMS_AUTH=<token>` to use a fixed token instead.

**Proxy prefix (`DOMINO_RUN_HOST_PATH`).** Domino serves an app under a path
prefix (`/apps/<app-id>/` in 6.2), exposed to the process as `DOMINO_RUN_HOST_PATH`,
and its proxy normally strips it before the request reaches the process. The
FastAPI apps (`rag-serve`, `rag-model-server`) set that value as `root_path`, so
they answer whether or not the prefix is on the wire, with no proxy layer in
front. OVMS cannot: its routes are fixed at `/v3/...` and it has no base-path
option. Check once, from a Workspace, what reaches the OVMS app:

```bash
curl -i -H "Authorization: Bearer $(curl -s http://localhost:8899/access-token)" \
  https://<domino-host>/apps/<ovms-app-id>/v3/models/bge-reranker-base
```

`200`: the prefix is stripped, nothing to do. `404`: the prefix reaches OVMS; set
`RAG_OVMS_PREFIX_PROXY=1` on the OVMS apps. `app.sh` then starts OVMS on
`127.0.0.1:9000` and nginx (in the `ovms` image) on `0.0.0.0:8888`, with the
settings in `domino/ovms-proxy.conf` (`__PREFIX__` is replaced at start-up):
`location <prefix>/ { proxy_pass http://ovms/; }` drops the prefix and forwards
over keep-alive HTTP/1.1. OVMS itself is unchanged (same binary,
config and threads); the cost is one loopback hop, well under a millisecond, plus
a little CPU on the same pod. Note it in the comparison and keep it identical
across the OVMS runs. Callers outside prefix every route with the App URL
(`https://<domino-host><DOMINO_RUN_HOST_PATH>/query`); `RAG_OVMS_RERANK_URL` /
`RAG_OVMS_EMBEDDINGS_URL` are the App URLs without a trailing slash.

**Tracing and agent deployment.** Agent deployments use the same hosting as
Apps (same `app.sh`, same `0.0.0.0:8888` rule) plus tracing through
`add_tracing` from `dominodatalab[agents]` (the `[domino]` extra, in the `rag`
image). With `RAG_DOMINO_TRACING=true` the workflow run behind both endpoints is
wrapped with `add_tracing(name="rag_query", autolog_frameworks=["langchain"])`
(`domino.aisystems.tracing`, falling back to `domino.agents.tracing`), so the
LangGraph run and the OpenAI call appear as traces; without the SDK the app logs
a warning and runs untraced. Deploy from Experiment Manager → **Deploy Agent**
with the same entry point, environment and tier as an App, and keep the flag
identical across compared runs (tracing adds a small per-request overhead).

**Running Locust.** From a Workspace in the same project (`rag` environment):

```bash
export RAG_LOADTEST_BEARER_TOKEN=$(curl -s http://localhost:8899/access-token)
make loadtest RUN_NAME=monolith        HOST=https://<domino-host><path of the monolith app>
make loadtest RUN_NAME=split-reranker  HOST=https://<domino-host><path of workflow app B>
make loadtest RUN_NAME=split-all       HOST=https://<domino-host><path of workflow app B, split-all project>
make compare                           # results/comparison.md
```

The run token expires after about 5 minutes and Locust reads it once per user at
start, so for longer runs (`RUN_TIME=10m`, `RAG_LOADTEST_SHAPE=step`) or from a laptop
use an API key instead: `export DOMINO_API_KEY=<Account settings → API Key>` (sent as
`X-Domino-Api-Key`). Keep `USERS`, `SPAWN_RATE`, `RUN_TIME` and the tiers identical across topologies.

| Symptom | Cause / fix |
| --- | --- |
| `401`/`403` from Locust | missing or expired `RAG_LOADTEST_BEARER_TOKEN` / `DOMINO_API_KEY`, or the app is not shared with that user |
| `404` on `/query` | `HOST` lacks the App path prefix, or has a trailing slash |
| `503 reranker_unavailable` / `embedder_unavailable` | OVMS app not ready (Dataset missing, wrong `RAG_OVMS_*_URL`, token rejected); check its logs and `GET <OVMS App URL>/v2/health/ready` |
| `503 vector_store_unavailable` / `/readyz` not ready | `RAG_ES_URL` unreachable or wrong credentials, or `make ingest` never ran against that cluster |
| `rag-serve: error: ...` in a workflow app's log | a settings mistake: a `RAG_OVMS_*_URL` missing for a port on a model server, or a backend that contradicts `RAG_TOPOLOGY`; the message names the variable and the fix |
| `app.sh[...]: RAG_TOPOLOGY='...' is not served by this role` | the role and the topology disagree: `monolith` serves `monolith`, `workflow` serves `split-reranker` and `split-all` |
| `app.sh[ovms-*]: OVMS config not found` | Dataset not mounted at `/mnt/data/ovms-models`; set `RAG_OVMS_MODELS_DIR` |
| `404` from an OVMS app on `/v3/...` | Domino forwards the path prefix; set `RAG_OVMS_PREFIX_PROXY=1` on that app (see Proxy prefix) |

### Step by step: first test run on Domino

Everything below is what changes between this repository and your Domino
instance. Replace the placeholders once; the code itself does not change.

| Placeholder / value | Where it goes |
| --- | --- |
| `<domino-host>` | your Domino URL, e.g. `https://domino.example.com` |
| `<registry>` | a registry Domino can pull environments from (only for step 1a) |
| `RAG_ES_URL`, `RAG_ES_API_KEY` | project environment variables, every project |
| `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for a gateway), or `RAG_LLM_BACKEND=fake` | project environment variables of the monolith and workflow projects |
| `RAG_ROLE` | project environment variable, one value per project (see step 2) |
| `RAG_OVMS_RERANK_URL`, `RAG_OVMS_EMBEDDINGS_URL` | workflow project, set after step 5 publishes the model apps |
| `RAG_OVMS_MODELS_DIR` | OVMS projects, only when the Dataset is not mounted at `/mnt/data/ovms-models` |
| `RAG_OVMS_PREFIX_PROXY=1` | OVMS projects, only if step 6 returns `404` |
| hardware tier | the same CPU tier (4+ cores) for every model app; one for the workflow apps |

**1. Environments (once).** Either build and push both images from the
project root, then create two Environments from them as custom base images:

```bash
docker build -f domino/Dockerfile --target rag  -t <registry>/rag-load-test-rag  .
docker build -f domino/Dockerfile --target ovms -t <registry>/rag-load-test-ovms .
docker push <registry>/rag-load-test-rag && docker push <registry>/rag-load-test-ovms
```

or (1b, no registry) create two Environments on the Domino Standard Environment
and paste the instructions of one `domino/Dockerfile` stage (everything after its
`FROM`) into each Environment's Dockerfile Instructions. The `rag` build downloads
the two HF models, so it needs Hub access at build time.

**2. Projects (one per app).** A Domino App inherits its project's environment
variables and each app needs its own `RAG_ROLE`, so import this git repository into
one project per app and set, in Project → Settings → Environment variables:

| Project | `RAG_ROLE` | Environment | Other variables |
| --- | --- | --- | --- |
| `rag-monolith` | `monolith` | `rag` | `RAG_ES_URL`, `RAG_ES_API_KEY`, `OPENAI_API_KEY` |
| `rag-ovms-reranker` | `ovms-reranker` | `ovms` | `RAG_OVMS_MODELS_DIR` if needed |
| `rag-ovms-embedder` | `ovms-embedder` | `ovms` | `RAG_OVMS_MODELS_DIR` if needed |
| `rag-fastapi-reranker` | `fastapi-reranker` | `rag` | none |
| `rag-fastapi-embedder` | `fastapi-embedder` | `rag` | none |
| `rag-workflow` | `workflow` | `rag` | `RAG_ES_URL`, `RAG_ES_API_KEY`, `OPENAI_API_KEY`, `RAG_OVMS_RERANK_URL` (+ `RAG_TOPOLOGY=split-all`, `RAG_OVMS_EMBEDDINGS_URL` for split-all), `RAG_RUN_LABEL` for a variant |

Skip the rows you do not test. Keep `RAG_DOMINO_TRACING` identical across projects
(unset, or `true` everywhere).

**3. OVMS models Dataset (once, OVMS projects only).** In `rag-ovms-reranker`
create a Dataset `ovms-models`, open a Workspace with the `rag` environment and run:

```bash
./ovms/export_models.sh /mnt/data/ovms-models      # needs HF Hub access; ~10 min on CPU
ls /mnt/data/ovms-models                            # config_reranker.json, config_embedder.json, model folders
```

Mount the same Dataset in `rag-ovms-embedder` (Data → Datasets → mount from another
project). DFS projects mount it at `/domino/datasets/local/ovms-models`: set
`RAG_OVMS_MODELS_DIR` there. The FastAPI projects need no Dataset.

**4. Index the corpus (once per Elasticsearch cluster).** From a Workspace in
`rag-monolith` (`rag` environment, `RAG_ES_URL` set):

```bash
make ingest                                         # 200 synthetic documents, local embedder
curl -s "$RAG_ES_URL/rag_passages/_count"          # {"count":200,...}
```

Every app reads this one index; the serving embedder must be the one used here
(`RAG_EMBEDDER_MODEL`, default `BAAI/bge-small-en-v1.5`, both locally and on OVMS).

**5. Publish the apps (Apps → Publish, entry point `domino/app.sh`).**
Order matters for the split topologies:

1. model apps first: `rag-ovms-reranker`, `rag-ovms-embedder` (and/or the two
   FastAPI projects); wait for "Running" and copy each App URL from the Apps view;
2. put those URLs, without a trailing slash, into `rag-workflow`'s
   `RAG_OVMS_RERANK_URL` / `RAG_OVMS_EMBEDDINGS_URL`, set `RAG_TOPOLOGY`
   (`split-reranker` or `split-all`) and, for a variant, `RAG_RUN_LABEL`
   (`split-reranker-fastapi`, ...), then publish it;
3. publish `rag-monolith`.

Share every model app with the owner of `rag-workflow` (its run token is the
identity that calls them) and share the monolith and workflow apps with whoever
runs Locust. "Anyone in Domino" is the simplest setting for a test instance.

**6. Verify before measuring.** From any Workspace:

```bash
TOKEN=$(curl -s http://localhost:8899/access-token)
curl -i -H "Authorization: Bearer $TOKEN" <URL of A>/v3/models/bge-reranker-base    # 200
curl -s -H "Authorization: Bearer $TOKEN" <URL of B>/readyz                          # every check ok
curl -s -H "Authorization: Bearer $TOKEN" -X POST <URL of B>/query \
  -H 'Content-Type: application/json' -d '{"question":"How many paid vacation days do employees accrue?"}'
```

A `404` on the first call means Domino forwards the path prefix to OVMS: set
`RAG_OVMS_PREFIX_PROXY=1` in both OVMS projects and republish those two apps
(see "Proxy prefix" above); the FastAPI apps need nothing. `/readyz` lists which
dependency is missing when it answers `503`.

**7. Load test (Workspace in `rag-monolith`, `rag` environment).** The run token
expires after about 5 minutes, so export it right before each run:

```bash
export RAG_LOADTEST_BEARER_TOKEN=$(curl -s http://localhost:8899/access-token)
make loadtest RUN_NAME=monolith        HOST=<URL of the monolith app>
make loadtest RUN_NAME=split-reranker  HOST=<URL of the workflow app>      # RAG_TOPOLOGY=split-reranker
make loadtest RUN_NAME=split-all       HOST=<URL of the workflow app>      # after switching it to split-all
make compare                                                               # results/comparison.md
```

For experiment 2 aim Locust at the model apps themselves:

```bash
make loadtest-models RUN_NAME=rerank-ovms    HOST=<URL of the OVMS reranker app>    TARGET=rerank
make loadtest-models RUN_NAME=rerank-fastapi HOST=<URL of the FastAPI reranker app> TARGET=rerank
rag-compare results/rerank-ovms results/rerank-fastapi --out results/rerank.md
```

`results/` is git-ignored: download the CSV/HTML files from the Workspace or copy
them into a Dataset before stopping it.

**8. What to change between runs, and nothing else.**

| Experiment | Change |
| --- | --- |
| monolith vs split-reranker vs split-all | which app `HOST` points at; the workflow project's `RAG_TOPOLOGY`, `RAG_OVMS_*_URL` |
| FastAPI vs OVMS behind the workflow | the workflow project's `RAG_OVMS_*_URL` (and a distinct `RAG_RUN_LABEL`) |
| FastAPI vs OVMS directly | `HOST` and `RUN_NAME` of `make loadtest-models` |
| cost of reranking | `RAG_RERANKER_BACKEND=none` on the monolith or workflow project |
| remove OpenAI latency | `RAG_LLM_BACKEND=fake` on the monolith and workflow projects |

Keep `USERS`, `SPAWN_RATE`, `RUN_TIME`, the hardware tiers, the corpus and
`RAG_DOMINO_TRACING` identical across the runs you compare.

## Development

```bash
make test          # pytest: fakes + httpx.MockTransport only; no weights, no network, no Elasticsearch
make lint          # ruff check + black --check over src tests locustfile.py
make format        # ruff --fix + black
make typecheck     # mypy src
```

`tests/conftest.py` sets `LOCUST_SKIP_MONKEY_PATCH=1` before any `import locust`
so the asyncio-based service tests are never gevent-patched.
