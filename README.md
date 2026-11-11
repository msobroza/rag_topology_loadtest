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

Around that comparison every stage has alternatives, each one setting away: the
vector store (Elasticsearch or Chroma), how it is searched (dense, lexical or
hybrid), a hosted API in place of a model of ours (OpenAI embeddings; a Cohere or
OpenAI-compatible rerank route) and what runs the in-process models (OpenVINO or
PyTorch). See [Retrieval, stores and hosted models](#retrieval-stores-and-hosted-models).

To run it on Domino, follow [Step by step: first test run on Domino](#step-by-step-first-test-run-on-domino).

## Workflow

```mermaid
flowchart LR
    L[Locust] -->|POST /query, /retrieve| API[FastAPI · rag-serve]
    API --> E["embed<br/>(EmbedderPort: in-process, OVMS /v3/embeddings or the OpenAI API)"]
    E --> R["retrieve<br/>(VectorStorePort: Elasticsearch or Chroma; dense, lexical or hybrid; RAG_TOP_K_RETRIEVE=20)"]
    R --> K["rerank<br/>(RerankerPort: in-process, OVMS /v3/rerank or a hosted rerank route; RAG_TOP_K_RERANK=5)"]
    K -->|mode = query| G["generate<br/>(ChatModelPort: OpenAI)"]
    K -->|mode = retrieve| Z([response + timings_ms])
    G --> Z
```

Each node is timed; responses carry `timings_ms{embed,retrieve,rerank,generate,total}`
and Locust turns those into per-stage statistics. `RagWorkflow(deps, parameters)`
takes the four ports and one `WorkflowParameters` (retrieval mode, both top-k
sizes, the query prefix and the prompts), built by
`RagSettings.workflow_parameters()`. The prompts live in
`src/rag_load_test/config/prompts.yaml`. With `RAG_RETRIEVAL_MODE=lexical` the
graph starts at retrieve: no vector is computed and `embed` is 0 ms.

## Code layout

```text
src/rag_load_test/
  __init__.py       package docstring, __version__
  domain/           no I/O, no framework; imports no other layer
    passages.py     Passage, ScoredPassage, RagResult, STAGES, top_by_score
    ports.py        the four async ports (abstract base classes) and RagDependencies
    errors.py       RagDependencyError, describe_error
    prompts.py      RagPrompts: system prompt + user template -> chat messages
    workflow_parameters.py  WorkflowParameters: retrieval mode, top-k sizes, query prefix and prompts for one workflow
    bm25.py         Bm25Index: ranked lexical search in memory, for the stores that have none of their own
    rank_fusion.py  reciprocal_rank_fusion: what a hybrid retrieval ends with
    thread_budget.py  ThreadBudget: how the virtual CPUs are split between model calls
    topology.py     the topology rule: backends per topology, contradictions, default run label
    corpus.py       synthetic handbook corpus, its questions, chunking
  application/      use cases over the ports; imports domain only
    workflow.py     RagWorkflow (LangGraph graph with timed nodes)
    ingest.py       embed passages in batches and upsert them
  config/           imports domain only
    settings.py     RagSettings: every RAG_* variable and tunable (pydantic-settings, .env and YAML aware)
    prompts.yaml    the system prompt and user template (RAG_PROMPTS_FILE picks another file)
    domino_rules.py apply_domino_rules: settings made fit for a Domino App (rag-serve --domino)
  adapters/         one subfolder per kind of outside system
    rag_ports.py    RagPorts, the composition root: settings in, ready ports out; owns the thread budget and pool, HTTP client, bearer token and their shutdown
                    (adapters take sizes, timeouts and names as constructor arguments; only rag_ports.py reads settings)
    local_models/   models loaded in this process
      sentence_transformer_models.py  SentenceTransformerEmbedder, CrossEncoderReranker, NoReranker, the model thread pool
      local_model.py  LocalModel: which files a model loads from and what runs it (OpenVINO or PyTorch)
      openvino_export.py  convert one embedder or reranker to OpenVINO files
      native_threads.py  NativeThreadScope: threadpoolctl scopes, one per model
      cpu_count.py  where the CPU count comes from: RAG_MODEL_CPUS, the Domino hardware tier, the container
    ovms/           models behind a model server
      ovms_adapters.py  BearerToken (static or Domino run token), OvmsEmbedder, OvmsReranker (httpx, timeout + retries)
      v3_contract.py  the /v3 contract: paths, request/response shapes (pydantic), index bookkeeping, UNKNOWN_MODEL_ERROR
    hosted_models/  models behind a hosted API
      openai_embedder.py  OpenAIEmbedder (the OpenAI embeddings API, or anything compatible)
      rerank_api.py   RerankApiReranker (the Cohere rerank shape: Cohere, or a gateway's /rerank)
    vector_store/
      elasticsearch_store.py  ElasticsearchVectorStore (kNN, and BM25 for lexical search)
      chroma_store.py  ChromaVectorStore (embedded or server; BM25 kept in this process)
    chat_model/
      openai_chat.py  OpenAIChatModel
    fakes/          the `fake` backends
      fake_models.py  FakeEmbedder, FakeReranker, FakeChatModel
      in_memory_vector_store.py  InMemoryVectorStore
    shared/
      http_policy.py  HttpPolicy: timeout, retries and backoff shared by every adapter that calls out
      json_post.py    RetryingJsonPost: the POST-with-retries the model-server and rerank adapters share
      json_log.py   log_json: the one log format
  entrypoints/      processes: FastAPI apps and CLIs
    workflow_api.py FastAPI app (/query /retrieve /healthz /readyz), Domino tracing, JSON logs, rag-serve
    model_server.py FastAPI model server with the OVMS /v3 contract (rag-model-server)
    served_models.py  rag-served-models: the served models from the settings defaults, as shell assignments
    export_openvino.py  rag-export-openvino: convert the in-process models to OpenVINO files
    ingest_cli.py   rag-ingest: JSONL loader, chunk, embed and upsert a corpus
  loadtest/
    request_plans.py  locust-free helpers (config, auth headers, request pool and plans, verdicts, target record)
    run_comparison.py rag-compare: Locust *_stats.csv files to markdown tables
  deploy/           rag-deploy on Domino 6.2
    settings.py     DeploySettings: deploy.yaml (shared defaults, one block per App), RAG_DEPLOY_* overrides
    publish.py      which Apps a topology needs, their entry scripts, App or Agent, and publishing them in order
    endpoint_url.py where an App answers: {host}/{owner}/{project}/r/notebookSession/{instance id}
    app_id_cache.py the App ids already seen (.rag-deploy-cache.json): an App is redeployed, never created twice
    status.py       rag-deploy status: one line per App, then the workflow App's /readyz
    cli.py          rag-deploy: the parser, url, --dry-run
    apps_client.py  the Domino 6.2 beta Apps API calls rag-deploy makes (find, create, add a version, get, list instances) and the wait until Running
src/fake_domino/    a stand-in for Domino 6.2's Apps API: rag-deploy's test double and what --dry-run runs.
                    It imports nothing from rag_load_test, and only deploy/cli.py imports it (both checked by
                    tests/structure/test_layering.py)
  api_shapes.py     the Apps API request/response shapes, transcribed from the 6.2 reference (what the fake serves)
  state.py          FakeDominoApps: the fake's Apps, versions and instances
  instances.py      InstanceScheduler: what decides an instance's status; by default the --behaviour scripts
  apps_routes.py    /api/apps/beta/apps: list, create, add a version, get, list a version's instances
  stand_ins.py      hardware tiers, /access-token and an App instance's /readyz
  server.py         create_fake_domino (FastAPI, Swagger at /docs), the in-process client, rag-fake-domino
  emulation/        TEMPORARY: the emulated Domino (proxy, App placement on machines), see emulation/README.md
emulation/          TEMPORARY: Terraform and scripts for an emulation of Domino on Google Cloud
locustfile.py       RagUser (FastHttpUser) + opt-in StepLoadShape; `locust` finds it in this directory
docker-compose.yml  Elasticsearch 9.0.0 (:9200) + OVMS reranker (:8001) + OVMS embedder (:8002)
ovms/               versions.env (single OVMS version pin), export_models.sh, models/ (exported, git-ignored)
domino/             app.sh (one entry point, role from RAG_ROLE), Dockerfile (targets `rag` and `ovms`)
tests/              pytest: no model weights, no network, no Elasticsearch; one folder per layer of src/
  domain/ application/ config/ adapters/ entrypoints/ loadtest/ deploy/   the tests of that sub-package
  scripts/          domino/app.sh and ovms/export_models.sh, run against fake binaries
  structure/        the layering and code-shape rules, read from the source with ast
  fake_domino/      the fake Domino's own tests; they import nothing from rag_load_test
  emulation/        TEMPORARY: the emulation's Terraform, parsed, and the rules that keep it removable
  helpers/          named fakes, fixtures and REPO_ROOT shared by several test files (on pytest's path)
```

## Quickstart (monolith on a laptop)

```bash
make install-dev              # pip install -e ".[dev]" (locust, pytest, ruff, black, mypy)
cp .env.example .env          # set OPENAI_API_KEY, or RAG_LLM_BACKEND=fake to skip the LLM
make infra-up                 # docker compose: Elasticsearch on :9200 (the OVMS containers need `make ovms-export` first)
make openvino-export          # once: convert both models to OpenVINO under models/openvino (2.7 GB of downloads, ~4.5 GB on disk)
make ingest                   # rag-ingest --synthetic 200 -> index rag_passages
make serve                    # rag-serve on http://0.0.0.0:8888
```

The project runs on Python 3.11, named once in `.python-version`. The `make`
targets use a `.venv` in this directory when there is one, without activating it
(`uv venv`, which reads `.python-version`, or `python3.11 -m venv .venv`, before
`make install-dev`); otherwise they use the commands on your `PATH`.

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

A request counts as a success only when the response is HTTP 200 with a JSON
object body. Any other status fails as `HTTP <status>: <first 200 characters of
the body>`, and a 200 whose body is not JSON, or is JSON but not an object, also
counts as a failure (it used to vanish from the statistics altogether).
Every successful response's `timings_ms` is re-emitted as Locust `STAGE` rows
(`/query:embed`, `/query:retrieve`, `/query:rerank`, `/query:generate`, same for
`/retrieve`; a stage timed at zero gets no row), so p50/p95/p99 per stage land
in the same CSV as the endpoint totals.

Each run also writes a target record, `results/<RUN_NAME>_target.json` (next to
the `--csv` files; nothing is written when Locust runs without `--csv`). Against
the workflow app it counts responses per `X-RAG-Topology` / `X-RAG-Run-Label`
pair the app reported, most responses first:

```json
{"target": "rag", "seen": [{"topology": "monolith", "run_label": "monolith", "responses": 2400}]}
```

A run straight at a model server (`make loadtest-models`) writes
`{"target": "rerank", "seen": []}` (or `embeddings`), since model servers send
no such headers.

`make compare` feeds every `results/*_stats.csv` prefix (sorted, so
`monolith` is the baseline) to `rag-compare`, which writes one markdown table
per endpoint and per stage: requests, failure %, RPS, p50, p95, p99 and Δp95
versus the first run. Pick another baseline by calling it directly:
`rag-compare results/split-all results/monolith --out results/comparison.md`.
When a run has a target record, the intro line names the topology and run label
it saw beside the run name, and a warning line
flags any run that saw more than one pair, because its numbers mix
configurations (for example an app redeployed mid-run). A run without the file
is listed by its run name alone, as before.

## Experiment 2: serving the models with FastAPI versus OVMS

`rag-model-server` serves one model per process behind the same `/v3` contract
as OpenVINO Model Server (`POST /v3/embeddings`, `POST /v3/rerank`,
`GET /v3/models/<alias>`), using the in-process sentence-transformers adapters.
The workflow app cannot tell the two servers apart, so the comparison needs no
code change: point `RAG_OVMS_RERANK_URL` / `RAG_OVMS_EMBEDDINGS_URL` at one or
the other. Both servers answer a request for an unknown model with the same
`404 {"error": "Mediapipe graph definition with requested name is not found"}`.

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
| `RAG_RUN_LABEL` | the default run label | names the measured configuration, echoed as `run_label` and `X-RAG-Run-Label`; default: the topology plus `+<port>=<backend>` for each backend off its default, then `+retrieval=<mode>` when not `dense` (e.g. `monolith+reranker=none`, `monolith+store=chroma+retrieval=hybrid`); letters, digits and `._+=:-`, 1 to 128 characters |
| `RAG_EMBEDDER_BACKEND` | from the topology | only `openai` or `fake` (or the topology's own value) may override it |
| `RAG_EMBEDDER_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | HF id (or a folder) for `local`, and what OVMS serves |
| `RAG_EMBEDDER_QUERY_PREFIX` | `query: ` | put in front of each question before `RAG_EMBEDDER_MODEL` embeds it (arctic-embed expects it on queries, never on passages); a `local` or `ovms` embedder gets it, an `openai` or `fake` one never does; set it empty for a model that takes none |
| `RAG_RERANKER_BACKEND` | from the topology | only `cohere`, `openai`, `fake` or `none` (or the topology's own value) may override it |
| `RAG_RERANKER_MODEL` | `antoinelouis/crossencoder-camembert-base-mmarcoFR` | HF id (or a folder) for `local`, and what OVMS serves |
| `RAG_LOCAL_MODEL_RUNTIME` | `openvino` | what runs the in-process models: `openvino` \| `torch` |
| `RAG_OPENVINO_MODELS_DIR` | `models/openvino` | where `rag-export-openvino` writes and the `openvino` runtime looks first, one folder per model |
| `RAG_OVMS_EMBEDDINGS_URL` | `http://localhost:8002` | base URL (App URL on Domino); the adapter appends `/v3/embeddings` |
| `RAG_OVMS_EMBEDDINGS_MODEL` | `snowflake-arctic-embed-l-v2.0` | servable name in the OVMS config |
| `RAG_OVMS_RERANK_URL` | `http://localhost:8001` | base URL; the adapter appends `/v3/rerank` |
| `RAG_OVMS_RERANK_MODEL` | `crossencoder-camembert-base-mmarcoFR` | servable name in the OVMS config |
| `RAG_OVMS_AUTH` | `` | empty = no auth; `domino` = the run's token from `RAG_DOMINO_TOKEN_URL`; anything else = a static bearer token |
| `RAG_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | embedding model for `RAG_EMBEDDER_BACKEND=openai` |
| `RAG_COHERE_RERANK_URL` | `https://api.cohere.com/v2/rerank` | rerank route for `RAG_RERANKER_BACKEND=cohere`; any server with the Cohere rerank shape works |
| `RAG_COHERE_RERANK_MODEL` | `rerank-v3.5` | model asked of that route |
| `RAG_OPENAI_RERANK_MODEL` | `` | model asked of `{OPENAI_BASE_URL}/rerank` for `RAG_RERANKER_BACKEND=openai`; required with it |
| `RAG_LLM_BACKEND` | `openai` | `openai` \| `fake` (echoes the question, no network) |
| `RAG_OPENAI_MODEL` | `gpt-4o-mini` | chat model id |
| `RAG_LLM_TEMPERATURE` | `0` | sampling temperature of the chat model, 0 to 2 |
| `RAG_VECTOR_STORE_BACKEND` | `elasticsearch` | `elasticsearch` \| `chroma` \| `fake` (in-memory, seeded with the synthetic corpus at start-up) |
| `RAG_RETRIEVAL_MODE` | `dense` | `dense` (by vector) \| `lexical` (BM25, no embed stage) \| `hybrid` (both, fused by reciprocal rank) |
| `RAG_HYBRID_RRF_K` | `60` | rank constant of the hybrid fusion (>= 1); smaller widens the gap between ranks |
| `RAG_CHROMA_URL` | `` | a Chroma server, e.g. `http://localhost:8000` (name the port); empty = embedded Chroma at `RAG_CHROMA_PATH` |
| `RAG_CHROMA_PATH` | `chroma` | folder of the embedded Chroma store |
| `RAG_CHROMA_COLLECTION` | `rag_passages` | Chroma collection holding the passages |
| `RAG_BM25_K1` / `RAG_BM25_B` | `1.2` / `0.75` | BM25 of the stores without ranked text search of their own (`chroma`, `fake`); Elasticsearch uses its own |
| `RAG_ES_URL` | `http://localhost:9200` | `http://user:password@host:9200` for basic auth |
| `RAG_ES_API_KEY` | `` | Elasticsearch API key (alternative to basic auth) |
| `RAG_ES_INDEX` | `rag_passages` | Elasticsearch index holding the passages |
| `RAG_ES_TEXT_ANALYZER` | `standard` | analyzer of the passages' text, used by lexical and hybrid search: `standard`, or a language's such as `french` (stemming, elision, stop words). An index keeps the analyzer it was created with, so changing it means ingesting into a new index |
| `RAG_DOMINO_TRACING` | `false` | wrap the workflow run with Domino `add_tracing` |
| `RAG_TOP_K_RETRIEVE` | `20` | candidates from the vector store (>= 1) |
| `RAG_TOP_K_RERANK` | `5` | passages kept after reranking and sent to the LLM (>= 1, <= `RAG_TOP_K_RETRIEVE`) |
| `RAG_SYSTEM_PROMPT` / `RAG_USER_PROMPT` | from `prompts.yaml` | override one prompt; the user template must hold `{context}` and `{question}` |
| `RAG_PROMPTS_FILE` | the packaged `config/prompts.yaml` | another YAML file with `system_prompt` and `user_prompt` (environment only, not `.env`) |
| `RAG_EMBEDDER_BATCH_SIZE` | `64` | texts per batch for the in-process embedder and per OVMS `/v3/embeddings` request |
| `RAG_RERANKER_BATCH_SIZE` | `32` | pairs per batch for the in-process cross-encoder |
| `RAG_MODEL_WORKERS` | `4` | at most this many in-process inference calls at once (fewer on a small CPU budget) |
| `RAG_MODEL_CPUS` | `0` | virtual CPUs for the thread budget; `0` = find out (Domino tier, else container), a positive value wins |
| `RAG_EMBEDDER_THREADS` / `RAG_RERANKER_THREADS` | `0` | native (OpenMP/BLAS) threads one embed / rerank call may use; `0` = the budget's share |
| `DOMINO_API_PROXY`, `DOMINO_HARDWARE_TIER_ID` | set by Domino | read as-is (no `RAG_` prefix) to look up the run's hardware tier |
| `RAG_HTTP_TIMEOUT_S` | `30` | timeout of every model-server and Domino-token request |
| `RAG_HTTP_RETRIES` | `2` | model-server retries on transport errors, 5xx and 401; hardware-tier lookup retries on transport errors |
| `RAG_HTTP_BACKOFF_S` | `0.2` | wait before the first retry, doubled for each next one |
| `RAG_DOMINO_TOKEN_URL` | `http://localhost:8899/access-token` | where a Domino run serves its token (`RAG_OVMS_AUTH=domino`) |
| `RAG_DOMINO_TOKEN_TTL_S` | `240` | how long a fetched Domino token is reused (tokens expire after about 5 minutes) |
| `RAG_FAKE_STORE_SEED_DOCUMENTS` | `200` | synthetic documents seeded into the `fake` vector store at start-up |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | | read by the OpenAI SDK; point `OPENAI_BASE_URL` at a Domino AI Gateway to swap providers. The `openai` embedder and reranker use the same two |
| `COHERE_API_KEY` | | sent as a bearer token to `RAG_COHERE_RERANK_URL`; empty sends none |
| `RAG_LOADTEST_BEARER_TOKEN`, `DOMINO_API_KEY` | | Locust side: auth header, see the load-test table |
| `RAG_LOADTEST_QUERY_WEIGHT`, `RAG_LOADTEST_RETRIEVE_WEIGHT`, `RAG_LOADTEST_SHAPE` | `1`, `3`, unset | Locust side: task weights and the step shape |
| `RAG_LOADTEST_QUESTIONS` | `50` | Locust side: distinct synthetic questions the users ask |
| `RAG_LOADTEST_QUESTIONS_FILE` | unset | Locust side: ask the questions of this file instead (JSON lines with a `question` key, as `rag-fetch-corpus` writes, or one question per line); `RAG_LOADTEST_QUESTIONS` then does not apply |

Every tunable is a field of `RagSettings` in `config/settings.py`; adapters get
the values as constructor arguments from `adapters/rag_ports.py`. Locust's rerank
users send `RAG_TOP_K_RETRIEVE` documents with `top_n=RAG_TOP_K_RERANK`, like the
workflow. Sources win in this order: constructor arguments, environment, `.env`,
then the prompts YAML. An invalid backend, topology, retrieval mode or runtime
name, a backend that contradicts the topology, a bad run label, a temperature
outside 0 to 2, `RAG_TOP_K_RERANK` above `RAG_TOP_K_RETRIEVE`, the `openai`
reranker without `OPENAI_BASE_URL` and `RAG_OPENAI_RERANK_MODEL`, a user prompt
missing a placeholder, or a missing prompts file fails at start-up with the
offending value.

**Thread budget.** When the first in-process model loads, the process counts its
virtual CPUs: `RAG_MODEL_CPUS` if positive; on Domino, the `cores` of the run's
hardware tier from `GET $DOMINO_API_PROXY/api/hardwaretiers/v1/hardwaretiers/$DOMINO_HARDWARE_TIER_ID`
(a 403 without `ViewHardwareTiers` only logs a warning); otherwise the
container's CPUs (`joblib.cpu_count()`, which honours the cgroup quota). Whole
CPUs are split as `workers = min(RAG_MODEL_WORKERS, cpus)` and
`threads = cpus // workers` per call (8 CPUs: 4 workers x 2 threads), each model
call running inside a threadpoolctl scope of that size. Autoscaled Apps start
more pods of the same tier, so every new process works its budget out again; the
result is one `thread_budget` JSON log line.

## Retrieval, stores and hosted models

**Retrieval mode** (`RAG_RETRIEVAL_MODE`). `dense` asks the store for the
passages nearest to the question's vector. `lexical` asks for the passages whose
words match best (BM25): no vector is needed, so the embed stage does not run
(`embed = 0 ms`; the embedder still loads, so `RAG_EMBEDDER_BACKEND=fake` saves
the weights). `hybrid` asks both ways at once, `RAG_TOP_K_RETRIEVE` each, and
fuses the two rankings by reciprocal rank (`1 / (RAG_HYBRID_RRF_K + rank)`,
summed per passage). The fusion runs in the workflow, so hybrid behaves the same
on every store and needs no Elasticsearch licence, at the price of two store
calls where a server-side fusion would make one. Passages are always ingested
with their vectors, so one index serves the three modes.

| Store | Dense | Lexical |
| --- | --- | --- |
| `elasticsearch` | kNN on the vector field | BM25 `match` on the `text` field |
| `chroma` | cosine, HNSW | BM25 kept in the workflow app's memory |
| `fake` | brute-force cosine | BM25 in memory |

**Chroma** (`RAG_VECTOR_STORE_BACKEND=chroma`, `pip install -e ".[chroma]"`).
Embedded by default: a folder (`RAG_CHROMA_PATH`), no server to run, filled by
`make ingest` like any store. `RAG_CHROMA_URL` points at a Chroma server instead.
Open-source Chroma filters text by substring but does not rank it (its ranked
Search API is Chroma Cloud only), so the workflow app builds a BM25 index from
the collection when it starts. That costs memory and start-up time in proportion
to the corpus, and a passage ingested after start-up is not found by `lexical`
or `hybrid` until the app restarts.

**Hosted model APIs.** They are allowed in any topology, like `fake` and `none`,
and name themselves in the default run label (`monolith+embedder=openai`):

| Setting | Calls | Needs |
| --- | --- | --- |
| `RAG_EMBEDDER_BACKEND=openai` | the OpenAI embeddings API with `RAG_OPENAI_EMBEDDING_MODEL` | `OPENAI_API_KEY`; `OPENAI_BASE_URL` for a gateway or another compatible server |
| `RAG_RERANKER_BACKEND=cohere` | `POST RAG_COHERE_RERANK_URL` with `RAG_COHERE_RERANK_MODEL` | `COHERE_API_KEY` |
| `RAG_RERANKER_BACKEND=openai` | `POST {OPENAI_BASE_URL}/rerank` with `RAG_OPENAI_RERANK_MODEL` | `OPENAI_BASE_URL` of a gateway that has that route (LiteLLM, vLLM, Infinity, Jina; api.openai.com has none) |

Both rerankers are one adapter: they send the Cohere rerank shape
(`{"model","query","documents","top_n"}` → `{"results":[{"index","relevance_score"}]}`),
time out after `RAG_HTTP_TIMEOUT_S` and retry `RAG_HTTP_RETRIES` times on
transport errors, 5xx and 429. A hosted embedder returns vectors of its own
size, so it needs its own index: ingest with the embedder you will serve with.
In a split topology `rag-deploy` still publishes the model server Apps the
topology names, even when a hosted backend leaves one unused.

**A French corpus.** The synthetic handbook is English. For a French,
domain-specific corpus, `rag-fetch-corpus quebec-insurance --out DIR` downloads
the Quebec automobile insurance corpus of GRAAL-Research (Beauchemin et al.,
NLLP 2024: about 1,200 passages and 82 questions) and writes `DIR/corpus.jsonl`
for `rag-ingest --corpus` and `DIR/questions.jsonl` for
`RAG_LOADTEST_QUESTIONS_FILE`. Its source repository states no licence: research
use, cite the paper, and commit none of it. With Elasticsearch, ingest it into
an index created with `RAG_ES_TEXT_ANALYZER=french`.

```bash
rag-fetch-corpus quebec-insurance --out data/quebec
RAG_ES_INDEX=rag_quebec RAG_ES_TEXT_ANALYZER=french rag-ingest --corpus data/quebec/corpus.jsonl
RAG_LOADTEST_QUESTIONS_FILE=data/quebec/questions.jsonl make loadtest RUN_NAME=monolith-fr HOST=http://localhost:8888
```

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
  topologies embed with `snowflake-arctic-embed-l-v2.0` (in-process or its int8 OVMS
  export), so one index serves every run. Its vectors have 1024 dimensions: an
  index filled by an earlier embedder (384 for `bge-small-en-v1.5`) cannot take
  them, so ingest into a new `RAG_ES_INDEX`.

## In-process models on OpenVINO

The in-process embedder and reranker (the `local` backend, and what
`rag-model-server` serves) run on OpenVINO by default
(`RAG_LOCAL_MODEL_RUNTIME=openvino`); `torch` runs them on PyTorch as before.
Neither default model ships OpenVINO files on the Hub, so they are converted
once:

```bash
make openvino-export          # rag-export-openvino: both models -> models/openvino/<model name>/
rag-export-openvino --only reranker --models-dir /domino/datasets/local/<dataset>/openvino
```

`rag-export-openvino` converts `RAG_EMBEDDER_MODEL` and `RAG_RERANKER_MODEL`
(`RAG_*` overrides honoured) into one folder each under
`RAG_OPENVINO_MODELS_DIR`, named after the last segment of the model id. A
folder holds everything the model loads from, so it works with the network off.
An export is written beside its target and swapped in at the end: a failed run
leaves the previous export in place.

Running it is optional. A process that finds a model's folder missing when it
first loads the model downloads it, converts it into that folder, logs one
`local_model_provisioned` line and carries on; later starts load the folder.
So the first start of an App is slow (the embedder is a 2.3 GB download) and
needs the Hub, about 5.5 GB of disk for both models and several GB of memory
while converting; with `HF_HUB_OFFLINE=1` or no network it fails, naming the
model and the folder. The command is for filling the folder ahead of time, as
the `rag` Docker image does at build. Each call uses the threads the thread
budget gives it (OpenVINO's `INFERENCE_NUM_THREADS`).

The OpenVINO packages come with the install (`sentence-transformers[openvino]`).
Their `optimum-intel` accepts transformers below 5.6 only, so pip holds
transformers and huggingface-hub at the versions it supports.

Checked on 2026-11-11, on an Apple-silicon Mac: the reranker converts (207 s
with its download, 424 MB), reloads offline and scores within 0.003 of PyTorch;
the thread count is honoured. **Not checked**: the embedder's conversion (it was
never run), anything on x86 Linux, which is what Domino runs, and speed there.
On the Mac, where OpenVINO defaults to half precision, it was slower than
PyTorch.

## OpenVINO Model Server

`ovms/versions.env` is the single pin of the OpenVINO Model Server version,
sourced by `export_models.sh`, read by `domino/Dockerfile` and passed to docker
compose with `--env-file`:

| Key | Value |
| --- | --- |
| `OVMS_VERSION` / `OVMS_IMAGE` | `2026.3.1` / `openvino/model_server:2026.3.1` |
| `OVMS_BINARY_URL` | `ovms_ubuntu22_2026.3.1_python_off.tar.gz` from the GitHub release `v2026.3.1` (Domino `ovms` image) |

The served models (source id and alias) are pinned once, by the `RagSettings`
defaults in `config/settings.py` (`RAG_RERANKER_MODEL` / `RAG_OVMS_RERANK_MODEL` =
`antoinelouis/crossencoder-camembert-base-mmarcoFR` / `crossencoder-camembert-base-mmarcoFR`, `RAG_EMBEDDER_MODEL` /
`RAG_OVMS_EMBEDDINGS_MODEL` = `Snowflake/snowflake-arctic-embed-l-v2.0` / `snowflake-arctic-embed-l-v2.0`).
`rag-served-models` prints them, `RAG_*` overrides included, as shell
assignments (`RERANKER_SOURCE_MODEL=...`) for `export_models.sh`; the `rag`
Domino image reads the source ids from the installed package.

**Export** (`make ovms-export` = `./ovms/export_models.sh [MODELS_DIR]`, default
`ovms/models`; run `make install` first so `rag-served-models` is on `PATH`, or
the script stops and says so): takes the served models from `rag-served-models`,
downloads `export_model.py` and its `requirements.txt` from the OVMS repository at tag `v${OVMS_VERSION}` into `<MODELS_DIR>/.export/` (cached),
`pip install`s them into `$PYTHON` (set `PYTHON=/path/to/venv/bin/python` for a
separate environment, which it needs: the exporter pins its own
sentence-transformers and OpenVINO versions and would downgrade this project's),
then runs `rerank_ov` and `embeddings_ov --pooling CLS`
with `--weight-format int8`. The result is `config_reranker.json`,
`config_embedder.json` and one directory per model; the configs reference the
model directories relatively, so the tree can be mounted anywhere.

**On Domino the export is optional.** An OVMS role (`RAG_ROLE=ovms-reranker` /
`ovms-embedder`) that finds its `config_*.json` missing under
`RAG_OVMS_MODELS_DIR` runs `export_models.sh` itself, once, before serving: it
needs `uv`, installs this project into `$HOME/rag-venv` when `rag-served-models`
is not on `PATH`, and gives the exporter its own `$HOME/ovms-exporter-venv`. The
prebuilt `ovms` image has no `uv`, so there the models still have to be mounted.
This path has not been run for real.

**Compose** (`make infra-up`) mounts `ovms/models` read-only at `/workspace` and
starts the reranker with `--rest_port 8001 --config_path /workspace/config_reranker.json`,
the embedder on `8002` with `config_embedder.json`. Verify:

```bash
curl -s localhost:8001/v2/health/ready && echo reranker ready
curl -s localhost:8001/v3/rerank -H 'Content-Type: application/json' \
  -d '{"model":"crossencoder-camembert-base-mmarcoFR","query":"vacation days","documents":["Employees get 25 vacation days.","Expense reports are due monthly."],"top_n":2}'
curl -s localhost:8002/v3/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"snowflake-arctic-embed-l-v2.0","input":["vacation policy"]}'
```

**Contracts** used by `adapters/ovms/ovms_adapters.py`, defined once in `adapters/ovms/v3_contract.py` (which the
FastAPI model server reads too):

| Adapter | Request | Response |
| --- | --- | --- |
| `OvmsReranker` → `POST {RAG_OVMS_RERANK_URL}/v3/rerank` | `RerankRequest` `{"model","query","documents":[...],"top_n"}` | `RerankResponse` `{"results":[{"index","relevance_score"}]}` |
| `OvmsEmbedder` → `POST {RAG_OVMS_EMBEDDINGS_URL}/v3/embeddings` | `EmbeddingsRequest` `{"model","input":[...]}` (batches of 64) | `EmbeddingsResponse` `{"data":[{"index","embedding":[...]}]}` |

Both map results back by `index` (an index outside the request, or a body that
does not fit the shape, is a malformed response), never send a field as `null`,
time out after 30 s and retry twice on transport errors, 5xx and 401 (the Domino
token is refetched first); other 4xx fail immediately, including the 404 for a
`model` that is not the server's alias. Readiness probes `GET /v3/models/{model}`.

**Bumping OVMS**: change `ovms/versions.env` only, re-run `make ovms-export`
(the exporter is fetched from the new tag), rebuild the `ovms` Domino image and
refill the models Dataset. **Changing a served model**: change its default in
`config/settings.py` only, then re-run `make ovms-export`, rebuild the `rag` Domino
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
root (commands in its header): `rag` starts from
`quay.io/domino/domino-standard-environment:ubuntu22-py3.10-r4.5-domino6.2-standard`,
installs this package with the `[loadtest,domino]` extras into a virtualenv of
its own on Python 3.11 (the image's Python is 3.10 and its packages clash with
ours), pre-downloads both HF models and sets `HF_HUB_OFFLINE=1`; `ovms` unpacks the
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
  https://<domino-host>/apps/<ovms-app-id>/v3/models/crossencoder-camembert-base-mmarcoFR
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
a warning and runs untraced. `make deploy` publishes the workflow as an Agent
with tracing on (see "App or Agent" below); by hand, deploy from Experiment
Manager → **Deploy Agent** with the same entry point, environment and tier as an
App. Keep the flag identical across compared runs (tracing adds a small
per-request overhead).

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

### Deploy on Domino in a few commands

From the repository root (for example `/repos/<project>`) in a Workspace of the one
project that hosts all the Apps:

```bash
cp deploy.example.yaml deploy.yaml   # set defaults.environment_id, topology, model server
make deploy                           # model server Apps first, then rag-workflow with their URLs
make status                           # each App's status and URL, then the workflow App's /readyz
make loadtest-domino RUN_NAME=split-all
make compare
```

`rag-deploy` (behind these targets) calls Domino's beta Apps API through
`DOMINO_API_PROXY` and publishes `rag-embedder`, `rag-reranker` and `rag-workflow`
as the topology needs. The Apps install their own dependencies when they start
(`domino/app.sh` with `RAG_BOOTSTRAP=1`), into a virtualenv on Python 3.11, so any
environment with `uv` works, but the first start is slower and needs access to
the package index and the Hugging Face Hub. An environment without Python 3.11
also needs access to GitHub, where `uv` downloads it from. For `ovms`, `ovms_bundle` points at the OVMS bundle (a folder with
`bin/ovms` and `lib/`, or the release `.tar.gz`) at a path the Apps can read, for
example `/domino/datasets/local/<dataset>/ovms.tar.gz`, and the environment needs
root or `sudo` for `apt-get`. Each App runs `domino/app.sh` by its absolute path in
the directory `rag-deploy` runs from, so run it from the repository root (the Apps
mount the repository at the same path); elsewhere it refuses to publish.

**`deploy.yaml`** (git-ignored; `deploy.example.yaml` documents every key) holds
every setting in one place. Top-level keys say what to deploy (`topology`,
`model_server`, `project_id`, `ovms_bundle`, `domino_host`) and how to talk to
Domino (`wait_timeout_s`, `poll_interval_s`, `no_instance_grace_s`,
`http_timeout_s`, `http_retries`, `run_token_url`); a `RAG_DEPLOY_<KEY>` variable
overrides one, e.g. `RAG_DEPLOY_TOPOLOGY=monolith make deploy`, and
`RAG_DEPLOY_FILE` names another file. `defaults` applies to every App and
`apps.<embedder|reranker|workflow>` to one, each App taking its own value, else the
default:

```yaml
defaults:
  environment_id: env-123          # required: the compute environment's id (in its page's URL)
  hardware_tier_id: small          # "" = the project's default tier
  settings: {RAG_ES_URL: https://es.example:9200}
apps:
  workflow:
    kind: agent                    # the default for the workflow; model servers are apps
    autoscaling: {min_replicas: 1, max_replicas: 4, target_cpu_pct: 70}
  reranker:
    hardware_tier_id: cpu-16
    settings: {RAG_RERANKER_MODEL: /domino/datasets/local/<dataset>/models/crossencoder-camembert-base-mmarcoFR}
```

`settings` are each App's environment variables (merged with the defaults', the
App's winning; YAML numbers and booleans are exported as strings), for example
`RAG_ES_URL`, `RAG_EMBEDDER_MODEL` / `RAG_RERANKER_MODEL` (a local folder such as
`/domino/datasets/local/<dataset>/models/snowflake-arctic-embed-l-v2.0` works as well as a
Hugging Face id) and `RAG_OVMS_MODELS_DIR` (for example
`/domino/datasets/local/<dataset>/ovms-models`). `rag-deploy` sets `RAG_ROLE`,
`RAG_TOPOLOGY`, `RAG_OVMS_RERANK_URL`, `RAG_OVMS_EMBEDDINGS_URL`,
`RAG_OVMS_BUNDLE` and `RAG_BOOTSTRAP` itself (they win), and `RAG_RUN_LABEL`
(`<topology>-<model server>`) unless the settings set it. Secrets
(`OPENAI_API_KEY`, `RAG_ES_API_KEY`, a bearer token in `RAG_OVMS_AUTH`) stay in
Domino's project or user environment variables, because `settings` are written
into the Apps' launch scripts; `rag-deploy` refuses them in any block. A misspelt
key is refused with its name.

**App or Agent.** `kind: agent` publishes the App with `configurationType:
AISYSTEM` (Domino: "Agents use the same hosting infrastructure as Domino Apps";
that field is the only API-level difference), `kind: app` with `STANDARD`. The
workflow is an Agent unless told otherwise: its launch script also exports
`RAG_DOMINO_TRACING=1` and `RAG_BOOTSTRAP_EXTRAS=domino` (unless its settings set
them), and `app.sh` then installs `.[domino]`, the extra that brings the tracing
library. An existing App of the other kind is not converted: `rag-deploy` stops
and says to delete it in Domino or change `kind`.

**Autoscaling.** `autoscaling` becomes the version's `autoscalingSpecification`
(`enabled`, `minReplicas`, `maxReplicas`, `targetCpuAvgUtilizationPct`). Replicas
are pods behind one instance, so scaling does not change the instance URL below.

**Redeploying, never a second App.** `rag-deploy` remembers each App's id in
`.rag-deploy-cache.json` (git-ignored, per project; a dry run keeps it in memory).
It looks an App up by that id first, then by exact name, and remembers it either
way; only when both miss does it create the App. An App that exists gets a new
version with whatever `deploy.yaml` says now (the 6.2 API has no restart call).
Delete the cache file and the Apps are found by name again.

**Where an App answers.** Apps are called on their running instance:
`{domino_host}/{owner}/{project}/r/notebookSession/{instance id}`, where owner and
project are the App's `project.ownerUsername` and `project.name` and the instance
is the `Running` one of its current version (`domino_host` "" = the scheme and host
of the App's own `url`). That URL is what the workflow App gets as
`RAG_OVMS_*_URL`, what `rag-deploy url <part>` prints, what `make status` shows and
checks `/readyz` on, and what the "Running" line after a deploy prints; the App's
own `url` stays in the messages that point at its page or logs. The instance id
changes at every redeploy, so after `make deploy-embedder` or `make
deploy-reranker` alone, `rag-deploy` reminds you that `rag-workflow` still calls
the previous instance: run `make deploy-workflow`. Every model server call in a
split topology goes through Domino's proxy: it is part of what the split
topologies measure on Domino.

One-time prerequisites that stay manual: install the package in the Workspace
(`make install-dev`, or `uv pip install -e ".[loadtest]"`), `make ingest`, and for
`ovms` the exported models.

#### Try the deploy without Domino

```bash
make deploy-dry-run      # rag-deploy --dry-run deploy all: nothing is sent to Domino
```

prints each App `rag-deploy` would publish (kind, environment, hardware tier,
autoscaling, entry script), with every check of a real deploy. One of them: no
value may still hold an `<angle-bracket>` placeholder from the example (in the
`settings` of the Apps being deployed, their `environment_id` and
`hardware_tier_id`, and `ovms_bundle`). The example copied unchanged stops on the
first one:

```
rag-deploy: error: environment_id in defaults of deploy.yaml still holds the example placeholder '<environment id>': replace it with a real value
```

To click through the API or watch a
deploy wait, serve the fake Domino and point `rag-deploy` at it:

```bash
make fake-domino BEHAVIOUR=slow     # Swagger at http://localhost:8899/docs
DOMINO_API_PROXY=http://localhost:8899 DOMINO_PROJECT_ID=local make deploy
DOMINO_API_PROXY=http://localhost:8899 DOMINO_PROJECT_ID=local make status
```

The fake listens on 8899, the port a Domino run's token service uses, so `make
status` gets its token from the fake too. Run it on your own machine, not in a
Domino Workspace, where that port is taken.

| Behaviour | What the fake Domino does |
|---|---|
| `starts` (default) | a new version's instance is `Pending` on the first GET, then `Running` |
| `slow` | `Queued`, `Pending`, `Preparing`, `Pulling`, then `Running` |
| `fails` | `Pending`, then `Failed` |
| `never-starts` | creates the App or version but no instance (6.2 has no start call) |
| `forbidden` | every Apps API call answers 403 with a `FailureEnvelopeV1` |

The fake is transcribed from the Domino 6.2 API reference, so it shows what
`rag-deploy` expects of Domino, not what a live Domino does. Besides the Apps
calls it lists a version's instances, answers `/readyz` at each instance's
`/fake-owner/fake-project/r/notebookSession/<instance id>` URL, and describes any
hardware tier (`GET /api/hardwaretiers/v1/hardwaretiers/<id>`) with a fixed
number of cores.

The fake only stands in for the Apps API. Imitating Domino at run time (the
proxy in front of every App, Apps placed on machines) is the job of a temporary
emulation, kept apart so that it can be deleted once the real Domino is in use:
see [emulation/README.md](emulation/README.md).

Implemented from the Domino 6.2 API reference but **not verified on a live
Domino**:

1. Creating an App or a new version starts it (6.2 has no start call; if an App
   comes up without an instance, `rag-deploy` says to start it once from the Apps
   page).
2. The inline `entryScript` runs (it starts `domino/app.sh` by absolute path, so
   whatever directory the App starts in does not matter).
3. The run token is accepted for calls between Apps. Domino's own guidance for 6.3
   says a script calling an App needs a Personal Access Token; if the run token is
   rejected (`503 reranker_unavailable` with HTTP 401 or a redirect in the detail),
   set `RAG_OVMS_AUTH=<personal access token>` as a project environment variable.
4. An App answers on `/{owner}/{project}/r/notebookSession/{instance id}` of the
   Domino host. This path is not in the API reference: it comes from using
   Domino. If it is wrong, every URL `rag-deploy` hands out is wrong; the App's
   page (its `url`) still works.
5. `configurationType: AISYSTEM` is what makes an App an Agent (the reference
   lists the value; that the UI's **Deploy Agent** sends it is an assumption).
6. The `[domino]` extra (`dominodatalab[agents]`) installs in the generic
   environment when an Agent starts.

If one of these fails, the step-by-step section below is the fallback.

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
(`RAG_EMBEDDER_MODEL`, default `Snowflake/snowflake-arctic-embed-l-v2.0`, both locally and on OVMS).

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
curl -i -H "Authorization: Bearer $TOKEN" <URL of A>/v3/models/crossencoder-camembert-base-mmarcoFR    # 200
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
make check         # every gate below, in order, stopping at the first failure
make test          # pytest: fakes + httpx.MockTransport only; no weights, no network, no Elasticsearch
make format        # ruff --fix + black
```

`make check` is the project's quality gate, and the GitHub Actions workflow
`.github/workflows/checks.yml` runs the same command on every pull request and on
every push to `main`. The verdict is deterministic: every gate tool is pinned to an
exact version in the `dev` extra, and every threshold is written in
`pyproject.toml`. Two platforms are supported and no other is targeted: Domino's
Ubuntu, and macOS for local work. CI runs the gates inside the image
`domino/Dockerfile` builds from (Domino's 6.2 standard environment, Ubuntu
22.04), as the `ubuntu` user Domino runs Apps as, and in a virtualenv on the
Python of `.python-version`, which is how the Apps start too. The workflow reads
the image from the Dockerfile's `BASE_IMAGE` line, so the two cannot drift. Each
gate is also a `make` target of its own:

| target | runs | passes when |
|---|---|---|
| `lint` | `ruff check`, `black --check`, `pydoclint` | no ruff finding, no file black would change, and every docstring in Google format with its Args, Returns and Raises |
| `typecheck` | `mypy src`, `ty check src` | neither reports a diagnostic |
| `complexity` | `complexipy` on `src` | no function has a cognitive complexity above 10 |
| `deadcode` | `vulture` on `src`, `tests`, `locustfile.py` | nothing is unused at confidence 60 or more, outside `vulture_whitelist.py` |
| `deps` | `deptry .` | no dependency is missing, unused or only transitive |
| `test` | `pytest` | every test passes, the layering and code-shape tests included |

The ruff rules go beyond style. A function has a cyclomatic complexity of at most
8, at most 8 branches, 6 return statements, 30 statements and 6 arguments
(arguments that travel together are bundled). A compared number has a name, or
is an `http.HTTPStatus`. Every signature has type hints. Tests are exempt from the
docstring, annotation, unused-argument, magic-number, argument-count and
line-length rules.

Docstrings follow the Google format, checked by three rules. ruff's `D` rules
require a docstring on every public module, class and function. `pydoclint`
requires each docstring's `Args`, `Returns` and `Yields` to match the signature
exactly, with constructor arguments documented on the class. ruff's `DOC501`
requires every exception a function raises to be in its `Raises` section. A
`Raises` entry may also name an exception raised through a helper, which is why
pydoclint's own raise check is off: it only accepts a `raise` written in the
function's own body.

`tests/structure/test_code_shape.py` reads the source with `ast` and checks three more
limits: no file in `src`, `tests` or `locustfile.py` has more than 500 lines; no
function in `src` or `locustfile.py` has a body longer than 25 lines, its docstring
not counted; and no function in `src` nests `if`, `for`, `while`, `try`, `with`
or `match` more than 2 deep, a nested function or class being measured on its
own.

A suppression (`noqa`, `type: ignore`, `ty: ignore`, a per-file ignore, a
whitelist entry) always says why on the same line or the line above.

`tests/conftest.py` sets `LOCUST_SKIP_MONKEY_PATCH=1` before any `import locust`
so the asyncio-based service tests are never gevent-patched.

`tests/` has one folder per layer of `src/rag_load_test` (`domain`, `application`,
`config`, `adapters`, `entrypoints`, `loadtest`, `deploy`), plus `scripts` for the
shell scripts, `structure` for the layering and code-shape rules and `fake_domino`
for the fake's own tests. The folders are not packages, so a test file's name is
unique across the tree. What several files share (named fakes, fixtures,
`REPO_ROOT`) is in `tests/helpers`, on pytest's path and imported by plain module
name. The Chroma store's tests run a real embedded Chroma in a temporary folder.
