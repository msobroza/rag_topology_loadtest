# query_processing — workflows declared in files, deployed on Domino in topologies, measured with Locust

This repository holds a toolkit, `query_processing`, and its first use case, a
RAG workflow. A use case is a directory of settings, and two of its files say
what runs and where. The RAG's workflow file,
`query_processing_rag_usecase/workflow.yaml`, without its comments (its
prompts are in `prompts.yaml`, beside it):

```yaml
components:
  embedder: {backend: local, source_id: Snowflake/snowflake-arctic-embed-l-v2.0, query_prefix: "query: "}
  reranker: {backend: local, source_id: antoinelouis/crossencoder-camembert-base-mmarcoFR}
  llm:
    backend: api
    api: {url: https://api.openai.com/v1/chat/completions, api_format: openai, model: gpt-4o-mini}
    http: {timeout_s: 120}
  vector_store:
    backend: elasticsearch
    elasticsearch: {url: "http://localhost:9200", index: rag_passages}
    chroma: {collection: rag_passages}
graph:
  stages:
    embed:    {call: embedder, reads: [standalone_query], writes: [query_embedding]}
    retrieve: {call: vector_store, retrieval_mode: dense, top_k: 20, reads: [query_embedding, metadata_filter], writes: [candidates]}
    rerank:   {call: reranker, top_k: 10, reads: [standalone_query, candidates], writes: [reranked]}
    generate: {call: llm, prompt: answer_from_context, reads: [conversational_query, reranked], writes: [answer]}
  edges:
    - [START, embed, retrieve, rerank]
    - [generate, END]
  conditional_edges:
    - {source: rerank, path: is_query_mode, reads: [mode], path_map: {true: generate, false: END}}
```

Its deploy file, `deploy.yaml`, is a copy of `deploy.example.yaml` edited for a
site. The example, shortened:

```yaml
topology: split-all
topology_alias:
  monolith:       {embedder: local,  reranker: local}
  split-reranker: {embedder: local,  reranker: served}
  split-all:      {embedder: served, reranker: served}
app_name_prefix: qp-rag
defaults: {environment_id: "<environment id>", server: fastapi}
apps:
  workflow: {kind: agent, autoscaling: {min_replicas: 1, max_replicas: 2}}
overrides:
  components:
    vector_store: {elasticsearch: {url: "https://<elasticsearch host>:9200"}}
```

The workflow file declares the components, each with its backend (`local` in
the process that calls it, `served` by a model server of ours, `api` at an
endpoint we do not run, `fake`, `none` for a reranker, or the vector store's
product), and the graph:
stages, each calling the component it names and stating the workflow state
fields it reads and writes, a plain call or an instance of a registered stage
type, linked by edges and by conditional edges that ask a registered edge
condition. The deploy file chooses a topology by its topology alias (a row that
puts each component it names `local` or `served`), says how each App runs on
Domino, and holds this site's values under `overrides`, in the workflow file's
shape.

The toolkit, `query_processing/`, holds all the code: the stage types, edge
conditions, components and backends, the workflow app (`qp-serve`), the model
server (`qp-model-server`), the deploy tool (`qp-deploy`), the load-test client
(a Locust file and `qp-compare`) and a fake Domino. It ships no use case's
models, prompts or sizes. A use case installs it and holds settings only: the
RAG's, `query_processing_rag_usecase/`, adds an environment example, a
`Makefile` of run targets and its tests to the two files. Its commands run in
its directory, where they find `workflow.yaml`, `deploy.yaml` and `.env`
(`QP_WORKFLOW_FILE` and `QP_DEPLOY_FILE` name others; with no workflow file, a
command stops and says where it looked). Every command of this README runs
there, and a file path is relative to the toolkit's directory, unless it says
otherwise. The design is in
[ADR 0003](docs/adr/0003-the-deploy-tool-resolves-the-topology.md) (the deploy
tool resolves the topology), [ADR 0004](docs/adr/0004-the-workflow-graph-is-a-file.md)
(the workflow graph is a file) and
[ADR 0005](docs/adr/0005-toolkit-and-use-case-are-two-distributions.md) (the
toolkit and a use case are two distributions), the vocabulary in
[CONTEXT.md](CONTEXT.md).

## Overview

```mermaid
flowchart LR
    subgraph UC["query_processing_rag_usecase/"]
        WF["workflow.yaml<br/>components and graph"]
        DF["deploy.yaml<br/>topology alias, Apps, overrides"]
    end
    WF --> QD["qp-deploy"]
    DF --> QD
    QD -->|"first: one App per served component"| MS["qp-rag-embedder, qp-rag-reranker<br/>model servers of a model-server kind: fastapi or ovms"]
    QD -->|"then: each component's backend and served URL"| WA["qp-rag-workflow<br/>the workflow app: qp-serve --domino"]
    WA -->|"POST /v3/embeddings, /v3/rerank"| MS
    WA -->|"api backend"| EP["the LLM's endpoint"]
    WA --> ES[("Elasticsearch")]
    LO["Locust"] -->|"POST /query, /retrieve"| WA
```

`qp-deploy` resolves the chosen topology alias against the workflow file's
components and publishes one model server App per served component, named
`<app_name_prefix>-<component>`, then the workflow App, whose launch script sets
each named component's backend and served URL, the topology alias and the run
label: `split-all` publishes three Apps, `monolith` one. Every App runs the
toolkit's `domino/app.sh`, which starts its command in the use case's directory,
where the workflow file is read; the git-ignored `deploy.yaml` stays where
`qp-deploy` ran (ADR 0003). On a laptop, `qp-serve` runs each component on the
backend its settings give. Inside the workflow app, LangGraph runs the graph of
the file. The question is both the conversational and the standalone query of
the workflow state, and the earlier turns the client sends, if any, its
conversational history; `/retrieve` ends after rerank, where
`is_query_mode` is false; `timings_ms` has one key per stage
([HTTP API](#http-api)). Each component has alternatives a setting away:
[Retrieval, stores and model endpoints](#retrieval-stores-and-model-endpoints).

## Quickstart

### Install

At the repository's root, create its one `.venv` on the Python of
`.python-version`, 3.11 (`uv venv`, or `python3.11 -m venv .venv`), then
install both distributions in one command, the toolkit first, since no package
index has it:

```bash
make install-dev    # the toolkit with locust, pytest and the gate tools, and the use case
```

The `make` targets use that `.venv` without activating it; a command typed by
name, such as `qp-serve`, needs its `bin/` on your `PATH`
(`. .venv/bin/activate`). `make install` leaves out locust and the gate tools,
which [Development](#development) describes.

### First run, on fake backends

No model weights, store, API key or network: the four variables put each
component of the RAG's workflow file on its `fake` backend. The fake store is
seeded with 200 synthetic documents as it opens, so there is nothing to ingest
(`qp-ingest` refuses it); the fake LLM echoes the start of its prompt.

```bash
cd query_processing_rag_usecase
export QP_COMPONENTS__EMBEDDER__BACKEND=fake QP_COMPONENTS__RERANKER__BACKEND=fake \
  QP_COMPONENTS__LLM__BACKEND=fake QP_COMPONENTS__VECTOR_STORE__BACKEND=fake
qp-serve --host 127.0.0.1 --port 18888    # leave it running
```

From a second shell, in the same directory (the second answer is cut where `...`
stands: the question, the mode, the fake answer and the ten passages kept):

```bash
curl -s localhost:18888/readyz
# {"status":"ready","checks":[{"name":"embedder","ok":true,"detail":"fake-embedder"},{"name":"vector_store","ok":true,"detail":"200 passages in memory"},{"name":"reranker","ok":true,"detail":"fake-reranker"}]}
curl -s -X POST localhost:18888/query -H 'Content-Type: application/json' \
  -d '{"question":"How many paid vacation days do employees accrue each year at Northwind Analytics?"}'
# {"request_id":"fd26bfa18f044416b2be1b4bd2b947c7","topology":null,"run_label":"embedder=fake+reranker=fake+llm=fake+vector_store=fake",...,
#  "timings_ms":{"embed":0.08879206143319607,"retrieve":0.46491692773997784,"rerank":0.30445889569818974,"generate":0.015625031664967537,"total":5.137166939675808}}
make loadtest RUN_NAME=first-run HOST=http://localhost:18888 USERS=2 SPAWN_RATE=2 RUN_TIME=5s
make compare
sed 's/<[^<>]*>/example/g' deploy.example.yaml > deploy.yaml
make deploy-dry-run
```

`topology` is `null` since no topology alias was given, and the run label lists
each component with its backend. Locust runs for five seconds and writes the
run's files under `results/`, among them `first-run_stats.csv` and the target
record `first-run_target.json`, which counts the responses under that run
label; `make compare` writes `results/comparison.md`, one table per endpoint and
per stage. `sed` copies the deploy example with each `<placeholder>` made a
value; the dry run runs every check of a real deploy against a fake Domino in
its own process, and sends nothing. It lists `qp-rag-embedder` and
`qp-rag-reranker`, model servers of kind `fastapi`, then `qp-rag-workflow`,
whose entry script holds:

```text
    export QP_TOPOLOGY_ALIAS=split-all
    export QP_COMPONENTS__EMBEDDER__BACKEND=served
    export QP_COMPONENTS__EMBEDDER__SERVED__URL=http://fake-domino.test/fake-owner/fake-project/r/notebookSession/app-1-v1-i/v3/embeddings
    export QP_COMPONENTS__RERANKER__BACKEND=served
    export QP_COMPONENTS__RERANKER__SERVED__URL=http://fake-domino.test/fake-owner/fake-project/r/notebookSession/app-2-v1-i/v3/rerank
    export QP_RUN_LABEL=split-all+embedder=fastapi+reranker=fastapi
```

Stop `qp-serve` with Ctrl-C and delete `deploy.yaml`, which holds made-up
values. The four variables stay set in the first shell: use a new one for the
real run.

### The real run

The models, Elasticsearch and an API key, from the use case's directory:

```bash
cp .env.example .env          # set QP_COMPONENTS__LLM__API__API_KEY there, or uncomment QP_COMPONENTS__LLM__BACKEND=fake
make infra-up                 # docker compose: Elasticsearch on :9200 (the OVMS containers need make ovms-export first)
make openvino-export          # once: both models from the Hugging Face Hub, converted under models/openvino
make ingest                   # qp-ingest --synthetic 200 -> index rag_passages
make serve                    # qp-serve on http://0.0.0.0:8888
```

The calls of the first run reach it on port 8888; `make infra-down` stops the
containers. `make ingest` embeds with the file's embedder (`qp-ingest
--embedder NAME` picks one of several) and upserts by passage id; `qp-ingest
--corpus docs.jsonl` loads a real corpus, one `{"id","title","text","metadata"}`
object per line. The export: [In-process models on OpenVINO](#in-process-models-on-openvino).

**The split topologies on a laptop.** A component is moved behind a model
server by its backend, `served`, and its served URL, and by nothing else. With
the models exported for OVMS (`make ovms-export`, see
[OpenVINO Model Server](#openvino-model-server)), `make infra-up` also serves
the reranker on :8001 and the embedder on :8002:

```bash
# split-reranker: the reranker served by OVMS, the embedder in-process
QP_COMPONENTS__RERANKER__BACKEND=served \
  QP_COMPONENTS__RERANKER__SERVED__URL=http://localhost:8001/v3/rerank make serve

# split-all: the reranker and the embedder served by OVMS
QP_COMPONENTS__RERANKER__BACKEND=served \
  QP_COMPONENTS__RERANKER__SERVED__URL=http://localhost:8001/v3/rerank \
  QP_COMPONENTS__EMBEDDER__BACKEND=served \
  QP_COMPONENTS__EMBEDDER__SERVED__URL=http://localhost:8002/v3/embeddings make serve
```

A `served` component is given only its URL, the full route: it is called in the
standard format of its type, under its alias, with no key off Domino. No
topology is reported, and the run label is
`embedder=local+reranker=served+llm=api+vector_store=elasticsearch` for the
first. Then [Load test and comparison](#load-test-and-comparison), and on Domino
[Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands) or,
by hand, [Step by step](#step-by-step-first-test-run-on-domino).

## Starting a second use case

A second use case is a directory of files, with no code (ADR 0005), in this
repository beside the toolkit, as `query_processing_<name>_usecase/`:
`qp-deploy` looks for the toolkit's launch script going up from where it runs
and refuses to publish from anywhere else, and every App runs its command in
that directory, where the workflow App installs the use case as it starts.
Start from a copy of the RAG's:

| File | What it states |
| --- | --- |
| `workflow.yaml` | The components and the graph ([The workflow file](#the-workflow-file)). Each component its `backend`; an embedder and a reranker their `source_id`, even on `fake`; a store on `elasticsearch` or `chroma` its `index` or `collection`. Each stage the component it `call`s, its `reads` and `writes`, and the settings its stage type or its component's request requires, which the toolkit leaves without a default: a retrieve stage's `retrieval_mode` and `top_k`, a rerank stage's `top_k`, a generate stage's `prompt`. No API key. |
| `prompts.yaml` | The prompts a stage names, each a list of messages with a `role` and a `content`, a Jinja2 template whose variables are the fields its stage reads ([The workflow file](#the-workflow-file)). |
| `deploy.example.yaml` | Its own `app_name_prefix`, so that its Apps can share a Domino project with the RAG's; its `topology_alias` table over its own components, and `topology`; `defaults.environment_id`; a block of `apps` per App part; this site's `overrides`. Copied to the git-ignored `deploy.yaml`. |
| `pyproject.toml` | A distribution of its own that depends on `query-processing`, with `packages = []`, which `domino/app.sh` installs with the toolkit. |
| `.env.example` | The API keys, by their paths; `QP_TOP_K_RETRIEVE` and `QP_TOP_K_RERANK` for a load test straight at a reranker. |
| `Makefile` | Its run targets. The RAG's names the toolkit as `../query_processing`, and the RAG's components in `deploy-embedder`, `deploy-reranker` and `SERVE ?= reranker`. |
| `tests/` | Its tests through the two seams, as the RAG's do: its workflow file answers through the workflow app's HTTP API, and a dry run of each topology alias publishes the expected Apps. |

The root `Makefile` and the CI install and test the RAG use case only: install
another with the toolkit in one command (`python -m pip install -e
./query_processing -e ./query_processing_<name>_usecase`) and run its tests in
its directory. The `rag` image of `domino/Dockerfile` takes
`--build-arg USE_CASE=<its directory>`. A minimal workflow file, run from a
scratch directory: a lexical search that keeps three passages, with no LLM:

```yaml
components:
  embedder: {backend: fake, source_id: example-org/example-embedder}
  reranker: {backend: fake, source_id: example-org/example-reranker}
  vector_store: {backend: fake}
graph:
  stages:
    retrieve: {call: vector_store, retrieval_mode: lexical, top_k: 20, reads: [standalone_query, metadata_filter], writes: [candidates]}
    rerank: {call: reranker, top_k: 3, reads: [standalone_query, candidates], writes: [reranked]}
  edges:
    - [START, retrieve, rerank, END]
```

`qp-serve` in that directory answers `/retrieve` and `/query` alike: three
passages, `"answer": null`, `timings_ms` with `retrieve`, `rerank` and `total`.
The `fake` store is seeded through the file's embedder, so the file declares
one that no stage calls: without it the app does not start (`vector_store
unavailable: could not open backend 'fake': ValueError: ingestion embeds with
one embedder component, and the workflow file declares 0 (its embedders:
none)`). With `app_name_prefix: qp-search` and the topology alias
`split-reranker: {reranker: served}`, a dry run from beside `query_processing/`
publishes `qp-search-reranker`, then `qp-search-workflow`; from the scratch
directory `qp-deploy` stops (`no query_processing/domino/app.sh in <directory>
or any directory above it`).

### What only the toolkit adds

Code goes in the toolkit, even when one use case alone needs it (rule 8 of
[Where a file goes](#where-a-file-goes)). Each of these is one new module, under
`src/query_processing/`, that registers itself with its registry's decorator
(rule 6); its package imports every module, so no list is edited. The module
also gets its line in [the tree](#the-tree) and a test.

| A new | Its module | Registered with | A test that adds one |
| --- | --- | --- | --- |
| stage type, for a stage that does more than call its component | `application/stages/<name>_stage.py`: a `BaseStage[<port>, <settings>]` that sets `component_type` and `settings_type`, and defines `check_fields` (the lists of fields it can use, with their types) and `run` | `@STAGE_TYPES.register("<name>")` | `tests/application/test_workflow_builder.py` (`copy`, `shout`, in a registry of the test's own) |
| edge condition | `application/edge_conditions/<name>.py`: a function of the values its conditional edge reads, in the order of its `reads`, that returns a bool, or a `str` label | `@EDGE_CONDITIONS.register("<name>")` | the same file (`is_a_question`) |
| backend | `composition/openers/<type>_on_<backend>.py`: an opener that returns the open port, whose implementation is in `adapters/<type>/<backend>/`; a block in `config/component_settings.py` if it has settings | `@RERANKER_OPENERS.register("<backend>")`, or `EMBEDDER_OPENERS`, `LLM_OPENERS`, `VECTOR_STORE_OPENERS` | `tests/composition/test_component_factory.py` (`test_a_new_backend_is_one_new_module`) |
| API format | a module of `adapters/<type>/api/`, such as `cohere_rerank_format.py`: a subclass of the type's format base class | `@RERANK_API_FORMATS.register("<name>")`, or `EMBEDDING_API_FORMATS`, `GENERATION_API_FORMATS` | `tests/entrypoints/test_qp_serve_api_formats.py` (`test_a_new_api_format_is_one_new_module`) |
| model-server kind | `deploy/model_server_kinds/<name>_kind.py`: a `BaseModelServerKind` with its `routes` and `probe_route`; and the toolkit's `domino/model_servers/<name>.sh`, which defines `bootstrap_model_server` and `serve_model_server` | `@MODEL_SERVER_KINDS.register("<name>")` | `tests/deploy/test_a_new_model_server_kind.py` |

The start-up checks and the graph builder read what a stage type declares, so
neither changes. A component type of its own, a new port, is more than one
module: its port, its settings, its registry of openers and its entry in
`application/component_types.py`.

## Code layout

### Where a file goes

The repository holds two distributions (ADR 0005), each a directory of its own
at its root:

- **The toolkit**, in `query_processing/`: every line of code. Every path of this
  section and of the tree below it is relative to that directory, except the use
  case's and the repository's own entries at the end of the tree
  (`query_processing_rag_usecase/`, `emulation/`, `docs/`, the root `Makefile`,
  `.python-version`, and this README, `CONTEXT.md`, `CLAUDE.md` and `.github/`).
  It ships no workflow file and no default that is a use case's model, prompt,
  size, index or collection: its tests run on a fixture workflow file of made-up
  models (`tests/helpers/fixture_workflow.yaml`).
- **A use case**, `query_processing_rag_usecase/` for the RAG: settings only, no
  Python code (its workflow file, deploy settings example, environment example,
  a Makefile of run targets) and its own tests, which import only what the
  toolkit installs. It depends on the toolkit; the toolkit never imports nor
  names a use case. Its commands run in its directory, where they find
  `workflow.yaml`, `deploy.yaml` and `.env`, and write what they write.

The toolkit is three applications over one shared domain. The domain
(`domain/`) is the language all three use: no I/O, no framework.

| Application | What it is | Where it lives |
| --- | --- | --- |
| RAG system | what runs on Domino and what prepares its data and models: the workflow app, the model server, ingestion, the corpus and the model export | split by layer: `application/`, `config/`, `adapters/`, `infrastructure/`, `composition/`, `entrypoints/` |
| deploy tool | what publishes the Apps of a Topology (`qp-deploy`) | one package, `deploy/` |
| load-test client | what drives a target and compares runs (Locust, `qp-compare`) | one package, `loadtest/`, and `locustfile.py` at the top, which a use case's Makefile hands to Locust (`-f`) |

1. **Imports.** A package imports only what this table allows.

   | Package | May import |
   | --- | --- |
   | `domain` | `domain` |
   | `application` | itself, `domain`, and the ports of the adapters (`adapters/<component>/base_<component>.py`), nothing else of them |
   | `config` | itself and `domain` |
   | `adapters` | `adapters`, `infrastructure` and `domain`, and a port only `domain` |
   | `infrastructure` | itself and `domain` |
   | `composition` | itself, `adapters`, `application`, `config`, `domain` and `infrastructure`: the one package that builds the adapters from the settings |
   | `entrypoints` | `composition`, `application`, `config`, `domain`, `infrastructure`; never `adapters`: a command gets its ports from the composition root |
   | `deploy` | `deploy` and `domain`; `qp_deploy.py` alone may also import the fake Domino, and `workflow_components.py` alone may also import `config`, to read the components of the workflow file a topology alias names and to check the overrides against the workflow settings (ADR 0003) |
   | `loadtest` | `loadtest` and `domain`; `loadtest_settings.py` alone may also import `config`, for the defaults of the two aliases a load test sends |
   | `testing` | every package: the tests the toolkit ships serve a user's component as the toolkit does; no package imports it, `extension_api` included, and a use case's tests import it as `query_processing.testing` |

2. **Commands.** A command lives in its own application, in a module named
   exactly after it with hyphens as underscores: the RAG system's six in
   `entrypoints/` (`qp-serve` is `entrypoints/qp_serve.py`), `qp-deploy` in
   `deploy/qp_deploy.py`, `qp-compare` in `loadtest/qp_compare.py`.
3. **One name, one module.** No two modules share a file name. A settings module
   is named after its settings class (`workflow_settings.py`, `deploy_settings.py`);
   a domain module that holds a glossary term of [CONTEXT.md](CONTEXT.md) is
   named after it (`backends.py`, `run_label.py`, `roles.py`,
   `api_endpoint.py`).
4. **Adapters.** `adapters/` holds the ports and their implementations, nothing
   else: one package per component, `embedder/`, `reranker/`, `llm/`
   (generation), `vector_store/` and `router/`. Its port, the abstract class the
   application depends on, is `base_<component>.py`; each implementation is a
   directory named after its Backend value, an embedder's `backend: local`
   being `embedder/local/`, and holds one implementation of the port. One
   directory serves two Backends: `api/` implements both `served` and `api`,
   which call an endpoint over HTTP the same way and differ only in how it is
   resolved (two directories would hold the same class twice). The router's
   `served` backend is built, so it has no directory, and its API format, the
   one written by hand, sits beside its port. Code that
   implements no port is not an adapter and lives in `infrastructure/`: what
   the `api` implementations share (`endpoint/`), what the `local` ones share
   (`local_models/`), `http/`, the corpora `qp-fetch-corpus` downloads
   (`corpora/`), `json_log.py` and `fake_text.py`. Building the implementations
   from the settings is not an adapter either: that is `composition/`, rule 5.
5. **Composition.** `composition/` is the composition root, the only code that
   reads the settings to build an adapter. Each job has one module:
   `resource_scope.py` opens each thing once and closes them all together;
   `http_resources.py` and `local_model_resources.py` hold what the
   implementations share; `openers/` holds one opener per component type and
   Backend, a module each named `<type>_on_<backend>.py`, registered by type
   and Backend (rule 6), and an opener takes its component's settings in the
   workflow file's shape (`config/component_settings.py`); the deploy tool,
   which may not import them, reads a type's Backends from these module names
   (`deploy/registered_backends.py`, held to the registries by a test); the one factory,
   `component_factory.py`, opens a component once with the opener of its type
   and Backend, so a new Backend is one new module in `openers/` and no
   factory is edited; `registered_names.py` refuses an unknown Backend or API
   format before a command opens anything; `composition_root.py` wires them
   and gives each command what it asks for.
6. **Registries.** A registry (`domain/registry.py`) holds named things of one
   sort and names the known ones when asked for an unknown one. Each entry
   registers itself with the registry's decorator, in a module of its own, and
   the package that holds the entries imports all of its modules from its
   `__init__.py` (`import_package_modules`), so a new entry is one new module
   and no table is edited. The registry is in `domain/` because every package
   that registers something may import the domain. Six kinds of entry today:
   an API format (each component's `api/` directory holds its formats, one
   module each, and its registry sits beside the formats' base class:
   `RERANK_API_FORMATS` in `base_rerank_api_format.py`), a stage type
   (`application/stages/`, one module each, `STAGE_TYPES` in `base_stage.py`),
   an edge condition (`application/edge_conditions/`, one module each,
   `EDGE_CONDITIONS` in `edge_condition.py`), a backend opener
   (`composition/openers/`, one module each, and the registry of each
   component type's openers sits beside what an opener is given:
   `RERANKER_OPENERS` in `component_openers.py`), a model-server kind
   (`deploy/model_server_kinds/`, one module each, `MODEL_SERVER_KINDS` in
   `base_model_server_kind.py`, and one launch script each in
   `domino/model_servers/`) and a component type's settings class, which a
   component is read with (`COMPONENT_TYPE_SETTINGS` in
   `config/component_settings.py`; the five types of the toolkit register
   their classes there by call, four of them kept in that module and the
   router's in `config/router_settings.py`, since the
   decorator's return type would hide their fields from the type checkers;
   an unknown type is refused, listing the known ones). A stage type and an edge
   condition call only ports, so they are in the application; the description
   of a graph is in the domain, which both the settings that write it and the
   application that builds it may import. A model-server kind is what the
   deploy tool publishes and refuses by, so it is in `deploy/`; the /v3 routes
   it declares are wire-contract constants of the domain, which the FastAPI
   model server answers on too.
7. **The tree below** lists every package and module of `src/query_processing/`; a
   new module gets a line.
8. **One way.** A use case depends on the toolkit, never the reverse: no file of
   the toolkit's `src/` names a use case (`query_processing_<name>_usecase`),
   in an import, a path or a default. What a use case needs that is code (a
   stage type, a backend) is added to the toolkit, even when one use case alone
   needs it; a use case adds files only.

`tests/structure/test_layering.py` checks rule 1 for every package, that
`locustfile.py` imports only the `loadtest` package, and rule 8;
`tests/structure/test_placement.py` checks rule 2, the file names of rule 3,
rule 4 (only component packages in `adapters/`, their layout with `api/` for
both `served` and `api`, and no port implemented anywhere else) and rule 7. The fake Domino (`src/fake_domino/`) is a
test double, not part of the product, and the rule does not apply to it.

### The tree

```text
query_processing/   the toolkit, one distribution: pyproject.toml, its Makefile (its install, its gates, fake-domino),
                    vulture_whitelist.py; no workflow file and no use case's file
  src/query_processing/
    __init__.py       package docstring, __version__
    extension_api.py  what a use case imports of the toolkit, every name it needs; its docstring holds the table of what can be added; no package imports it
    domain/           no I/O, no framework; imports no other package
      passages.py     Passage, ScoredPassage, RagResult, TOTAL_TIMING_KEY, top_by_score
      api_endpoint.py  ApiEndpoint (URL, API format, API key, model name), the model servers' routes
      standard_formats.py  the request and answer shapes of the standard formats (pydantic): OpenAI embeddings, Cohere rerank, OpenAI chat; STANDARD_API_FORMATS, the format name of each component type
      errors.py       RagDependencyError, describe_error
      registry.py     Registry: named things of one sort, filled by a decorator (rule 6), its names; import_package_modules; import_package_tree, which an extension is loaded by
      base_component.py  BaseComponent: a request in, an answer out (ainvoke, invoke); COMPONENT_TYPES, filled on each component type's class line; each type's backends, filled on each implementation's class line (backend=, settings=); from_settings
      component_resources.py  ComponentResources: what an implementation's from_settings may use besides its own block
      routes.py       Route, RouteRequest, RouteChoice: a route of routes.yaml (name, description, utterances), and the router's request and answer
      api_format.py   BaseApiFormat: a component type's request and answer to the wire and back; COMPONENT_API_FORMATS; api_format_of, the registered format or the one that mirrors the request and the answer
      component_requests.py  the request and the answer of the four existing component types: EmbedRequest, RerankRequest, GenerateRequest, SearchRequest and their answers
      conversation.py  HistoryMessage: one message of the conversational history a request carries, its role, content and conditions; check_history_conditions
      metadata_filter.py  MetadataCondition, MetadataFieldInfo, check_conditions: a metadata filter's conditions, checked against the fields a vector store declares
      prompts.py      PromptMessage, PromptPlaceholder, Prompt: a prompt of prompts.yaml, a list of messages with roles and placeholders
      graph_description.py  GraphDescription, StageDescription, ConditionalEdge: a graph's stages and edges, START and END
      retrieval_mode.py  RetrievalMode: dense, lexical or hybrid, a setting of the retrieve stage
      rank_fusion.py  reciprocal_rank_fusion: what a hybrid retrieval ends with
      thread_budget.py  ThreadBudget: how the virtual CPUs are split between model calls
      calibration.py  StageTuning, halvings, fastest, calibration_sample: the rules of calibration, no clock
      backends.py     the Runtime names (what runs an in-process model), and what a Backend is
      run_label.py    the run label: its default (each component with its backend), what a valid one is
      roles.py        the Role each App is published as (Role: workflow or model-server), AppPart, WORKFLOW_APP
      corpus.py       synthetic handbook corpus, its questions, chunking
    application/      use cases over the ports; imports domain and the adapters' ports only
      rag_dependencies.py  RagDependencies: the components a workflow calls, open, by the names its stages call them; their probes and model names
      workflow_state.py  WorkflowState: the one typed state of every graph; initial_state, merge_timings, REQUEST_FIELDS
      state_fields.py  field_types_of: each field the request and the stages write, with its type; state_type: the LangGraph state built from them; WorkflowGraphError
      component_types.py  COMPONENT_TYPES, as the five ports register it: each component type's port, which the start-up checks read before any component opens
      graph_check.py  check_graph: a graph that cannot run refused before anything opens, from what each stage type declares or each plain call states; WorkflowGraphError
      stage_binding.py  stage_type_of: a stage's stage type, or none for a plain call; bind_call: a plain call's keys, reads and writes bound to its component's request and answer
      edge_check.py   check_edges: the edges and conditional edges of a graph that cannot run, refused; field_labels: the labels each field can take
      workflow_builder.py  build_workflow: a graph description and the components in, the LangGraph graph out; asked_of_state
      workflow.py     CompiledWorkflow: one request through the compiled graph, its RagResult out; stage_timings
      prompt_rendering.py  render_prompt, prompt_variables: a prompt rendered into chat messages, each content in Jinja2's sandbox (the one module that uses Jinja2)
      output_schemas.py  OUTPUT_SCHEMAS: the pydantic models a generate stage can ask the LLM for, each field written to the field of its name
      stages/         the stage types (rule 6) and the plain call; importing it imports every module, so each stage type registers itself
        base_stage.py  BaseStage (component type, settings, state fields read and written), StageSettings, StageFields, FieldTypes, StageContext (the prompts, given to every stage), STAGE_TYPES
        call_stage.py  CallStage: a plain call, of no stage type: its request built from its keys and reads, its answer written to its writes
        retrieve_stage.py  RetrieveStage: candidates from the vector store, dense, lexical or hybrid
        generate_stage.py  GenerateStage: the prompt its settings name, rendered from the fields it reads, answered by the LLM
      edge_conditions/  the edge conditions (rule 6); importing it imports every module
        edge_condition.py  EdgeCondition: a function of the values its conditional edge reads that answers yes or no, or a label; EDGE_CONDITIONS; answers_with_label
        is_query_mode.py  is_query_mode: whether the request goes on to generate, from the mode
        has_history.py  has_history: whether the request carries a conversational history, from that field
        label.py      label: answers with the one field it reads, such as a route
      ingest.py       embed passages in batches and upsert them
    config/           the settings; imports domain only
      workflow_settings.py  WorkflowSettings: every top-level key of the workflow file (the components, the graph, the shared blocks), the topology alias and the run label (two labels), Domino's own run variables (pydantic-settings, .env and workflow file aware)
      component_settings.py  ComponentSettings and each type's (EmbedderSettings, ..., VectorStoreSettings): a component's type, backend, source id and alias, its settings, one block per backend (local, served, api; elasticsearch, chroma, fake)
      settings_blocks.py  SettingsBlock, ServedSettings, ComponentHttpSettings, ComponentSettings: the blocks every component's settings are made of
      registered_component_settings.py  RegisteredComponentSettings: a component of a type registered in COMPONENT_TYPES (an extension's), its type's keys and each backend's block read with the class its implementation names
      searched_embedder.py  searched_embedder: the embedder a vector store's searches use, the one called by the stage that writes what a retrieve stage reads (it seeds a fake store)
      extensions.py   load_extensions: every module of each package the workflow file names under extensions, imported before its components are read
      shared_settings.py  LocalModelsSettings, CalibrationSettings, DefaultHttpSettings, DominoSettings: what the components share, four top-level blocks of the workflow file
      workflow_file.py  workflow_file (QP_WORKFLOW_FILE, else ./workflow.yaml, a use case's, else a refusal naming both), the settings source that reads it, declared_alias
      prompts_file.py  prompts_file (QP_PROMPTS_FILE, else ./prompts.yaml, else none), the settings source that gives it as the prompts setting; a prompts key in the workflow file refused
      routes_file.py  routes_file (QP_ROUTES_FILE, else ./routes.yaml, else none), the settings source that gives its catalogue as the routes setting; a routes key in the workflow file refused
      router_settings.py  RouterSettings: a router's embedder and routes, two at least; a route the routes file lacks refused
      endpoint_urls.py  model_endpoint_url and missing_url_message: the URL a model component on served or api is called at, and the refusal of a missing one
    adapters/         the ports and their implementations, nothing else (rule 4)
                      (an adapter takes sizes, timeouts and names as constructor arguments and never reads the settings)
      embedder/       the embedder
        base_embedder.py  BaseEmbedder, the port: embed_query (the model's query prefix in front) and embed_documents (as they are)
        local/        backend: local
          sentence_transformer_embedder.py  SentenceTransformerEmbedder: a sentence-transformers model in this process
        api/          backends: served and api; importing it imports every module, so each format registers itself
          api_embedder.py  ApiEmbedder, and open_api_embedder: the embedder of one endpoint
          base_embedding_api_format.py  BaseEmbeddingApiFormat (a request body builder and an answer reader, no HTTP), EMBEDDING_API_FORMATS: the formats by name
          openai_embeddings_format.py  the openai embeddings format ({model, input} in, data[].embedding out)
        fake/         backend: fake
          fake_embedder.py  FakeEmbedder: a hashed bag of words
      reranker/       the reranker
        base_reranker.py  BaseReranker, the port
        local/        backend: local
          cross_encoder_reranker.py  CrossEncoderReranker: a cross-encoder model in this process
        api/          backends: served and api; importing it imports every module, so each format registers itself
          api_reranker.py  ApiReranker, and open_api_reranker: the reranker of one endpoint
          base_rerank_api_format.py  BaseRerankApiFormat, RERANK_API_FORMATS: the formats by name
          cohere_rerank_format.py  the cohere rerank format ({model, query, documents, top_n} in, results[] out)
        fake/         backend: fake
          fake_reranker.py  FakeReranker: token overlap with the query
        none/         backend: none
          no_reranker.py  NoReranker: retrieval order kept
      llm/            the LLM of the generation component
        base_llm.py   BaseLLM, the port
        api/          backends: served and api; importing it imports every module, so each format registers itself
          api_llm.py  ApiLLM, and open_api_llm: the LLM of one endpoint
          base_generation_api_format.py  BaseGenerationApiFormat, GENERATION_API_FORMATS: the formats by name
          openai_chat_format.py  the openai chat completions format (messages in, choices[0] out)
        fake/         backend: fake
          fake_llm.py  FakeLLM: an echo
      router/         the router
        base_router.py  BaseRouter, the port: a query's vector in, a route choice out; its routes are the labels of the route chosen
        router_api_format.py  RouterApiFormat, the router's API format written by hand ({embedding} in, {route, confidence} out)
        fake/         backend: fake
          fake_router.py  FakeRouter: the first of its routes, confidence 1
      vector_store/   the vector store
        base_vector_store.py  BaseVectorStore and BaseLexicalSearch, the ports
        base_metadata_filter_translator.py  BaseMetadataFilterTranslator, the port that turns a metadata filter into a store's own
        elasticsearch/  backend: elasticsearch
          elasticsearch_store.py  ElasticsearchVectorStore (kNN, and Elasticsearch's BM25 for lexical search)
          elasticsearch_filter_translator.py  ElasticsearchFilterTranslator: a metadata filter as filter clauses
        chroma/       backend: chroma
          chroma_store.py  ChromaVectorStore (embedded or server; dense only, it ranks no text)
          chroma_filter_translator.py  ChromaFilterTranslator: a metadata filter as a where
        fake/         backend: fake
          in_memory_vector_store.py  InMemoryVectorStore (its lexical ranking counts shared words)
          in_memory_filter_translator.py  InMemoryFilterTranslator: a metadata filter as a predicate over a passage
    infrastructure/   technical code that implements no port; imports itself and domain only
      endpoint/       what every api implementation shares, on the served and api backends
        api_endpoint_client.py  ApiEndpointClient: auth, timeout, retries and the error shape of every endpoint call
        bearer_token.py  BearerToken: no Authorization header, a static key, or the Domino run token
        open_endpoint_client.py  open_endpoint_client: the client of one endpoint, with a bearer token of its own
        served_call.py  ServedCallMixin: a registered component answered by a model server of ours, on its call route
      local_models/   what the local implementations share
        local_model.py  LocalModel: which files a model loads from and what runs it (OpenVINO or PyTorch)
        model_pool.py  model_executor, run_blocking: the one bounded thread pool the models run on
        model_provisioning.py  provision_local_model: an in-process model's OpenVINO export, made on the first start that lacks it
        openvino_export.py  convert one embedder or reranker to OpenVINO files
        native_threads.py  NativeThreadScope: threadpoolctl scopes, one per model
        stage_calibration.py  BaseCalibratedStage, calibrate: time a CPU-bound stage under each candidate, keep the fastest
        cpu_count.py  where the CPU count comes from: QP_LOCAL_MODELS__CPUS, the Domino hardware tier, the container
      http/           what every adapter that calls out shares
        http_policy.py  HttpPolicy: timeout, retries and backoff shared by every adapter that calls out
        json_post.py    RetryingJsonPost: the POST-with-retries the endpoint adapters share
      corpora/        an outside source: the public corpora qp-fetch-corpus downloads
        catalog.py    CORPORA: the corpora it knows, one CorpusDefinition each
        download.py   download a corpus's files under one HttpPolicy and convert them
        conversion.py a fetched file's text into documents or questions; CorpusFormatError names the file and line
        jsonl_files.py  write JSON lines atomically: a temporary file, then a rename
      fake_text.py    tokenize, cosine: the text helpers the fake implementations share
      json_log.py     log_json: the one log format
    composition/      the composition root: builds the adapters from the settings (rule 5)
      composition_root.py  CompositionRoot: settings in, ready ports out; what each command asks for
      resource_scope.py  ResourceScope: each resource opened once, all closed together
      http_resources.py  HttpResources: the shared HTTP client, its policy, the run token source, served_endpoint (the run token on Domino)
      local_model_resources.py  LocalModelResources: the thread budget, the model pool, a local model's files
      calibrated_stages.py  CalibratedRerankStage, CalibratedEmbedStage: the CPU-bound stages calibration times, and their candidates
      component_factory.py  ComponentFactory: the port supplied under a component's name, else what the opener of its type and Backend opens, once
      registered_names.py  check_registered_names: an unknown Backend or API format refused before a command opens anything
      registered_components.py  open_registered: a component of a type registered in COMPONENT_TYPES, built by the implementation its Backend names
      openers/        one opener per component type and Backend; importing it imports every module, so each registers itself
        component_openers.py  SharedResources (what an opener is given), ComponentOpeners, the four registries (EMBEDDER_OPENERS, ...) and OPENERS_BY_TYPE
        embedder_on_local.py  the in-process embedder, its threads calibrated at 0; calibrate_embedder_on_local for qp-calibrate
        embedder_on_served.py  the embedder of a model server of ours: its served URL, the standard format, its alias
        embedder_on_api.py  the embedder of an endpoint we do not run; open_embedder_at, which both use
        embedder_on_fake.py  FakeEmbedder
        reranker_on_local.py  the in-process reranker, calibrated at 0; calibrate_reranker_on_local for qp-calibrate
        reranker_on_served.py  the reranker of a model server of ours: its served URL, the standard format, its alias
        reranker_on_api.py  the reranker of an endpoint we do not run; open_reranker_at, which both use
        reranker_on_fake.py  FakeReranker
        reranker_on_none.py  NoReranker
        llm_on_served.py  the LLM of a model server of ours (none serves one yet): its served URL, the standard format, its alias
        llm_on_api.py  the LLM of an endpoint we do not run, under its own timeout; open_llm_at, which both use
        llm_on_fake.py  FakeLLM
        vector_store_on_elasticsearch.py  Elasticsearch, its client closed with the scope
        vector_store_on_chroma.py  Chroma, a server or embedded
        vector_store_on_fake.py  the in-memory store, seeded through the embedder
    entrypoints/      the RAG system's commands: FastAPI apps and CLIs
      qp_serve.py     qp-serve: the FastAPI app (/query /retrieve /healthz /readyz), --domino, Domino tracing, JSON logs, a refused graph as one line
      qp_model_server.py  qp-model-server: the FastAPI model server of one named component on OpenVINO Model Server's /v3 routes, its answers, UNKNOWN_MODEL_ERROR
      qp_served_models.py  qp-served-models: each embedder and reranker of the workflow file (name, type, source id, alias), one line of shell words each
      qp_export_openvino.py  qp-export-openvino: convert the in-process models of the embedders and rerankers to OpenVINO files
      qp_calibrate.py  qp-calibrate: calibrate the in-process models here, print the environment paths that pin them
      qp_ingest.py    qp-ingest: JSONL loader, chunk, embed and upsert a corpus
      qp_fetch_corpus.py  qp-fetch-corpus: download a public corpus for qp-ingest and Locust
    deploy/           the deploy tool: qp-deploy on Domino 6.2
      deploy_settings.py  DeploySettings: deploy.yaml (the topology aliases, shared defaults, one block per App, the overrides), QP_DEPLOY_* overrides, require_deploy_file (a refusal naming where it looked); Placement (local or served); override_values, each override by its path
      workflow_components.py  the workflow file the workflow App reads: its components (name, type, backend), the variable of a value's path and the path of a variable, the overrides' keys walked and their values checked with the workflow settings' models
      overrides.py    the overrides, this site's values of the workflow file: checked before anything is published (and a workflow setting refused in a free list), then each App's exports
      registered_backends.py  registered_backends: a component type's backends, read from the names of its opener modules in composition/openers/ (rule 5) without importing them
      topology_alias.py  resolve_topology: the chosen topology alias against the workflow file, its Apps, each served component's model-server kind and route, and run label, or a refusal (a backend a component's type lacks included)
      model_server_kinds/  the model-server kinds (rule 6), each with its launch script domino/model_servers/<kind>.sh; importing it imports every module
        base_model_server_kind.py  BaseModelServerKind (the types it serves, the route of each, its probe, what its script reads, what it refuses), MODEL_SERVER_KINDS, model_server_kind
        fastapi_kind.py  fastapi: qp-model-server, an embedder or a reranker on the /v3 routes
        ovms_kind.py  ovms: OpenVINO Model Server on its /v3 routes, from the OVMS bundle it refuses when missing
      publish.py      where every App starts (LaunchPaths: the toolkit's launch script, found going up from the use case's directory, and that directory), the entry scripts of a topology's Apps (the workflow app, or a model server told its component and kind), App or Agent, and publishing them in order
      endpoint_url.py where an App answers: {host}/{owner}/{project}/r/notebookSession/{instance id}
      app_id_cache.py the App ids already seen (.qp-deploy-cache.json): an App is redeployed, never created twice
      status.py       qp-deploy status: one line per App, then the workflow App's /readyz
      qp_deploy.py    qp-deploy: the parser, url, --dry-run
      apps_client.py  the Domino 6.2 beta Apps API calls qp-deploy makes (find, create, add a version, get, list instances) and the wait until Running
    loadtest/         the load-test client, with locustfile.py
      loadtest_settings.py  LoadTestSettings: every QP_LOADTEST_* variable, the four values a model-server request carries (two sizes, required by the rerank target, and two aliases, the workflow file's), Domino auth
      request_plans.py  what to send: the request pool (synthetic questions or a questions file) and one plan per request
      response_checks.py  what to record: a response becomes a failure or STAGE rows, one per stage its timings_ms names
      target_record.py  the target record (_target.json): TargetRecord, written as a run stops and read by qp-compare
      qp_compare.py   qp-compare: Locust *_stats.csv files to markdown tables
    testing/          the tests the toolkit ships, which a user runs on a component of their own; a use case's tests import them from query_processing.testing, no package imports them
      component_contract.py  ComponentContract: subclassed with make_component and make_request, it checks the answer's shape, the API format both ways, and the served class through an in-process model server
      router_contract.py  RouterContract: ComponentContract for a router, with routes to write; the route chosen is one of them, the confidence between 0 and 1
  src/fake_domino/    a stand-in for Domino 6.2's Apps API: qp-deploy's test double and what --dry-run runs.
                      It imports nothing from query_processing, and only deploy/qp_deploy.py imports it (both checked by
                      tests/structure/test_layering.py)
    api_shapes.py     the Apps API request/response shapes, transcribed from the 6.2 reference (what the fake serves)
    state.py          FakeDominoApps: the fake's Apps, versions and instances
    instances.py      BaseInstanceScheduler: what decides an instance's status; by default the --behaviour scripts
    apps_routes.py    /api/apps/beta/apps: list, create, add a version, get, list a version's instances
    stand_ins.py      hardware tiers, /access-token and an App instance's /readyz
    server.py         create_fake_domino (FastAPI, Swagger at /docs), the in-process client, qp-fake-domino
    emulation/        TEMPORARY: the emulated Domino (proxy, App placement on machines), see emulation/README.md
  locustfile.py       RagUser (FastHttpUser) + opt-in StepLoadShape; a use case's Makefile runs it (locust -f) from the use case's directory
  docker-compose.yml  Elasticsearch 9.0.0 (:9200) + OVMS reranker (:8001) + OVMS embedder (:8002), a use case's directory its project directory
  ovms/               versions.env (single OVMS version pin), export_models.sh (into ovms/models of the working directory, a use case's)
  domino/             app.sh (one entry point: the workflow app or a model server, from QP_ROLE, run in the use case's directory, QP_USE_CASE_DIR), model_servers/ (one launch script per model-server kind: fastapi.sh, ovms.sh), Dockerfile (targets `rag` and `ovms`)
  tests/              pytest: no model weights, no network, no Elasticsearch; one folder per package of src/query_processing/
    domain/ application/ config/ adapters/ infrastructure/ composition/ entrypoints/ loadtest/ deploy/   the tests of that sub-package
    scripts/          domino/app.sh and ovms/export_models.sh, run against fake binaries
    structure/        the import, placement and code-shape rules, read from the source with ast
    fake_domino/      the fake Domino's own tests; they import nothing from query_processing
    emulation/        TEMPORARY: the emulation's Terraform, parsed, and the rules that keep it removable
    helpers/          named fakes, fixtures, fixture_workflow.yaml (the tests' workflow file: made-up models) and its fixture_prompts.yaml, and the two roots (REPO_ROOT, TOOLKIT_ROOT) shared by several test files (on pytest's path)
query_processing_rag_usecase/   the RAG use case, one distribution that installs the toolkit: settings only, no code
  pyproject.toml      the distribution (query-processing-rag-usecase), its one dependency the toolkit; packages = []
  workflow.yaml       the RAG's workflow file: its components (models, endpoint, store) and its graph (stages, sizes)
  prompts.yaml        the RAG's prompts, which its generate stage names (answer_from_context)
  deploy.example.yaml  its deploy settings example: the topology aliases, the Apps, this site's overrides (cp to deploy.yaml)
  .env.example        its environment example, the load-test sizes of a rerank request included (cp to .env)
  Makefile            its run targets (serve, ingest, loadtest, compare, the exports, calibrate, deploy, status, infra-up, ...), run here
  tests/              its own tests, through the two seams: its workflow file answers, a dry run of each topology alias
  results/            load-test output (git-ignored), as data/, models/, ovms/models/ and chroma/ written here
emulation/          TEMPORARY: Terraform and scripts for an emulation of Domino on Google Cloud
docs/adr/           the decisions
Makefile            installs both distributions in one command; hands check and each gate to query_processing/, then runs the use case's tests; the one .venv here
.python-version     the project's Python, read by uv and by domino/app.sh
```

## Load test and comparison

One headless Locust run per topology, then a comparison table:

```bash
make loadtest RUN_NAME=monolith HOST=http://localhost:8888
make loadtest RUN_NAME=split-reranker HOST=http://localhost:8888
make loadtest RUN_NAME=split-all HOST=http://localhost:8888
make compare                          # results/comparison.md
```

`make loadtest` runs `locust --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME)`
(defaults `USERS=20 SPAWN_RATE=5 RUN_TIME=2m`) with the toolkit's `locustfile.py`
(`locust -f ../query_processing/locustfile.py`), from the use case's directory, and
writes there `results/<RUN_NAME>_stats.csv`, `_failures.csv` and
`.html`. `HOST` is the service base URL without a trailing slash (on Domino: the
App URL). Locust rewrites the stats file once a second and not again as it shuts
down, so `locustfile.py` rewrites it once more when the run stops:
`_stats.csv`, which `qp-compare` reads, holds the whole run, its last second
included. `_failures.csv` is Locust's own and may lack a failure of that last
second; the failure count in `_stats.csv` and the HTML report have it.

| Knob | Effect |
| --- | --- |
| `USERS`, `SPAWN_RATE`, `RUN_TIME` | Makefile variables passed to `locust -u -r -t` |
| `QP_LOADTEST_TARGET` | the user class: `rag` (default), `rerank` or `embeddings` |
| `QP_LOADTEST_QUERY_WEIGHT` / `QP_LOADTEST_RETRIEVE_WEIGHT` | task weights, default `1` / `3`; a whole number of 0 or more |
| `QP_LOADTEST_WAIT_MIN_S` / `QP_LOADTEST_WAIT_MAX_S` | each user waits a random time in this range between requests, default `0.2` / `1.0` s |
| `QP_LOADTEST_SHAPE=step` | step load 5 → 10 → 20 → 40 users, 60 s each (set `RUN_TIME=4m` or more); any other value but unset is refused |
| `QP_LOADTEST_STEP_USERS` / `QP_LOADTEST_STEP_SECONDS` | the steps of the step shape, default `5,10,20,40` / `60` |
| `QP_LOADTEST_QUESTIONS` | distinct synthetic questions the users ask, default `50` |
| `QP_LOADTEST_BEARER_TOKEN` | sent as `Authorization: Bearer ...` (Domino run token) |
| `DOMINO_API_KEY` | sent as `X-Domino-Api-Key` (from outside Domino) |

A request counts as a success only when the response is HTTP 200 with a JSON
object body. Any other status fails as `HTTP <status>: <first 200 characters of
the body>`, and a 200 whose body is not JSON, or is JSON but not an object, also
counts as a failure (it used to vanish from the statistics altogether).
Every successful response's `timings_ms` is re-emitted as Locust `STAGE` rows,
one per stage it names, in its order: `/query:embed`, `/query:retrieve`,
`/query:rerank`, `/query:generate` for this graph, same for `/retrieve`; the
total and a stage timed at zero (generate on `/retrieve`) get no row. The client
reads the stages from the response and keeps no list of them, so a graph with
other stages is measured as it is, and p50/p95/p99 per stage land in the same
CSV as the endpoint totals.

Each run also writes a target record, `results/<RUN_NAME>_target.json` (next to
the `--csv` files; nothing is written when Locust runs without `--csv`). Against
the workflow app it counts responses per `X-RAG-Topology` / `X-RAG-Run-Label`
pair the app reported, most responses first:

```json
{"target": "rag", "seen": [{"topology": "monolith", "run_label": "monolith", "responses": 2400}]}
```

A workflow app given no topology alias (a laptop run, unless `QP_TOPOLOGY_ALIAS`
is set) sends no `X-RAG-Topology`, and its responses are counted with
`"topology": null`, never under a topology nobody chose.

A run straight at a model server (`make loadtest-models`) writes
`{"target": "rerank", "seen": []}` (or `embeddings`), since model servers send
no such headers.

`make compare` feeds every `results/*_stats.csv` prefix (sorted, so
`monolith` is the baseline) to `qp-compare`, which writes one markdown table
per endpoint and per stage the runs contain: requests, failure %, RPS, p50, p95,
p99 and Δp95 versus the first run. Runs of graphs with different stages
compare: a run that lacks a stage reads `n/a` in that stage's table, not 0.
Pick another baseline by calling it directly:
`qp-compare results/split-all results/monolith --out results/comparison.md`.
When a run has a target record, the intro line names the topology alias (when
the app reported one) and run label it saw beside the run name, and a warning line
flags any run that saw more than one pair, because its numbers mix
configurations (for example an app redeployed mid-run). A run without the file
is listed by its run name alone, as before.

## Experiment 2: serving the models with FastAPI versus OVMS

`qp-model-server --serve NAME` serves one component of the workflow file per
process, an embedder or a reranker, on the same routes as OpenVINO Model Server
(`POST /v3/embeddings`, `POST /v3/rerank`, `GET /v3/models/<alias>`), using
the in-process sentence-transformers adapters: it loads the weights the
component's `source_id` names and serves them under its `alias`, whatever its
backend says.
Both servers speak the same API formats (`openai` for embeddings, `cohere` for
rerank), so the workflow app cannot tell them apart and the comparison needs no
code change: point `QP_COMPONENTS__RERANKER__SERVED__URL` /
`QP_COMPONENTS__EMBEDDER__SERVED__URL` of a served component at one or the
other. Both servers answer a request for an unknown model with the same
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
qp-compare results/rerank-ovms results/rerank-fastapi --out results/rerank.md
qp-compare results/embed-ovms results/embed-fastapi --out results/embeddings.md
```

End-to-end comparison (the split topologies with either server behind them):

```bash
QP_COMPONENTS__RERANKER__BACKEND=served QP_RUN_LABEL=split-reranker-fastapi \
  QP_COMPONENTS__RERANKER__SERVED__URL=http://localhost:8011/v3/rerank make serve
make loadtest RUN_NAME=split-reranker-fastapi HOST=http://localhost:8888
```

On Domino, the model-server kind of each model server App picks the server:
`server: fastapi` in its block of `deploy.yaml` (or in `defaults`) publishes the
FastAPI server (no OVMS binary needed), `server: ovms` OVMS, and the two can
serve the embedder and the reranker of one deploy side by side
(`split-all+embedder=fastapi+reranker=ovms`). `qp-deploy` points the workflow
app's `QP_COMPONENTS__RERANKER__SERVED__URL` /
`QP_COMPONENTS__EMBEDDER__SERVED__URL` at each App URL plus the route its kind
declares, `/v3/rerank` / `/v3/embeddings` for both.
`QP_LOADTEST_TARGET` selects the Locust user class: `rag` (default,
`/query` + `/retrieve`), `rerank` or `embeddings`.

## HTTP API

| Route | Body → result |
| --- | --- |
| `POST /query` | `{question, conversational_history?, metadata_filter?}` → `{request_id, topology, run_label, question, mode, answer, passages[], timings_ms{embed,retrieve,rerank,generate,total}, metadata_filter}`; `conversational_history` holds the earlier turns, `{role: user\|assistant, content, metadata_filter?}`, a user message's conditions being read by the prompts only, never applied to retrieval; the response's `metadata_filter` is the conditions applied; `topology` is the topology alias, `null` when none was given; `timings_ms` has one key per stage of the graph, in its order, then `total` (lexical retrieval: no `embed`) |
| `POST /retrieve` | same request → same response with `answer: null` (`generate` skipped, `generate = 0`) |
| `GET /healthz` | `{"status":"ok"}` |
| `GET /readyz` | `{"status":"ready"\|"not_ready","checks":[{name,ok,detail}]}`, one check per vector store (index count > 0), embedder and reranker a stage calls, named after the component (`vector_store`, `embedder`, `reranker`); `503` when a check fails |

`X-RAG-Run-Label` (the run label) and `X-Request-ID` are set on every response
(an incoming `X-Request-ID` is propagated), and `X-RAG-Topology` (the topology
alias, `QP_TOPOLOGY_ALIAS`) on every response when one was given: `qp-deploy`
gives the alias it deploys, and without one the header is absent. Errors: dependency failure → `503`
`{"error":"<component>_unavailable","detail","target"}`, upstream timeout →
`504` `{"error":"<component>_timeout",...}`, validation → `422`. `root_path`
follows `DOMINO_RUN_HOST_PATH`, so the app answers with or without Domino's
prefix on the wire and `/docs` works behind the proxy.
`qp-serve --host 0.0.0.0 --port 8888` are the defaults (host and port are
flags, not settings); `qp-serve --domino` (what `domino/app.sh` runs) says the
app runs as a Domino App, so every component on the `served` backend, another
App behind Domino's proxy, is called with the run token; a component on `api`
is called with its own key, on Domino or not. A settings mistake ends `qp-serve`
with a one-line usage error, not a traceback, before any component opens; so
does a graph that cannot run, a stage setting its stage type refuses (such as a
user prompt without `{context}`) included. Only what needs an open component,
a store that ranks no text under a lexical or hybrid retrieve stage, is found
as the graph is built: the app then fails to start before it serves, and the
server logs the refusal as one line, `qp-serve: cannot start: stage
'retrieve': retrieval_mode='lexical' needs a store that ranks text, …`.

## Configuration (the workflow file, `QP_*` environment variables or `.env`)

### The workflow file

The workflow is declared in a workflow file: its components, each with its
backend, and its graph, whose stages each name the component they call and hold
their own settings, its plain edges as chains, and its conditional edges in a
block of their own, spelled with LangGraph's names (ADR 0004), each stating in
`reads` the fields its edge condition is asked about (ADR 0007). A command finds
it as `qp-deploy` finds `deploy.yaml`: the file `QP_WORKFLOW_FILE` names (read
from the environment, not from `.env`), else `workflow.yaml` in the working
directory, a use case's; with neither, it stops, naming both places. The toolkit
ships none and defaults none of a use case's models, prompts, sizes, index or
collection. The RAG use case's, `query_processing_rag_usecase/workflow.yaml`, is
the first example of this README.

A component can take any name. Its type is the port it is used through,
`embedder`, `reranker`, `llm` or `vector_store`, and defaults to its name, so a
second component of one type names it: `coarse: {type: reranker, backend:
local, source_id: ...}`, which a second rerank stage calls with `call:
coarse`.

Every stage states `reads` and `writes`, the workflow state fields it reads and
writes. A stage is of the stage type its `type` names, else of the stage type
of its own name (`retrieve` and `generate`), else a plain call, which only calls
its component: a key of the stage that names an attribute of the component's
request fills it, such as `top_k: 10`; `reads` fill the other attributes in
order; and `writes` take the answer's attributes in order, and can stop early.
The RAG's `embed` and `rerank` stages are plain calls. A `retrieve` stage reads
`[query_embedding]` for `dense`, `[standalone_query]` for `lexical` and both,
in that order, for `hybrid`, and writes one field; a `generate` stage reads the
question, then the passages, and writes one field.

The vector store is a component like the models: its backend is named
after its product, `elasticsearch` or `chroma`, or is `fake`, each with its
block, and `qp-ingest` fills the one the file declares.

| Component type | Key | In the RAG's file | Meaning |
| --- | --- | --- | --- |
| every type | `backend` | `local`, `local`, `api`, `elasticsearch` | how it runs, one of the openers registered for its type: `local` \| `served` \| `api` \| `fake` for an embedder, the same and `none` (no rerank) for a reranker, `served` \| `api` \| `fake` for an LLM, `elasticsearch` \| `chroma` \| `fake` for a vector store; `served` is a model server of ours, `api` an endpoint we do not run; an unknown one is refused at start-up, naming the known ones. Nothing else decides it: `qp-deploy` sets it for each component the topology alias names, and on a laptop you set it |
| every type | `type` | not written: the component's name | `embedder` \| `reranker` \| `llm` \| `vector_store` |
| every model | `source_id` | the two models' Hugging Face ids | the weights, a Hugging Face id or a folder; required of an embedder and a reranker |
| every model | `alias` | not written: the last part of `source_id` | the name a model server of ours serves the model under (`snowflake-arctic-embed-l-v2.0`), and the model name a `served` component sends; an LLM on `served` needs one, having no `source_id` |
| `embedder` | `query_prefix` | `query: ` | put in front of each query before it is embedded, never of a passage (arctic-embed expects it); the `local`, `served` and `api` embedders apply it, the `fake` one has none, and a model server adds none to what it receives; empty for a model that takes none |
| `embedder` | `local.batch_size`, `local.threads` | not written: `64`, `0` | texts per encode call (>= 1); native (OpenMP/BLAS) threads one call may use, `0` = calibrated at start-up, up to the budget's share |
| `embedder` | `local.normalize` | not written: `false` | scale each vector the in-process (`local`) embedder returns to unit length; off is safe because every store compares vectors by cosine, which ignores their length. It does not change OVMS: its own export normalises on its side (the exporter's default) |
| `reranker` | `local.batch_size`, `local.threads` | not written: `0`, `0` | passages per predict batch and native threads per call; `0` = calibrated at start-up, the batch size up to `local.calibration_passages` |
| `reranker` | `local.calibration_passages` | not written: `20` | passages in calibration's sample rerank request, all kept; also the largest batch size it tries (>= 1). The reranker's own: calibration reads no stage's setting |
| every model | `served.url` | not written | the full route of the model server of ours that serves it, e.g. `http://localhost:8001/v3/rerank`; required with the `served` backend. The rest follows: the standard format of its type, the alias as the model name, the Domino run token under `qp-serve --domino` and no key elsewhere |
| `embedder` | `served.batch_size` | not written: `64` | texts per model-server request (>= 1) |
| `llm` | `served.temperature` | not written: `0` | sampling temperature, 0 to 2 |
| every model | `api.url` | the LLM's: OpenAI's chat route | the full route of an endpoint we do not run, e.g. `https://api.cohere.com/v2/rerank`; required with the `api` backend |
| every model | `api.api_format` | not written: `openai`, `cohere`, `openai` | the request and response shapes the route speaks |
| every model | `api.model` | the LLM's: `gpt-4o-mini` | the model name sent; empty = the component's alias |
| every model | `api.api_key` | never written | from the environment only: empty = no `Authorization` header; `domino` = the Domino run's token from `domino.token_url`; anything else = a bearer token |
| `embedder` | `api.batch_size` | not written: `64` | texts per endpoint request (>= 1) |
| `llm` | `api.temperature` | not written: `0` | sampling temperature, 0 to 2 |
| every model | `http.timeout_s` | the LLM's: `120`; not written for the others: the top-level `http.timeout_s` (`30`) | timeout of one request to the component's endpoint on `served` or `api` (> 0), and of the Domino token fetched for it; the LLM's is longer, since an answer takes longer than a model-server call |
| every model | `http.retries` | not written: the top-level `http.retries` (`2`) | requests after the first (>= 0), on a transport error, a 5xx, a 429, and a 401 when the call carries the Domino run token; any other 4xx fails at once |
| every model | `http.backoff_s` | not written: the top-level `http.backoff_s` (`0.2`) | wait before the first retry, doubled before each next one (>= 0) |
| `vector_store` | `elasticsearch.url` | `http://localhost:9200` | the cluster; `http://user:password@host:9200` for basic auth |
| `vector_store` | `elasticsearch.index` | `rag_passages` | the index holding the passages; required on `elasticsearch`: no default, start-up refuses an empty one |
| `vector_store` | `elasticsearch.api_key` | never written | from the environment only (`QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`): an Elasticsearch API key, the alternative to basic auth; empty = none |
| `vector_store` | `elasticsearch.text_analyzer` | not written: `standard` | analyzer of the passages' text, used by lexical and hybrid search: `standard`, or a language's such as `french` (stemming, elision, stop words). An index keeps the analyzer it was created with, so changing it means ingesting into a new index |
| `vector_store` | `chroma.url` | not written: empty | a Chroma server, e.g. `http://localhost:8000` (name the port); empty = embedded Chroma at `chroma.path` |
| `vector_store` | `chroma.path`, `chroma.collection` | not written: `chroma`; `rag_passages` | the folder of the embedded Chroma store, in the working directory, and the collection holding the passages; the collection is required on `chroma`: no default |
| `vector_store` | `fake.seed_documents` | not written: `200` | synthetic documents seeded into the `fake` store when it opens (>= 1) |

A key the type does not have, a block of a backend it does not have included
(an LLM's `local`), is refused at start-up naming the key and the type's keys.

The `http` block says how patiently the workflow app calls a component on
`served` or `api`. It belongs to the component, not to one backend: a
topology that moves a component from `local` to `served` keeps it, and a
component on another backend ignores it. Each value it leaves out is the
default every component shares, the workflow file's top-level `http` block
(`QP_HTTP__TIMEOUT_S`, `QP_HTTP__RETRIES`, `QP_HTTP__BACKOFF_S`), so
setting one of these changes it for every component that does not set its
own. With every default in place the embedder and the reranker wait 30 s per
request and the LLM 120 s, and each makes up to three requests, 0.2 s then
0.4 s apart:

```bash
QP_COMPONENTS__LLM__HTTP__RETRIES=0 make serve   # the LLM: one request, no retry
QP_HTTP__RETRIES=0 make serve   # no retry for every component that sets none
```

What several components share is four more top-level blocks of the file, each
named after what it is about. The RAG's file writes none of them, so each
value is the default below; a workflow file may write any of them, and a
variable named by its path wins over the file:

```yaml
local_models: {runtime: openvino, openvino_dir: models/openvino, workers: 4, cpus: 0}
calibration:  {budget_s: 90, rounds: 1}
http:         {timeout_s: 30, retries: 2, backoff_s: 0.2}
domino:       {token_url: "http://localhost:8899/access-token", token_ttl_s: 240, tracing: false}
```

| Block | Key | Default | Meaning |
| --- | --- | --- | --- |
| `local_models` | `runtime` | `openvino` | what runs the in-process models, those of the components on `local`: `openvino` \| `torch` |
| `local_models` | `openvino_dir` | `models/openvino` | where `qp-export-openvino` writes and the `openvino` runtime looks first, one folder per model |
| `local_models` | `workers` | `4` | under the `torch` runtime, at most this many in-process inference calls at once (fewer on a small CPU budget); `openvino` ignores it and runs one call at a time (>= 1) |
| `local_models` | `cpus` | `0` | virtual CPUs for the thread budget; `0` = find out (Domino tier, else container), a positive value wins |
| `calibration` | `budget_s` | `90` | per CPU-bound stage, the seconds calibration may search (at start-up or in `qp-calibrate`); no new candidate starts after them (> 0) |
| `calibration` | `rounds` | `1` | timed requests per calibration candidate, after one warm-up (>= 1) |
| `http` | `timeout_s` | `30` | the default timeout of a request to a component on `served` or `api` and of the Domino token fetched for it, for every component whose `http.timeout_s` is not set (the RAG's LLM sets `120`); also the timeout of the hardware-tier lookup |
| `http` | `retries` | `2` | the default retries of a component on `served` or `api` whose `http.retries` is not set, on transport errors, 5xx, 429, and 401 when the call carries the Domino run token; also the hardware-tier lookup's retries, on transport errors |
| `http` | `backoff_s` | `0.2` | the default wait before the first retry, doubled for each next one, for every component whose `http.backoff_s` is not set, and for the hardware-tier lookup |
| `domino` | `token_url` | `http://localhost:8899/access-token` | where a Domino run serves its token: for every `served` component under `qp-serve --domino`, and for an `api` key set to `domino` |
| `domino` | `token_ttl_s` | `240` | how long a fetched Domino token is reused (tokens expire after about 5 minutes) |
| `domino` | `tracing` | `false` | wrap the workflow run with Domino `add_tracing` |

```bash
QP_LOCAL_MODELS__RUNTIME=torch QP_LOCAL_MODELS__CPUS=4 make serve
```

| Stage | Setting | In the RAG's file | Meaning |
| --- | --- | --- | --- |
| every stage | `call` | | the component it calls: `embedder`, `vector_store`, `reranker` or `llm` |
| every stage | `type` | not written: the stage's name | the stage type it is an instance of; a stage of no stage type is a plain call |
| every stage | `reads`, `writes` | each stage's | the workflow state fields it reads and writes, in the order its stage type or its component takes and gives them; required |
| `retrieve` | `retrieval_mode` | `dense` | `dense` (by vector) \| `lexical` (by words, ranked by the store) \| `hybrid` (both, fused by reciprocal rank); `lexical` and `hybrid` need `elasticsearch` or `fake`, and are refused with `chroma` when the graph is built |
| `retrieve` | `top_k` | `20` | candidates from the vector store (>= 1) |
| `retrieve` | `rrf_k` | not written: `60` | rank constant of the hybrid fusion (>= 1); smaller widens the gap between ranks |
| `rerank` | `top_k` | `10` | passages kept after reranking and sent to the LLM (>= 1) |
| `generate` | `prompt` | `answer_from_context` | the prompt of `prompts.yaml` it renders; the fields of its `reads` are the prompt's variables, any number of them, and it writes one field, the answer |

The prompts are in a file of their own, `prompts.yaml`, beside the workflow
file: the one `QP_PROMPTS_FILE` names, else `prompts.yaml` in the working
directory. It feeds the top-level setting `prompts`, and a `prompts` key in the
workflow file is refused. Each prompt, by its name, is a list of messages with a
`role` (`system`, `user` or `assistant`) and a `content`, as OpenAI writes them;
an entry `placeholder: <field>` splices the messages of that field there, with
only their role and content. Each content is a Jinja2 template, rendered on its
own in Jinja2's sandbox with `trim_blocks` and `lstrip_blocks`, its variables
the fields its stage reads; an undefined variable is an error. Start-up refuses
a prompt that is named and not defined, a variable the stage does not read, and
a template that cannot be read, naming its message. The RAG's:

```yaml
answer_from_context:
  - role: system
    content: You answer questions using only the numbered context passages. …
  - role: user
    content: |-
      Context:
      {% for passage in reranked %}
      {{ "\n" if not loop.first }}[{{ loop.index }}] {{ passage.text }}
      {%- endfor %}


      Question: {{ conversational_query }}
```

Each top-level key of the file is a setting, so every value in it has an
environment variable named by its path, its levels joined by `__`, and the
variable wins over the file; a variable named by a prompt's path replaces the
whole prompt, written as JSON:

```bash
QP_GRAPH__STAGES__RERANK__TOP_K=5 make serve
QP_PROMPTS__ANSWER_FROM_CONTEXT='[{"role": "user", "content": "Answer in French: {{ conversational_query }}"}]' make serve
QP_COMPONENTS__RERANKER__BACKEND=none QP_COMPONENTS__LLM__API__API_KEY=<key> make serve
```

An API key is read from the environment or `.env` only, by its path
(`QP_COMPONENTS__LLM__API__API_KEY`): the workflow file is versioned and never
holds one.

Sources win in this order: constructor arguments, environment variables, `.env`,
the workflow file and the prompts file, then the defaults of the code. A top-level key of the file
that is no setting, and a key a stage or the graph does not know, are refused at
start-up with the key named, as is a stage without `call`, `reads` or `writes`.
A stage checks only its own settings, so nothing checks any more that the rerank stage's `top_k`
stays within the retrieve stage's: above it, the reranker keeps every candidate,
and a reranker endpoint is asked for a `top_n` above the documents it is sent.

Lexical retrieval is a workflow file without the embed stage, its components
the same (ingestion still embeds the passages, with the embedder):

```yaml
components: {…}
graph:
  stages:
    retrieve: {call: vector_store, retrieval_mode: lexical, top_k: 20, reads: [standalone_query, metadata_filter], writes: [candidates]}
    rerank:   {call: reranker, top_k: 10, reads: [standalone_query, candidates], writes: [reranked]}
    generate: {call: llm, prompt: answer_from_context, reads: [conversational_query, reranked], writes: [answer]}
  edges:
    - [START, retrieve, rerank]
    - [generate, END]
  conditional_edges:
    - {source: rerank, path: is_query_mode, reads: [mode], path_map: {true: generate, false: END}}
```

`QP_GRAPH__STAGES__RETRIEVE__RETRIEVAL_MODE=lexical` over the RAG's file is
refused at start-up: the retrieve stage would read the query's vector as its
words. With `QP_GRAPH__STAGES__RETRIEVE__READS='["standalone_query", "metadata_filter"]'` too it is
refused still, since the embed stage would write a vector that nothing reads.

### What start-up refuses in a graph

`qp-serve` checks the graph as it reads its settings, before any component
opens, and refuses one that cannot run with a usage error naming what is wrong
and what was expected (`application/graph_check.py`). The checks know no
particular stage: they read the `reads` and `writes` each stage states, and
what each stage type declares, which is the type of component it calls, its own
settings, and the lists it can use (the retrieve stage reads what its retrieval
mode needs). They refuse:

- a stage whose `type` names no stage type, naming the known ones;
- a stage whose settings or lists its stage type refuses, naming what it
  expects;
- a plain call whose keys and `reads` do not fill its component's request
  exactly once, or whose `writes` list more fields than its answer has, showing
  what the component takes and gives;
- a stage whose `call` names no component, naming the components, or a
  component of another type than its stage type calls, naming that type;
- a stage named `START`, `END` or `total`: `total` is the key of the whole
  run's time in `timings_ms`, where the stage's timing would collide with it;
- an edge from or to a name that is no stage (`START` only starts an edge,
  `END` only ends one), and a graph with no edge from `START`;
- a conditional edge without `reads`, whose `path` is no registered edge
  condition, whose `reads` list another number of fields than its edge
  condition takes, whose edge condition is already used on the same source,
  or whose `path_map` has other rows than its answers: a `true` and a `false`
  target for a yes-or-no edge condition; for `label`, one row for each label
  the writer of the one field it reads declares, a field that declares none
  being refused (ADR 0007; on LangGraph 1.2.21 a label missing from
  `path_map` would raise only when a request takes it). A conditional edge
  may start at `START` (`application/edge_check.py`);
- a stage, or a conditional edge once its source has run, that reads a field
  of the workflow state that is not sure to be set when it runs. A field is sure to be set when the request provides it (the
  conversational query and history, the standalone query, the mode, the
  metadata filter, the timings),
  or when a stage writes it on every path from `START` to the stage. A dense
  or hybrid retrieve stage with no embed stage before it on every path is
  refused by this rule:

```
qp-serve: error: stage 'retrieve' reads the workflow state field 'query_embedding', which the request does not provide and is written by no stage of the graph (expected a field the request provides, one of: 'conversational_history', 'conversational_query', 'metadata_filter', 'mode', 'standalone_query', 'timings_ms', or one a stage writes on every path from START to 'retrieve')
```

- a stage that reads a field as another type than what writes it: the
  request, or a stage, whose type is that of the answer attribute or the stage
  type's output it is bound to. A lexical retrieve stage that still reads
  `query_embedding` is refused so:

```
qp-serve: error: stage 'retrieve' reads 'query_embedding' as str, but 'embed' writes it as list[float] (expected a field written as str)
```

- two writers, the request or stages, that write one field as two types,
  naming both;
- a field a stage writes that no stage and no response reads (the response
  reads `answer` and `reranked`), as a likely typo;
- a key of a plain call that does not fit the type of the request attribute it
  fills, such as `top_k: ten`, naming the key, its value and the type expected.

The workflow state is built from the same fields: each field the request or a
stage writes, with its type, and `timings_ms`
(`application/state_fields.py`).

Only whether the vector store ranks text needs the store open, so a lexical
or hybrid retrieve stage over `chroma` is refused as the graph is built, still
before the app serves, with one line in the server's log.

### Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `QP_TOPOLOGY_ALIAS` | unset | the topology alias deployed, which `qp-deploy` sets: a label only, it changes no backend; echoed as `topology` and `X-RAG-Topology`, and unset (or empty) the header is left out and `topology` is `null`; letters, digits and `._+=:-`, 1 to 128 characters |
| `QP_RUN_LABEL` | the default run label | names the measured configuration, echoed as `run_label` and `X-RAG-Run-Label`; default: each component with its backend, in the workflow file's order, joined by `+` (`embedder=local+reranker=local+llm=api+vector_store=elasticsearch` for the RAG's file); `qp-deploy` sets it to the topology alias plus each served component's model-server kind; a variant the backends do not show, such as a retrieval mode or a hardware tier, sets its own (`run_label` under the overrides of `deploy.yaml`, with `qp-deploy`); letters, digits and `._+=:-`, 1 to 128 characters |
| `QP_WORKFLOW_FILE` | unset | the workflow file to read, before a `workflow.yaml` in the working directory (environment only, not `.env`); with neither, a command stops, naming both places |
| `QP_COMPONENTS__<NAME>__...` | from the workflow file | one value of a component of the workflow file, named by its path, such as `QP_COMPONENTS__RERANKER__LOCAL__THREADS`, `QP_COMPONENTS__LLM__API__API_KEY` or `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` (see the workflow file above) |
| `QP_GRAPH__<KEY>__...` | from the workflow file | one value of the workflow file's `graph` block, named by its path, such as `QP_GRAPH__STAGES__RERANK__TOP_K` (see the workflow file above) |
| `QP_LOCAL_MODELS__...`, `QP_CALIBRATION__...`, `QP_HTTP__...`, `QP_DOMINO__...` | from the workflow file, else the defaults | one value of a block the components share, named by its path, such as `QP_LOCAL_MODELS__CPUS` or `QP_DOMINO__TRACING` (see the shared blocks above) |
| `DOMINO_API_PROXY`, `DOMINO_HARDWARE_TIER_ID` | set by Domino | Domino's own run variables, not settings of ours: read as-is (no `QP_` prefix) to look up the run's hardware tier |
| `QP_LOADTEST_TARGET` | `rag` | Locust side: what a run drives: the workflow app (`rag`), or a model server directly (`rerank`, `embeddings`) |
| `QP_TOP_K_RETRIEVE`, `QP_TOP_K_RERANK` | none: required by the `rerank` target | Locust side only: a `rerank` target is sent this many documents and this `top_n`; the use case's sizes, which its `.env.example` states (the RAG's: `20`, `10`, its stages' `top_k`), and the workflow app does not read either. The aliases a model-server target sends default to the workflow file's, which only that target reads |
| `QP_LOADTEST_BEARER_TOKEN`, `DOMINO_API_KEY` | | Locust side: auth header, see the load-test table |
| `QP_LOADTEST_QUERY_WEIGHT`, `QP_LOADTEST_RETRIEVE_WEIGHT`, `QP_LOADTEST_SHAPE` | `1`, `3`, unset | Locust side: task weights (0 or more) and the step shape (`step` or unset) |
| `QP_LOADTEST_WAIT_MIN_S`, `QP_LOADTEST_WAIT_MAX_S` | `0.2`, `1.0` | Locust side: the range each user waits between requests, in seconds |
| `QP_LOADTEST_STEP_USERS`, `QP_LOADTEST_STEP_SECONDS` | `5,10,20,40`, `60` | Locust side: the users of each step and the seconds per step of `QP_LOADTEST_SHAPE=step` |
| `QP_LOADTEST_QUESTIONS` | `50` | Locust side: distinct synthetic questions the users ask |
| `QP_LOADTEST_QUESTIONS_FILE` | unset | Locust side: ask the questions of this file instead (JSON lines with a `question` key, as `qp-fetch-corpus` writes, or one question per line); `QP_LOADTEST_QUESTIONS` then does not apply. A JSON line can also carry `conversational_history` and `metadata_filter`, in the shape of the `/query` body, and the request sends them: the file decides the mix of first messages and follow-ups. Any other key of a line is left out |

Every tunable is a setting of a component, a setting the components share,
or a setting of a stage of the graph, each read from the workflow file and its
environment path; only the topology alias and the run label, two labels, are
flat. They are the fields of `WorkflowSettings` in `config/workflow_settings.py`; adapters
get the values as constructor arguments from the `composition/` package. Sources
win in the order the workflow file's section gives. An invalid runtime or
component type, a run label or topology alias no header can carry (a default
run label too long or with a component name outside those characters included:
set `QP_RUN_LABEL` then), a temperature outside 0 to 2, a component on
the `served` or `api` backend whose URL (`served.url` or `api.url`, the block
of its backend) does not start with `http://` or `https://`, a `served`
component without an alias, a missing workflow file or an unknown key in it
fails as the settings are read. A graph that cannot run, a stage setting its stage type refuses included
(an unknown key, a retrieval mode, a size below 1, a user prompt missing a
placeholder), is refused as `qp-serve` reads its settings
(see [What start-up refuses in a graph](#what-start-up-refuses-in-a-graph)),
and lexical or hybrid retrieval on a store that ranks no text fails as the
graph is built. All are at start-up, with the offending value. For example:

```
QP_COMPONENTS__RERANKER__BACKEND='served' needs QP_COMPONENTS__RERANKER__SERVED__URL, the full route of the endpoint (for example http://localhost:8001/v3/rerank); got ''
```

An unknown backend or API format stops every command that opens components
(`qp-serve`, `qp-model-server`, `qp-ingest`, `qp-calibrate`) before it opens
any, as a settings mistake: the settings cannot see the openers and the
formats, which are registered in the composition and the adapters, so each
command asks their registries as it reads its settings, and the registry
refuses the name, after the variable it is read from, and lists the known ones:

```
qp-serve: error: QP_COMPONENTS__RERANKER__API__API_FORMAT: unknown rerank API format 'jina' (expected one of: 'cohere')
```

Earlier versions named some settings differently: see
[Moving from the old names](#moving-from-the-old-names).

**Thread budget.** When the first in-process model loads, the process counts its
virtual CPUs: `QP_LOCAL_MODELS__CPUS` if positive; on Domino, the `cores` of the run's
hardware tier from `GET $DOMINO_API_PROXY/api/hardwaretiers/v1/hardwaretiers/$DOMINO_HARDWARE_TIER_ID`
(a 403 without `ViewHardwareTiers` only logs a warning); otherwise the
container's CPUs (`joblib.cpu_count()`, which honours the cgroup quota). How
the whole CPUs are used depends on `QP_LOCAL_MODELS__RUNTIME`. Under `openvino`
one model call runs at a time and each model uses every CPU (8 CPUs: 1 worker x
8 threads): an OpenVINO model fixes its threads when it loads and holds one
inference request, so a second call on it while one runs fails with `Infer
Request is busy`. Under `torch` the CPUs are split as
`workers = min(QP_LOCAL_MODELS__WORKERS, cpus)` and `threads = cpus // workers` per
call (8 CPUs: 4 workers x 2 threads), each model call running inside a
threadpoolctl scope of that size. A non-zero `local.threads` of a component
(`QP_COMPONENTS__EMBEDDER__LOCAL__THREADS`) replaces that model's threads under
either runtime.
Autoscaled Apps start more pods of the same tier, so every new process works its
budget out again; the result is one `thread_budget` JSON log line, which names
the runtime.

**Calibration.** The budget's share is an upper bound, not always the fastest
choice. So when an in-process model loads, the process times its stage (embed
or rerank) under each candidate and keeps the fastest. It does so only for a
parameter of a component's `local` block left at `0`: a reranker's
`batch_size` tries its `calibration_passages` (20, the size of the sample
request), then half of it, down to 1; the `threads` of a reranker or an
embedder try the budget's share, then half of it, down to 1. The embedder's batch size is not
searched: a query is one text. Batch sizes are timed first, on the budget's
share, then each smaller thread count with the best batch size: a load is the
expensive part. Each candidate serves one untimed warm-up request, then
`QP_CALIBRATION__ROUNDS` timed ones, through the same adapter a real request
uses. It costs start-up time, up to `QP_CALIBRATION__BUDGET_S` per stage (plus
the trial running when it runs out); no new App instance is needed. The result
is one `stage_calibrated` JSON log line per stage: the choice, every trial and
its seconds, the time spent, and whether the budget cut the search short.

Why: on a 2-CPU App machine (one core, two hyper-threads), under `openvino`, a
rerank request of 20 passages took 5.49 s with one batch of 20, 3.38 s with
batches of 2, and 2.94 s with batches of 2 on 2 real threads: 0.340 requests
per second instead of 0.182. OpenVINO had kept to the physical core until it was
allowed hyper-threads (`ENABLE_HYPER_THREADING`, now always set). Two copies of
the model with one thread each gave nothing more.

Limits. It times one request at a time, which is how a stage runs under
`openvino`. With a fixed number of users, latency is the users divided by the
requests per second, so the fastest request is the best at any number of users.
Its passages are synthetic, cut to 150 to 800 characters. An App does not
remember the result: every start and every replica calibrates again, about 70 s
for the reranker on a 2-CPU machine. `qp-calibrate` finds the values once
instead (below). Under `torch` with several workers, calls share the CPUs, and
calibration does not time them together.

**Pinning the result** (`qp-calibrate`). Run it once on a machine of the App's
hardware tier that holds the models. It calibrates the component `--only`
names, else every embedder and reranker of the workflow file, and loads each
in-process model as an App would (the component's `source_id`,
`QP_LOCAL_MODELS__RUNTIME`, `QP_LOCAL_MODELS__OPENVINO_DIR`, the thread budget),
whatever its backend says. It
searches every parameter, even one already pinned: revisiting it is the point.
`QP_CALIBRATION__BUDGET_S` and `QP_CALIBRATION__ROUNDS` apply as set. Standard
output holds only `.env` lines: for each component a comment, a second comment
with its entry for `overrides.components` of `deploy.yaml`, then the
environment paths of its `local` block that pin the choice. The logs go to
standard error.

```bash
make calibrate                                  # qp-calibrate: every embedder and reranker
QP_CALIBRATION__ROUNDS=3 QP_CALIBRATION__BUDGET_S=600 qp-calibrate --only reranker
```

On a 2-CPU machine under `openvino` it printed this in 96 s (GCP sitting of
2026-11-15, shown with today's variable names and with the `deploy.yaml`
comments it did not print yet; the seconds vary):

```
# embedder (embed): timed on 2 CPUs under openvino, 0.193 s per request; tried 2 candidates in 24.8 s
# deploy.yaml, under overrides.components: embedder: {local: {threads: 1}}
QP_COMPONENTS__EMBEDDER__LOCAL__THREADS=1
# reranker (rerank): timed on 2 CPUs under openvino, 4.691 s per request; tried 6 candidates in 68.2 s
# deploy.yaml, under overrides.components: reranker: {local: {threads: 2, batch_size: 1}}
QP_COMPONENTS__RERANKER__LOCAL__THREADS=2
QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE=1
```

There a batch size of 1 took 4.69 s and one of 2 took 4.77 s, against 7.06 s for
one batch of 20: the two smallest are within a few percent, so either may win a
run. An App of the same tier chose 2 at its own start that day.

To pin them on Domino, put each component's entry, the text after
`overrides.components:` in its second comment, under `overrides.components` of
`deploy.yaml`, beside the component's other overrides (`qp-deploy` refuses the
variables themselves in a `settings` list, naming this place):

```yaml
overrides:
  components:
    embedder: {local: {threads: 1}}
    reranker: {source_id: /domino/datasets/local/<dataset>/models/..., local: {threads: 2, batch_size: 1}}
```

Every App then gets them, and on a laptop the `KEY=VALUE` lines go in `.env`
as they are (`qp-calibrate >> .env`). Either way an App starts without
calibrating, and every replica of the tier runs the same values. The
embedder prints `QP_COMPONENTS__EMBEDDER__LOCAL__THREADS` only: its batch size is
not searched. A name that is no embedder or reranker of the workflow file is a
usage error.
A failure it can explain (a bad setting, a model it cannot load or export) is
one line on standard error and exit code 1. OVMS has its own settings and is
not calibrated by it.

## Moving from the old names

<!-- old-names:begin. The one part of this README that names the toolkit's names
before the rename: its package, its commands and its environment prefix
(query_processing/tests/structure/test_old_names.py allows them here only). -->

The toolkit was `rag_load_test` (its distribution `rag-load-test`), its commands
were `rag-*` and its environment prefix `RAG_`. They are now `query_processing`
(`query-processing`), `qp-*` and `QP_`, and the old names are not read any more.
What a load test sees does not change: `/query`, `/retrieve` and their headers
(`X-RAG-Topology`, `X-RAG-Run-Label`, `X-Request-ID`) keep their names.

- **The first deploy after the rename creates new Apps.** `qp-deploy` finds an
  App by its name, and the RAG use case names its Apps `qp-rag-workflow`,
  `qp-rag-embedder` and `qp-rag-reranker` (`app_name_prefix: qp-rag`; a deploy
  file must now name its prefix). The old `rag-workflow`, `rag-embedder` and
  `rag-reranker` keep running until they are deleted by hand in Domino.
- **Every `RAG_` variable is now `QP_`**, with the same path after the prefix
  (`RAG_COMPONENTS__LLM__API__API_KEY` is `QP_COMPONENTS__LLM__API__API_KEY`).
  The Project environment variables that hold the API keys on Domino must be
  created again under their new names, and a local `.env` or `deploy.yaml`
  written before the rename must be renamed the same way: a variable under the
  old prefix is ignored, not refused.
- **Every `rag-*` command is now `qp-*`** (`rag-serve` is `qp-serve`,
  `rag-deploy` is `qp-deploy`, and so on), and both distributions must be
  installed again (`make install` or `make install-dev` at the repository's
  root).

Earlier versions named some settings differently; none of the old names is read
any more by the workflow app:

| Old | New |
| --- | --- |
| `RAG_TOPOLOGY` (`monolith` \| `split-reranker` \| `split-all`, which moved the components named `embedder` and `reranker` from `local` to `served`) | each component's own backend and served URL (`QP_COMPONENTS__RERANKER__BACKEND=served`, `QP_COMPONENTS__RERANKER__SERVED__URL`), which `qp-deploy` writes; `QP_TOPOLOGY_ALIAS` for the `X-RAG-Topology` header, a label that moves nothing |
| the default run label `<topology>+<choice>=<value>` for each choice off its default (`monolith+rerank=none`, `monolith+retrieval=hybrid`) | each component with its backend (`embedder=local+reranker=none+llm=api+vector_store=elasticsearch`); a retrieval mode is named by setting `QP_RUN_LABEL` |
| `RAG_RETRIEVAL_MODE` | the retrieve stage's `retrieval_mode` in the workflow file (`QP_GRAPH__STAGES__RETRIEVE__RETRIEVAL_MODE`); a lexical workflow file has no embed stage |
| `RAG_TOP_K_RETRIEVE`, `RAG_TOP_K_RERANK` | the retrieve and rerank stages' `top_k` (`QP_GRAPH__STAGES__RETRIEVE__TOP_K`, `QP_GRAPH__STAGES__RERANK__TOP_K`); Locust still reads `QP_TOP_K_RETRIEVE` and `QP_TOP_K_RERANK` for a `rerank` target |
| `RAG_HYBRID_RRF_K` | the retrieve stage's `rrf_k` |
| `RAG_SYSTEM_PROMPT`, `RAG_USER_PROMPT`, `RAG_PROMPTS_FILE` (`config/prompts.yaml`) | the use case's `prompts.yaml` (`QP_PROMPTS_FILE`, else `./prompts.yaml`), which the generate stage names by its `prompt` |
| the generate stage's `system_prompt` and `user_prompt` in the workflow file (`QP_GRAPH__STAGES__GENERATE__SYSTEM_PROMPT`, `__USER_PROMPT`), the user template holding `{context}` and `{question}` | a prompt of `prompts.yaml` (the RAG's `answer_from_context`), a list of messages whose contents are Jinja2 templates over the fields the stage reads (the passages by a loop over `reranked`), named by the stage's `prompt`; a variable by its path replaces the whole prompt, as JSON (`QP_PROMPTS__ANSWER_FROM_CONTEXT`) |
| the calibration request sized by `RAG_TOP_K_RETRIEVE`, kept to `RAG_TOP_K_RERANK` | `QP_COMPONENTS__RERANKER__LOCAL__CALIBRATION_PASSAGES` passages, all kept |
| `RAG_EMBEDDING_*`, `RAG_RERANK_*`, `RAG_GENERATION_*`, and before them `RAG_EMBEDDER_*`, `RAG_RERANKER_*` | each model component's entry in the workflow file, by its path: `RAG_EMBEDDING_MODEL` is the embedder's `source_id` (`QP_COMPONENTS__EMBEDDER__SOURCE_ID`), `RAG_RERANK_API_MODEL` its alias as a model server serves it (`QP_COMPONENTS__RERANKER__ALIAS`) or the name sent to an endpoint (`api.model`), `RAG_RERANK_THREADS` `QP_COMPONENTS__RERANKER__LOCAL__THREADS`, `RAG_EMBEDDING_NORMALIZE` `QP_COMPONENTS__EMBEDDER__LOCAL__NORMALIZE`, `RAG_GENERATION_TIMEOUT_S` `QP_COMPONENTS__LLM__HTTP__TIMEOUT_S` |
| `RAG_OVMS_EMBEDDINGS_URL`, `RAG_OVMS_RERANK_URL` (a base URL with a localhost default) | `QP_COMPONENTS__EMBEDDER__SERVED__URL`, `QP_COMPONENTS__RERANKER__SERVED__URL` (the full route, no default) |
| `RAG_OVMS_EMBEDDINGS_MODEL`, `RAG_OVMS_RERANK_MODEL` | the components' aliases (`QP_COMPONENTS__EMBEDDER__ALIAS`, `QP_COMPONENTS__RERANKER__ALIAS`), the model name a `served` component sends |
| `RAG_OVMS_AUTH` (one for both) | none: a `served` component is called with the run token under `qp-serve --domino` and with no key elsewhere |
| a model server of ours on the `api` backend: `RAG_COMPONENTS__EMBEDDER__API__URL`, `RAG_COMPONENTS__RERANKER__API__URL`, their key defaulted to `domino` by `rag-serve --domino` | the `served` backend: `QP_COMPONENTS__EMBEDDER__SERVED__URL`, `QP_COMPONENTS__RERANKER__SERVED__URL`; `api` is an endpoint we do not run, and `--domino` no longer changes its key |
| the LLM's `http` block with a timeout only, `120` when not written | every component's `http` block: `timeout_s`, `retries`, `backoff_s`, each the top-level `http` block's when not written; the LLM's `120` is written in the RAG's workflow file, and a workflow file that gives its LLM none calls it with `QP_HTTP__TIMEOUT_S` |
| `RAG_LLM_BACKEND` (`openai` \| `fake`) | `QP_COMPONENTS__LLM__BACKEND` (`api` \| `fake`) |
| `RAG_VECTOR_STORE_BACKEND` | the vector store component's `backend` (`QP_COMPONENTS__VECTOR_STORE__BACKEND`) |
| `RAG_ES_URL`, `RAG_ES_API_KEY`, `RAG_ES_INDEX`, `RAG_ES_TEXT_ANALYZER` | its `elasticsearch` block: `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__INDEX`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__TEXT_ANALYZER` |
| `RAG_CHROMA_URL`, `RAG_CHROMA_PATH`, `RAG_CHROMA_COLLECTION` | its `chroma` block: `QP_COMPONENTS__VECTOR_STORE__CHROMA__URL`, `__PATH`, `__COLLECTION` |
| `RAG_FAKE_STORE_SEED_DOCUMENTS` | its `fake` block: `QP_COMPONENTS__VECTOR_STORE__FAKE__SEED_DOCUMENTS` |
| `RAG_LOCAL_MODEL_RUNTIME`, `RAG_OPENVINO_MODELS_DIR`, `RAG_MODEL_WORKERS`, `RAG_MODEL_CPUS` | the `local_models` block: `QP_LOCAL_MODELS__RUNTIME`, `QP_LOCAL_MODELS__OPENVINO_DIR`, `QP_LOCAL_MODELS__WORKERS`, `QP_LOCAL_MODELS__CPUS` |
| `RAG_CALIBRATION_BUDGET_S`, `RAG_CALIBRATION_ROUNDS` | the `calibration` block: `QP_CALIBRATION__BUDGET_S`, `QP_CALIBRATION__ROUNDS` |
| `RAG_HTTP_TIMEOUT_S`, `RAG_HTTP_RETRIES`, `RAG_HTTP_BACKOFF_S` | the top-level `http` block: `QP_HTTP__TIMEOUT_S`, `QP_HTTP__RETRIES`, `QP_HTTP__BACKOFF_S` |
| `RAG_DOMINO_TOKEN_URL`, `RAG_DOMINO_TOKEN_TTL_S`, `RAG_DOMINO_TRACING` | the `domino` block: `QP_DOMINO__TOKEN_URL`, `QP_DOMINO__TOKEN_TTL_S`, `QP_DOMINO__TRACING` |
| `RAG_OPENAI_MODEL`, `RAG_LLM_TEMPERATURE` | `QP_COMPONENTS__LLM__API__MODEL`, `QP_COMPONENTS__LLM__API__TEMPERATURE` |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | `QP_COMPONENTS__LLM__API__URL` (the full route), `QP_COMPONENTS__LLM__API__API_KEY` |
| backends `ovms`, `openai`, `cohere` | backend `served` for a model server of ours (OVMS among them), `api` for an endpoint in an API format |
| `RAG_OPENAI_EMBEDDING_MODEL`, `RAG_COHERE_RERANK_URL`, `RAG_COHERE_RERANK_MODEL`, `RAG_OPENAI_RERANK_MODEL`, `COHERE_API_KEY`, `RAG_BM25_K1`, `RAG_BM25_B` | removed: set the component's `api` block instead; the BM25 tunables have no replacement |

<!-- old-names:end -->

## Retrieval, stores and model endpoints

**Retrieval mode** (the retrieve stage's `retrieval_mode`). `dense` asks the
store for the passages nearest to the question's vector. `lexical` asks for the
passages whose words match best, as the store ranks them: no vector is needed,
so a lexical workflow file has no embed stage (no `embed` timing, and the
workflow app opens no embedder unless a fake store is seeded through it). `hybrid` asks
both ways at once, the retrieve stage's `top_k` each, and fuses the two rankings
by reciprocal rank (`1 / (rrf_k + rank)`, summed per passage). The fusion runs in the
workflow, so it needs no Elasticsearch licence, at the price of two store calls
where a server-side fusion would make one. Passages are always ingested with
their vectors, so one index serves the three modes.

Lexical and hybrid retrieval need a store that ranks text. With `chroma` they
are refused at start-up.

| Store | Dense | Lexical |
| --- | --- | --- |
| `elasticsearch` | kNN on the vector field | Elasticsearch's own BM25 (`match` on the `text` field) |
| `chroma` | cosine, HNSW | refused at start-up |
| `fake` | brute-force cosine | a stand-in: passages ranked by how many distinct words they share with the question |

**Chroma** (`QP_COMPONENTS__VECTOR_STORE__BACKEND=chroma`, the toolkit's `[chroma]` extra).
Embedded by default: a folder (its `chroma.path`, in the use case's directory), no server to run, filled by
`make ingest` like any store. Its `chroma.url` points at a Chroma server instead.
Chroma serves dense retrieval only. Open-source Chroma filters text by substring
but does not rank it (its ranked Search API is Chroma Cloud only), so a
`lexical` or `hybrid` retrieve stage with `chroma` stops `qp-serve` at
start-up.

**Model endpoints.** Two backends call a model over HTTP, through one
implementation per component (its `api/` directory), and differ only in how
the endpoint is resolved. A component on `served` is served by a model server
of ours, OVMS or `qp-model-server`, and its `served` block gives only the
`url` (the full route, not a base URL): it is called in the standard format of
its type, under its alias, and with the Domino run token when `qp-serve
--domino` says the app runs on Domino, with no key elsewhere. A component on
`api` calls an endpoint we do not run, and its `api` block gives four
settings: `url`, `api_format` (which request and response shapes the route
speaks), `api_key` (from the environment only) and `model` (the model name
sent, the component's alias when empty), each set by its path, such as
`QP_COMPONENTS__RERANKER__API__URL`. An API format is a wire shape, not a
vendor. The formats today:

| Component | API format | Request → response | Spoken by |
| --- | --- | --- | --- |
| embedding | `openai` | `{"model","input":[...]}` → `{"data":[{"index","embedding":[...]}]}` | OVMS and the FastAPI model server (`/v3/embeddings`), OpenAI, vLLM, LiteLLM |
| rerank | `cohere` | `{"model","query","documents","top_n"}` → `{"results":[{"index","relevance_score"}]}` | OVMS and the FastAPI model server (`/v3/rerank`), Cohere (`/v2/rerank`), the `/rerank` routes of LiteLLM, vLLM, Jina and Infinity |
| generation | `openai` | `{"model","messages","temperature"}` → `{"choices":[{"message":{"content"}}]}` | OpenAI, vLLM, LiteLLM, the Domino AI Gateway |

Each component has one standard format, the one in the table, and it is the
same whether the endpoint is OVMS, the FastAPI model server or a gateway. An API
that speaks another shape is reached through an adapter class that implements
`BaseEmbedder` or `BaseReranker`, either in the workflow process or behind the
model server's route. The shapes of the standards are pydantic classes in
`src/query_processing/domain/standard_formats.py`. A new API format is one new
module in the component's `api/` directory, its class decorated with the
registry beside the base class, such as `@RERANK_API_FORMATS.register("cohere")`;
no table or list is edited.

An `api` key is the same for every component: empty sends no `Authorization`
header, `domino` sends the Domino run's token, anything else is sent as a
bearer token. Every call times out (`QP_HTTP__TIMEOUT_S`,
`QP_COMPONENTS__LLM__HTTP__TIMEOUT_S` for generation) and is retried
`QP_HTTP__RETRIES` times on transport errors, 5xx and 429, and on 401 when it
carries the run token (after the token is fetched again); any other 4xx fails
at once. A `served` or `api` embedder or reranker reports ready in `/readyz`
after one real minimal request to its endpoint, sent once without retry; on a
metered API that request is billed like any other.

An OVMS or FastAPI model server is called as a `served` component, as in the
split topologies on a laptop ([The real run](#the-real-run)). Nothing of ours
serves an LLM yet: the LLM's `served` backend takes a URL and
an alias (`QP_COMPONENTS__LLM__ALIAS`, since an LLM has no `source_id`) and is
called in OpenAI's chat format, for when something does.

An OpenAI-compatible gateway, for generation and embeddings:

```bash
QP_COMPONENTS__LLM__API__URL=https://gateway.example/v1/chat/completions \
  QP_COMPONENTS__LLM__API__API_KEY=<key> QP_COMPONENTS__LLM__API__MODEL=<model name> \
  QP_COMPONENTS__EMBEDDER__BACKEND=api QP_COMPONENTS__EMBEDDER__API__URL=https://gateway.example/v1/embeddings \
  QP_COMPONENTS__EMBEDDER__API__API_KEY=<key> QP_COMPONENTS__EMBEDDER__API__MODEL=<embedding model> make serve
```

The `served` and `api` embedders send `QP_COMPONENTS__EMBEDDER__QUERY_PREFIX`
in front of each question too: set it empty for a model that takes none.

Cohere's rerank route:

```bash
QP_COMPONENTS__RERANKER__BACKEND=api QP_COMPONENTS__RERANKER__API__URL=https://api.cohere.com/v2/rerank \
  QP_COMPONENTS__RERANKER__API__API_KEY=<Cohere key> QP_COMPONENTS__RERANKER__API__MODEL=rerank-v3.5 make serve
```

Any component may be on any of its backends, `api` like `fake` and `none`, and
the default run label names each component with its own
(`embedder=local+reranker=api+llm=api+vector_store=elasticsearch` for the
command above). An embedding endpoint serving another
model returns vectors of its own size, so it needs its own index: ingest with
the embedder you will serve with. In a split topology `qp-deploy` still
publishes the model server Apps the topology names, even when an endpoint
elsewhere leaves one unused.

**A French corpus.** The synthetic handbook is English. For a French,
domain-specific corpus, `qp-fetch-corpus quebec-insurance --out DIR` downloads
the Quebec automobile insurance corpus of GRAAL-Research (Beauchemin et al.,
NLLP 2024: about 1,200 passages and 82 questions) and writes `DIR/corpus.jsonl`
for `qp-ingest --corpus` and `DIR/questions.jsonl` for
`QP_LOADTEST_QUESTIONS_FILE`. Its source repository states no licence: research
use, cite the paper, and commit none of it. With Elasticsearch, ingest it into
an index created with the `french` text analyzer.

```bash
qp-fetch-corpus quebec-insurance --out data/quebec
QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__INDEX=rag_quebec \
  QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__TEXT_ANALYZER=french \
  qp-ingest --corpus data/quebec/corpus.jsonl
QP_LOADTEST_QUESTIONS_FILE=data/quebec/questions.jsonl make loadtest RUN_NAME=monolith-fr HOST=http://localhost:8888
```

## Elasticsearch

* The store is LangChain's `AsyncElasticsearchStore` over index `rag_passages`,
  created on the first `qp-ingest` with a dense-vector kNN mapping; documents
  hold `text`, `metadata` and the vector, and the `_id` is the passage id
  (`<doc id>:<chunk>`), so re-ingesting overwrites in place.
* Auth: basic credentials inside the `elasticsearch.url` of the vector store
  (`https://user:password@host:9200`) or its `elasticsearch.api_key`, from the
  environment only (`QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`). Any Elasticsearch 8 or 9 cluster works; the compose file
  runs a single 9.0.0 node with security disabled, for development only.
* Failures surface as `503 vector_store_unavailable`; `/readyz` reports
  `not_ready` while the index is missing or empty.
* On Domino every workflow app points at one shared cluster (`QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`
  in the project's environment variables), so the corpus is ingested once. All
  topologies embed with `snowflake-arctic-embed-l-v2.0` (in-process or its int8 OVMS
  export), so one index serves every run. Its vectors have 1024 dimensions: an
  index filled by an earlier embedder (384 for `bge-small-en-v1.5`) cannot take
  them, so ingest into a new index (`QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__INDEX`).

## In-process models on OpenVINO

The in-process embedder and reranker (the `local` backend, and what
`qp-model-server` serves) run on OpenVINO by default
(`QP_LOCAL_MODELS__RUNTIME=openvino`); `torch` runs them on PyTorch as before.
Neither default model ships OpenVINO files on the Hub, so they are converted
once:

```bash
make openvino-export          # qp-export-openvino: every embedder and reranker -> models/openvino/<model name>/
qp-export-openvino --only reranker --models-dir /domino/datasets/local/<dataset>/openvino
```

`qp-export-openvino` converts the `source_id` of every embedder and reranker of
the workflow file, or of the one `--only` names (`QP_*` overrides honoured,
such as `QP_COMPONENTS__RERANKER__SOURCE_ID`), into one folder each under
`QP_LOCAL_MODELS__OPENVINO_DIR`, named after the last segment of the model id. A
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
budget gives it, or fewer when calibration finds them faster (OpenVINO's
`INFERENCE_NUM_THREADS`, with `ENABLE_HYPER_THREADING`).

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

The served models (source id and alias) are pinned once, by the embedders and
rerankers of the workflow file, whatever their names (here `reranker`,
`antoinelouis/crossencoder-camembert-base-mmarcoFR`, served as
`crossencoder-camembert-base-mmarcoFR`, and `embedder`,
`Snowflake/snowflake-arctic-embed-l-v2.0`, served as
`snowflake-arctic-embed-l-v2.0`: each alias is the source id's last part).
`qp-served-models` prints them, `QP_*` overrides included, one line of
shell-quoted words each (name, type, source id, alias) for `export_models.sh`;
the `rag` Domino image reads the source ids from the use case's workflow file,
which it copies beside the use case it installs.

**Export** (`make ovms-export` in the use case's directory = the toolkit's
`ovms/export_models.sh [MODELS_DIR]`, default `ovms/models` in the working
directory, the use case's, whose workflow file `qp-served-models` reads; run
`make install` at the root first so `qp-served-models` is on `PATH`, or the
script stops and says so): takes the served models from `qp-served-models`,
downloads `export_model.py` and its `requirements.txt` from the OVMS repository at tag `v${OVMS_VERSION}` into `<MODELS_DIR>/.export/` (cached),
`pip install`s them into `$PYTHON` (set `PYTHON=/path/to/venv/bin/python` for a
separate environment, which it needs: the exporter pins its own
sentence-transformers and OpenVINO versions and would downgrade this project's),
then runs `rerank_ov` for each reranker and `embeddings_ov --pooling CLS` for
each embedder, with `--weight-format int8`. The result is one
`config_<component>.json` per component (`config_reranker.json`,
`config_embedder.json`) and one directory per model; the configs reference the
model directories relatively, so the tree can be mounted anywhere.

**On Domino the export is optional.** A model server App of kind `ovms`
(`domino/model_servers/ovms.sh`) that finds its component's
`config_<component>.json` missing under
`QP_OVMS_MODELS_DIR` runs `export_models.sh` itself, once, before serving: it
needs `uv`, installs the toolkit and the use case into `$HOME/qp-venv` when
`qp-served-models` is not on `PATH`, runs in the use case's directory, and gives the exporter its own `$HOME/ovms-exporter-venv`. The
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

**API formats.** OVMS speaks the `cohere` rerank format on `/v3/rerank` and the
`openai` embeddings format on `/v3/embeddings`, the standard formats, so the
workflow calls it as a `served` component, through the generic implementations
of `adapters/<component>/api/`, with the full routes as URLs. The routes are wire-contract constants in `domain/api_endpoint.py`; the
FastAPI model server answers on the same ones, and each model-server kind
declares the route of each component type it serves
(`deploy/model_server_kinds/`), from which `qp-deploy` builds a served URL.

| Adapter | Request | Response |
| --- | --- | --- |
| `ApiReranker` (`cohere`) → `POST QP_COMPONENTS__RERANKER__SERVED__URL` | `{"model","query","documents":[...],"top_n"}` | `{"results":[{"index","relevance_score"}]}` |
| `ApiEmbedder` (`openai`) → `POST QP_COMPONENTS__EMBEDDER__SERVED__URL` | `{"model","input":[...]}` (`QP_COMPONENTS__EMBEDDER__SERVED__BATCH_SIZE` texts per request) | `{"data":[{"index","embedding":[...]}]}` |

Both map results back by `index` (an index outside the request, or a body that
does not fit the shape, is a malformed response), never send a field as `null`,
time out after `QP_HTTP__TIMEOUT_S` (30 s) and retry `QP_HTTP__RETRIES` (2)
times on transport errors, 5xx and 429, and on 401 when the call carries the
run token (the token is fetched again first); other 4xx fail immediately,
including the 404 for a `model` that is not the server's alias. Readiness is one real minimal
request (one text embedded, one document scored), not a probe of
`GET /v3/models/{model}`.

**Bumping OVMS**: change `ovms/versions.env` only, re-run `make ovms-export`
(the exporter is fetched from the new tag), rebuild the `ovms` Domino image and
refill the models Dataset. **Changing a served model**: change its `source_id`
in the workflow file only, then re-run `make ovms-export`, rebuild the `rag` Domino
image (it pre-downloads the source ids) and refill the models Dataset.

## Deploying on Domino 6.2

Every app is published with the same entry point, the toolkit's `domino/app.sh`,
which installs the toolkit and the use case (`QP_BOOTSTRAP=1`), then runs its
role's command in the use case's directory (`QP_USE_CASE_DIR`, the directory
`qp-deploy` ran from), where that command finds the use case's workflow file,
reads its role from `QP_ROLE` and listens on `0.0.0.0:8888` (a Domino
requirement). Only environment variables differ between
topologies, so the Locust numbers stay comparable. Apps published before the
toolkit took its current names are not found again: see
[Moving from the old names](#moving-from-the-old-names).

An App has one of two roles: the workflow app (`QP_ROLE=workflow`, the
default), or a model server (`QP_ROLE=model-server`), which is told the
component it serves (`QP_MODEL_SERVER_COMPONENT`, any embedder or reranker of
the workflow file) and its model-server kind (`QP_MODEL_SERVER_KIND`).
`app.sh` sources the kind's own launch script, `domino/model_servers/<kind>.sh`,
which holds its bootstrap and its command: `fastapi` runs the FastAPI model
server (`qp-model-server`, the component's local model on the `/v3` routes and
formats), `ovms` runs OpenVINO Model Server. Each kind is also one registered
class (`deploy/model_server_kinds/`) that says which component types it serves,
on which route, and how a served model is probed; a new kind is one class and
one script, and neither `qp-deploy`'s flow nor `app.sh` changes. `server` in
an App's block of `deploy.yaml` sets its kind, `defaults.server` the kind of
every other model server App, `fastapi` by default. Which Apps a topology needs
follows from its topology alias in `deploy.yaml` (see "Topology aliases"
below): one model server App per component it serves, then the workflow App.
The table shows the three aliases of `deploy.example.yaml`.

| Topology | App | `QP_ROLE` (kind: OVMS / FastAPI) | Environment variables | Image: OVMS / FastAPI |
| --- | --- | --- | --- | --- |
| `monolith` | one app: workflow + embedder + reranker | `workflow` (default) | `QP_COMPONENTS__LLM__API__API_KEY`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` (+ `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`) | `rag` |
| `split-reranker` | **A** model server: reranker | `model-server` (`ovms` / `fastapi`) | `QP_MODEL_SERVER_COMPONENT=reranker`, `QP_MODEL_SERVER_KIND`; OVMS: `QP_OVMS_MODELS_DIR` when the Dataset is not at `/mnt/data/ovms-models`, `QP_OVMS_PREFIX_PROXY=1` if the prefix reaches OVMS; FastAPI: none (weights from `QP_COMPONENTS__RERANKER__SOURCE_ID`) | `ovms` / `rag` |
| | **B** workflow + embedder | `workflow` | `QP_COMPONENTS__RERANKER__BACKEND=served`, `QP_COMPONENTS__RERANKER__SERVED__URL=<URL of A>/v3/rerank`, `QP_COMPONENTS__LLM__API__API_KEY`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` | `rag` |
| `split-all` | **A** model server: reranker | `model-server` (`ovms` / `fastapi`) | as above | `ovms` / `rag` |
| | **C** model server: embedder | `model-server` (`ovms` / `fastapi`) | as above, `QP_MODEL_SERVER_COMPONENT=embedder` (FastAPI: weights from `QP_COMPONENTS__EMBEDDER__SOURCE_ID`) | `ovms` / `rag` |
| | **B** workflow | `workflow` | `QP_COMPONENTS__RERANKER__BACKEND=served`, `QP_COMPONENTS__RERANKER__SERVED__URL=<URL of A>/v3/rerank`, `QP_COMPONENTS__EMBEDDER__BACKEND=served`, `QP_COMPONENTS__EMBEDDER__SERVED__URL=<URL of C>/v3/embeddings`, `QP_COMPONENTS__LLM__API__API_KEY`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` | `rag` |

`qp-deploy` also gives the workflow App `QP_TOPOLOGY_ALIAS`, the alias it
deploys, which the app only reports as `X-RAG-Topology`. The workflow role runs
`qp-serve --domino`, which runs each component on the backend its variables give
and calls each served component with the run token; a component on `api`, the
LLM's included, keeps its own key. The run label ends up in every response, so
set a distinct `QP_RUN_LABEL` when testing several hardware tiers or model
servers (`run_label` under the overrides of `deploy.yaml`, with `qp-deploy`).

**Environments.** `domino/Dockerfile` has two targets built from the repository's
root, the build context (commands in its header): `rag` starts from
`quay.io/domino/domino-standard-environment:ubuntu22-py3.10-r4.5-domino6.2-standard`,
installs this package with the `[loadtest,domino]` extras into a virtualenv of
its own on Python 3.11 (the image's Python is 3.10 and its packages clash with
ours), pre-downloads both HF models and sets `HF_HUB_OFFLINE=1`; `ovms` unpacks the
pinned binary tarball under `/opt/ovms`. Push both images and create two Domino
Environments from them (custom base image), or paste one stage's instructions
into the Dockerfile Instructions field. Workspaces that ingest or run Locust use
the `rag` environment.

**Models Dataset.** An OVMS model server reads its component's
`config_<component>.json` (`config_reranker.json`, `config_embedder.json`) from
`QP_OVMS_MODELS_DIR` (default `/mnt/data/ovms-models`), exporting it first when
it can and exiting with a clear message when it cannot. Create a Dataset named
`ovms-models` (Data → Datasets; git-based projects mount it at
`/mnt/data/ovms-models`, DFS projects at `/domino/datasets/local/ovms-models`, so
set `QP_OVMS_MODELS_DIR` there) and fill it from a Workspace, in the use case's
directory, with `../query_processing/ovms/export_models.sh /mnt/data/ovms-models`,
or upload the contents of a local `ovms/models/` (the use case's). Apps inherit the project's Dataset mounts.

**Elasticsearch and corpus.** Put `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` (and `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`) of a cluster the
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
started with `qp-serve --domino`, the workflow app fetches it for every
component on the `served` backend, caches it for 4 minutes, refreshes it on
`401` and sends it as `Authorization: Bearer` on every model server call. To
call an App with a fixed token instead, put the component on `api` with that
token as its key (`QP_COMPONENTS__RERANKER__API__API_KEY`); the key `domino`
there sends the run token too.

**Proxy prefix (`DOMINO_RUN_HOST_PATH`).** Domino serves an app under a path
prefix (`/apps/<app-id>/` in 6.2), exposed to the process as `DOMINO_RUN_HOST_PATH`,
and its proxy normally strips it before the request reaches the process. The
FastAPI apps (`qp-serve`, `qp-model-server`) set that value as `root_path`, so
they answer whether or not the prefix is on the wire, with no proxy layer in
front. OVMS cannot: its routes are fixed at `/v3/...` and it has no base-path
option. Check once, from a Workspace, what reaches the OVMS app:

```bash
curl -i -H "Authorization: Bearer $(curl -s http://localhost:8899/access-token)" \
  https://<domino-host>/apps/<ovms-app-id>/v3/models/crossencoder-camembert-base-mmarcoFR
```

`200`: the prefix is stripped, nothing to do. `404`: the prefix reaches OVMS; set
`QP_OVMS_PREFIX_PROXY=1` on the OVMS apps. `app.sh` then starts OVMS on
`127.0.0.1:9000` and nginx (in the `ovms` image) on `0.0.0.0:8888`, with the
settings in `domino/ovms-proxy.conf` (`__PREFIX__` is replaced at start-up):
`location <prefix>/ { proxy_pass http://ovms/; }` drops the prefix and forwards
over keep-alive HTTP/1.1. OVMS itself is unchanged (same binary,
config and threads); the cost is one loopback hop, well under a millisecond, plus
a little CPU on the same pod. Note it in the comparison and keep it identical
across the OVMS runs. Callers outside prefix every route with the App URL
(`https://<domino-host><DOMINO_RUN_HOST_PATH>/query`); `QP_COMPONENTS__RERANKER__SERVED__URL` /
`QP_COMPONENTS__EMBEDDER__SERVED__URL` are the App URLs followed by `/v3/rerank` /
`/v3/embeddings`.

**Tracing and agent deployment.** Agent deployments use the same hosting as
Apps (same `app.sh`, same `0.0.0.0:8888` rule) plus tracing through
`add_tracing` from `dominodatalab[agents]` (the `[domino]` extra, in the `rag`
image). With `QP_DOMINO__TRACING=true` the workflow run behind both endpoints is
wrapped with `add_tracing(name="rag_query", autolog_frameworks=["langchain"])`
(`domino.aisystems.tracing`, falling back to `domino.agents.tracing`), so the
LangGraph run and the generation call appear as traces; without the SDK the app logs
a warning and runs untraced. `make deploy` publishes the workflow as an Agent
with tracing on (see "App or Agent" below); by hand, deploy from Experiment
Manager → **Deploy Agent** with the same entry point, environment and tier as an
App. Keep the flag identical across compared runs (tracing adds a small
per-request overhead).

**Running Locust.** From a Workspace in the same project (`rag` environment):

```bash
export QP_LOADTEST_BEARER_TOKEN=$(curl -s http://localhost:8899/access-token)
make loadtest RUN_NAME=monolith        HOST=https://<domino-host><path of the monolith app>
make loadtest RUN_NAME=split-reranker  HOST=https://<domino-host><path of workflow app B>
make loadtest RUN_NAME=split-all       HOST=https://<domino-host><path of workflow app B, split-all project>
make compare                           # results/comparison.md
```

The run token expires after about 5 minutes and Locust reads it once per user at
start, so for longer runs (`RUN_TIME=10m`, `QP_LOADTEST_SHAPE=step`) or from a laptop
use an API key instead: `export DOMINO_API_KEY=<Account settings → API Key>` (sent as
`X-Domino-Api-Key`). Keep `USERS`, `SPAWN_RATE`, `RUN_TIME` and the tiers identical across topologies.

| Symptom | Cause / fix |
| --- | --- |
| `401`/`403` from Locust | missing or expired `QP_LOADTEST_BEARER_TOKEN` / `DOMINO_API_KEY`, or the app is not shared with that user |
| `404` on `/query` | `HOST` lacks the App path prefix, or has a trailing slash |
| `503 reranker_unavailable` / `embedder_unavailable` | model server app not ready (Dataset missing, wrong `QP_COMPONENTS__RERANKER__SERVED__URL` / `QP_COMPONENTS__EMBEDDER__SERVED__URL`, token rejected); check its logs and `GET <OVMS App URL>/v2/health/ready` |
| `503 vector_store_unavailable` / `/readyz` not ready | `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` unreachable or wrong credentials, or `make ingest` never ran against that cluster |
| `qp-serve: error: ...` in a workflow app's log | a settings mistake: a `QP_COMPONENTS__RERANKER__SERVED__URL` / `QP_COMPONENTS__EMBEDDER__SERVED__URL` missing for a component on the `served` backend, or an unknown backend or API format; the message names the variable and the fix |
| `app.sh[...]: unknown QP_ROLE '...'` | an old role (`monolith`, `ovms-reranker`, ...): the roles are `workflow` and `model-server`, the latter with `QP_MODEL_SERVER_COMPONENT` and `QP_MODEL_SERVER_KIND` |
| `app.sh[model-server ovms ...]: OVMS config not found` | Dataset not mounted at `/mnt/data/ovms-models`; set `QP_OVMS_MODELS_DIR` |
| `404` from an OVMS app on `/v3/...` | Domino forwards the path prefix; set `QP_OVMS_PREFIX_PROXY=1` on that app (see Proxy prefix) |

### Deploy on Domino in a few commands

From the use case's directory (for example
`/repos/<project>/query_processing_rag_usecase`) in a Workspace of the one project
that hosts all the Apps:

```bash
cp deploy.example.yaml deploy.yaml   # set defaults.environment_id, topology, each App's server
make deploy                           # model server Apps first, then the workflow App with their URLs
make status                           # each App's status and URL, then the workflow App's /readyz
make loadtest-domino RUN_NAME=split-all
make compare
```

`qp-deploy` (behind these targets) calls Domino's beta Apps API through
`DOMINO_API_PROXY` and publishes `qp-rag-embedder`, `qp-rag-reranker` and `qp-rag-workflow`
as the topology alias needs (their names follow `app_name_prefix`: see "App
names" below). The Apps install their own dependencies when they start
(`domino/app.sh` with `QP_BOOTSTRAP=1`), into a virtualenv on Python 3.11, so any
environment with `uv` works, but the first start is slower and needs access to
the package index and the Hugging Face Hub. An environment without Python 3.11
also needs access to GitHub, where `uv` downloads it from. For an App served by `ovms`, `ovms_bundle` points at the OVMS bundle (a folder with
`bin/ovms` and `lib/`, or the release `.tar.gz`) at a path the Apps can read, for
example `/domino/datasets/local/<dataset>/ovms.tar.gz`, and the environment needs
root or `sudo` for `apt-get`. Run `qp-deploy` from the use case's directory: it
finds the toolkit's launch script going up from there (the first directory that
holds `query_processing/domino/app.sh`, refusing to publish when there is none),
and every App runs that script by its absolute path and its role's command in
that directory (`QP_USE_CASE_DIR`, one of `qp-deploy`'s own exports), the Apps
mounting the repository at the same path as the Workspace.

**`deploy.yaml`** (git-ignored; `deploy.example.yaml` documents every key) holds
every setting in one place. Top-level keys say what to deploy (`topology`,
`topology_alias`, `project_id`, `app_name_prefix`, `ovms_bundle`,
`domino_host`) and how to talk to Domino (`wait_timeout_s`, `poll_interval_s`,
`no_instance_grace_s`, `http_timeout_s`, `http_retries`, `run_token_url`); a
`QP_DEPLOY_<KEY>` variable overrides one, e.g.
`QP_DEPLOY_TOPOLOGY=monolith make deploy`, and
`QP_DEPLOY_FILE` names another file. `defaults` applies to every App and
`apps.<workflow|served component>` to one, each App taking its own value, else the
default (a block keyed by anything else is refused). `overrides` holds this
site's values of the workflow file, in the workflow file's own shape, for every
App:

```yaml
defaults:
  environment_id: env-123          # required: the compute environment's id (in its page's URL)
  environment_revision_id: rev-7   # optional: one build of env-123; "" = Domino chooses
  hardware_tier_id: small          # "" = the project's default tier
  server: fastapi                  # every model server App's kind, unless its own block says
apps:
  workflow:
    kind: agent                    # the default for the workflow; model servers are apps
    autoscaling: {min_replicas: 1, max_replicas: 4, target_cpu_pct: 70}
  reranker:
    server: ovms                   # this model server App alone is OVMS (needs ovms_bundle)
    environment_id: env-456        # its own environment, so not rev-7: Domino chooses
    hardware_tier_id: cpu-16
    settings: {QP_OVMS_MODELS_DIR: /domino/datasets/local/<dataset>/ovms-models}
overrides:                         # the workflow file's shape
  http: {timeout_s: 30, retries: 2, backoff_s: 0.2}
  components:
    llm:          {http: {timeout_s: 120}}
    vector_store: {elasticsearch: {url: "https://es.example:9200"}}
    reranker:     {source_id: /domino/datasets/local/<dataset>/models/crossencoder-camembert-base-mmarcoFR}
```

**Overrides.** The store's URL, the model paths (a local folder such as
`/domino/datasets/local/<dataset>/models/snowflake-arctic-embed-l-v2.0` works as well as a
Hugging Face id), the timeouts and retries of calls, the values `qp-calibrate`
prints: what differs per site is written under `overrides`, so no one
hand-writes a nested variable name. Before anything is published, `qp-deploy`
checks them against the workflow settings of the workflow file the workflow App
reads: each key must be a value of that file, in the block it is in (a misspelt
key is refused, naming it and its place: `overrides.components.llm.http in
deploy.yaml has no key 'timout_s'`), a component or a stage must be one the
file declares, and the file with the overrides over it must make settings the
workflow app accepts (each value's type and range, a URL for a component on
`api`). A stage's own settings (`graph.stages.rerank.top_k`) are its stage
type's, which checks them when the workflow app starts. Then every App, the
workflow App and each model server App, exports each override under the
variable the workflow settings read for its path (`QP_` and the path joined by
`__`: `components.llm.http.timeout_s` is
`QP_COMPONENTS__LLM__HTTP__TIMEOUT_S`), so an override wins over the workflow
file; a list, such as `graph.edges`, is exported as JSON, which the settings
read back.

`settings` are each App's environment variables for what its launch script
reads that is no value of the workflow file (merged with the defaults', the
App's winning; YAML numbers and booleans are exported as strings), for example
`QP_OVMS_MODELS_DIR` (for example `/domino/datasets/local/<dataset>/ovms-models`),
`QP_BOOTSTRAP_EXTRAS` or `QP_WORKFLOW_FILE`. A value of the workflow file
written there, such as `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, is
refused before anything is published, naming the place under `overrides` where
it belongs. `qp-deploy` sets `QP_ROLE`,
a model server's `QP_MODEL_SERVER_COMPONENT` and `QP_MODEL_SERVER_KIND`, what
its kind's script reads (`QP_OVMS_BUNDLE` for `ovms`), the workflow App's
`QP_TOPOLOGY_ALIAS`, `QP_COMPONENTS__<COMPONENT>__BACKEND` of each component the
topology alias names and `QP_COMPONENTS__<COMPONENT>__SERVED__URL` of each served
one (the model server App's URL plus the route its kind declares for its type,
`/v3/rerank` or `/v3/embeddings`), and `QP_BOOTSTRAP` itself (they win over the
settings and the overrides, which come first in the launch script), and
`QP_RUN_LABEL` unless the overrides set `run_label`. Secrets stay in
Domino's project or user environment variables, because `settings` and
`overrides` are written into the Apps' launch scripts. `qp-deploy` refuses, in
any block, a setting whose name ends in `_API_KEY` (such as
`QP_COMPONENTS__LLM__API__API_KEY` or
`QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`), and an `api_key` under
`overrides`, unless its value is empty or `domino`, which names the run's
token and is not a secret. A misspelt key is refused with its name.

**Topology aliases.** Where each component runs is a deployment matter, so the
topologies live in `deploy.yaml` (ADR 0003): `topology` chooses one by its alias,
and `topology_alias` defines each alias as a row that says `local` (in the
workflow App's process) or `served` (by a model server App of its own) for the
components it names. `deploy.example.yaml` ships the three compared here, shown
at the top of this README; a fourth is one more row and no change in code, such
as `split-embedder: {embedder: served, reranker: local}`. No alias is defined in
code: a `deploy.yaml` without the table deploys nothing.
`qp-deploy` reads the workflow file the workflow App reads (`QP_WORKFLOW_FILE`
in its settings, else the one `qp-serve` finds in the use case's directory) for
the components' names and types. It publishes one model server App per served
component, in the workflow file's order, then the workflow App, whose launch
script gets `QP_COMPONENTS__<COMPONENT>__BACKEND` for every component the row
names and `QP_COMPONENTS__<COMPONENT>__SERVED__URL` for every served one. A
component the row does not name keeps the backend the workflow file (or the
`overrides`) gives it: to deploy the reranker on `none` or an `api`
endpoint, leave it out of the row, since a named component's backend is the
row's and wins over `overrides`. A cell says only where a component runs; how it
is served (`server`, the model-server kind, the environment, the hardware tier)
is in its App's block. A served component may have any name: its App is
`<app_name_prefix>-<component>` and is told the component. The run label is the
alias, then each served component and the model-server kind of its own App:
`split-all+embedder=fastapi+reranker=ovms`, or `monolith`.

The workflow app holds no topology rule: the backends written are the only
thing that places a component. The launch script also gets
`QP_TOPOLOGY_ALIAS`, the alias deployed, whatever it is, which the app reports
as its `X-RAG-Topology` header, so a fourth alias deploys with its own name in
the header and its own run label. The workflow App has one Role whatever the
topology, and `app.sh` passes `QP_TOPOLOGY_ALIAS` through without checking it.
The topology variable of earlier versions is not read: a workflow App published before that change
keeps the backends its launch script writes, but reports no topology until it
is redeployed.

Before anything is published, `qp-deploy` refuses a `topology` that is not a
key of the table, a row naming a component the workflow file does not declare, a
cell other than `local` or `served`, a row putting a component on a backend its
type does not have (naming the type's backends: the vector store or the LLM on
`local`, which the workflow App would otherwise refuse when it starts; a type's
backends are its registered openers, rule 5), a model-server kind no class is registered
under (naming the kinds), a served component its App's kind cannot serve (naming
the kinds that can, or, when none can, each kind and what it serves: a served
LLM is refused, since neither `fastapi` nor `ovms` serves one), and an alias
whose run label the workflow app would refuse (letters, digits and `._+=:-`, up
to 128 characters). Each message names the value and what was expected.

**App or Agent.** `kind: agent` publishes the App with `configurationType:
AISYSTEM` (Domino: "Agents use the same hosting infrastructure as Domino Apps";
that field is the only API-level difference), `kind: app` with `STANDARD`. The
workflow is an Agent unless told otherwise: its launch script also exports
`QP_DOMINO__TRACING=1` (unless the overrides set `domino.tracing`) and
`QP_BOOTSTRAP_EXTRAS=domino` (unless its settings set it), and `app.sh` then
installs `.[domino]`, the extra that brings the tracing
library. An existing App of the other kind is not converted: `qp-deploy` stops
and says to delete it in Domino or change `kind`.

**Environment revision.** `environment_revision_id` holds an App to one build
of its environment: `qp-deploy` sends it in the App's version as
`environmentRevisionId`, the name Domino's Apps API gives it, and sends nothing
when it is unset, so that Domino chooses the revision. An App's own revision
applies to its environment, its own or the default one. The default revision
goes only with the default environment, so an App that sets its own
`environment_id` does not take it. A default revision without a default
`environment_id` is refused before anything is published, and so is an App with
a revision but no environment.

**Autoscaling.** `autoscaling` becomes the version's `autoscalingSpecification`
(`enabled`, `minReplicas`, `maxReplicas`, `targetCpuAvgUtilizationPct`). Replicas
are pods behind one instance, so scaling does not change the instance URL below.

**App names.** Each App is named `<app_name_prefix>-<part>`, the part being
`workflow` or the name of a served component: `qp-rag-embedder`, `qp-rag-reranker`
and `qp-rag-workflow` with the RAG use case's prefix, `qp-rag`. There is no default
prefix: a deploy file without one is refused. `qp-deploy` creates, finds and
reports every App by that name (a deploy, `make status`, `qp-deploy url` and the
note after a model server is redeployed), so two use cases with two prefixes share
one Domino project, each with its own Apps. The `make` targets keep the part's
name (`make deploy-workflow`). A prefix is lowercase letters and digits in words
joined by single hyphens, starting with a letter (`rag`, `qp-rag`); any other is
refused before anything is published. That rule is `qp-deploy`'s own, not
Domino's: the Apps API reference gives an App's name as a plain string, and we
found no rule of Domino's for it. Since an App is found by its name, a new prefix
creates new Apps, and the Apps under the old one keep running until they are
deleted by hand in Domino.

**Redeploying, never a second App.** `qp-deploy` remembers each App's id in
`.qp-deploy-cache.json` (git-ignored, per project; a dry run keeps it in memory).
It looks an App up by that id first, then by exact name, and remembers it either
way; only when both miss does it create the App. An App that exists gets a new
version with whatever `deploy.yaml` says now (the 6.2 API has no restart call).
Delete the cache file and the Apps are found by name again.

**Where an App answers.** Apps are called on their running instance:
`{domino_host}/{owner}/{project}/r/notebookSession/{instance id}`, where owner and
project are the App's `project.ownerUsername` and `project.name` and the instance
is the `Running` one of its current version (`domino_host` "" = the scheme and host
of the App's own `url`). That URL, followed by `/v3/rerank` or `/v3/embeddings`,
is what the workflow App gets as `QP_COMPONENTS__RERANKER__SERVED__URL` / `QP_COMPONENTS__EMBEDDER__SERVED__URL`;
the URL alone is what `qp-deploy url <part>` prints, what `make status` shows and
checks `/readyz` on, and what the "Running" line after a deploy prints; the App's
own `url` stays in the messages that point at its page or logs. The instance id
changes at every redeploy, so after `make deploy-embedder` or `make
deploy-reranker` alone, `qp-deploy` reminds you that the workflow App
(`qp-rag-workflow` for the RAG use case) still calls the previous instance: run `make
deploy-workflow`. Every model server call in a split topology goes through
Domino's proxy: it is part of what the split topologies measure on Domino.

One-time prerequisites that stay manual: install the toolkit and the use case in
the Workspace (`make install-dev` at the root, or `uv pip install -e
"./query_processing[loadtest]" -e ./query_processing_rag_usecase`, both in one
command), `make ingest`, and for `ovms` the exported models.

#### Try the deploy without Domino

```bash
make deploy-dry-run      # qp-deploy --dry-run deploy all: nothing is sent to Domino
```

prints each App `qp-deploy` would publish for the chosen topology alias (kind,
environment, environment revision, hardware tier, autoscaling, entry script:
what the App receives, its `settings`, then every override, then what
`qp-deploy` sets itself), with every check of a real deploy;
`QP_DEPLOY_TOPOLOGY=<alias> make deploy-dry-run` shows another alias. One of
them: no value may still hold an `<angle-bracket>` placeholder from the example
(in the `settings` of the Apps being deployed, their `environment_id`,
`environment_revision_id` and `hardware_tier_id`, `ovms_bundle`, and the
`overrides`, which every App gets). The example copied unchanged stops on the
first one:

```
qp-deploy: error: environment_id in defaults of deploy.yaml still holds the example placeholder '<environment id>': replace it with a real value
```

To click through the API or watch a
deploy wait, serve the fake Domino and point `qp-deploy` at it:

```bash
make -C ../query_processing fake-domino BEHAVIOUR=slow     # Swagger at http://localhost:8899/docs
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
`qp-deploy` expects of Domino, not what a live Domino does. Besides the Apps
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
   comes up without an instance, `qp-deploy` says to start it once from the Apps
   page).
2. The inline `entryScript` runs (it starts `domino/app.sh` by absolute path, so
   whatever directory the App starts in does not matter).
3. The run token is accepted for calls between Apps. Domino's own guidance for 6.3
   says a script calling an App needs a Personal Access Token; if the run token is
   rejected (`503 reranker_unavailable` with HTTP 401 or a redirect in the detail),
   a served component cannot take another key: set the reranker and the embedder
   on `api` instead (`QP_COMPONENTS__RERANKER__BACKEND=api`,
   `QP_COMPONENTS__RERANKER__API__URL` the same URL), with a personal access token
   as `QP_COMPONENTS__RERANKER__API__API_KEY` (the embedder's likewise), as
   project environment variables.
4. An App answers on `/{owner}/{project}/r/notebookSession/{instance id}` of the
   Domino host. This path is not in the API reference: it comes from using
   Domino. If it is wrong, every URL `qp-deploy` hands out is wrong; the App's
   page (its `url`) still works.
5. `configurationType: AISYSTEM` is what makes an App an Agent (the reference
   lists the value; that the UI's **Deploy Agent** sends it is an assumption).
6. The `[domino]` extra (`dominodatalab[agents]`) installs in the generic
   environment when an Agent starts.
7. A version sent with `environmentRevisionId` starts on that revision of its
   environment (the field is in Domino's Public API spec 6.3.1, optional there).
8. Domino accepts every App name that `app_name_prefix`'s rule allows. The rule is
   `qp-deploy`'s own; a name Domino refuses stops the deploy at that App's
   creation, with Domino's answer.

If one of these fails, the step-by-step section below is the fallback.

### Step by step: first test run on Domino

Everything below is what changes between this repository and your Domino
instance. Replace the placeholders once; the code itself does not change.

| Placeholder / value | Where it goes |
| --- | --- |
| `<domino-host>` | your Domino URL, e.g. `https://domino.example.com` |
| `<registry>` | a registry Domino can pull environments from (only for step 1a) |
| `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY` | project environment variables, every project |
| `QP_COMPONENTS__LLM__API__API_KEY` (and `QP_COMPONENTS__LLM__API__URL` for a gateway), or `QP_COMPONENTS__LLM__BACKEND=fake` | project environment variables of the monolith and workflow projects |
| `QP_ROLE`, and a model server's `QP_MODEL_SERVER_COMPONENT` and `QP_MODEL_SERVER_KIND` | project environment variables, one set per project (see step 2) |
| `QP_USE_CASE_DIR` | project environment variables, every project: the use case's directory as the Apps see the repository, e.g. `/mnt/code/query_processing_rag_usecase`, where `domino/app.sh` runs the role's command (`qp-deploy` sets it itself) |
| `QP_COMPONENTS__RERANKER__SERVED__URL`, `QP_COMPONENTS__EMBEDDER__SERVED__URL` | workflow project, set after step 5 publishes the model apps |
| `QP_OVMS_MODELS_DIR` | OVMS projects, only when the Dataset is not mounted at `/mnt/data/ovms-models` |
| `QP_OVMS_PREFIX_PROXY=1` | OVMS projects, only if step 6 returns `404` |
| hardware tier | the same CPU tier (4+ cores) for every model app; one for the workflow apps |

**1. Environments (once).** Either build and push both images from the
repository's root, then create two Environments from them as custom base images:

```bash
docker build -f query_processing/domino/Dockerfile --target rag  -t <registry>/query-processing-rag  .
docker build -f query_processing/domino/Dockerfile --target ovms -t <registry>/query-processing-ovms .
docker push <registry>/query-processing-rag && docker push <registry>/query-processing-ovms
```

or (1b, no registry) create two Environments on the Domino Standard Environment
and paste the instructions of one `domino/Dockerfile` stage (everything after its
`FROM`) into each Environment's Dockerfile Instructions. The `rag` build downloads
the two HF models, so it needs Hub access at build time.

**2. Projects (one per app).** A Domino App inherits its project's environment
variables and each app needs its own `QP_ROLE` (and a model server its
component and kind), so import this git repository into one project per app and
set, in Project → Settings → Environment variables:

| Project | `QP_ROLE` | Environment | Other variables |
| --- | --- | --- | --- |
| `rag-monolith` | `workflow` | `rag` | `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`, `QP_COMPONENTS__LLM__API__API_KEY`, `QP_TOPOLOGY_ALIAS=monolith` to report the topology |
| `rag-ovms-reranker` | `model-server` | `ovms` | `QP_MODEL_SERVER_COMPONENT=reranker`, `QP_MODEL_SERVER_KIND=ovms`, `QP_OVMS_MODELS_DIR` if needed |
| `rag-ovms-embedder` | `model-server` | `ovms` | `QP_MODEL_SERVER_COMPONENT=embedder`, `QP_MODEL_SERVER_KIND=ovms`, `QP_OVMS_MODELS_DIR` if needed |
| `rag-fastapi-reranker` | `model-server` | `rag` | `QP_MODEL_SERVER_COMPONENT=reranker`, `QP_MODEL_SERVER_KIND=fastapi` |
| `rag-fastapi-embedder` | `model-server` | `rag` | `QP_MODEL_SERVER_COMPONENT=embedder`, `QP_MODEL_SERVER_KIND=fastapi` |
| `rag-workflow` | `workflow` | `rag` | `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`, `QP_COMPONENTS__LLM__API__API_KEY`, `QP_COMPONENTS__RERANKER__BACKEND=served`, `QP_COMPONENTS__RERANKER__SERVED__URL` (+ `QP_COMPONENTS__EMBEDDER__BACKEND=served`, `QP_COMPONENTS__EMBEDDER__SERVED__URL` for split-all), `QP_TOPOLOGY_ALIAS=split-reranker` (or `split-all`) to report the topology, `QP_RUN_LABEL` for a variant |

Skip the rows you do not test. Keep `QP_DOMINO__TRACING` identical across projects
(unset, or `true` everywhere).

**3. OVMS models Dataset (once, OVMS projects only).** In `rag-ovms-reranker`
create a Dataset `ovms-models`, open a Workspace with the `rag` environment and run:

```bash
cd query_processing_rag_usecase                     # the use case's directory, where its workflow file is
../query_processing/ovms/export_models.sh /mnt/data/ovms-models   # needs HF Hub access; ~10 min on CPU
ls /mnt/data/ovms-models                            # config_reranker.json, config_embedder.json, model folders
```

Mount the same Dataset in `rag-ovms-embedder` (Data → Datasets → mount from another
project). DFS projects mount it at `/domino/datasets/local/ovms-models`: set
`QP_OVMS_MODELS_DIR` there. The FastAPI projects need no Dataset.

**4. Index the corpus (once per Elasticsearch cluster).** From a Workspace in
`rag-monolith` (`rag` environment, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` set),
in the use case's directory (`query_processing_rag_usecase/`):

```bash
make ingest                                         # 200 synthetic documents, local embedder
curl -s "$QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL/rag_passages/_count"   # {"count":200,...}
```

Every app reads this one index; the serving embedder must be the one used here
(the embedder's `source_id`, `Snowflake/snowflake-arctic-embed-l-v2.0` in the RAG's workflow file, both locally and on OVMS).

**5. Publish the apps (Apps → Publish, entry point `query_processing/domino/app.sh`).**
Order matters for the split topologies:

1. model apps first: `rag-ovms-reranker`, `rag-ovms-embedder` (and/or the two
   FastAPI projects); wait for "Running" and copy each App URL from the Apps view;
2. put those URLs, followed by `/v3/rerank` / `/v3/embeddings`, into
   `rag-workflow`'s `QP_COMPONENTS__RERANKER__SERVED__URL` / `QP_COMPONENTS__EMBEDDER__SERVED__URL`,
   set the backend of each of those components to `served`
   (`QP_COMPONENTS__RERANKER__BACKEND`, `QP_COMPONENTS__EMBEDDER__BACKEND`),
   `QP_TOPOLOGY_ALIAS` (`split-reranker` or `split-all`, a label only) and,
   for a variant, `QP_RUN_LABEL` (`split-reranker-fastapi`, ...), then publish it;
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
`QP_OVMS_PREFIX_PROXY=1` in both OVMS projects and republish those two apps
(see "Proxy prefix" above); the FastAPI apps need nothing. `/readyz` lists which
dependency is missing when it answers `503`.

**7. Load test (Workspace in `rag-monolith`, `rag` environment, from the use case's directory).** The run token
expires after about 5 minutes, so export it right before each run:

```bash
export QP_LOADTEST_BEARER_TOKEN=$(curl -s http://localhost:8899/access-token)
make loadtest RUN_NAME=monolith        HOST=<URL of the monolith app>
make loadtest RUN_NAME=split-reranker  HOST=<URL of the workflow app>      # the reranker on served
make loadtest RUN_NAME=split-all       HOST=<URL of the workflow app>      # after switching it to split-all
make compare                                                               # results/comparison.md
```

For experiment 2 aim Locust at the model apps themselves:

```bash
make loadtest-models RUN_NAME=rerank-ovms    HOST=<URL of the OVMS reranker app>    TARGET=rerank
make loadtest-models RUN_NAME=rerank-fastapi HOST=<URL of the FastAPI reranker app> TARGET=rerank
qp-compare results/rerank-ovms results/rerank-fastapi --out results/rerank.md
```

`results/` is git-ignored: download the CSV/HTML files from the Workspace or copy
them into a Dataset before stopping it.

**8. What to change between runs, and nothing else.**

| Experiment | Change |
| --- | --- |
| monolith vs split-reranker vs split-all | which app `HOST` points at; the workflow project's backend and served URL of each served component (`QP_COMPONENTS__RERANKER__BACKEND`, `QP_COMPONENTS__RERANKER__SERVED__URL`, `QP_COMPONENTS__EMBEDDER__BACKEND`, `QP_COMPONENTS__EMBEDDER__SERVED__URL`) and `QP_TOPOLOGY_ALIAS` |
| FastAPI vs OVMS behind the workflow | the workflow project's `QP_COMPONENTS__RERANKER__SERVED__URL` / `QP_COMPONENTS__EMBEDDER__SERVED__URL` (and a distinct `QP_RUN_LABEL`) |
| FastAPI vs OVMS directly | `HOST` and `RUN_NAME` of `make loadtest-models` |
| cost of reranking | `QP_COMPONENTS__RERANKER__BACKEND=none` on the monolith or workflow project |
| remove generation latency | `QP_COMPONENTS__LLM__BACKEND=fake` on the monolith and workflow projects |

Keep `USERS`, `SPAWN_RATE`, `RUN_TIME`, the hardware tiers, the corpus and
`QP_DOMINO__TRACING` identical across the runs you compare.

## Development

```bash
make check         # every gate below, in order, stopping at the first failure, then the use case's tests
make test          # pytest: fakes + httpx.MockTransport only; no weights, no network, no Elasticsearch
make format        # ruff --fix + black
```

These run from the repository's root, whose `Makefile` hands them to
`query_processing/`'s, or from `query_processing/` itself, over the toolkit alone
(`make -C query_processing check`). At the root, `make check` then runs the RAG
use case's own tests, from `query_processing_rag_usecase/`, linted with ruff and
black under the toolkit's configuration: its workflow file answers through the
workflow app's HTTP API, and a dry run of each topology alias of its
`deploy.example.yaml` publishes the expected Apps on the fake Domino.
`make check` is the project's quality gate, and the GitHub Actions workflow
`.github/workflows/checks.yml` runs the same command on every pull request and on
every push to `main`. The verdict is deterministic: every gate tool is pinned to an
exact version in the `dev` extra, and every threshold is written in
`query_processing/pyproject.toml`. Two platforms are supported and no other is
targeted: Domino's Ubuntu, and macOS for local work. CI runs the gates inside the
image `query_processing/domino/Dockerfile` builds from (Domino's 6.2 standard
environment, Ubuntu 22.04), as the `ubuntu` user Domino runs Apps as, and in a
virtualenv on the Python of `.python-version`, which is how the Apps start too.
The workflow reads the image from the Dockerfile's `BASE_IMAGE` line, so the two
cannot drift. Each gate is also a `make` target of its own, and its paths below
are in `query_processing/`:

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

`tests/` has one folder per layer of `src/query_processing` (`domain`, `application`,
`config`, `adapters`, `entrypoints`, `loadtest`, `deploy`), plus `scripts` for the
shell scripts, `structure` for the layering and code-shape rules and `fake_domino`
for the fake's own tests. The folders are not packages, so a test file's name is
unique across the tree. What several files share (named fakes, fixtures,
`REPO_ROOT` and `TOOLKIT_ROOT`) is in `tests/helpers`, on pytest's path and
imported by plain module name. The Chroma store's tests run a real embedded Chroma in a temporary folder.
