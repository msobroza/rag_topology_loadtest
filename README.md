# query_processing — workflows declared in files, deployed on Domino in topologies, measured with Locust

This repository holds a toolkit, `query_processing`, and its first use case, a
RAG workflow. A use case is a directory of settings, and two of its files say
what runs and where. The RAG's workflow file,
`query_processing_rag_usecase/workflow.yaml`, without its comments (its
prompts are in `prompts.yaml` and its routes in `routes.yaml`, beside it):

```yaml
extensions: [query_processing_rag_usecase]
local_models: {runtime: openvino}
components:
  embedder: {backend: local, source_id: Snowflake/snowflake-arctic-embed-l-v2.0, query_prefix: "query: "}
  router_embedder: {type: embedder, backend: local, source_id: OrdalieTech/Solon-embeddings-large-0.1, query_prefix: "query : "}
  router:
    backend: local
    embedder: router_embedder
    routes: [self_contained, leans_on_history]
    local: {pipeline_file: router_pipeline.joblib, n_neighbors: 3, metric: cosine}
  reranker: {backend: local, source_id: antoinelouis/crossencoder-camembert-base-mmarcoFR}
  llm:
    backend: api
    api: {url: https://api.openai.com/v1/chat/completions, api_format: openai, model: gpt-4o-mini}
    http: {timeout_s: 120}
  vector_store:
    backend: elasticsearch
    elasticsearch: {url: "http://localhost:9200", index: rag_passages}
    metadata_field_info:
      - name: source
        type: string
        description: The reference the passage comes from.
        allowed_values: [A-25, AMF, CHAD, GAA, code_securite_routiere, educaloi, infoassurances]
graph:
  stages:
    embed_for_route: {call: router_embedder, reads: [conversational_query], writes: [routing_embedding]}
    route:    {call: router, reads: [routing_embedding], writes: [route]}
    rewrite:
      type: generate
      call: llm
      prompt: rewrite_as_standalone
      output_schema: standalone_query_with_filter
      metadata_fields_of: vector_store
      reads: [conversational_query, conversational_history]
      writes: [standalone_query, metadata_filter]
    embed:    {call: embedder, reads: [standalone_query], writes: [query_embedding]}
    retrieve: {call: vector_store, retrieval_mode: dense, top_k: 20, reads: [query_embedding, metadata_filter], writes: [candidates]}
    rerank:   {call: reranker, top_k: 10, reads: [standalone_query, candidates], writes: [reranked]}
    generate: {call: llm, prompt: answer_from_context, reads: [conversational_query, reranked], writes: [answer]}
  edges:
    - [embed_for_route, route]
    - [rewrite, embed, retrieve, rerank]
    - [generate, END]
  conditional_edges:
    - {source: START, path: has_history, reads: [conversational_history], path_map: {true: embed_for_route, false: embed}}
    - {source: route, path: label, reads: [route], path_map: {self_contained: embed, leans_on_history: rewrite}}
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

The toolkit, `query_processing/`, holds what any use case can assemble: the
component types with their requests and answers, the stage types, the edge
conditions, the API formats, the backends of the embedder, the reranker, the
LLM and the vector store, the workflow app (`qp-serve`), the model server
(`qp-model-server`), the deploy tool (`qp-deploy`), the load-test client (a
Locust file, `qp-compare`, `qp-record-answers` and `qp-compare-answers`) and a
fake Domino. It ships no use case's
models, prompts or sizes. A use case installs it and holds its settings and the
implementations, trained models and data that are its own, loaded by the
toolkit from the packages its workflow file names under `extensions`. The
RAG's, `query_processing_rag_usecase/`, adds to the two files its prompts and
routes, an environment example, a `Makefile` of run targets, its tests, and a
package of its own: a kNN router, the router's `local` backend, and
`qp-rag-train-router`, the command that trains it. Its commands run in
its directory, where they find `workflow.yaml`, `deploy.yaml` and `.env`
(`QP_WORKFLOW_FILE` and `QP_DEPLOY_FILE` name others; with no workflow file, a
command stops and says where it looked). Every command of this README runs
there, and a file path is relative to the toolkit's directory, unless it says
otherwise. The design is in
[ADR 0003](docs/adr/0003-the-deploy-tool-resolves-the-topology.md) (the deploy
tool resolves the topology), [ADR 0004](docs/adr/0004-the-workflow-graph-is-a-file.md)
(the workflow graph is a file),
[ADR 0005](docs/adr/0005-toolkit-and-use-case-are-two-distributions.md) (the
toolkit and a use case are two distributions),
[ADR 0006](docs/adr/0006-a-use-case-holds-the-implementations-that-are-its-own.md)
(a use case holds the implementations that are its own),
[ADR 0007](docs/adr/0007-an-edge-condition-can-answer-with-a-label.md) (an
edge condition can answer with a label),
[ADR 0008](docs/adr/0008-every-component-is-called-the-same-way.md) (every
component is called the same way) and
[ADR 0009](docs/adr/0009-a-served-llm-is-asked-for-as-any-served-component.md)
(a served LLM is asked for as any served component is), the vocabulary in
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
conversational history. A first message goes straight to `embed`; one with a
conversational history goes to the router, and when it leans on that history,
the `rewrite` stage makes it a standalone query and may add conditions to the
metadata filter. `/retrieve` ends after rerank, where
`is_query_mode` is false; `timings_ms` has one key per stage, and `tokens`
one per stage whose LLM reported its counts, `rewrite` and `generate`
([HTTP API](#http-api)). Each component has alternatives a setting away:
[Retrieval, stores and model endpoints](#retrieval-stores-and-model-endpoints).

## Quickstart

### Install

At the repository's root, create its one `.venv` on the Python of
`.python-version`, 3.11 (`uv venv`, or `python3.11 -m venv .venv`), then
install both distributions in one command, the toolkit first, since no package
index has it:

```bash
make install-dev    # the toolkit with locust, rouge-score, Textual, pytest and the gate tools, the use case, and the charts package
```

The `make` targets use that `.venv` without activating it; a command typed by
name, such as `qp-serve`, needs its `bin/` on your `PATH`
(`. .venv/bin/activate`). `make install` leaves out locust and rouge-score
(the toolkit's `loadtest` extra: `pip install -e "./query_processing[loadtest]"`), the gate tools,
which [Development](#development) describes, and the charts package
(`experiment_charts/`, which draws `experiments-explained/` and is not part of
the product).
It also leaves out Textual, which draws the list `qp-deploy tiers` and
`qp-deploy environments` open in a terminal (the toolkit's `terminal` extra:
`pip install -e "./query_processing[terminal]"`); without it they print their
table there.

### First run, on fake backends

No model weights, store, API key or network: the six variables put each
component of the RAG's workflow file on its `fake` backend. The fake store is
seeded with 200 synthetic documents as it opens, so there is nothing to ingest
(`qp-ingest` refuses it); the fake LLM echoes the start of its prompt; the fake
router always chooses the first of its routes.

```bash
cd query_processing_rag_usecase
export QP_COMPONENTS__EMBEDDER__BACKEND=fake QP_COMPONENTS__RERANKER__BACKEND=fake \
  QP_COMPONENTS__LLM__BACKEND=fake QP_COMPONENTS__VECTOR_STORE__BACKEND=fake \
  QP_COMPONENTS__ROUTER_EMBEDDER__BACKEND=fake QP_COMPONENTS__ROUTER__BACKEND=fake
qp-serve --host 127.0.0.1 --port 18888    # leave it running
```

From a second shell, in the same directory (the second answer is cut where `...`
stands: the question, the mode, the fake answer and the ten passages kept):

```bash
curl -s localhost:18888/readyz
# {"status":"ready","checks":[{"name":"router_embedder","ok":true,"detail":"fake-embedder"},{"name":"embedder","ok":true,"detail":"fake-embedder"},{"name":"vector_store","ok":true,"detail":"200 passages in memory"},{"name":"reranker","ok":true,"detail":"fake-reranker"}]}
curl -s -X POST localhost:18888/query -H 'Content-Type: application/json' \
  -d '{"question":"How many paid vacation days do employees accrue each year at Northwind Analytics?"}'
# {"request_id":"e1e69f34fa1844e19be2beec8510fcdf","topology":null,"run_label":"embedder=fake+router_embedder=fake+router=fake+reranker=fake+llm=fake+vector_store=fake",...,
#  "timings_ms":{"embed_for_route":0.0,"route":0.0,"rewrite":0.0,"embed":0.2180000301450491,"retrieve":0.8875830098986626,"rerank":0.3361250273883343,"generate":0.3732079640030861,"total":9.341874858364463},"tokens":{"generate":{"prompt":605,"completion":32}},"metadata_filter":[]}
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
value; against a fake Domino in its own process, the dry run runs every check of
a real deploy that needs no answer from Domino, and sends nothing (a hardware tier's
name and a revision's number are shown as written: their ids are asked of
Domino only at a deploy). It lists `qp-rag-embedder` and
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
values. The six variables stay set in the first shell: use a new one for the
real run.

### The real run

The models, Elasticsearch and an API key, from the use case's directory:

```bash
cp .env.example .env          # set QP_COMPONENTS__LLM__API__API_KEY there, or uncomment QP_COMPONENTS__LLM__BACKEND=fake
make infra-up                 # docker compose: Elasticsearch on :9200 (the OVMS containers need make ovms-export first)
make openvino-export          # once: the two embedders and the reranker from the Hugging Face Hub, converted under models/openvino
make ingest                   # qp-ingest --synthetic 200 --embedder embedder -> index rag_passages
make serve                    # qp-serve on http://0.0.0.0:8888
```

The router's pipeline file, `router_pipeline.joblib`, and its manifest,
`router_pipeline.json`, are committed together in the use case's directory,
trained on Solon's vectors of the utterances of `routes.yaml`; start-up loads
them as they are. They are trained again, with `make train-router`, only after
a change the manifest refuses: an utterance or a route of the router, its
`n_neighbors` or `metric`, its embedder's alias or query prefix, or the
scikit-learn pin. Start-up accepts the pipeline whatever the weight format,
runtime or backend of its embedder: the manifest records the embedder's alias
and query prefix, never a runtime, since the format of a served embedder is not
visible to the workflow app. The use case's tests read the committed pipeline as start-up
does and fail on such a change, naming the command. It runs on a machine that
has the router's embedder's model: it refuses to train on the fake embedder's
vectors, which no model gives, and names the variable to unset. Commit both
files of a training together. The calls of the
first run reach it on port 8888; `make infra-down` stops the
containers. `make ingest` embeds with the component `embedder`, which it names
since the file declares two embedders (`qp-ingest --embedder NAME`), and upserts by passage id; `qp-ingest
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
`embedder=local+router_embedder=local+router=local+reranker=served+llm=api+vector_store=elasticsearch`
for the first. Then [Load test and comparison](#load-test-and-comparison), and on Domino
[Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands) or,
by hand, [Step by step](#step-by-step-first-test-run-on-domino).

## Starting a second use case

A second use case is a distribution of its own, in this repository beside the
toolkit, as `query_processing_<name>_usecase/`: `qp-deploy` looks for the
toolkit's launch script going up from where it runs and refuses to publish
from anywhere else, and every App runs its command in that directory, where
the workflow App installs the use case as it starts. It holds its settings,
and, when the toolkit has no implementation that fits, a package of its own
with the implementations, trained models and data that are its own (ADR 0006).
Start from a copy of the RAG's:

| File | What it states |
| --- | --- |
| `workflow.yaml` | The components and the graph ([The workflow file](#the-workflow-file)). `extensions`, the packages of the use case whose classes register before the components are read, when it has any. Each component its `backend`; an embedder and a reranker their `source_id`, even on `fake`; a store on `elasticsearch` or `chroma` its `index` or `collection`; a router its `embedder` and `routes`. Each stage the component it `call`s, its `reads` and `writes`, and the settings its stage type or its component's request requires, which the toolkit leaves without a default: a retrieve stage's `retrieval_mode` and `top_k`, a rerank stage's `top_k`, a generate stage's `prompt`. No API key. |
| `prompts.yaml` | The prompts a stage names, each a list of messages with a `role` and a `content`, a Jinja2 template whose variables are the fields its stage reads ([The workflow file](#the-workflow-file)). |
| `routes.yaml` | When it has a router: the routes its routers choose between, each with a `name`, a `description` and its `utterances` ([The router](#the-router)). |
| `deploy.example.yaml` | Its own `app_name_prefix`, so that its Apps can share a Domino project with the RAG's; its `topology_alias` table over its own components, and `topology`; `defaults.environment_id`; a block of `apps` per App part; this site's `overrides`. Copied to the git-ignored `deploy.yaml`. |
| `pyproject.toml` | A distribution of its own that depends on `query-processing`, which `domino/app.sh` installs with the toolkit: its package under `src/`, the libraries its implementations need and the commands it installs. With no code, `packages = []`. |
| `src/query_processing_<name>_usecase/` | When it has code: its package, in the toolkit's layers (`domain`, `config`, `adapters`, `application`, `infrastructure`, `entrypoints`) and with no `composition`, since a class builds itself. It imports the toolkit through `query_processing.extension_api` alone, and its `__init__.py` files stay empty. The RAG's holds its kNN router (`adapters/router/local/knn_router.py`), its settings, the pipeline file it loads, and `qp-rag-train-router`. |
| `.env.example` | The API keys, by their paths; `QP_TOP_K_RETRIEVE` and `QP_TOP_K_RERANK` for a load test straight at a reranker. |
| `Makefile` | Its run targets. The RAG's names the toolkit as `../query_processing`, the RAG's components in `deploy-embedder`, `deploy-reranker`, `deploy-llm` and `SERVE ?= reranker`, and its own command in `train-router`. |
| `tests/` | Its tests through the two seams, as the RAG's do: its workflow file answers through the workflow app's HTTP API, and a dry run of each topology alias publishes the expected Apps. An implementation of its own is tested by a subclass of a test class the toolkit ships, and its imports by a structure test of its own (the RAG's `tests/test_layering.py`). |

The root `Makefile` and the CI install and test the RAG use case only: install
another with the toolkit in one command (`python -m pip install -e
./query_processing -e ./query_processing_<name>_usecase`) and run its tests in
its directory. The `rag` image of `domino/Dockerfile` takes
`--build-arg USE_CASE=<its directory>`. A minimal workflow file, run from a
scratch directory: a lexical search that keeps three passages, with no LLM and
no code:

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
none): name one, with qp-ingest --embedder`). With `app_name_prefix: qp-search` and the topology alias
`split-reranker: {reranker: served}`, a dry run from beside `query_processing/`
publishes `qp-search-reranker`, then `qp-search-workflow`; from the scratch
directory `qp-deploy` stops (`no query_processing/domino/app.sh in <directory>
or any directory above it`).

### What a use case adds

The toolkit holds what any use case can assemble, and a use case holds what is
its own (rule 8 of [Where a file goes](#where-a-file-goes)). A use case adds
each of these in a module of its package, which the toolkit imports because
the workflow file names the package under `extensions`. Every name it needs is
in `query_processing.extension_api`, whose docstring holds this table:

| To add | Subclass | Write |
| --- | --- | --- |
| An implementation of a component type | its abstract class, such as `BaseRouter` | `invoke`, and `from_settings` when it has settings |
| A whole component type | `BaseComponent[Request, Answer]` | its request and its answer |
| A stage type | `BaseStage` | what the stage does beyond calling its component |
| An edge condition | a function under the toolkit's decorator | the function |
| An output schema | a pydantic model under the toolkit's decorator | its fields |

Two spellings fill the same registries: a keyword on the class line for a
component, a decorator for the other entries (`STAGE_TYPES`, `EDGE_CONDITIONS`,
`OUTPUT_SCHEMAS`, `COMPONENT_API_FORMATS`). A component type names its type on
its class line, `component_type="anonymizer"`, and an implementation its
backend and its settings block, `backend="local", settings=...`. A class is
refused when it is defined, and the message says what to write, if it is a
component type whose request or answer is no pydantic model, or an
implementation that writes neither `invoke` nor `ainvoke`, names a backend its
component type already has, names `settings=` and writes no `from_settings`,
or writes a `from_settings` that is not `@classmethod from_settings(cls,
settings, resources)`. A use case adds registrations and never replaces one of
the toolkit's: a name registered twice stops the load, naming both classes.
Two more names are in `extension_api` for a use case's code:
`MetadataCondition`, the type of an output schema's field that holds a
metadata filter, and `RagDependencyError`, which a component raises, naming
itself, when its model or service fails, and which the app answers with 503,
or 504 with `kind="timeout"`.

An LLM of a use case subclasses `BaseLLM`, also in `extension_api`, and
writes `generate` rather than `invoke`. Besides the messages and the output
schema, `generate` takes `max_tokens`, the cap on the answer that a generate
stage or a plain call states, None for none: the API LLM sends it to its
endpoint, the fake LLM ignores it. It answers a `GenerateAnswer`: the
text, and the token counts its model reported, `prompt_tokens` and
`completion_tokens`, each None when the model reported none. `ainvoke`
returns that answer as it is. A plain call on an LLM fills every attribute of
its request, so its keys state `output_schema: null` and `max_tokens`, `null`
or a cap; its `reads` fill the messages, and it can write the counts to fields
of its own, listed after the text in its `writes`. A generate stage writes the
text alone, or its schema's fields, and reports the counts in the response's
`tokens` ([HTTP API](#http-api)).
The toolkit's API LLM reads the counts from the endpoint's `usage`; when it
asks again for an answer that did
not match its output schema, it reports the sum over every attempt, since each
was paid for, and None for a count any attempt did not report.

Every component type made this way has a `served` backend with no code: the
toolkit builds the class that calls a model server of ours on the type's call
route, and an API format that mirrors its request and its answer, and
`qp-model-server --serve <component>` serves it on the `fastapi` model-server
kind. A use case registers a `BaseApiFormat` under the type's name to redefine
the format, and writes a `served` class, a subclass of `ServedCallMixin` and the
type, only to redefine the built one.

The toolkit ships two test classes in `query_processing.testing`, which a use
case's tests import from there and not from `extension_api`, so that no command
loads what they serve a component with. `ComponentContract`, subclassed with
`make_component` and `make_request`, checks that the class names a backend,
that the request and the answer are pydantic models, the answer's shape, the
API format both ways, and the served class through an in-process model server;
given `settings_block`, the backend's block as the workflow file states it, it
also builds the component from it through `from_settings`, as the app does,
and compares the two answers. `RouterContract` adds `routes`, and checks that
the route chosen is one of them and that the confidence is between 0 and 1.
The RAG's `tests/test_knn_router.py` subclasses it. The same package ships
`directory_size_findings`, the measure of rule 9 of
[Where a file goes](#where-a-file-goes): given a use case's root, and its
exceptions when it has any, it returns each directory that breaks the rule as
a sentence. The limit is the rule's 12 unless another is given; the RAG's
`tests/test_layering.py` calls it with the root alone.

#### One component type from end to end

The toolkit's own tests define, as a use case would, a component type
`shouter` in `query_processing/tests/helpers/shout_extension/`; a test checks
that every block below is still what those files hold. A module of the use
case's package defines the request and the answer, two pydantic models, the
component type that names them, and one implementation with its settings
block, which `from_settings` reads:

```python
from __future__ import annotations

from pydantic import BaseModel

from query_processing.extension_api import (
    BaseComponent,
    ComponentResources,
    SettingsBlock,
)


class ShoutRequest(BaseModel):
    text: str


class Shouted(BaseModel):
    text: str


class BaseShouter(BaseComponent[ShoutRequest, Shouted], component_type="shouter"):
    """Says a text louder."""


class UpperShouterSettings(SettingsBlock):
    suffix: str = ""


class UpperShouter(BaseShouter, backend="local", settings=UpperShouterSettings):
    """Says the text in capitals, its suffix after it."""

    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    @classmethod
    def from_settings(
        cls, settings: UpperShouterSettings, resources: ComponentResources
    ) -> UpperShouter:
        del resources  # a shouter needs nothing besides its own block
        return cls(settings.suffix)

    def invoke(self, request: ShoutRequest) -> Shouted:
        return Shouted(text=request.text.upper() + self.suffix)
```

An implementation with no settings names no `settings=` and writes no
`from_settings`; one that calls a service writes `async def ainvoke` instead of
`invoke`. A component type whose answer a conditional edge on `label` reads
declares the labels of that answer attribute with `answer_labels` (ADR 0007),
as the test extension's `grader` does:

```python
class BaseGrader(BaseComponent[ShoutRequest, Grade], component_type="grader"):
    """Grades how loud a text is: quiet or loud."""

    @classmethod
    def answer_labels(
        cls, type_settings: BaseModel | None
    ) -> dict[str, tuple[str, ...]]:
        del type_settings  # every grader answers one of the same two labels
        return {"level": ("quiet", "loud")}
```

The workflow file names the package under `extensions`, each component's
backend and the block named after it, and the stages that call them: a plain
call's `reads` fill the request's fields and its `writes` take the answer's.
Here the shouter answers only a question its grader finds quiet:

```yaml
extensions: [shout_extension]
components:
  grader: {backend: local}
  shouter:
    backend: local
    local: {suffix: "!"}
graph:
  stages:
    grade: {call: grader, reads: [standalone_query], writes: [level]}
    shout: {call: shouter, reads: [standalone_query], writes: [answer]}
  edges:
    - [START, grade]
    - [shout, END]
  conditional_edges:
    - {source: grade, path: label, reads: [level], path_map: {quiet: shout, loud: END}}
```

Its test imports `ComponentContract` from `query_processing.testing` and says
how to make one component, one request and, to have the component built as
the app builds it, its block:

```python
class TestUpperShouter(ComponentContract):
    def make_component(self) -> BaseShouter:
        return UpperShouter(suffix="!")

    def make_request(self) -> ShoutRequest:
        return ShoutRequest(text="a")

    def settings_block(self) -> dict[str, object]:
        return {"suffix": "!"}
```

The contract's tests are coroutines: the use case's `pyproject.toml` runs them
with pytest-asyncio, as the toolkit's does:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### What only the toolkit adds

What any use case can assemble goes in the toolkit. A stage type and an edge
condition can go either way: in the toolkit when another use case could use
them, else in the use case's package, as above. The other three rows of the
table below only the toolkit can add: the embedder, the reranker, the LLM and
the vector store keep their openers in the composition (ADR 0008), so a use
case cannot yet add a backend or an API format to one of them, and a
model-server kind has a launch script in the toolkit's `domino/`. Each of
these is one new module, under `src/query_processing/`, that registers itself
with its registry's decorator (rule 6); its package imports every module, so
no list is edited. The module also gets its line in [the tree](#the-tree) and
a test.

| A new | Its module | Registered with | A test that adds one |
| --- | --- | --- | --- |
| stage type, for a stage that does more than call its component | `application/stages/<name>_stage.py`: a `BaseStage[<port>, <settings>]` that sets `component_type` and `settings_type`, and defines `check_fields` (the lists of fields it can use, with their types) and `run` | `@STAGE_TYPES.register("<name>")` | `tests/application/test_workflow_builder.py` (`copy`, `shout`, in a registry of the test's own) |
| edge condition | `application/edge_conditions/<name>.py`: a function of the values its conditional edge reads, in the order of its `reads`, that returns a bool, or a `str` label | `@EDGE_CONDITIONS.register("<name>")` | the same file (`is_a_question`) |
| backend of one of the four existing component types | `composition/openers/<type>_on_<backend>.py`: an opener that returns the open port, whose implementation is in `adapters/<type>/<backend>/`; a block in `config/components/component_settings.py` if it has settings | `@RERANKER_OPENERS.register("<backend>")`, or `EMBEDDER_OPENERS`, `LLM_OPENERS`, `VECTOR_STORE_OPENERS` | `tests/composition/test_component_factory.py` (`test_a_new_backend_is_one_new_module`) |
| API format of one of the four | a module of `adapters/<type>/api/`, such as `cohere_rerank_format.py`: a subclass of the type's format base class | `@RERANK_API_FORMATS.register("<name>")`, or `EMBEDDING_API_FORMATS`, `GENERATION_API_FORMATS` | `tests/entrypoints/qp_serve/test_qp_serve_api_formats.py` (`test_a_new_api_format_is_one_new_module`) |
| model-server kind | `deploy/model_server_kinds/<name>_kind.py`: a `BaseModelServerKind` with its `routes` and `probe_route`; and the toolkit's `domino/model_servers/<name>.sh`, which defines `bootstrap_model_server` and `serve_model_server` | `@MODEL_SERVER_KINDS.register("<name>")` | `tests/deploy/model_server_kinds/test_a_new_model_server_kind.py` |

The start-up checks and the graph builder read what a stage type declares, or
what a plain call's component takes and gives, so neither changes. A
component type of the toolkit's own is one abstract class in
`adapters/<component>/base_<component>.py`, as the router's is, with its
request and its answer in the domain, and its module named in
`TOOLKIT_COMPONENT_TYPES` in `config/extensions.py`, which every command
imports before the extensions, so that the start-up checks find it
registered. Its implementations sit beside it, one directory per backend
(rule 4), and a settings class of the type's own keys is registered in
`COMPONENT_TYPE_SETTINGS` in `config/components/component_settings.py`, as
the router's `RouterSettings` is, kept in `config/components/router_settings.py`.

## Code layout

### Where a file goes

The repository holds the product's two distributions (ADR 0005), each a
directory of its own at its root:

- **The toolkit**, in `query_processing/`: what any use case can assemble.
  Every path of this section and of the tree below it is relative to that
  directory, except the use case's and the repository's own entries at the end
  of the tree (`query_processing_rag_usecase/`, `experiment_charts/`,
  `emulation/`, `experiments/`, `experiments-explained/`, `docs/`, the root
  `Makefile`, `.python-version`, and this README, `CONTEXT.md`, `CLAUDE.md` and
  `.github/`). It ships no workflow file and no default that is a use case's
  model, prompt, size, index or collection: its tests run on a fixture workflow
  file of made-up models (`tests/helpers/fixture_workflow.yaml`), and on a
  component type of their own, `shouter`, defined in
  `tests/helpers/shout_extension/` as a use case would define one.
- **A use case**, `query_processing_rag_usecase/` for the RAG: its settings (its
  workflow file, prompts, routes, deploy settings example, environment example,
  a Makefile of run targets), its own tests, and the implementations, trained
  models and data that are its own (ADR 0006), in a package under `src/` that
  its workflow file names under `extensions`. It depends on the toolkit; the
  toolkit never imports nor names a use case. Its commands run in its
  directory, where they find `workflow.yaml`, `deploy.yaml` and `.env`, and
  write what they write.

The toolkit is three applications over one shared domain. The domain
(`domain/`) is the language all three use: no I/O, no framework.

| Application | What it is | Where it lives |
| --- | --- | --- |
| RAG system | what runs on Domino and what prepares its data and models: the workflow app, the model server, ingestion, the corpus and the model export | split by layer: `application/`, `config/`, `adapters/`, `infrastructure/`, `composition/`, `entrypoints/` |
| deploy tool | what publishes the Apps of a Topology (`qp-deploy`) | one package, `deploy/` |
| load-test client | what drives a target and compares runs (Locust, `qp-compare`), and records and compares its answers (`qp-record-answers`, `qp-compare-answers`) | one package, `loadtest/`, and `locustfile.py` at the top, which a use case's Makefile hands to Locust (`-f`) |

Two parts of the toolkit face a use case's code and belong to none of the
three. `extension_api.py` is the one module a use case imports the toolkit
through, and no package of the toolkit imports it. `testing/` holds the test
classes a use case's tests subclass, imported as `query_processing.testing`,
and no package imports it either, `extension_api` included.

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
   exactly after it with hyphens as underscores: the RAG system's eight in
   `entrypoints/` (`qp-serve` is `entrypoints/qp_serve.py`), `qp-deploy` in
   `deploy/qp_deploy.py`, `qp-compare`, `qp-record-answers` and
   `qp-compare-answers` in `loadtest/` (`loadtest/qp_compare.py`, ...).
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
   (`corpora/`), the calibration loops (`calibration/`), OpenVINO Model Server's
   graph, serving folder and process (`ovms/`), `openvino_cpu_properties.py`,
   `one_line_failure.py`, `json_log.py` and `fake_text.py`. Building the implementations
   from the settings is not an adapter either: that is `composition/`, rule 5.
5. **Composition.** `composition/` is the composition root, the only code that
   reads the settings to build an adapter. Each job has one module:
   `resource_scope.py` opens each thing once and closes them all together;
   `http_resources.py` and `local_model_resources.py` hold what the
   implementations share; `openers/` holds one opener per component type and
   Backend, a module each named `<type>_on_<backend>.py`, registered by type
   and Backend (rule 6), and an opener takes its component's settings in the
   workflow file's shape (`config/components/component_settings.py`); the
   deploy tool, which may not import them, reads a type's Backends from these
   module names (`deploy/registered_backends.py`, held to the registries by a
   test); the one factory,
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
   `config/components/component_settings.py`; the five types of the toolkit
   register their classes there by call, four of them kept in that module and
   the router's in `config/components/router_settings.py`, since the
   decorator's return type would hide their fields from the type checkers;
   an unknown type is refused, listing the known ones). A stage type and an edge
   condition call only ports, so they are in the application; the description
   of a graph is in the domain, which both the settings that write it and the
   application that builds it may import. A model-server kind is what the
   deploy tool publishes and refuses by, so it is in `deploy/`; the /v3 routes
   it declares are wire-contract constants of the domain, which the FastAPI
   model server answers on too. Four more kinds serve the components that are
   called the same way (ADR 0008): an output schema (`OUTPUT_SCHEMAS` in
   `application/output_schemas.py`), a structured output method
   (`STRUCTURED_OUTPUT_METHODS` in `adapters/llm/api/base_structured_output_method.py`),
   the API format of a component type registered on its class line
   (`COMPONENT_API_FORMATS` in `domain/endpoints/api_format.py`), and the component
   types with their implementations, which register with no decorator: a
   component type names itself on its class line (`component_type="router"`)
   and an implementation its backend and settings (`backend="local",
   settings=...`), filling `COMPONENT_TYPES` and each type's backends in
   `domain/components/base_component.py`. A use case's package fills the same registries:
   the toolkit imports every module of each package the workflow file names
   under `extensions` (`config/extensions.py`, with `import_package_tree`)
   before it reads the components, and a name registered twice stops the
   load, naming both.
7. **The tree below** lists every package and module of `src/query_processing/`; a
   new module gets a line.
8. **One way.** A use case depends on the toolkit, never the reverse: no file of
   the toolkit's `src/` names a use case (`query_processing_<name>_usecase`),
   in an import, a path or a default; a use case's package name is a value of
   its own workflow file. What any use case can assemble is added to the
   toolkit: a component type with its request and answer, a stage type, an
   edge condition, an API format. A use case holds what is its own (ADR 0006):
   its implementations of a component type, its trained models and its data.
   Its package imports the toolkit through `query_processing.extension_api`
   alone and follows the toolkit's layers, with no `composition`, since a
   class builds itself; its adapters may import its own `config`, as a class
   names its settings on its class line.
9. **Twelve modules a directory.** A directory under a distribution's `src/`
   or `tests/` holds at most 12 modules, the `.py` files directly inside it,
   `__init__.py` not counted and `conftest.py` counted. A crowded directory
   splits by what its modules share, in the order the code already uses: a
   component type, then a Backend, a command, or a glossary area, the
   sub-package named after that thing with a noun of [CONTEXT.md](CONTEXT.md),
   never `utils`, `common` or `misc`. A moved module keeps its full name
   (rule 3), and tests mirror `src/`: a command or module with many test files
   gets a folder named after it. Two kinds of folder are excepted, each listed
   by the test that measures it. A one-pattern folder, whose every module is
   one entry of one sort, may be as long as its list, since its listing reads
   as a table, and every module of it must match its pattern:
   `composition/openers/` (`<type>_on_<backend>.py`, and
   `component_openers.py`), `tests/entrypoints/qp_serve/`
   (`test_qp_serve.py` and `test_qp_serve_<aspect>.py`) and the charts package's `charts/`
   (`<finding>_chart.py`). A folder not yet split (`tests/helpers/`, the
   emulated Domino's code and its tests) may not
   pass the count it had when listed, and its entry fails as stale once the
   folder is back at 12 or under, so the list only shrinks. The measure is
   `directory_size_findings` in `query_processing.testing`, which a use case's
   tests call on the use case.

`tests/structure/test_layering.py` checks rule 1 for every package, that
`locustfile.py` imports only the `loadtest` package, that no package imports
`extension_api` or `testing`, and the toolkit's side of rule 8; the use case's
`tests/test_layering.py` checks its side;
`tests/structure/test_placement.py` checks rule 2, the file names of rule 3,
rule 4 (only component packages in `adapters/`, their layout with `api/` for
both `served` and `api`, and no port implemented anywhere else) and rule 7;
`tests/structure/test_directory_size_rule.py` checks rule 9 on the toolkit, the use
case's `tests/test_layering.py` on the use case, with no exception, and the
charts package's `tests/test_experiment_charts_directory_size.py` on that
package. The fake Domino (`src/fake_domino/`) is a
test double, not part of the product, and rules 1 to 8 do not apply to it.

Measurements go in `experiments/`, at the repository's root, not in the code:
one folder per sitting, with its question, its settings and the statistics of
its runs (see [experiments/README.md](experiments/README.md)). They are records,
which keep the names of their day, and the rules above do not apply to them.

`experiments-explained/` explains those measurements to a reader who does not know the
domain: a tour with one chart per finding (see
[experiments-explained/README.md](experiments-explained/README.md)). Its charts and the
table behind each are drawn from the registry by `experiment_charts/`, a third
folder at the repository's root with a distribution of its own
(`make experiment-charts`, from the root). That package is not part of the
product: it depends on matplotlib and PyYAML, imports nothing from
`query_processing`, `fake_domino` or a use case (its own
`tests/test_experiment_charts_imports.py` checks it), and of the rules above
only rule 9 applies to it.

### The tree

```text
query_processing/   the toolkit, one distribution: pyproject.toml, its Makefile (its install, its gates, pin-answers, fake-domino),
                    vulture_whitelist.py; no workflow file and no use case's file
  src/query_processing/
    __init__.py       package docstring, __version__
    extension_api.py  what a use case imports of the toolkit, every name it needs; its docstring holds the table of what can be added; no package imports it
    domain/           no I/O, no framework; imports no other package
      errors.py       RagDependencyError, describe_error
      registry.py     Registry: named things of one sort, filled by a decorator (rule 6), its names; import_package_modules; import_package_tree, which an extension is loaded by
      conversation.py  HistoryMessage: one message of the conversational history a request carries, its role, content and conditions; check_history_conditions
      prompts.py      PromptMessage, PromptPlaceholder, Prompt: a prompt of prompts.yaml, a list of messages with roles and placeholders
      query_request.py  QueryRequest: the body of /query and /retrieve, which the app and the load test's questions file both check
      graph_description.py  GraphDescription, StageDescription, ConditionalEdge: a graph's stages and edges, START and END
      opened_graph.py  OpenedGraph, OpenedConditionalEdge: what an orchestrator is built from, the description with each stage opened and timed, each conditional edge with what it asks, the state's field types and reducers; merged, how the state takes a stage's update
      run_label.py    the run label: its default (each component with its backend), what a valid one is
      roles.py        the Role each App is published as (Role: workflow or model-server), AppPart, WORKFLOW_APP
      components/     what a component is and how it is called
        base_component.py  BaseComponent: a request in, an answer out (ainvoke, invoke); COMPONENT_TYPES, filled on each component type's class line; each type's backends, filled on each implementation's class line (backend=, settings=); from_settings
        component_requests.py  the request and the answer of the four existing component types: EmbedRequest, RerankRequest, GenerateRequest, SearchRequest and their answers
        component_resources.py  ComponentResources: what an implementation's from_settings may use besides its own block
        routes.py     Route, RouteRequest, RouteChoice: a route of routes.yaml (name, description, utterances), and the router's request and answer
      endpoints/      what an Endpoint is and what is said to it
        api_endpoint.py  ApiEndpoint (URL, API format, API key, model name), the model servers' routes
        api_format.py   BaseApiFormat: a component type's request and answer to the wire and back; COMPONENT_API_FORMATS; api_format_of, the registered format or the one that mirrors the request and the answer
        standard_formats.py  the request and answer shapes of the standard formats (pydantic): OpenAI embeddings, Cohere rerank, OpenAI chat; STANDARD_API_FORMATS, the format name of each component type
      retrieval/      what is searched and what comes back
        passages.py   Passage, ScoredPassage, RagResult, TOTAL_TIMING_KEY, top_by_score
        metadata_filter.py  MetadataCondition, MetadataFieldInfo, check_conditions: a metadata filter's conditions, checked against the fields a vector store declares
        retrieval_mode.py  RetrievalMode: dense, lexical or hybrid, a setting of the retrieve stage
        rank_fusion.py  reciprocal_rank_fusion: what a hybrid retrieval ends with
        corpus.py     synthetic handbook corpus, its questions, chunking
      runtime/        what runs an in-process model and how it is sized
        backends.py   the Runtime names (what runs an in-process model), the Weight format names (what an OpenVINO export stores its weights as), and what a Backend is
        thread_budget.py  ThreadBudget: how the virtual CPUs are split between model calls
        calibration.py  StageTuning (threads, batch size, streams), halvings, fastest, calibration_sample: the rules of calibration, no clock
    application/      use cases over the ports; imports domain and the adapters' ports only
      rag_dependencies.py  RagDependencies: the components a workflow calls, open, by the names its stages call them; their probes and model names
      workflow_state.py  WorkflowState, what a request carries from stage to stage by field name (the state's type is built in state_fields); REQUEST_FIELD_TYPES, the fields the request writes; RESPONSE_FIELDS, those the response reads; REQUEST_FIELDS, those a stage may read wherever it is; MEASURED_FIELDS, timings_ms and tokens, which no stage writes; initial_state, sum_timings_by_stage, sum_tokens_by_stage
      state_fields.py  field_types_of: each field the request and the stages write, with its type; state_field_types and STATE_REDUCERS: every field of the state and the reducers of those that add up, as the orchestrator is given them; WorkflowGraphError
      graph_check.py  check_graph: a graph that cannot run refused before anything opens, from what each stage type declares or each plain call states: its stages, the field rule and the type rules, the edges through edge_check; fields_set_before
      stage_binding.py  stage_type_of: a stage's stage type, or none for a plain call; bind_call: a plain call's keys, reads and writes bound to its component's request and answer
      edge_check.py   check_edges: the edges and conditional edges of a graph that cannot run, refused; field_labels: the labels each field can take
      workflow_builder.py  build_workflow: a graph description, the components and the orchestrator's class in, the opened graph built on that orchestrator out; asked_of_state
      workflow.py     CompiledWorkflow: one request through the orchestrator, its RagResult out; stage_timings
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
      workflow_settings.py  WorkflowSettings: every top-level key of the workflow file (the components, the graph, the orchestrator, the shared blocks), the topology alias and the run label (two labels), Domino's own run variables (pydantic-settings, .env and workflow file aware)
      searched_embedder.py  searched_embedder: the embedder a vector store's searches use, the one called by the stage that writes what a retrieve stage reads (it seeds a fake store)
      extensions.py   load_extensions: the toolkit's own component types, then every module of each package the workflow file names under extensions, imported before its components are read
      shared_settings.py  LocalModelsSettings, CalibrationSettings, DefaultHttpSettings, DominoSettings: what the components share, four top-level blocks of the workflow file
      ovms_settings.py  OvmsSettings: an OVMS model server App's own QP_OVMS_* variables (its streams, its threads, its start timeout, an LLM's two batching options), no value of the workflow file
      components/     each component's settings
        settings_blocks.py  SettingsBlock, ServedSettings, ComponentHttpSettings, ComponentSettings: the blocks every component's settings are made of
        component_settings.py  ComponentSettings and each type's (EmbedderSettings, ..., VectorStoreSettings): a component's type, backend, source id and alias, its settings, one block per backend (local, served, api; elasticsearch, chroma, fake)
        registered_component_settings.py  RegisteredComponentSettings: a component of a type registered in COMPONENT_TYPES (an extension's), its type's keys and each backend's block read with the class its implementation names
        router_settings.py  RouterSettings: a router's embedder and routes, two at least; a route the routes file lacks refused
        endpoint_urls.py  model_endpoint_url and missing_url_message: the URL a model component on served or api is called at, and the refusal of a missing one
      files/          the three files a use case writes, and the settings sources that read them
        workflow_file.py  workflow_file (QP_WORKFLOW_FILE, else ./workflow.yaml, a use case's, else a refusal naming both), the settings source that reads it, declared_alias
        prompts_file.py  prompts_file (QP_PROMPTS_FILE, else ./prompts.yaml, else none), the settings source that gives it as the prompts setting; a prompts key in the workflow file refused
        routes_file.py  routes_file (QP_ROUTES_FILE, else ./routes.yaml, else none), the settings source that gives its catalogue as the routes setting; a routes key in the workflow file refused
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
        api/          backends: served and api; importing it imports every module, so each format and structured output method registers itself
          api_llm.py  ApiLLM, and open_api_llm: the LLM of one endpoint
          base_generation_api_format.py  BaseGenerationApiFormat, GENERATION_API_FORMATS: the formats by name
          openai_chat_format.py  the openai chat completions format (messages in, choices[0] out)
          base_structured_output_method.py  BaseStructuredOutputMethod, STRUCTURED_OUTPUT_METHODS: how a schema reaches the endpoint, by name
          json_schema_method.py  json_schema: the schema as OpenAI's response_format (the default)
          format_instructions_method.py  format_instructions: the schema in the messages, LangChain core's parser reads the answer
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
      orchestrator/   the library that runs the workflow graph; importing it imports every module under it, so each orchestrator registers itself
        base_orchestrator.py  BaseOrchestrator, the port: check_graph_runs refuses a graph before anything opens, built from an opened graph, ainvoke runs a request from its first state to its last; ORCHESTRATORS
        langgraph/    orchestrator: langgraph, the default
          langgraph_orchestrator.py  LangGraphOrchestrator: LangGraph's StateGraph of the opened graph, refusing no graph, the one module importing LangGraph; state_type, langgraph_name, path_map
        pydantic_graph/  orchestrator: pydantic_graph, pydantic-graph (Pydantic AI's graph library)
          pydantic_graph_orchestrator.py  PydanticGraphOrchestrator: pydantic-graph's graph of the opened graph, a step per stage that merges its update, a step per conditional edge named <source>:<edge condition> then a decision; refuses more than one way out at once and a loop; the one module importing pydantic-graph
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
        stage_calibration.py  BaseCalibratedStage, calibrate: time a CPU-bound stage under each candidate batch size and thread count, keep the fastest
        cpu_count.py  where the CPU count comes from: QP_LOCAL_MODELS__CPUS, the Domino hardware tier, the container
      http/           what every adapter that calls out shares
        http_policy.py  HttpPolicy: timeout, retries and backoff shared by every adapter that calls out
        json_post.py    RetryingJsonPost: the POST-with-retries the endpoint adapters share
      corpora/        an outside source: the public corpora qp-fetch-corpus downloads
        catalog.py    CORPORA: the corpora it knows, one CorpusDefinition each
        download.py   download a corpus's files under one HttpPolicy and convert them
        conversion.py a fetched file's text into documents or questions; CorpusFormatError names the file and line
        jsonl_files.py  write JSON lines atomically: a temporary file, then a rename
      calibration/    the calibration loops
        timed_trials.py  CalibrationLimits, TimedTrials: the trials, the budget rule, the report and its stage_calibrated line
        served_stage_calibration.py  BaseServedStage, calibrate_served: one model server start per candidate, streams then threads
      ovms/           OpenVINO Model Server, run by us
        ovms_graph.py  ovms_plugin_config, graph_with_plugin_config: the graph's one plugin_config line for a tuning
        ovms_serving_folder.py  write_ovms_serving_folder: the export linked, a graph of our own; the export is never written; recorded_weight_format: what the export says its weights are stored as
        ovms_process.py  OvmsProcess: one ovms on a free local port for an async with, waited for and stopped
      openvino_cpu_properties.py  openvino_cpu_properties: INFERENCE_NUM_THREADS and ENABLE_HYPER_THREADING for a thread count
      one_line_failure.py  one_line_failure: how a command reports a failure an operator can act on, on one stderr line
      fake_text.py    tokenize, cosine: the text helpers the fake implementations share
      json_log.py     log_json: the one log format
    composition/      the composition root: builds the adapters from the settings (rule 5)
      composition_root.py  CompositionRoot: settings in, ready ports out, and the orchestrator's class the settings name; what each command asks for; check_workflow_graph: the toolkit's checks of the graph, then the orchestrator's
      resource_scope.py  ResourceScope: each resource opened once, all closed together
      http_resources.py  HttpResources: the shared HTTP client, its policy, the run token source, served_endpoint (the run token on Domino)
      local_model_resources.py  LocalModelResources: the thread budget, the model pool, a local model's files
      calibrated_stages.py  CalibratedRerankStage, CalibratedEmbedStage: the CPU-bound stages calibration times, and their candidates
      ovms_stages.py  OvmsRerankStage, OvmsEmbedStage: the stages OVMS serves, as calibration times them
      ovms_calibration.py  OvmsCalibration: an OVMS App's streams and threads, at start-up or on demand, and its serving folder; require_asked_weight_format: an export of another weight format than the settings ask, refused
      component_factory.py  ComponentFactory: the port supplied under a component's name, else what the opener of its type and Backend opens, once
      registered_names.py  check_registered_names: an unknown Backend, API format or orchestrator refused before a command opens anything
      registered_components.py  open_registered: a component of a type registered in COMPONENT_TYPES, built by the implementation its Backend names; component_resources, what it is given besides its block; component_type_settings, each component's type keys
      chosen_orchestrator.py  chosen_orchestrator: the orchestrator's class the settings name; check_orchestrator_runs_graph: its refusal of the graph, after its name
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
        llm_on_served.py  the LLM of a model server of ours (OVMS alone serves one): its served URL, the standard format, its alias
        llm_on_api.py  the LLM of an endpoint we do not run, under its own timeout; open_llm_at, which both use
        llm_on_fake.py  FakeLLM
        vector_store_on_elasticsearch.py  Elasticsearch, its client closed with the scope
        vector_store_on_chroma.py  Chroma, a server or embedded
        vector_store_on_fake.py  the in-memory store, seeded through the embedder
    entrypoints/      the RAG system's commands: FastAPI apps and CLIs
      qp_serve.py     qp-serve: the FastAPI app (/query /retrieve /healthz /readyz), --domino, Domino tracing, JSON logs, a refused graph as one line
      qp_model_server.py  qp-model-server: the FastAPI model server of one named component on OpenVINO Model Server's /v3 routes, or on its call route (POST /call/<component type>) for a registered type; its answers, UNKNOWN_MODEL_ERROR
      qp_served_models.py  qp-served-models: each embedder, reranker and LLM of the workflow file (name, type, source id, alias, then the weight format), one line of shell words each
      qp_export_openvino.py  qp-export-openvino: convert the in-process models of the embedders and rerankers to OpenVINO files
      qp_calibrate.py  qp-calibrate: calibrate the in-process models (or OVMS) here, print the environment paths (or the OVMS App's variables) that pin them
      qp_ovms_config.py  qp-ovms-config: the config an OVMS App starts with, its export held to the weight format asked, its streams and threads calibrated first; with --serve-llm, an LLM's batching flags
      qp_ingest.py    qp-ingest: JSONL loader, chunk, embed and upsert a corpus
      qp_fetch_corpus.py  qp-fetch-corpus: download a public corpus for qp-ingest and Locust
    deploy/           the deploy tool: qp-deploy on Domino 6.2
      deploy_settings.py  DeploySettings: deploy.yaml (the topology aliases, shared defaults, one block per App, each writing its hardware tier by name or id and its environment revision by number or id, the overrides), QP_DEPLOY_* overrides, require_deploy_file (a refusal naming where it looked), project_or_workspace; Placement (local or served); override_values, each override by its path
      deploy_refusals.py  what is refused in the deploy settings before anything is published: a secret in a settings block, a value still holding an example placeholder
      workflow_components.py  the workflow file the workflow App reads: its components (name, type, backend), the variable of a value's path and the path of a variable, the overrides' keys walked and their values checked with the workflow settings' models
      overrides.py    the overrides, this site's values of the workflow file: checked before anything is published (and a workflow setting refused in a free list), then each App's exports
      registered_backends.py  registered_backends: a component type's backends, read from the names of its opener modules in composition/openers/ (rule 5) without importing them
      topology_alias.py  resolve_topology: the chosen topology alias against the workflow file, its Apps, each served component's model-server kind and route, and run label, or a refusal (a backend a component's type lacks included)
      model_server_kinds/  the model-server kinds (rule 6), each with its launch script domino/model_servers/<kind>.sh; importing it imports every module
        base_model_server_kind.py  BaseModelServerKind (the types it serves, the route of each, its probe, what its script reads, what it refuses), MODEL_SERVER_KINDS, model_server_kind
        fastapi_kind.py  fastapi: qp-model-server, an embedder or a reranker on the /v3 routes, a component of a registered type on its call route
        ovms_kind.py  ovms: OpenVINO Model Server on its /v3 routes, from the OVMS bundle it refuses when missing
      publish.py      where every App starts (LaunchPaths: the toolkit's launch script, found going up from the use case's directory, and that directory), the entry scripts of a topology's Apps (the workflow app, or a model server told its component and kind), App or Agent, and publishing them in order
      endpoint_url.py where an App answers: {host}/{owner}/{project}/r/notebookSession/{instance id}
      app_id_cache.py the App ids already seen (.qp-deploy-cache.json): an App is redeployed, never created twice
      status.py       qp-deploy status: one line per App, then the workflow App's /readyz
      qp_deploy.py    qp-deploy: the parser, url, --dry-run, tiers and environments (refused in a dry run)
      apps_client.py  the Domino 6.2 beta Apps API calls qp-deploy makes (find, create, add a version, get, list instances) and the wait until Running; send_to_domino, domino_json and describe_failure, shared by every call to Domino
      domino_lookups/  what qp-deploy reads of Domino besides its Apps
        paged_list.py   read_pages: a list of Domino's documented API read page after page, a 401 or 403 refused with the permission it needs
        hardware_tiers.py  HardwareTier (its size, every field Domino gave), list_hardware_tiers, tier_named: one tier by its exact name, or a refusal listing the tiers Domino has
        environments.py  EnvironmentRevision, DominoEnvironment (its latest, active and restricted revisions, every field Domino gave), list_environments
        revision_sources.py  BaseRevisionSource, DocumentedRevisions (one environment's three revisions, asked once), revision_numbered: a revision by its number from the sources in order, or a refusal saying what each holds
        internal_revisions.py  InternalRevisions: every revision of an environment with its build date, through the internal route Domino's own pages use (not in Domino's reference, not verified on a live Domino); a route that fails gives no revision and is told, never raised
        base_domino_ids.py  AppWithIds, BaseDominoIds (an App's tier name and revision number turned into the ids its version sends), where_set, find_every_id: every App's ids found, then printed, before the first App is published
        domino_ids_asked.py  DominoIdsAsked: the ids asked of Domino, the tiers listed once, a revision from the documented call then the internal route
        domino_ids_not_asked.py  DominoIdsNotAsked: a dry run's ids, the name and the number as written, nothing asked
      domino_listings/  what qp-deploy tiers and qp-deploy environments show of Domino's answers
        aligned_table.py  aligned_table: plain text in columns padded to their widest value, some aligned right, no colour
        listing_names.py  only_named: the items of the names given, in Domino's order, or a refusal listing the names Domino has; details_yaml: every field Domino gave as YAML, the name first
        listing_looks.py  Look, chosen_look: the details when asked, else the table when asked or with no terminal, else the list
        listing_settings.py  ListingSettings: what a listing reads of deploy.yaml, how to reach Domino and nothing else, so no App block needs to be valid
        tiers_listing.py  tiers_table (the default tier marked), tiers_details
        environments_listing.py  environments_table (the active and latest numbers), shown_revisions (every revision of the internal route with its documented status, or the three and a note), revisions_table (dates in UTC, marks), revision_lines (an open environment's revisions in the list), environments_details
        opening_list.py  OpeningItem, OpeningList, run_opening_list: the list that opens in a terminal, drawn by Textual (the extra terminal), the only module importing it; an item's lines asked when first opened, / matching its name
        listing_commands.py  show_tiers, show_environments: ask Domino, then show the look, the list without Textual the table and how to install it; tier_items, environment_items, list_is_installed; the internal route asked only for an environment named or opened
    loadtest/         the load-test client, with locustfile.py
      loadtest_settings.py  LoadTestSettings: every QP_LOADTEST_* variable, the four values a model-server request carries (two sizes, required by the rerank target, and two aliases, the workflow file's), Domino auth
      request_plans.py  what to send: the request pool (synthetic questions or a questions file) and one plan per request
      response_checks.py  what to record: a response becomes a failure or STAGE rows, one per stage its timings_ms names
      target_record.py  the target record (_target.json): TargetRecord, written as a run stops and read by qp-compare
      qp_compare.py   qp-compare: Locust *_stats.csv files to markdown tables
      answer_record.py  the answer record (_answers.json): AnswerRecord, what a target answered without what changes from run to run, written by qp-record-answers and read by qp-compare-answers
      target_client.py  TargetClient: the JSON POST qp-record-answers makes, its timeout, retries and backoff from the load test's settings
      qp_record_answers.py  qp-record-answers: each request of the request pool sent once, its answer recorded
      answer_comparison.py  what differs between two answer records of the same requests: exactly, within the score tolerance, the answers' text with a metric on demand
      rouge_l.py      ROUGE-L between two answers an LLM wrote, the one module importing rouge-score (the loadtest extra), with a tokenizer that keeps accented letters
      qp_compare_answers.py  qp-compare-answers: answer records against the first, to markdown; status 1 when an answer differs
    testing/          the tests the toolkit ships, which a user runs on a component of their own; a use case's tests import them from query_processing.testing, no package imports them
      component_contract.py  ComponentContract: subclassed with make_component and make_request, it checks the answer's shape, the API format both ways, and the served class through an in-process model server
      router_contract.py  RouterContract: ComponentContract for a router, with routes to write; the route chosen is one of them, the confidence between 0 and 1
      directory_size.py  directory_size_findings, OnePatternFolder, FolderNotYetSplit: rule 9 measured under a distribution's src/ and tests/, each breach a sentence naming the folder, its count and what to do
      pinned_answer_record.py  PinnedAnswerRecord, check_pinned_answers: the workflow app recorded in process on fake backends and compared with its committed answer record, or pinned again with --pin-answers
  src/fake_domino/    a stand-in for Domino 6.2's Apps API: qp-deploy's test double and what --dry-run runs.
                      It imports nothing from query_processing, and only deploy/qp_deploy.py imports it (both checked by
                      tests/structure/test_layering.py)
    api_shapes.py     the Apps API, hardware tier and environment shapes, transcribed from the 6.2 reference, and the published shape of an environment's every revision (what the fake serves)
    state.py          FakeDominoApps: the fake's Apps, versions and instances, its hardware tiers and its environments (FakeEnvironment, with every revision)
    instances.py      BaseInstanceScheduler: what decides an instance's status; by default the --behaviour scripts
    apps_routes.py    /api/apps/beta/apps: list, create, add a version, get, list a version's instances
    stand_ins.py      hardware tiers (the list, one by its id), environments (the list, one with its three revisions), fake_tier, fake_environment, and the stand-ins: every revision of an environment, /access-token and an App instance's /readyz
    server.py         create_fake_domino (FastAPI, Swagger at /docs), the in-process client, qp-fake-domino
    emulation/        TEMPORARY: the emulated Domino (proxy, App placement on machines), see emulation/README.md
  locustfile.py       RagUser (FastHttpUser) + opt-in StepLoadShape; a use case's Makefile runs it (locust -f) from the use case's directory
  docker-compose.yml  Elasticsearch 9.0.0 (:9200) + OVMS reranker (:8001) + OVMS embedder (:8002), a use case's directory its project directory
  ovms/               versions.env (single OVMS version pin), export_models.sh (into ovms/models of the working directory, a use case's)
  domino/             app.sh (one entry point: the workflow app or a model server, from QP_ROLE, run in the use case's directory, QP_USE_CASE_DIR), model_servers/ (one launch script per model-server kind: fastapi.sh, ovms.sh), Dockerfile (targets `rag` and `ovms`)
  tests/              pytest: no model weights, no network, no Elasticsearch; one folder per package of src/query_processing/, and in a crowded folder one sub-folder per sub-package, module or command (rule 9)
    domain/ application/ config/ adapters/ infrastructure/ composition/ entrypoints/ loadtest/ deploy/ testing/   the tests of that sub-package
    testing/answer_records/   the pinned answer record of the fixture workflow on fake backends (make -C query_processing pin-answers)
    scripts/          domino/app.sh and ovms/export_models.sh, run against fake binaries
    structure/        the import, placement and code-shape rules, read from the source with ast, and the directory-size rule (rule 9)
    fake_domino/      the fake Domino's own tests; they import nothing from query_processing
    emulation/        TEMPORARY: the emulation's Terraform, parsed, and the rules that keep it removable
    helpers/          named fakes, fixtures, fixture_workflow.yaml (the tests' workflow file: made-up models) and its fixture_prompts.yaml, shout_extension/ (a component type of the tests' own, defined as a use case would), and the two roots (REPO_ROOT, TOOLKIT_ROOT) shared by several test files (on pytest's path)
query_processing_rag_usecase/   the RAG use case, one distribution that installs the toolkit: its settings, and its kNN router (ADR 0006)
  pyproject.toml      the distribution (query-processing-rag-usecase): the toolkit, scikit-learn pinned exactly and joblib; its package under src/; qp-rag-train-router
  workflow.yaml       the RAG's workflow file: its extension, its components (models, router, endpoint, store) and its graph (stages, sizes, edges)
  prompts.yaml        the RAG's prompts, which its generate stages name (answer_from_context, rewrite_as_standalone)
  routes.yaml         the routes its router chooses between, each with its utterances
  router_pipeline.joblib, router_pipeline.json   the router's trained pipeline and its manifest, committed together (make train-router)
  src/query_processing_rag_usecase/   its package, named under extensions; it imports the toolkit through extension_api alone
    config/knn_router_settings.py     KnnRouterSettings: the router's local block (pipeline_file, n_neighbors, metric)
    adapters/router/local/knn_router.py  KnnRouter: the router's local backend, a trained kNN pipeline
    infrastructure/knn_pipeline.py    the scikit-learn pipeline: a normalizer and a k-nearest-neighbours classifier
    infrastructure/pipeline_artifact.py  the pipeline file and its manifest: written, and refused at load when the manifest does not match
    entrypoints/train_router.py       qp-rag-train-router: embeds the utterances, fits and scores the pipeline, writes the file (make train-router)
  deploy.example.yaml  its deploy settings example: the topology aliases, the Apps, this site's overrides (cp to deploy.yaml)
  .env.example        its environment example, the load-test sizes of a rerank request included (cp to .env)
  Makefile            its run targets (serve, ingest, train-router, loadtest, compare, the exports, calibrate, deploy, status, infra-up, pin-answers, ...), run here
  tests/              its own tests, through the two seams: its workflow file answers, a dry run of each topology alias; its router through RouterContract; its imports; helpers/ on pytest's path; answer_records/, its pinned answer records on fake backends, one for each way a follow-up takes, with their questions file (make pin-answers)
  results/            load-test output (git-ignored), as data/, models/, ovms/models/ and chroma/ written here
experiment_charts/  the charts of the measurements, one distribution (experiment-charts): not the product, it draws experiments-explained/ from experiments/
  pyproject.toml      the distribution: matplotlib and PyYAML, nothing of the product; its package under src/
  src/experiment_charts/   its package; it imports nothing from query_processing, fake_domino or a use case
    registry_reading.py  ExperimentRegistry: every results.csv row and every experiments*.yaml entry of the registry
    chart_table.py    ChartTable: the table behind one chart, written to experiments-explained/data/
    chart_theme.py    ChartTheme (light and dark) and the themed figure, axes, titles and legend
    chart_marks.py    the marks several charts share: grouped bars, lines with ringed markers, value labels
    chart_selections.py  the run selections the catalog passes to a chart
    base_experiment_chart.py  BaseExperimentChart: a slug, a title, table() and draw()
    charts/           *_chart.py: one module per chart, each an implementation of BaseExperimentChart
    chart_catalog.py  experiment_charts(): the charts in the tour's order, the one place that names experiment ids
    write_experiment_charts.py  writes every chart's two figures and its table; python -m experiment_charts
  tests/              its own tests: its tables against the registry, every chart drawn, data/ kept current, its imports; helpers/ on pytest's path
emulation/          TEMPORARY: Terraform and scripts for an emulation of Domino on Google Cloud
experiments/        the registry of every measurement: one folder per sitting (README.md, experiments*.yaml, results.csv)
experiments-explained/  the measurements told to a newcomer: README.md, and figures/ and data/ drawn by experiment_charts/
docs/adr/           the decisions
docs/plans/         dated records: a plan as it was approved, and what became of it
Makefile            installs the toolkit and the use case in one command, and the charts package with them for development; hands check and each gate to query_processing/, then runs them over the use case and over the charts package, each followed by its tests; experiment-charts; the one .venv here
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
| `QP_LOADTEST_REQUEST_TIMEOUT_S` | how long a user waits for one answer before Locust records a failure (its `network_timeout`), default `60` s, above 0 |
| `QP_LOADTEST_SHAPE=step` | step load 5 → 10 → 20 → 40 users, 60 s each (set `RUN_TIME=4m` or more); any other value but unset is refused |
| `QP_LOADTEST_STEP_USERS` / `QP_LOADTEST_STEP_SECONDS` | the steps of the step shape, default `5,10,20,40` / `60` |
| `QP_LOADTEST_QUESTIONS` | distinct synthetic questions the users ask, default `50` |
| `QP_LOADTEST_ANSWER_ROUGE_L` | `qp-compare-answers` holds the answers an LLM wrote with ROUGE-L, default `false`: their text is not compared |
| `QP_LOADTEST_ANSWER_ROUGE_L_THRESHOLD` | the lowest ROUGE-L F-measure at which an answer is the baseline's, default `0.9`; a number between 0 and 1 |
| `QP_LOADTEST_BEARER_TOKEN` | sent as `Authorization: Bearer ...` (Domino run token) |
| `DOMINO_API_KEY` | sent as `X-Domino-Api-Key` (from outside Domino) |

A request counts as a success only when the response is HTTP 200 with a JSON
object body. Any other status fails as `HTTP <status>: <first 200 characters of
the body>`, and a 200 whose body is not JSON, or is JSON but not an object, also
counts as a failure (it used to vanish from the statistics altogether).
Every successful response's `timings_ms` is re-emitted as Locust `STAGE` rows,
one per stage it names, in its order: `/query:embed`, `/query:retrieve`,
`/query:rerank`, `/query:generate` for a first message of the RAG's graph,
`/query:embed_for_route`, `/query:route` and `/query:rewrite` before them for
a follow-up that leans on its history, same for `/retrieve`; the total and a
stage timed at zero (generate on `/retrieve`, a stage the request did not take)
get no row, nor does a stage whose time is no number. The client
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

The record also holds `tokens` when any response reported a stage's tokens:
each such stage, named as its STAGE row is (`/query:generate`,
`/query:rewrite`, `/retrieve:rewrite`), with `responses`, the successful
responses whose `tokens` map gave the stage both counts and whose `timings_ms`
timed it above zero, their `prompt` and `completion` tokens, and `stage_ms`,
the stage's time over those same responses:

```json
{"target": "rag", "seen": [], "tokens": {"/query:generate": {"responses": 100, "prompt": 189000, "completion": 14200, "stage_ms": 1420000.0}}}
```

A stage whose counts or time are unusable is left out alone; the other stages
of the response still count. A record written before tokens were counted has
no `tokens` and still reads; one main's load test wrote, whose `tokens` is
`{"answers", "prompt", "completion", "generate_ms"}`, reads as the one stage
`/query:generate`.

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

When a run's record has tokens, `qp-compare` adds a `## Tokens` table, one row
per such run and stage, in the runs' order and then by the stage's name:
responses, prompt and completion tokens per response, tokens a second as a
caller sees them (the stage's completion tokens over its `stage_ms`) and tokens
a second served (the stage's completion tokens over the run's seconds, from the
`Aggregated` row's requests and RPS, so a failed request adds none and the rows
of one run add up to what its LLM wrote a second). With the LLM on `fake`
the counts are words of the echo, not tokens of a model, so its token figures
say nothing of a server. The settings of a run that measures the LLM: "The
waits on the path of an answer" in [Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands).

### Answer records: the same requests, the same passages

The load test says how fast a target answers; an answer record says what it
answered. The two commands are the check that a refactor, or a move to
another topology, changed nothing a caller sees: record a deployment before
and after, then compare.

The check is self-contained, in three parts. The corpus is the synthetic
corpus the toolkit generates, indexed into the deployment's store by `make
ingest`. The requests are the load test's synthetic questions, built in
process (`QP_LOADTEST_QUESTIONS` of them, 50 by default). The models are the
ones the deployment already has: on Domino they are on a Dataset, where a
model's `source_id` is a folder and the OpenVINO and OVMS model folders live
(`/domino/datasets/local/<dataset>/...`, see the `overrides` and `settings`
of [Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands)).
The check fetches no corpus and no model: `qp-record-answers` reads no
questions file unless given one, and calls nothing but its target.

```bash
make ingest                           # the synthetic corpus, into the deployment's store
make record-answers RUN_NAME=monolith HOST=http://localhost:8888
make record-answers RUN_NAME=split-all HOST=http://localhost:8888
make compare-answers                  # results/answers-comparison.md
make record-answers-domino RUN_NAME=split-all   # the workflow App on Domino
```

`qp-record-answers --host <url> --out <prefix>` sends each request of the load
test's request pool once, in the pool's order, one after the other: each body
to `/retrieve` and then to `/query`. The pool, the credentials and the wait for
one answer are the load test's (`QP_LOADTEST_QUESTIONS`,
`QP_LOADTEST_BEARER_TOKEN` or `DOMINO_API_KEY`,
`QP_LOADTEST_REQUEST_TIMEOUT_S`). `QP_LOADTEST_QUESTIONS_FILE` replaces the
synthetic questions with the reader's own file, already on the machine; `make
record-answers` sets none. An endpoint whose load-test weight is 0 is
not sent: `QP_LOADTEST_QUERY_WEIGHT=0` records `/retrieve` alone (a deployment
without a working LLM key, or one where each `/query` costs an LLM call), and
`QP_LOADTEST_RETRIEVE_WEIGHT=0` `/query` alone; both at 0 is refused. A
request the target never answers is sent again, `QP_LOADTEST_RETRIES` times,
after `QP_LOADTEST_BACKOFF_S` doubled before each next retry (default 2 and
0.2 s, the defaults of the workflow file's shared `http` block); then the command
stops with status 3 and writes nothing, since half a record would pass for a
whole one. Any response the target gives is recorded, an error status
included; a 200 whose JSON object is not of the answer's shape (a passage
without an `id`, say) stops it with status 4, naming the endpoint and the
field, and nothing is written either.

It writes `<prefix>_answers.json`, the answer record. For each request it
holds the endpoint, the body sent and the status, and from a 200's JSON object
the `mode`, the `metadata_filter` applied, the `answer`, and each passage in
order with its `id`, the SHA-256 of its `text`, its `metadata`, its
`retrieval_score` and its `rerank_score`; a response that is not a 200 with a
JSON object is recorded with its status alone. It also holds, under `seen`,
each topology alias and run label the responses reported, as the target
record does. It keeps none of `timings_ms`, `tokens`, `request_id`,
`run_label` or the passages' text: they change from run to run, or a hash
says as much.

`qp-compare-answers <prefix> <prefix>... [--out <file>]` compares each run's
record with the first, the baseline, and names a run by the last component of
its prefix, as `qp-compare` does:

| What | How |
| --- | --- |
| status, mode, metadata filter, the passages' ids in order, their text hashes and metadata | exactly; an absent metadata filter equals an empty one, since both apply no condition (an app from before the field records none) |
| the retrieval and rerank scores | within `QP_LOADTEST_SCORE_TOLERANCE` (default `0.0001`, 0 or more); an absent score equals only an absent one |
| the answer the LLM wrote | only with `QP_LOADTEST_ANSWER_ROUGE_L=true`, with ROUGE-L's F-measure against the baseline's at or above `QP_LOADTEST_ANSWER_ROUGE_L_THRESHOLD` (default `0.9`); two equal texts are the same, two empty ones included, as are two absent ones (on `/retrieve`), and a text on one side only is different |

What the deterministic components give is compared exactly. The answer an
LLM wrote never is: asked twice, even at temperature 0, it can word an answer
otherwise, so its text is held with ROUGE-L (from `rouge-score`, of the
toolkit's `loadtest` extra, given a tokenizer that keeps accented letters), a
metric computed the same way every time. Run the LLM at temperature 0, the
default of its `served` and `api` blocks. Off, the report says in one line
that the text was not compared, so "identical" never reads as covering the
LLM; on, each run's row adds the lowest and the mean ROUGE-L of its answers,
and each answer under the threshold is listed with its score and both texts.
Asked for without the `loadtest` extra, the command stops with status 2,
naming the extra to install.

The report lists the runs with the topology alias and run label each saw, one
row per run (requests, identical, different), then every answer that differs
with its endpoint, its question, the field, the baseline's value and the run's.
The exit status is 0 when every run answers as the baseline, 1 when an answer
differs, and 2 when the records cannot be compared: a file that is not an
answer record (the message names the file, the field and the shape expected),
two prefixes of one last component, or records of different requests (their number, their order, an endpoint or a
body), refused before any comparison, naming the first request that differs.
`make compare-answers` compares every `results/*_answers.json`, sorted;
`BASELINE=split-all` puts that run first.

The check on real models runs on demand only, where they run: of it, `make
check` and CI run its own tests and the pinned answer records below, on
fakes. What to expect: the same code on two topologies answers the same, since
a component served over HTTP returns what it returns in process. Two model
servers (FastAPI and OVMS) or two weight formats may not: their scores move,
and a passage can change places. Compare each with a baseline of its own.

#### Pinned answer records: the answers on fake backends, in the gate

What the workflow app answers on fake backends is committed as answer
records, the "before" a change to how the graph runs is held to. A test of each
distribution starts the app in process with every component on its fake
backend (no socket, no download), records it with `qp-record-answers`, and
compares the record with the committed one with `qp-compare-answers`: every
answer identical, its text included, with ROUGE-L at a threshold of 1 (the
fake LLM's echo is a function of its prompt), and the scores within the score
tolerance. `make check` runs both.

| Distribution | Records | Requests |
| --- | --- | --- |
| the toolkit | `tests/testing/answer_records/fixture_workflow_answers.json`, the fixture workflow | ten synthetic questions, one per topic of the synthetic corpus |
| the RAG use case | `tests/answer_records/follow_ups_stand_alone_answers.json` and `follow_ups_rewritten_answers.json`, its own `workflow.yaml` | `tests/answer_records/questions.jsonl`: a first message, then two follow-ups with a conversational history |

Each request goes to `/retrieve` and to `/query`. The fake router chooses the
first of its routes, so the use case's two records cover the three paths of
its graph: in both, the first message takes no route; with the routes in the
file's order a follow-up stands alone, and with `leans_on_history` first
(`QP_COMPONENTS__ROUTER__ROUTES`) it is rewritten. The code both tests share is
`query_processing.testing`'s (`PinnedAnswerRecord`, `check_pinned_answers`), so
a use case's tests pin their own records the same way.

A change that alters the answers on purpose makes the records again, from the
repository's root, then commits them; the failing test names the command:

```bash
make -C query_processing pin-answers              # the fixture workflow's record
make -C query_processing_rag_usecase pin-answers  # the use case's two records
```

Each runs its test with `--pin-answers`, in process on fake backends, which
writes the record instead of comparing with it. A record holds nothing that
changes from one process to the next: the fakes hash with SHA-256, never with
`hash()`, and the synthetic corpus and questions come from a seeded
`random.Random`.

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

**Compare servers on the same weight format.** The FastAPI model server runs the
in-process export and OVMS its own export; both read the workflow file's
`local_models.openvino_weight_format` (`QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT`:
`fp32` or `int8`, see [Weight format](#weight-format)). Measured on 15 November 2026,
one reranker replica on a 2-CPU machine: the reranker's export is 442 MB at
`fp32` and 111 MB at `int8`. OVMS on `int8` looked 38 to 65% faster than the
FastAPI model server on `fp32`, and the whole lead was the 8-bit weights: on the
same weights both served the same, 0.24 requests a second each at `fp32` and
0.36 to 0.38 at `int8`. Ranking quality on `int8` weights was not checked.

`docs/model-servers.md` sets the two servers side by side: what the documentation
of OVMS and of FastAPI says about serving, how each is set up here, and what the
sittings measured (streams, batch size, order of service).

## HTTP API

| Route | Body → result |
| --- | --- |
| `POST /query` | `{question, conversational_history?, metadata_filter?}` → `{request_id, topology, run_label, question, mode, answer, passages[], timings_ms{embed_for_route,route,rewrite,embed,retrieve,rerank,generate,total}, tokens{rewrite?,generate?}, metadata_filter}`; `conversational_history` holds the earlier turns, `{role: user\|assistant, content, metadata_filter?}`, a user message's conditions being read by the prompts only, never applied to retrieval; the response's `metadata_filter` is the conditions applied; `topology` is the topology alias, `null` when none was given; `timings_ms` has one key per stage of the graph, in its order, then `total` (lexical retrieval: no `embed`); `tokens` has one key per stage whose LLM reported both its counts in this request, in the graph's order, each `{"prompt": n, "completion": m}`, the sum over the attempts of an answer asked again; a stage that did not run, or whose LLM reported a count as none, has no key, so a request whose stages reported none has `{}`; a stage that ran more than once in the request, in a loop or where two branches join at different depths, reports in both the sum of its runs |
| `POST /retrieve` | same request → same response with `answer: null` (`generate` skipped, `generate = 0`, no `generate` key in `tokens`: `{}` unless `rewrite` ran) |
| `GET /healthz` | `{"status":"ok"}` |
| `GET /readyz` | `{"status":"ready"\|"not_ready","checks":[{name,ok,detail}]}`, one check per vector store (index count > 0), embedder and reranker a stage calls, named after the component (`vector_store`, `embedder`, `router_embedder`, `reranker`); `503` when a check fails |

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
prompt `prompts.yaml` does not define, or one using a field its stage does not
read) included. Only what needs an open component,
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
`embedder`, `reranker`, `llm`, `vector_store` or `router`, or a component type
an extension registers, and defaults to its name, so a second component of one
type names it: the RAG's `router_embedder: {type: embedder, ...}`, which the
`embed_for_route` stage calls with `call: router_embedder`. Every component is
called the same way, a request in and an answer out (ADR 0008), and its
`backend` says which of its type's registered classes runs it.

Every stage states `reads` and `writes`, the workflow state fields it reads and
writes. A stage is of the stage type its `type` names, else of the stage type
of its own name (`retrieve` and `generate`), else a plain call, which only calls
its component: a key of the stage that names an attribute of the component's
request fills it, such as `top_k: 10`; `reads` fill the other attributes in
order; and `writes` take the answer's attributes in order, and can stop early.
The RAG's `embed_for_route`, `route`, `embed` and `rerank` stages are plain
calls. A `retrieve` stage reads `[query_embedding]` for `dense`,
`[standalone_query]` for `lexical` and both, in that order, for `hybrid`, then
`metadata_filter`, and writes one field; a `generate` stage reads any fields,
each the prompt variable of the same name, and writes one field, the text, or
the fields of its output schema.

The vector store is a component like the models: its backend is named
after its product, `elasticsearch` or `chroma`, or is `fake`, each with its
block, and `qp-ingest` fills the one the file declares.

| Component type | Key | In the RAG's file | Meaning |
| --- | --- | --- | --- |
| every type | `backend` | `local`, `local`, `local`, `local`, `api`, `elasticsearch` | how it runs, one of the backends registered for its type: `local` \| `served` \| `api` \| `fake` for an embedder, the same and `none` (no rerank) for a reranker, `served` \| `api` \| `fake` for an LLM, `elasticsearch` \| `chroma` \| `fake` for a vector store, `served` \| `fake` for a router and `local` when the use case's extension registers it, as the RAG's does; `served` is a model server of ours, `api` an endpoint we do not run; an unknown one is refused at start-up, naming the known ones. Nothing else decides it: `qp-deploy` sets it for each component the topology alias names, and on a laptop you set it |
| every type | `type` | the router's embedder's: `embedder`; not written for the others: the component's name | `embedder` \| `reranker` \| `llm` \| `vector_store` \| `router`, or a type an extension registers |
| every model | `source_id` | the three models' Hugging Face ids | the weights, a Hugging Face id or a folder; required of an embedder and a reranker; an LLM's is the model OVMS pulls when it serves the LLM ("A served LLM" in [Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands)), and the RAG's file writes none |
| every model | `alias` | not written: the last part of `source_id` | the name a model server of ours serves the model under (`snowflake-arctic-embed-l-v2.0`), and the model name a `served` component sends |
| `embedder` | `query_prefix` | `query: ` | put in front of each query before it is embedded, never of a passage (arctic-embed expects it); the `local`, `served` and `api` embedders apply it, the `fake` one has none, and a model server adds none to what it receives; empty for a model that takes none |
| `embedder` | `local.batch_size`, `local.threads` | not written: `64`, `0` | texts per encode call (>= 1); native (OpenMP/BLAS) threads one call may use, `0` = calibrated at start-up, up to the budget's share |
| `embedder` | `local.normalize` | not written: `false` | scale each vector the in-process (`local`) embedder returns to unit length; off is safe because every store compares vectors by cosine, which ignores their length. The RAG's embedder ends its own pipeline with a normalisation step, so its vectors are unit-length whatever this says (measured: length 1.000000 with it off and on). It does not change OVMS: its own export normalises on its side (the exporter's default) |
| `reranker` | `local.batch_size`, `local.threads` | not written: `0`, `0` | passages per predict batch and native threads per call; `0` = calibrated at start-up, the batch size up to `local.calibration_passages` |
| `reranker` | `local.calibration_passages` | not written: `20` | passages in calibration's sample rerank request, all kept; also the largest batch size it tries (>= 1). The reranker's own: calibration reads no stage's setting |
| every model | `served.url` | not written | the full route of the model server of ours that serves it, e.g. `http://localhost:8001/v3/rerank`; required with the `served` backend. The rest follows: the standard format of its type, the alias as the model name, the Domino run token under `qp-serve --domino` and no key elsewhere |
| `embedder` | `served.batch_size` | not written: `64` | texts per model-server request (>= 1) |
| `llm` | `served.temperature` | not written: `0` | sampling temperature, 0 to 2 |
| `llm` | `served.structured_output`, `api.structured_output` | not written: `json_schema` | how an output schema reaches the endpoint: `json_schema` (OpenAI's `response_format`, strict) or `format_instructions` (the schema in the messages, for an endpoint that imposes none); an unknown one is refused before a command opens anything ([Structured output](#structured-output)) |
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
| `vector_store` | `metadata_field_info` | `source`, a string of seven values | the metadata fields a request's metadata filter, and an output schema narrowed to this store, can name: each a `name`, a `type` (`string`, `integer`, `float`, `boolean`), a `description` and optionally its `allowed_values`; none declared accepts only an empty filter ([The metadata filter](#the-metadata-filter-and-the-prefilter)) |
| `vector_store` | `prefilter` | not written: none | conditions every search applies before the request's own, on fields a client cannot name; the RAG's `deploy.example.yaml` shows one under `overrides` |
| `router` | `embedder` | `router_embedder` | the embedder component whose vectors the router reads ([The router](#the-router)) |
| `router` | `routes` | `self_contained`, `leans_on_history` | the routes of `routes.yaml` it chooses between, two at least; a route the file lacks is refused |
| `router` | `local.pipeline_file`, `local.n_neighbors`, `local.metric` | `router_pipeline.joblib`, `3`, `cosine` | the RAG's kNN router's own block, no default: the pipeline file `make train-router` writes, with its manifest beside it, and the values it is trained with |

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
local_models: {runtime: openvino, openvino_dir: models/openvino, openvino_weight_format: fp32, workers: 4, cpus: 0}
calibration:  {budget_s: 90, rounds: 1, users: 4}
http:         {timeout_s: 30, retries: 2, backoff_s: 0.2}
domino:       {token_url: "http://localhost:8899/access-token", token_ttl_s: 240, tracing: false}
```

| Block | Key | Default | Meaning |
| --- | --- | --- | --- |
| `local_models` | `runtime` | `openvino` | what runs the in-process models, those of the components on `local`: `openvino` \| `torch` |
| `local_models` | `openvino_dir` | `models/openvino` | where `qp-export-openvino` writes and the `openvino` runtime looks first, one folder per model and weight format |
| `local_models` | `openvino_weight_format` | `fp32` | what both OpenVINO exports (in-process and OVMS) store the weights as: `fp32` \| `int8` (8-bit weight compression only); changing it exports again, into `<model>-int8` in-process. See [Weight format](#weight-format) |
| `local_models` | `workers` | `4` | under the `torch` runtime, at most this many in-process inference calls at once (fewer on a small CPU budget); `openvino` ignores it and runs one call at a time (>= 1) |
| `local_models` | `cpus` | `0` | virtual CPUs for the thread budget; `0` = find out (Domino tier, else container), a positive value wins |
| `calibration` | `budget_s` | `90` | per CPU-bound stage, the seconds calibration may search (at start-up or in `qp-calibrate`); no new candidate starts after them, and an OVMS trial still running then is abandoned (> 0) |
| `calibration` | `rounds` | `1` | timed requests per calibration candidate, after one warm-up (>= 1) |
| `calibration` | `users` | `4` | requests in flight while a model server's candidate is timed: the load OVMS's stream count is chosen for (>= 1); in-process stages are timed on one request |
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
| every stage | `call` | | the component it calls, by its name: `router_embedder`, `router`, `llm`, `embedder`, `vector_store`, `reranker` |
| every stage | `type` | not written: the stage's name | the stage type it is an instance of; a stage of no stage type is a plain call |
| every stage | `reads`, `writes` | each stage's | the workflow state fields it reads and writes, in the order its stage type or its component takes and gives them; required |
| `retrieve` | `retrieval_mode` | `dense` | `dense` (by vector) \| `lexical` (by words, ranked by the store) \| `hybrid` (both, fused by reciprocal rank); `lexical` and `hybrid` need `elasticsearch` or `fake`, and are refused with `chroma` when the graph is built |
| `retrieve` | `top_k` | `20` | candidates from the vector store (>= 1) |
| `retrieve` | `rrf_k` | not written: `60` | rank constant of the hybrid fusion (>= 1); smaller widens the gap between ranks |
| `rerank` | `top_k` | `10` | passages kept after reranking and sent to the LLM (>= 1) |
| `generate` | `prompt` | `rewrite_as_standalone`, `answer_from_context` | the prompt of `prompts.yaml` it renders; the fields of its `reads` are the prompt's variables, any number of them, and it writes one field, the text, unless it names an output schema |
| `generate` | `output_schema` | the rewrite's: `standalone_query_with_filter`; not written for `generate` | a registered output schema the LLM answers in; the stage writes each of its fields to the field of `writes` of the same name, and `writes` lists them all ([Structured output](#structured-output)) |
| `generate` | `metadata_fields_of` | the rewrite's: `vector_store` | a vector store whose declared fields narrow the output schema's conditions, and which the prompt gets as `metadata_field_info` |
| `generate` | `max_tokens` | not written: no cap | the most tokens the stage's answer may hold (>= 1), sent to the LLM's endpoint as `max_tokens`; the stage's, not the LLM's: a cap on `generate` leaves `rewrite`, whose answer must match its schema, uncut |

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

The orchestrator is the library that runs the graph along its edges (ADR
0010): `langgraph`, the default, or another name registered in `ORCHESTRATORS`
(`adapters/orchestrator/`). It is the top-level key `orchestrator`, a name and
not a block, and the graph does not change with it. A workflow file states it
only to run on another one. Like any value of the file it is set three ways:

```bash
QP_ORCHESTRATOR=langgraph make serve
```

```yaml
# or as a top-level key of a workflow file
orchestrator: langgraph
```

```yaml
# or for the Apps of a deploy, in deploy.yaml, with the run label that names the variant
overrides:
  orchestrator: langgraph
  run_label: monolith+orchestrator=langgraph
```

An unknown name stops every command that opens components, as an unknown
backend does (see the variables below). The default run label does not name the
orchestrator: a run on another one is a variant, named with `run_label`. The
workflow app's `rag_startup` log line names the orchestrator it runs on.

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
  expects: for a `generate` stage, a prompt that is not defined or uses a
  variable the stage does not read, an output schema that is not registered or
  whose fields differ from its `writes`, and a `metadata_fields_of` that names
  no vector store;
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
qp-serve: error: stage 'retrieve' reads the workflow state field 'query_embedding', which the request does not provide and is written by no stage of the graph (expected a field the request provides, one of: 'conversational_history', 'conversational_query', 'metadata_filter', 'mode', 'standalone_query', 'timings_ms', 'tokens', or one a stage writes on every path from START to 'retrieve')
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
  reads `answer`, `reranked` and `metadata_filter`), as a likely typo;
- a stage that writes `timings_ms` or `tokens`, which the workflow measures
  for each stage under its name;
- a key of a plain call that does not fit the type of the request attribute it
  fills, such as `top_k: ten`, naming the key, its value and the type expected.

Once these checks accept the graph, the orchestrator the workflow file chooses
is asked whether it runs it as LangGraph does, still before any component
opens, and refuses what it does not run: its refusal is a usage error after its
name, `qp-serve: error: orchestrator '<name>': ...`, naming the stages
(`composition/chosen_orchestrator.py`). It is asked only about a graph the
checks above accept. `langgraph`, the default and the reference, refuses none.
`pydantic_graph` refuses the two shapes its library does not run as LangGraph
does (seen on pydantic-graph 2.70.0):

- `START` or a stage with more than one way out at once: two plain edges, a
  plain edge and a conditional edge, or two conditional edges on one source.
  pydantic-graph ends a run when the first branch reaches `END` and cancels
  the others, where LangGraph waits for every branch;
- a loop, found with the standard library's `graphlib`: pydantic-graph has no
  limit that stops one that never ends.

```
qp-serve: error: orchestrator 'pydantic_graph': 'rerank' has more than one way out at once; pydantic-graph ends a run when the first reaches END and cancels the others (expected one plain edge or one conditional edge out of each stage, or the orchestrator 'langgraph')
qp-serve: error: orchestrator 'pydantic_graph': the stages 'draft' -> 'draft' form a loop; pydantic-graph has no limit that stops one (expected a graph without a loop, or the orchestrator 'langgraph')
```

The workflow state is built from the same fields: each field the request or a
stage writes, with its type, and `timings_ms` and `tokens`
(`application/state_fields.py`).

Only whether the vector store ranks text needs the store open, so a lexical
or hybrid retrieve stage over `chroma` is refused as the graph is built, still
before the app serves, with one line in the server's log.

### Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `QP_TOPOLOGY_ALIAS` | unset | the topology alias deployed, which `qp-deploy` sets: a label only, it changes no backend; echoed as `topology` and `X-RAG-Topology`, and unset (or empty) the header is left out and `topology` is `null`; letters, digits and `._+=:-`, 1 to 128 characters |
| `QP_RUN_LABEL` | the default run label | names the measured configuration, echoed as `run_label` and `X-RAG-Run-Label`; default: each component with its backend, in the workflow file's order, joined by `+` (`embedder=local+router_embedder=local+router=local+reranker=local+llm=api+vector_store=elasticsearch` for the RAG's file); `qp-deploy` sets it to the topology alias plus each served component's model-server kind; a variant the backends do not show, such as a retrieval mode or a hardware tier, sets its own (`run_label` under the overrides of `deploy.yaml`, with `qp-deploy`); letters, digits and `._+=:-`, 1 to 128 characters |
| `QP_ORCHESTRATOR` | `langgraph` | the library that runs the workflow graph, a name registered in `ORCHESTRATORS`; also the workflow file's top-level `orchestrator` and a key of the overrides of `deploy.yaml` (see the workflow file above); an unknown name is refused before any component opens, listing the registered ones |
| `QP_WORKFLOW_FILE` | unset | the workflow file to read, before a `workflow.yaml` in the working directory (environment only, not `.env`); with neither, a command stops, naming both places |
| `QP_PROMPTS_FILE`, `QP_ROUTES_FILE` | unset | the prompts file and the routes file to read, before a `prompts.yaml` and a `routes.yaml` in the working directory; with neither, there are no prompts or no routes, and a stage or a router that names one is refused at start-up |
| `QP_COMPONENTS__<NAME>__...` | from the workflow file | one value of a component of the workflow file, named by its path, such as `QP_COMPONENTS__RERANKER__LOCAL__THREADS`, `QP_COMPONENTS__LLM__API__API_KEY` or `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL` (see the workflow file above) |
| `QP_GRAPH__<KEY>__...` | from the workflow file | one value of the workflow file's `graph` block, named by its path, such as `QP_GRAPH__STAGES__RERANK__TOP_K` (see the workflow file above) |
| `QP_LOCAL_MODELS__...`, `QP_CALIBRATION__...`, `QP_HTTP__...`, `QP_DOMINO__...` | from the workflow file, else the defaults | one value of a block the components share, named by its path, such as `QP_LOCAL_MODELS__CPUS` or `QP_DOMINO__TRACING` (see the shared blocks above) |
| `QP_OVMS_NUM_STREAMS` | `0` | an OVMS model server App's own, no value of the workflow file: requests OVMS runs at once on its model (`NUM_STREAMS`); `0` = calibrated at start-up, 1, 2, 4, ... up to the CPUs. Set in that App's block of `deploy.yaml` (`apps.<component>.settings`) |
| `QP_OVMS_THREADS` | `0` | an OVMS App's own: threads its inference shares, all streams together (`INFERENCE_NUM_THREADS`); `0` = calibrated at start-up, the CPUs down to 1; a pinned value below a pinned `QP_OVMS_NUM_STREAMS` is refused |
| `QP_OVMS_START_TIMEOUT_S` | `120` | an OVMS App's own: how long one OVMS start may take to answer before its calibration fails (> 0) |
| `QP_OVMS_DYNAMIC_SPLIT_FUSE` | `true` | an LLM's OVMS App's own: `false` has OVMS read a prompt in one pass (`--dynamic_split_fuse false`); refused on an embedder's or a reranker's App. See "An LLM on OVMS" in [OpenVINO Model Server](#openvino-model-server) |
| `QP_OVMS_MAX_BATCHED_TOKENS` | `0` | an LLM's OVMS App's own: the tokens OVMS batches in one step (`--max_num_batched_tokens`), `0` = OVMS's own choice, at least `0`; refused on an embedder's or a reranker's App, like `QP_OVMS_DYNAMIC_SPLIT_FUSE` |
| `DOMINO_API_PROXY`, `DOMINO_HARDWARE_TIER_ID` | set by Domino | Domino's own run variables, not settings of ours: read as-is (no `QP_` prefix) to look up the run's hardware tier |
| `QP_LOADTEST_TARGET` | `rag` | Locust side: what a run drives: the workflow app (`rag`), or a model server directly (`rerank`, `embeddings`) |
| `QP_TOP_K_RETRIEVE`, `QP_TOP_K_RERANK` | none: required by the `rerank` target | Locust side only: a `rerank` target is sent this many documents and this `top_n`; the use case's sizes, which its `.env.example` states (the RAG's: `20`, `10`, its stages' `top_k`), and the workflow app does not read either. The aliases a model-server target sends default to the workflow file's, which only that target reads |
| `QP_LOADTEST_BEARER_TOKEN`, `DOMINO_API_KEY` | | Locust side: auth header, see the load-test table |
| `QP_LOADTEST_QUERY_WEIGHT`, `QP_LOADTEST_RETRIEVE_WEIGHT`, `QP_LOADTEST_SHAPE` | `1`, `3`, unset | Locust side: task weights (0 or more) and the step shape (`step` or unset) |
| `QP_LOADTEST_WAIT_MIN_S`, `QP_LOADTEST_WAIT_MAX_S` | `0.2`, `1.0` | Locust side: the range each user waits between requests, in seconds |
| `QP_LOADTEST_STEP_USERS`, `QP_LOADTEST_STEP_SECONDS` | `5,10,20,40`, `60` | Locust side: the users of each step and the seconds per step of `QP_LOADTEST_SHAPE=step` |
| `QP_LOADTEST_QUESTIONS` | `50` | Locust side: distinct synthetic questions the users ask |
| `QP_LOADTEST_QUESTIONS_FILE` | unset | Locust side: ask the questions of this file instead (JSON lines with a `question` key, as `qp-fetch-corpus` writes, or one question per line); `QP_LOADTEST_QUESTIONS` then does not apply. A JSON line can also carry `conversational_history` and `metadata_filter`, in the shape of the `/query` body, and the request sends them: the file decides the mix of first messages and follow-ups. Any other key of a line is left out |

Every tunable of the workflow is a setting of a component, a setting the components share,
or a setting of a stage of the graph, each read from the workflow file and its
environment path; only the topology alias and the run label, two labels, and
the orchestrator, a name, are flat. They are the fields of `WorkflowSettings` in `config/workflow_settings.py`; adapters
get the values as constructor arguments from the `composition/` package. An OVMS
model server App's own variables (`QP_OVMS_NUM_STREAMS`, `QP_OVMS_THREADS`,
`QP_OVMS_START_TIMEOUT_S`, and an LLM's `QP_OVMS_DYNAMIC_SPLIT_FUSE` and
`QP_OVMS_MAX_BATCHED_TOKENS`) are no value of the workflow file: they belong to one
App and its hardware, so they are flat, the fields of `OvmsSettings` in
`config/ovms_settings.py`, and set in that App's block of `deploy.yaml`. Sources
win in the order the workflow file's section gives. An invalid runtime or
component type, a run label or topology alias no header can carry (a default
run label too long or with a component name outside those characters included:
set `QP_RUN_LABEL` then), a temperature outside 0 to 2, a component on
the `served` or `api` backend whose URL (`served.url` or `api.url`, the block
of its backend) does not start with `http://` or `https://`, a `served`
component without an alias, a missing workflow file or an unknown key in it
fails as the settings are read. A graph that cannot run, a stage setting its stage type refuses included
(an unknown key, a retrieval mode, a size below 1, a prompt the prompts file
does not define or one using a field its stage does not read), is refused as `qp-serve` reads its settings
(see [What start-up refuses in a graph](#what-start-up-refuses-in-a-graph)),
and lexical or hybrid retrieval on a store that ranks no text fails as the
graph is built. All are at start-up, with the offending value. For example:

```
QP_COMPONENTS__RERANKER__BACKEND='served' needs QP_COMPONENTS__RERANKER__SERVED__URL, the full route of the endpoint (for example http://localhost:8001/v3/rerank); got ''
```

An unknown backend, API format or orchestrator stops every command that opens
components (`qp-serve`, `qp-model-server`, `qp-ingest`, `qp-calibrate`) before
it opens any, as a settings mistake: the settings cannot see the openers, the
formats and the orchestrators, which are registered in the composition and the
adapters, so each
command asks their registries as it reads its settings, and the registry
refuses the name, after the variable it is read from, and lists the known ones:

```
qp-serve: error: QP_COMPONENTS__RERANKER__API__API_FORMAT: unknown rerank API format 'jina' (expected one of: 'cohere')
qp-serve: error: QP_ORCHESTRATOR: unknown orchestrator 'pydantic' (expected one of: 'langgraph', 'pydantic_graph')
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

**Calibrating OVMS.** An OVMS App is calibrated too, on its two parameters: its
stream count (`QP_OVMS_NUM_STREAMS`, requests OVMS runs at once on its model)
and its thread count (`QP_OVMS_THREADS`, the threads all its streams share).
A lone request is always fastest on one stream, since it gets every thread.
Streams only pay when several requests are in flight, so a stream count is
chosen for a stated load: `QP_CALIBRATION__USERS` requests at once, 4 by
default. A trial is the time until all of them are answered. OVMS takes its
settings only when it starts, so each candidate is one start of `ovms`. Every
stream count (1, 2, 4, ... up to the CPUs, 1 first) is timed on all the CPUs;
then each smaller thread count, as long as it is at least the best stream
count, with that stream count. The warm-up and rounds are the in-process ones,
and so is the sample: a reranker's `local.calibration_passages` passages, all
kept, or one question an embedder embeds as a query.
The budget also bounds a trial: one still running when it is spent is
abandoned, its `ovms` stopped, and the search ends there. The fastest completed
trial wins; if none completed, the first candidate is kept untimed: one stream,
which is what the export runs. So calibration lasts about the budget plus one
start, and a slow candidate never fails the App. The requests go through the
workflow app's own `api` adapters, with no retry: a retried request would be
timed as one slow request. Each waits up to `QP_CALIBRATION__BUDGET_S`, not
the component's HTTP timeout (`http.timeout_s`), since it may wait behind the
others in flight. The result is one `stage_calibrated` line, which also names
the streams and the requests, then one `ovms_serving_config` line.

Run on real machines on 15 November 2026 (OVMS 2026.3.1, one reranker of 8 and of
16 CPUs), before the commands and variables took the names of today. For 4
requests in flight the whole search took 151 seconds on 8 CPUs,
six candidates, so the default budget of 90 seconds covers the stream counts and
not the thread counts. On both machines one request already used every core, and
streams gained 2% at most: 1, 2 and 4 streams were within a few percent of each
other, and more were slower. With one timed round such candidates swap places
from run to run: the same machine kept 1 stream at one start and 4 at another.
Raise `QP_CALIBRATION__ROUNDS` for a steadier choice. `docs/model-servers.md` has
the figures.

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
make calibrate                                  # qp-calibrate: every embedder and reranker, the router's embedder included
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
one line on standard error and exit code 1.

With `--ovms-models DIR` it calibrates OVMS instead, on the export in DIR
(`ovms` must be on `PATH`), in a temporary folder it removes; `--only` still
picks one component. It searches both parameters, even pinned ones. Each
component prints the same two variables, the OVMS App's own, since one OVMS App
serves one component: pin them in that App's block of `deploy.yaml`
(`apps.<component>.settings`, which its second comment names), not under
`overrides`, which every App gets.

```bash
qp-calibrate --ovms-models /mnt/data/ovms-models --only reranker
```

On an 8-CPU machine on 15 November 2026, with 16 requests in flight
(`QP_CALIBRATION__USERS=16`) and a budget of 1,500 seconds, it printed this
(shown with today's variable names, and with the component's name and the
`deploy.yaml` comment it did not print yet):

```
# reranker (rerank): timed on 8 CPUs under ovms, 30.812 s for 16 requests at once; tried 6 candidates in 503.5 s
# deploy.yaml, under apps.reranker.settings: QP_OVMS_NUM_STREAMS: 2, QP_OVMS_THREADS: 8
QP_OVMS_NUM_STREAMS=2
QP_OVMS_THREADS=8
```

```yaml
apps:
  reranker:
    server: ovms
    settings: {QP_OVMS_NUM_STREAMS: 2, QP_OVMS_THREADS: 8}
```

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
| `RAG_LOCAL_MODEL_RUNTIME`, `RAG_OPENVINO_MODELS_DIR`, `RAG_OPENVINO_WEIGHT_FORMAT`, `RAG_MODEL_WORKERS`, `RAG_MODEL_CPUS` | the `local_models` block: `QP_LOCAL_MODELS__RUNTIME`, `QP_LOCAL_MODELS__OPENVINO_DIR`, `QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT`, `QP_LOCAL_MODELS__WORKERS`, `QP_LOCAL_MODELS__CPUS` |
| `RAG_CALIBRATION_BUDGET_S`, `RAG_CALIBRATION_ROUNDS`, `RAG_CALIBRATION_USERS` | the `calibration` block: `QP_CALIBRATION__BUDGET_S`, `QP_CALIBRATION__ROUNDS`, `QP_CALIBRATION__USERS` |
| `RAG_OVMS_NUM_STREAMS`, `RAG_OVMS_THREADS`, `RAG_OVMS_START_TIMEOUT_S` | `QP_OVMS_NUM_STREAMS`, `QP_OVMS_THREADS`, `QP_OVMS_START_TIMEOUT_S`: an OVMS model server App's own variables, no value of the workflow file, set in that App's block of `deploy.yaml` (`apps.<component>.settings`) |
| the calibration request OVMS was sent, sized by `RAG_TOP_K_RETRIEVE` and kept to `RAG_TOP_K_RERANK` | the reranker's `local.calibration_passages` passages, all kept, as for its in-process calibration |
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
start-up. On Chroma, `ne` and `nin` keep a passage that lacks the field, where
the other stores drop it
([The metadata filter](#the-metadata-filter-and-the-prefilter)).

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
`src/query_processing/domain/endpoints/standard_formats.py`. A new API format is one new
module in the component's `api/` directory, its class decorated with the
registry beside the base class, such as `@RERANK_API_FORMATS.register("cohere")`;
no table or list is edited.

A component type registered on its class line, such as the router, has no
outside API format: on `served` it is called on its model server's call route,
`POST /call/<component type>`, in a format of our own, the one registered for
its type in `COMPONENT_API_FORMATS`, else one the toolkit builds that mirrors
its request and its answer. The router's is written by hand, `{"embedding"}` →
`{"route","confidence"}` (`adapters/router/router_api_format.py`).

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
split topologies on a laptop ([The real run](#the-real-run)). The LLM's
`served` backend takes a URL and an alias (`QP_COMPONENTS__LLM__ALIAS`, else
the last part of its `source_id`) and is called in OpenAI's chat format; OVMS
is the one model server of ours that serves an LLM, in an App of its own on
Domino ("A served LLM" in [Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands)).

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
(`embedder=local+router_embedder=local+router=local+reranker=api+llm=api+vector_store=elasticsearch` for the
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

## The router, the metadata filter and structured output

### The router

A router chooses, for a query's vector, one route out of its own list, with a
confidence. It is the toolkit's fifth component type: the toolkit gives its
port (`BaseRouter`), its `fake` backend, which always chooses the first of its
routes, and its `served` backend, built with no code; how it chooses is a use
case's. Its settings name the embedder whose vectors it reads (`embedder`) and
the routes it chooses between (`routes`), two at least, out of the catalogue of
`routes.yaml`, found as `prompts.yaml` is: the file `QP_ROUTES_FILE` names,
else `routes.yaml` in the working directory. A route is a `name`, a
`description` and its `utterances`; a route a router lists and the file lacks
is refused at start-up. Its stage is a plain call, and a conditional edge on
the field it writes asks the edge condition `label`, whose `path_map` has one
row for each of the router's routes (ADR 0007).

The RAG's router, on `local`, is its own kNN router: `make train-router`
(`qp-rag-train-router`) embeds the utterances of its routes through the
router's embedder, with its query prefix, fits a normalizer and a
k-nearest-neighbours classifier with the values of its `local` block, scores
them by leave-one-out, and writes `router_pipeline.joblib` with a manifest of
what it was trained on. Both files are committed, trained on Solon's vectors
(see [The real run](#the-real-run)). The router refuses a pipeline file whose
manifest records other values, routes, utterances, embedder or scikit-learn
version than its own: train it again after any of them changes, on a machine
that has the embedder's model, and commit both files together.

### The metadata filter and the prefilter

A request can carry a metadata filter, `metadata_filter` in the body of
`/query` and `/retrieve`: a flat list of conditions that must all hold, each an
`attribute`, a `comparator` (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`,
`nin`) and a `value`, a list for `in` and `nin`. A condition is checked against
the fields its vector store declares under `metadata_field_info`: an unknown
field, a value outside the allowed ones, or a comparator the field's type does
not have (an order only for a number) is answered with a 422 that names it. A
stage can add conditions, as the RAG's `rewrite` does, and never removes the
client's; the response's `metadata_filter` is the conditions applied.

A vector store can also state a `prefilter`, conditions in the same shape that
every search applies before the request's own: which passages a workflow can
read at all. Its fields are not declared, so neither a client nor the LLM can
name them, and the response does not report them. `qp-ingest` writes none of
them, so a store filled by it returns nothing under a prefilter; the RAG's
workflow file states none, and its `deploy.example.yaml` shows one under
`overrides`.

Each store turns the conditions into its own filter:

| Store | The filter | A passage that lacks the field |
| --- | --- | --- |
| `elasticsearch` | filter clauses on `metadata.<attribute>`, a string on its `.keyword` sub-field; inside the `knn` clause of a vector search, which the server applies during the search, and in the filter of a text search | dropped, whatever the comparator |
| `chroma` | a `where` with Chroma's operators, applied before its vector search | dropped, except that `ne` and `nin` keep it, since Chroma has no operator that asks for a field |
| `fake` | a predicate over a passage's metadata | dropped, whatever the comparator |

### Structured output

A `generate` stage can name `output_schema`, a registered pydantic model: the
LLM then answers one instance of it, and the stage writes each of its fields
to the field of `writes` of the same name. The toolkit registers one,
`standalone_query_with_filter`, a `standalone_query` and a `metadata_filter`,
which the RAG's `rewrite` stage names; `metadata_fields_of: vector_store`
narrows its conditions to the fields and the allowed values that store
declares. A use case registers its own with `@OUTPUT_SCHEMAS.register(...)` in
its package. A use case's output schema must forbid extra keys and have no
default to be accepted by an endpoint's strict mode.

How the schema reaches the endpoint is a setting of the LLM, `structured_output`
in its `api` or `served` block: `json_schema`, the default, sends it as
OpenAI's `response_format`, strict; `format_instructions` writes it in the
messages and parses the answer with LangChain core's parser, for an endpoint
that imposes no schema. Under both, an answer that is not an instance of the
schema is asked again within the LLM's `http.retries`, then the request fails,
showing the text. The fake LLM answers a valid instance.

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
  topologies embed with `snowflake-arctic-embed-l-v2.0` (in-process or its OVMS
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

### Weight format

The workflow file's `local_models.openvino_weight_format`
(`QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT`) says what an OpenVINO export stores
its weights as: `fp32` (32-bit floats, the default) or `int8` (weights stored as
8-bit integers; weight compression only, not full quantization). It is one
setting read by both exports: `qp-export-openvino` and a first start of the
in-process models, and `ovms/export_models.sh` through `qp-served-models`.
`fp32` is the default because it is what the in-process models have always run,
and ranking quality on `int8` weights has not been checked.

In-process, each weight format has its own folder: `<model>` for `fp32` (every
export made before the setting holds 32-bit weights and keeps its place) and
`<model>-int8` for `int8`. An export of another format is never loaded. A start
that finds no folder for its format exports one, so changing the setting
exports again from the Hub. The `torch` runtime ignores the setting. To export
`int8` ahead of time, from the use case's directory:

```bash
QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT=int8 make openvino-export    # -> models/openvino/<model name>-int8/
QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT=int8 ../query_processing/ovms/export_models.sh ovms/models-int8
```

An OVMS models folder holds one export, of one weight format, recorded in its
`weight-format` file (see [OpenVINO Model Server](#openvino-model-server)).

An OVMS App holds its folder to what the settings ask. When the variable is set,
as `qp-deploy` sets it from the overrides of `deploy.yaml`
(`overrides: {local_models: {openvino_weight_format: int8}}`), its launch script
checks at once. Unset, the workflow file or `.env` may be what states the
format, which only Python reads: the check is then `qp-ovms-config`'s, with the
settings. Only an OVMS App with neither that command nor `uv` to install it (the
prebuilt `ovms` image) sees the variable alone and asks for `fp32` when it is
unset: there, set the variable.

Verified on an 8-CPU machine on 15 November 2026, before the setting and the
commands took the names of today. The in-process `int8` export
of the reranker took 13 seconds and holds 110.6 MB of 8-bit weights, against
442.5 MB of 32-bit floats at `fp32`. `ovms/export_models.sh` exported `fp32` into
a new folder and recorded it, and an OVMS export made before the setting was
served as `int8`. On `int8` both servers served the same again, 1.56 and 1.57
requests a second on one reranker alone, against 1.0 each at `fp32`. Ranking
quality on `int8` weights is still unchecked.

## OpenVINO Model Server

`ovms/versions.env` is the single pin of the OpenVINO Model Server version,
sourced by `export_models.sh`, read by `domino/Dockerfile` and passed to docker
compose with `--env-file`:

| Key | Value |
| --- | --- |
| `OVMS_VERSION` / `OVMS_IMAGE` | `2026.4.1` / `openvino/model_server:2026.4.1` |
| `OVMS_BINARY_URL` | `ovms_ubuntu22_2026.4.1_python_off.tar.gz` from the GitHub release `v2026.4.1` (Domino `ovms` image) |

The pin moved from 2026.3.1 to 2026.4.1, for every OVMS App. That version was
run on a machine once, by hand, serving a language model (the sitting of
16 November 2026 on text generation, in `experiments/`). The reranker's and
the embedder's OVMS at 2026.4.1 have not been run, nor the exporter fetched at
that tag (`export_models.sh` fetches `export_model.py` from `v${OVMS_VERSION}`),
so reranker and embedder figures taken after the move do not compare with those
of 15 November 2026 without a re-run.

The served models (source id and alias) are pinned once, by the embedders and
rerankers of the workflow file, whatever their names (here `reranker`,
`antoinelouis/crossencoder-camembert-base-mmarcoFR`, served as
`crossencoder-camembert-base-mmarcoFR`, and `embedder`,
`Snowflake/snowflake-arctic-embed-l-v2.0`, served as
`snowflake-arctic-embed-l-v2.0`: each alias is the source id's last part).
`qp-served-models` prints them, `QP_*` overrides included, one line of
shell-quoted words each (name, type, source id, alias, then the weight format,
the same on every line) for `export_models.sh`, with a line for each LLM
too, which the exporter skips (see "An LLM on OVMS" below);
the `rag` Domino image reads the source ids from the use case's workflow file,
which it copies beside the use case it installs.

**Known defect: OVMS does not start on the RAG's embedder.** On a real machine
OVMS 2026.3.1, the pin at the time, crashed with a segmentation fault while it
loaded `snowflake-arctic-embed-l-v2.0`, on the export exactly as written, so an
OVMS App of the embedder failed at start (the check of 16 November 2026 in
`experiments/`; the cause is not established). It has not been tried again at
2026.4.1, the pin now. OVMS serves the reranker without trouble. Until this is
fixed, serve the embedder with the FastAPI model server or in the workflow's
process.

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
each embedder, the router's embedder included, with `--weight-format` set to
the weight format `qp-served-models` prints
(`QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT`: `fp32` by default, or `int8`). The
result is one `config_<component>.json` per component (`config_reranker.json`,
`config_embedder.json`, `config_router_embedder.json`), one directory per model
and a one-line `weight-format` file, written last; the configs reference the
model directories relatively, so the tree can be mounted anywhere.

A models folder holds one export, of one weight format. A folder holding a
config but no `weight-format` file was exported before the setting existed and
holds `int8`. The script refuses to export into a folder of another weight
format, since OVMS's exporter skips a model folder that exists and would leave
the old weights under the new label: export into another folder instead. An
OVMS App whose folder holds another format than the settings ask
(`local_models.openvino_weight_format`) stops before OVMS starts and
names both ways out: set `QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT` to the format
found, or point `QP_OVMS_MODELS_DIR` at another folder, where the asked format
is exported on first start. See [Weight format](#weight-format) for who checks.

**On Domino the export is optional.** A model server App of kind `ovms`
(`domino/model_servers/ovms.sh`) that finds its component's
`config_<component>.json` missing under
`QP_OVMS_MODELS_DIR` runs `export_models.sh` itself, once, before serving: it
needs `uv`, installs the toolkit and the use case into `$HOME/qp-venv` when
`qp-served-models` is not on `PATH`, runs in the use case's directory, and gives the exporter its own `$HOME/ovms-exporter-venv`. The
prebuilt `ovms` image has no `uv`, so there the models still have to be mounted.
This path has not been run for real.

**How an OVMS App starts.** Once the export is there,
`domino/model_servers/ovms.sh` runs
`qp-ovms-config --serve <component> --models-dir "$QP_OVMS_MODELS_DIR" --serving-dir <a fresh temporary folder>`
and starts `ovms` on the config it prints. That command first refuses an export
of another weight format than the settings ask, then calibrates each of
`QP_OVMS_NUM_STREAMS` and `QP_OVMS_THREADS` left at `0` (see "Calibrating
OVMS" above), and writes the serving folder for the choice: a copy of the
export's `config_<component>.json`, and a `<alias>/` folder that links every
file of the export's but `graph.pbtxt`. The graph is written there with its one
`plugin_config` line set to `NUM_STREAMS`, `INFERENCE_NUM_THREADS` and
`ENABLE_HYPER_THREADING`, and no `PERFORMANCE_HINT` (OVMS refuses it with
`NUM_STREAMS`). The export is never written, so it can stay on a read-only
Dataset shared by several Apps. With both variables pinned nothing is
calibrated and the App starts at once. When `qp-ovms-config` is not on `PATH`
(the `ovms` kind's bootstrap installs no project), the script installs the
toolkit and the use case with `uv` into `$HOME/qp-venv` first. Without `uv` it
serves the export's own config, one stream and not calibrated, and says so on
standard error; a pinned `QP_OVMS_NUM_STREAMS` or `QP_OVMS_THREADS` then stops
the App, since it cannot be applied. If `qp-ovms-config` fails, the App stops:
the command's own line says why, and the script's adds that pinning both
variables skips a calibration that failed. The prefix
proxy (`QP_OVMS_PREFIX_PROXY=1`) serves the same config. Verified on a real
OVMS on 15 November 2026, with the command under its name of that day: OVMS
follows the linked model files of the serving folder, and it takes
`NUM_STREAMS`, `INFERENCE_NUM_THREADS` and `ENABLE_HYPER_THREADING` together.
The prefix proxy and the start without `uv` have only run against stand-ins,
and so has all of it since the command took the toolkit's names and settings.

**An LLM on OVMS.** An `ovms` App of an LLM has no config to find: the script
asks `qp-served-models` the component's type, then has OVMS pull its
`source_id` (an OpenVINO conversion on the Hub, `OpenVINO/<model>-int4-ov`)
into `QP_OVMS_MODELS_DIR`, which it creates
first, and serve it under its `alias`: `ovms --source_model <source id>
--model_name <alias> --model_repository_path <folder> --task text_generation
--target_device CPU`. An empty `source_id` stops the App, naming the component
and the field. Nothing is exported or calibrated, and
a pinned `QP_OVMS_NUM_STREAMS` or `QP_OVMS_THREADS` on an LLM's App is
refused. Two options of the App's block of `deploy.yaml`, checked by
`qp-ovms-config --serve-llm` before OVMS starts, set OVMS's batching:
`QP_OVMS_DYNAMIC_SPLIT_FUSE` (default `true`; `false` reads a prompt in one
pass) and `QP_OVMS_MAX_BATCHED_TOKENS` (default `0`, OVMS's own choice; at
least `0`). Each flag is passed only away from OVMS's default; on an
embedder's or a reranker's App either option is refused. It needs the
generic environment with `QP_BOOTSTRAP=1` and `QP_OVMS_BUNDLE`, not the
prebuilt `ovms` image. What a served LLM is, how it is deployed and what is not
verified of it: "A served LLM" in [Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands).

**Compose** (`make infra-up`) mounts `ovms/models` read-only at `/workspace` and
starts the reranker with `--rest_port 8001 --config_path /workspace/config_reranker.json`,
the embedder on `8002` with `config_embedder.json`. It serves the export as
exported: one stream, not calibrated. Verify:

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
of `adapters/<component>/api/`, with the full routes as URLs. The routes are wire-contract constants in `domain/endpoints/api_endpoint.py`; the
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
component it serves (`QP_MODEL_SERVER_COMPONENT`, any component of the
workflow file its kind serves, an LLM on `ovms` alone) and its model-server kind (`QP_MODEL_SERVER_KIND`).
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
at least 4 cores (the OVMS models and the in-process models are CPU-bound;
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
| `... holds an OVMS export at weight format ...`, from `app.sh[model-server ovms ...]` or from `qp-ovms-config` | the folder's export is of another weight format than the settings ask; set `QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT` to the format found (`overrides.local_models.openvino_weight_format` in `deploy.yaml`), or point `QP_OVMS_MODELS_DIR` at another folder |
| `app.sh[model-server ovms ...]: qp-ovms-config could not write the OVMS config` | the reason is `qp-ovms-config`'s own line, just above; when it is OVMS's calibration that failed, pin `QP_OVMS_NUM_STREAMS` and `QP_OVMS_THREADS` in that App's `settings` to start without it |
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
  environment_revision: 7          # optional: build v7 of env-123, its id asked of Domino; left out, Domino chooses
  hardware_tier_id: small-k8s      # a tier by its id; left out, the project's default tier
  server: fastapi                  # every model server App's kind, unless its own block says
apps:
  workflow:
    kind: agent                    # the default for the workflow; model servers are apps
    autoscaling: {min_replicas: 1, max_replicas: 4, target_cpu_pct: 70}
  reranker:
    server: ovms                   # this model server App alone is OVMS (needs ovms_bundle)
    environment_id: env-456        # its own environment, so not revision 7: Domino chooses
    hardware_tier: cpu-16          # a tier by its name in Domino, its id asked of Domino
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
`QP_BOOTSTRAP_EXTRAS` or `QP_WORKFLOW_FILE`, and an OVMS App's own stream and
thread counts once calibrated (`QP_OVMS_NUM_STREAMS`, `QP_OVMS_THREADS`, in that
App's block: `qp-calibrate --ovms-models` prints them). A value of the workflow file
written there, such as `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, is
refused before anything is published, naming the place under `overrides` where
it belongs. `qp-deploy` sets `QP_ROLE`,
a model server's `QP_MODEL_SERVER_COMPONENT` and `QP_MODEL_SERVER_KIND`, what
its kind's script reads (`QP_OVMS_BUNDLE` for `ovms`), the workflow App's
`QP_TOPOLOGY_ALIAS`, `QP_COMPONENTS__<COMPONENT>__BACKEND` of each component the
topology alias names and `QP_COMPONENTS__<COMPONENT>__SERVED__URL` of each served
one (the model server App's URL plus the route its kind declares for its type,
`/v3/rerank`, `/v3/embeddings` or, for an LLM, `/v3/chat/completions`), and
`QP_BOOTSTRAP` itself (they win over the
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
the kinds that can, such as `ovms` for an LLM on `fastapi`, or, when none can,
each kind and what it serves), and an alias
whose run label the workflow app would refuse (letters, digits and `._+=:-`, up
to 128 characters). Each message names the value and what was expected.

**A served LLM.** The LLM can run in a model server App of its own, served by
OpenVINO Model Server: a row that says `llm: served`, such as the example's
commented `split-all-llm`, and `server: ovms` in `apps.llm` (or
`defaults.server: ovms`), with `ovms_bundle` set. `ovms` is the one kind that
serves an LLM, on its chat completions route; `fastapi` serves none. `qp-deploy`
publishes `<app_name_prefix>-llm` (`qp-rag-llm`) after the other model servers
and before the workflow App, which gets `QP_COMPONENTS__LLM__BACKEND=served` and
the App's URL plus `/v3/chat/completions`, and calls it with the run token as it
calls any served component; the run label gains `+llm=ovms`, and `make
deploy-llm` publishes that App alone. The model OVMS serves is the LLM's
`source_id` and the name it is served and asked for under its `alias`, both under
`overrides.components.llm`, in one place. How the App starts and its two
batching options: "An LLM on OVMS" in [OpenVINO Model Server](#openvino-model-server).
The workflow App's `/readyz` does not check the LLM, so it can report ready
while OVMS still pulls the model on the LLM App's first start; a run started
then records failed requests.

**Not verified.** An LLM App on OVMS has never run on a machine: not the App,
not its launch script (its tests run it against a stand-in `ovms`), not the
deploy. What has run is the bare `ovms` command, by hand, on 16 November 2026
(`experiments/`): OVMS 2026.4.1 pulled the model `deploy.example.yaml` names (2.0 GB) and
served it on `/v3/chat/completions` 60 s after its first start and 5 s
after a later one; an App's first start adds its bootstrap and the install of
the project. OVMS writes its `graph.pbtxt` again at every start with
`--source_model`, so a changed option takes effect on a model already pulled.
Whether OVMS honours `response_format` is unknown: the RAG's rewrite sends its
schema that way by default. For an endpoint that does not, the setting is the
LLM's `structured_output: format_instructions`
(`overrides.components.llm.served.structured_output` for a served LLM;
[Structured output](#structured-output)).

**The waits on the path of an answer.** A `/query` that generates passes these
waits, in order:

| Wait | Default | Set by |
| --- | --- | --- |
| The load-test user | 60 s | `QP_LOADTEST_REQUEST_TIMEOUT_S` |
| Domino's proxy, to the workflow App, then to a served LLM's App | Domino's own; the emulated Domino's 30 s | not a setting of this project; on the emulation, `forward_timeout_seconds` ([emulation/README.md](emulation/README.md)) |
| One call to the LLM | 120 s in the RAG's workflow file | the LLM's `http.timeout_s` |
| Attempts of that call | 3 (2 retries), 0.2 s then 0.4 s between | the LLM's `http.retries` and `http.backoff_s`, else the top-level `http` block's |
| nginx in front of OVMS, with `QP_OVMS_PREFIX_PROXY=1` | 300 s | `proxy_read_timeout` in `domino/ovms-proxy.conf`, not a setting |

A call that outlasts its timeout or a proxy's wait is sent again: a timeout is
a transport error and a proxy that gave up answers a 5xx, and both are
retried. No log line records a retry that succeeded: the stage's time is the sum
of every attempt, and so is the time that tokens a second as a caller sees them
are computed from. The user, who is not retried, has left
at 60 s, while the workflow App may keep the call going for three attempts of
120 s. A run that measures the LLM therefore sets the LLM's retries to `0`,
so each answer is asked once, one that misses its schema too ([Structured
output](#structured-output)); the setting is the LLM's alone, and the
embedder's and the reranker's calls keep theirs. It also raises the user's patience together
with the LLM's timeout, so the user does not leave an answer the workflow App
still waits for; on the emulation, raise `forward_timeout_seconds` too.
`deploy.example.yaml` shows these settings, commented, under
`overrides.components.llm`.

**App or Agent.** `kind: agent` publishes the App with `configurationType:
AISYSTEM` (Domino: "Agents use the same hosting infrastructure as Domino Apps";
that field is the only API-level difference), `kind: app` with `STANDARD`. The
workflow is an Agent unless told otherwise: its launch script also exports
`QP_DOMINO__TRACING=1` (unless the overrides set `domino.tracing`) and
`QP_BOOTSTRAP_EXTRAS=domino` (unless its settings set it), and `app.sh` then
installs `.[domino]`, the extra that brings the tracing
library. An existing App of the other kind is not converted: `qp-deploy` stops
and says to delete it in Domino or change `kind`.

**Environment revision.** `environment_revision` holds an App to one build of
its environment, by the revision's number as Domino shows it (v7 is `7`, written
bare: `"7"`, `v7` or `0` is refused when the file is read).
`environment_revision_id` takes the revision's id itself instead; a block that
sets both, each not empty, is refused, and `""` counts as not set. `qp-deploy`
sends the id in the App's version as `environmentRevisionId`, the name Domino's
Apps API gives it, and sends nothing when neither is set, so that Domino chooses
the revision. A number is found in two places, in this order, the first that
holds it giving the id: the documented call for the environment
(`GET /api/environments/v1/environments/{environmentId}`, `ViewEnvironment`),
which gives only three revisions, the latest, the active and the restricted;
then, only when none of those has the number, the internal route Domino's own
pages use, which lists every revision (not in Domino's reference: see the list
below). Each is asked once an environment, the internal route in two calls
when its first answer is not its whole list. An App's own revision
applies to its environment, its own or the default one, and wins over the
default's, written either way. The default revision goes only with the default
environment, so an App that sets its own `environment_id` does not take it. A
default revision without a default `environment_id` is refused before anything
is published, and so is an App with a revision but no environment. A number no
source has is refused before anything is published:

```
qp-deploy: error: environment_revision 9 of App qp-rag-reranker (apps.reranker or defaults in deploy.yaml) is no revision Domino gives for environment '623132867a0af0281c01a69c': the documented API gives its latest (7), active (7) and restricted (none) revisions; the internal route GET /v4/environments/623132867a0af0281c01a69c/paginatedRevisionData lists 3 revisions (7 to 5). Write the revision's id as environment_revision_id
```

When the internal route fails, the message says what it answered in place of
its revisions. An environment Domino does not have (404) is refused naming its
id, and a 401 or 403 with Domino's answer and the permission `ViewEnvironment`;
the internal route is not asked in place of a documented call that failed.

**Hardware tier.** `hardware_tier` names an App's hardware tier by its name in
Domino, exactly, capitals included; `hardware_tier_id` takes the tier's id
instead, under the same rules as the revision's two keys (one of the two a
block, `""` not set, the App's own winning over the default's, a value still
holding an `<angle-bracket>` placeholder refused). Left out, the App runs on the
project's default tier. When an App being deployed names a tier, `qp-deploy`
lists Domino's tiers once (`GET /api/hardwaretiers/v1/hardwaretiers`, page
after page, `ViewHardwareTiers`) and finds each name there. Before it publishes
the first App, it prints one line for each tier and revision it found:

```
qp-rag-reranker: hardware tier cpu-16 (id large-k8s): 16 cores, 64 GiB
qp-rag-reranker: environment revision 7 (id 62313cfd7a0af0281c01a6a6)
```

A tier with GPUs adds `, 1 GPU`, and the memory keeps the unit Domino gives. A
wrong name stops the deploy with nothing published, listing every tier Domino
has:

```
qp-deploy: error: hardware_tier 'cpu-32' of App qp-rag-reranker (apps.reranker or defaults in deploy.yaml) names no hardware tier of Domino, which has: cpu-8 (id small-k8s, 8 cores, 32 GiB); cpu-16 (id large-k8s, 16 cores, 64 GiB); gpu-1 (id gpu-k8s, 8 cores, 32 GiB, 1 GPU)
```

A name two tiers share is refused with their ids, saying to write
`hardware_tier_id`; a 401 or 403 on the tiers' list names `ViewHardwareTiers`
and says that `hardware_tier_id` needs no such call. Every name and number of
every App being deployed is found before the first App is published. An App
whose block gives ids is published as before, with no call to find them, so its
tier's size is not printed; `qp-deploy status` and `qp-deploy url` look nothing
up. The calls take `http_timeout_s` and `http_retries`, as every call of
`qp-deploy` does.

**What Domino has.** Two commands only read, and show what to write in
`deploy.yaml`: `make tiers` (`qp-deploy tiers`) Domino's hardware tiers with
their size, and `make environments` (`qp-deploy environments`) the environments
the person can see, with their revisions. Each takes names to narrow what it
shows (an unknown name is refused with the names Domino has), and reads
`deploy.yaml` only for how to reach Domino (`project_id`, `http_timeout_s`,
`http_retries`), so no App block needs to be valid. Both are refused with
`--dry-run`, since they ask Domino. Each has three looks:

| Look | When | What it is |
|---|---|---|
| The list | input and output both a terminal | the arrows move, the space bar opens or closes an item, `/` narrows the items to the names holding the text typed, `enter` then gives the keys back to the narrowed list (the cursor on its first item), `escape` brings every item back, `q` quits |
| The table | anywhere else (a pipe, a file, a script), or with `--table` | plain text in aligned columns, no colour |
| The details | with `--details` | every field Domino gave, as YAML, the name first |

```bash
make tiers                                # qp-deploy tiers
make tiers ARGS="cpu-16 --details"        # every field Domino gave for cpu-16
make environments ARGS="qp-rag --table"   # every revision of qp-rag, with its build date
```

The tiers' table, in Domino's order, the default tier marked:

```
NAME    CORES  MEMORY  GPUS  DEFAULT  ID
cpu-8       8  32 GiB     -  yes      small-k8s
cpu-16     16  64 GiB     -           large-k8s
gpu-1       8  32 GiB     1           gpu-k8s
```

The environments' table gives each environment's active and latest revision
numbers. An environment named on the command line is shown with every
revision, newest first, through the internal route (asked for that environment
alone), its build date in UTC and its marks; the status is known
for the three revisions the documented API gives, and left empty for the
others:

```
qp-rag   id 623132867a0af0281c01a69c   private, owner fake-owner

REVISION         BUILT (UTC)       STATUS     ID
7 active latest  2026-11-12 09:41  succeeded  62313cfd7a0af0281c01a6a6
6                2026-11-03 16:02             6229a1b07a0af0281c01a511
5                2026-10-21 11:15             62219e447a0af0281c01a402
```

With `--details` the same environment is its documented fields as YAML, then
`revisions`, each with its `number`, `id`, `built`, `marks` and `status`. When
the internal route fails, the documented fields and the three documented
revisions are still shown, followed by a line saying what the route answered
(`revisions_note` with `--details`). In a terminal, the list, here with
`qp-rag` opened (an environment's revisions are asked of Domino when it is
first opened, and items named on the command line start open; `>` and `v` stand
for the closed and open marks Textual draws):

```
up/down move   space open/close   / filter   enter to the list   esc all items   q quit

  > Domino Standard Environment Py3.10 R4.5   id 6231328a7a0af0281c01a69a
  v qp-rag                                    id 623132867a0af0281c01a69c
      id 623132867a0af0281c01a69c
      v7  active latest  built 2026-11-12 09:41  id 62313cfd7a0af0281c01a6a6
      v6                 built 2026-11-03 16:02  id 6229a1b07a0af0281c01a511
      v5                 built 2026-10-21 11:15  id 62219e447a0af0281c01a402
      - name: qp-rag
        ...
```

An open tier shows every field Domino gave for it. The list is drawn by Textual,
in the toolkit's optional `terminal` extra (also in `dev`), so an App never
installs it. In a terminal without it, the command prints the table and, on the
error output:

```
qp-deploy: the interactive list needs the toolkit's "terminal" extra (pip install -e "query_processing[terminal]" from the repository's root); here is the table
```

Neither command writes anything, to Domino or to `deploy.yaml`.

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
changes at every redeploy, so after `make deploy-embedder`, `make
deploy-reranker` or `make deploy-llm` alone, `qp-deploy` reminds you that the workflow App
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
`qp-deploy` sets itself), with every check of a real deploy but the two that ask
Domino: a dry run asks Domino nothing, so it does not look up a hardware tier's
name or a revision's number, and shows them as written in place of their ids:

```
  environmentRevisionId: (revision 7: its id is asked of Domino at deploy)
  hardwareTierId: (tier cpu-16: its id is asked of Domino at deploy)
```

The rules of the two keys (one way a block, a whole number from 1 up) are
checked as in a deploy. `QP_DEPLOY_TOPOLOGY=<alias> make deploy-dry-run` shows
another alias. One of the checks: no value may still hold an `<angle-bracket>`
placeholder from the example (in the `settings` of the Apps being deployed,
their `environment_id`, `environment_revision_id`, `hardware_tier` and
`hardware_tier_id`, `ovms_bundle`, and the `overrides`, which every App gets).
The example copied unchanged stops on the first one:

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
| `forbidden` | every Apps API call, the list of hardware tiers and the environments answer 403 with a `FailureEnvelopeV1` |

The fake is transcribed from the Domino 6.2 API reference, so it shows what
`qp-deploy` expects of Domino, not what a live Domino does. Besides the Apps
calls it lists a version's instances, answers `/readyz` at each instance's
`/fake-owner/fake-project/r/notebookSession/<instance id>` URL, and describes any
hardware tier (`GET /api/hardwaretiers/v1/hardwaretiers/<id>`) with a fixed
number of cores. For the hardware tiers and the environments it lists the tiers
of its state by pages (`GET /api/hardwaretiers/v1/hardwaretiers`), lists the
environments of its state (`GET /api/environments/beta/environments`), and
gives one environment with its three revisions
(`GET /api/environments/v1/environments/<id>`, 404 for an id it does not have),
each in the reference's shape. Among its stand-ins for what is not Domino's
documented API (the run token, `/readyz`), it serves the internal route that
lists every revision of an environment with its build date
(`GET /v4/environments/<id>/paginatedRevisionData`), in the shape the 6.3.1
spec publishes. The tests give its state tiers and environments; served by
`make fake-domino` it has none, so `qp-deploy tiers` and `qp-deploy
environments` list nothing there.

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
9. The tiers' list (`GET /api/hardwaretiers/v1/hardwaretiers`) answers the
   person who deploys, who needs `ViewHardwareTiers`; if it answers 403,
   `hardware_tier_id` needs no such call.
10. The environment calls (`GET /api/environments/beta/environments` and
    `GET /api/environments/v1/environments/{id}`) answer through
    `DOMINO_API_PROXY`.
11. The internal route that lists every revision of an environment: its path,
    `GET /v4/environments/{id}/paginatedRevisionData`, is remembered, not
    published (only its shape, `PaginatedRevisionData`, is in the 6.3.1 spec);
    its two parameters, `page` and `pageSize` (`qp-deploy` sends `pageSize`
    alone, and none on its first call); that it answers through
    `DOMINO_API_PROXY`; and that `created` counts milliseconds. From a
    Workspace, these two commands print its status and the names of its fields,
    not their values:

    ```bash
    curl -s -o /tmp/revisions.json -w "HTTP %{http_code}\n" "$DOMINO_API_PROXY/v4/environments/<environment id>/paginatedRevisionData"
    ```

    ```bash
    python3 -c 'import json; shape = lambda v: {k: shape(x) for k, x in v.items()} if isinstance(v, dict) else [shape(v[0])] if isinstance(v, list) and v else type(v).__name__; print(json.dumps(shape(json.load(open("/tmp/revisions.json"))), indent=1))'
    ```

    Expected: `HTTP 200`, then `revisions` with `id`, `number`, `created`, and
    `pageInfo` with `totalPages`, `currentPage`, `pageSize`. Otherwise the
    browser's network panel on the Revisions tab of an environment's page shows
    the real request, and the path is corrected in the one module that holds
    it (`deploy/domino_lookups/internal_revisions.py`). Until then a number the
    three documented revisions lack is refused with what the route answered.
12. The list that opens draws and takes its keys in the terminal of a Domino
    Workspace: its tests drive it with Textual's test pilot, and it was never
    run in a Workspace's terminal. Opening an environment waits on the internal
    route, so the list does not answer keys while the call runs (up to
    `http_timeout_s`).

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
each embedder and reranker of the use case's workflow file (the RAG's three HF
models: its two embedders and its reranker), so it needs Hub access at build
time. It holds the use case's package and the workflow, prompts and routes
files its settings read.

**2. Projects (one per app).** A Domino App inherits its project's environment
variables and each app needs its own `QP_ROLE` (and a model server its
component and kind), so import this git repository into one project per app and
set, in Project → Settings → Environment variables:

| Project | `QP_ROLE` | Environment | Other variables |
| --- | --- | --- | --- |
| `rag-monolith` | `workflow` | `rag` | `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`, `QP_COMPONENTS__LLM__API__API_KEY`, `QP_TOPOLOGY_ALIAS=monolith` to report the topology |
| `rag-ovms-reranker` | `model-server` | `ovms` | `QP_MODEL_SERVER_COMPONENT=reranker`, `QP_MODEL_SERVER_KIND=ovms`, `QP_OVMS_MODELS_DIR` if needed |
| `rag-ovms-embedder` | `model-server` | `ovms` | `QP_MODEL_SERVER_COMPONENT=embedder`, `QP_MODEL_SERVER_KIND=ovms`, `QP_OVMS_MODELS_DIR` if needed |
| `rag-ovms-llm` | `model-server` | the generic environment `qp-deploy` publishes into, with `uv` (not `ovms`: see "An LLM on OVMS" in [OpenVINO Model Server](#openvino-model-server)) | `QP_MODEL_SERVER_COMPONENT=llm`, `QP_MODEL_SERVER_KIND=ovms`, `QP_BOOTSTRAP=1`, `QP_OVMS_BUNDLE`, `QP_COMPONENTS__LLM__SOURCE_ID` and `QP_COMPONENTS__LLM__ALIAS` (the model OVMS pulls and the name it serves it under), `QP_OVMS_MODELS_DIR` (a folder the App can write) if needed, `QP_OVMS_DYNAMIC_SPLIT_FUSE` and `QP_OVMS_MAX_BATCHED_TOKENS` to change its batching |
| `rag-fastapi-reranker` | `model-server` | `rag` | `QP_MODEL_SERVER_COMPONENT=reranker`, `QP_MODEL_SERVER_KIND=fastapi` |
| `rag-fastapi-embedder` | `model-server` | `rag` | `QP_MODEL_SERVER_COMPONENT=embedder`, `QP_MODEL_SERVER_KIND=fastapi` |
| `rag-workflow` | `workflow` | `rag` | `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL`, `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY`, `QP_COMPONENTS__LLM__API__API_KEY`, `QP_COMPONENTS__RERANKER__BACKEND=served`, `QP_COMPONENTS__RERANKER__SERVED__URL` (+ `QP_COMPONENTS__EMBEDDER__BACKEND=served`, `QP_COMPONENTS__EMBEDDER__SERVED__URL` for split-all), `QP_TOPOLOGY_ALIAS=split-reranker` (or `split-all`) to report the topology, `QP_RUN_LABEL` for a variant |

Skip the rows you do not test. Keep `QP_DOMINO__TRACING` identical across projects
(unset, or `true` everywhere). With a `rag-ovms-llm` project made by hand, the
`rag-workflow` project, whose App asks the LLM, must hold what `qp-deploy` would
export for an alias whose row says `llm: served`: `QP_COMPONENTS__LLM__BACKEND=served`,
`QP_COMPONENTS__LLM__SERVED__URL` (the LLM App's URL plus
`/v3/chat/completions`), `QP_COMPONENTS__LLM__ALIAS` set to the same name in
both projects (the name the LLM is served and asked for under), and
`QP_TOPOLOGY_ALIAS` set to that alias (such as `split-all-llm`) to report the
topology. A served LLM is called with the run token, so that project needs no
`QP_COMPONENTS__LLM__API__API_KEY` for it. What a served LLM is and what of it is
not verified: "A served LLM" in [Deploy on Domino in a few commands](#deploy-on-domino-in-a-few-commands).

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
make check         # every gate below, in order, stopping at the first failure, then the use case's tests and the charts package's
make test          # pytest: fakes + httpx.MockTransport only; no weights, no network, no Elasticsearch
make format        # ruff --fix + black
```

These run from the repository's root, whose `Makefile` hands them to
`query_processing/`'s, or from `query_processing/` itself, over the toolkit alone
(`make -C query_processing check`). At the root, `make check` then runs the RAG
use case's own tests, from `query_processing_rag_usecase/`, linted with ruff and
black under the toolkit's configuration: its workflow file answers through the
workflow app's HTTP API, and a dry run of each topology alias of its
`deploy.example.yaml` publishes the expected Apps on the fake Domino. Last, it
runs lint, typecheck, complexity and deps over `experiment_charts/` under the
toolkit's configuration, then that package's tests, which also hold the
committed tables of `experiments-explained/` to the registry: after a change to
`experiments/`, `make experiment-charts` at the root draws the figures and the
tables again.
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
for the fake's own tests. A crowded folder holds one sub-folder per sub-package,
module or command (`tests/entrypoints/qp_serve/`, rule 9 of
[Where a file goes](#where-a-file-goes)). The folders are not packages, so a test file's name is
unique across the tree. What several files share (named fakes, fixtures,
`REPO_ROOT` and `TOOLKIT_ROOT`) is in `tests/helpers`, on pytest's path and
imported by plain module name. The Chroma store's tests run a real embedded Chroma in a temporary folder.
