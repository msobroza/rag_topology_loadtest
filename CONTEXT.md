# RAG topology load test

One text RAG workflow, deployed in several topologies on Domino, and a Locust suite that measures each so they can be compared.

## Language

### Deployment shape

**Topology**:
Where the embedder and the reranker run relative to the workflow app. A closed set: `monolith` (both in its process), `split-reranker` (reranker behind a model server), `split-all` (both behind model servers).
_Avoid_: deployment, layout, and using "topology" for a free-form name (that is a run label)

**Backend**:
The kind of adapter behind one port: `local`, `ovms`, `fake`, `none`, `elasticsearch`, `openai`. The topology decides the embedder and reranker backends; `fake` and `none` are experiment variants allowed in any topology.
_Avoid_: provider, implementation, driver

**Workflow app**:
The process that serves the RAG workflow over HTTP.
_Avoid_: service, API, RAG app, server

**Model server**:
A process that serves one model behind the /v3 contract: OpenVINO Model Server or the FastAPI model server.
_Avoid_: inference service, OVMS app (when the FastAPI one is also meant)

**/v3 contract**:
The HTTP paths and JSON shapes a model server speaks for embeddings, reranking and the model probe, as OpenVINO Model Server defines them.
_Avoid_: OVMS API, model API

**Served model**:
A model as a model server exposes it: a source id paired with an alias.

**Source id**:
The Hugging Face id of a served model's weights.
_Avoid_: model name, weights id

**Alias**:
The name clients send to a model server to select a served model.
_Avoid_: servable, model name

**Role**:
What one Domino app is published as: a workflow app (`monolith` for the monolith topology, `workflow` for a split topology) or a model server (an OpenVINO Model Server or a FastAPI model server, for the reranker or the embedder).
_Avoid_: mode, app type

### Workflow

**Stage**:
One step of the workflow, in order: embed, retrieve, rerank, generate. Timings are reported per stage.
_Avoid_: step, node, phase

**Mode**:
How far a request runs through the stages: `query` runs all four, `retrieve` stops after rerank.

### Load test

**Run**:
One headless Locust execution against one target, producing one set of result files.
_Avoid_: test, session, experiment

**Run name**:
The name the load-test side gives a run; it names the run's result files.
_Avoid_: run label

**Run label**:
The free-form name the workflow app reports for the configuration being measured. Defaults to the topology plus every backend that differs from that topology's default; set explicitly to name a variant such as model-server kind or hardware tier.
_Avoid_: deployment, topology label, run name

**Target record**:
What a run observed its target reporting about itself: each topology and run label seen, and how often.

**Target**:
What a run drives: the workflow app, or a model server directly.

**STAGE row**:
A row in a run's statistics that holds one stage's timings for one endpoint.

**Baseline**:
The first run given to a comparison; every delta is relative to it.
