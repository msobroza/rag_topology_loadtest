# RAG topology load test

One text RAG workflow, deployed in several topologies on Domino, and a Locust suite that measures each so they can be compared.

## Language

### Applications

The code is three applications over one shared domain; the README's "Where a file goes" places each.

**RAG system**:
What runs on Domino and what prepares its data and models: the workflow app, the model server, ingestion, the corpus and the model export.
_Avoid_: the app, the service, the product (the product is all three applications)

**Deploy tool**:
What publishes the Apps of a topology on Domino: `rag-deploy`.
_Avoid_: deployer, the CLI

**Load-test client**:
What drives a target and compares runs: the Locust file and `rag-compare`.
_Avoid_: load tester, the client, Locust (that is the tool it runs on)

### Code structure

**Port**:
The abstract class one part of the workflow is used through: `BaseEmbedder`, `BaseReranker`, `BaseVectorStore` with `BaseLexicalSearch`, `BaseLLM`.
_Avoid_: interface, protocol, and "ports" for what builds them (that is the composition root)

**Adapter**:
An implementation of a port, one per backend. Only ports and their implementations are adapters.
_Avoid_: adapter for code that implements no port (that is infrastructure)

**Infrastructure**:
Technical code that implements no port: the HTTP client pieces, the local model runtime, the corpora download, the log format.
_Avoid_: shared, utils, helpers

**Composition root**:
The one place that builds the adapters from the settings and owns what they share.
_Avoid_: factory (a factory builds one port), container, wiring

### Deployment shape

**Topology**:
Where the embedder and the reranker run relative to the workflow app. A closed set: `monolith` (both in its process), `split-reranker` (reranker behind a model server), `split-all` (both behind model servers).
_Avoid_: deployment, layout, and using "topology" for a free-form name (that is a run label)

**Component**:
One of the three model steps of the workflow: embedding, rerank, generation. Each has its own backend and its own endpoint settings, shaped the same way.
_Avoid_: model, service, and "embedder" or "reranker" for the setting (those are the objects that do the work)

**Backend**:
How a component, or the vector store, runs: `local` (a model loaded in the workflow app's process), `api` (an endpoint called over HTTP), `fake`, `none` (rerank only), and for the store `elasticsearch`, `chroma`, `fake`. The topology decides the embedding and rerank backends; `api`, `fake` and `none` load no model of ours into the process and are variants allowed in any topology. Generation is `api` or `fake`.
_Avoid_: provider, implementation, driver, and a vendor's name for a backend (a vendor is reached through an endpoint)

**Endpoint**:
Where a component on the `api` backend is called: the full route, the API format it speaks, an API key and the model name sent. A model server of ours, a gateway and a vendor's API are all endpoints.
_Avoid_: base URL (an endpoint is the full route), hosted API, external model, remote model

**API format**:
The request and response shapes an endpoint speaks, named after whoever defined them: `openai` for embeddings and generation, `cohere` for rerank. A format is a wire shape, not a vendor: any server that speaks it is called the same way.
_Avoid_: provider, vendor, protocol, contract

**Standard format**:
The one API format each component uses on the wire: OpenAI's for embedding and generation, Cohere's for rerank. OVMS, our model server and any gateway are called in it, and an in-process model follows the same contract. An API that speaks another shape is reached through an adapter that presents it as the standard.
_Avoid_: canonical format, native format, default format

**Runtime**:
What runs an in-process model: `openvino` on files exported beforehand, or `torch`.
_Avoid_: backend (that is the adapter kind), engine, device

**Weight format**:
What an OpenVINO export stores its weights as: `fp32` (32-bit floats) or `int8` (weights stored as 8-bit integers). One setting, `RAG_OPENVINO_WEIGHT_FORMAT`, read by both exports: the in-process one and OVMS's. Servers are compared on the same weight format.
_Avoid_: precision alone, quantization (`int8` here compresses the weights only)

**Workflow app**:
The process that serves the RAG workflow over HTTP.
_Avoid_: service, API, RAG app, server

**Model server**:
A process of ours that serves one model on the /v3 routes: OpenVINO Model Server or the FastAPI model server. To the workflow app it is an endpoint like any other.
_Avoid_: inference service, OVMS app (when the FastAPI one is also meant)

**/v3 routes**:
The HTTP paths a model server answers on, as OpenVINO Model Server defines them: embeddings and rerank, each in an API format, and the model probe.
_Avoid_: /v3 contract, OVMS API, model API

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

**App**:
What Domino publishes and serves behind its proxy, in one role; an Agent is an App with tracing.
_Avoid_: service, deployment

**App part**:
Which App of a topology it is: the workflow app (`workflow`), or the model server of the embedder (`embedder`) or of the reranker (`reranker`). A topology's App parts follow from the backends it implies; the App is named `rag-<part>`.
_Avoid_: role (a role also says what serves the model, and which workflow app), app name

**Instance**:
What runs an App's current version, as one replica or several; an App answers at its instance's address.
_Avoid_: pod, and "instance" for a cloud VM (that is a machine)

**Replica**:
One of the identical processes of an instance, each on the App's hardware tier and all behind the instance's one address; an App's autoscaling says how many run. In the emulation each replica has a machine to itself.
_Avoid_: pod, copy, instance (the instance is all of them)

**Hardware tier**:
A named size of compute that Domino runs an App or a Workspace on. In the emulation each tier is a pool of machines of one size.
_Avoid_: machine type, instance type, flavour

**Dataset**:
A folder of files Domino shares with the Apps and Workspaces of a project; the exported models live in one.
_Avoid_: volume, shared drive, bucket

**Project environment variable**:
A value Domino hands to every App and Workspace of a project; where secrets are kept.
_Avoid_: secret (not every one is), setting (those are per App)

**Workspace**:
Where a person works inside Domino: deploys, ingestion and runs are started from there.
_Avoid_: client, dev box, laptop (when the Domino side is meant)

**Fake Domino**:
A stand-in for Domino's Apps API, for tests and for trying a deploy without Domino.
_Avoid_: mock Domino, Domino simulator

### Emulation

Temporary: these terms leave with the emulation, once the real Domino is in use.

**Emulation**:
A set of cloud machines standing in for a Domino deployment: an emulated Domino with its Workspace, the vector store, and the Apps of one topology.
_Avoid_: staging, test environment, GCP deployment

**Emulated Domino**:
The fake Domino extended to behave like Domino at run time: it sits in front of each App as Domino's proxy does and places Apps on machines.
_Avoid_: fake Domino (that is the Apps API stand-in alone), mini Domino

**Machine**:
One cloud VM of the emulation.
_Avoid_: instance, node, server

**Balancing rule**:
How the emulated Domino's proxy chooses the replica that answers a request: the next one in turn for each request, or one drawn at random for each caller connection. Domino documents that requests spread over replicas, not by which rule.
_Avoid_: load balancer (there is no separate one), routing policy

**Cache bucket**:
Where the emulation keeps what outlives its machines: the exported models, the index snapshots, the result archives.
_Avoid_: cache (alone), state bucket (that is Terraform's)

### Workflow

**Stage**:
One step of the workflow, in order: embed, retrieve, rerank, generate. Timings are reported per stage.
_Avoid_: step, node, phase

**CPU-bound stage**:
A stage whose work runs in the App's own process, on its CPUs: embed and rerank when their model is local. Retrieve and generate wait on another service.
_Avoid_: heavy stage, compute stage

**Calibration**:
Timing a CPU-bound stage under each candidate batch size and thread count, and keeping the fastest. An App does it at start-up, only for a parameter the settings leave at 0; `rag-calibrate` does it on demand for every parameter and prints the settings that pin the choice.
_Avoid_: tuning, auto-tune, finder, benchmark (that is a run)

**Mode**:
How far a request runs through the stages: `query` runs all four, `retrieve` stops after rerank.

**Retrieval mode**:
How the retrieve stage finds candidates: `dense` (by vector), `lexical` (by words, ranked by the store) or `hybrid` (both, fused by reciprocal rank). A lexical workflow has no embed stage.
_Avoid_: search type, and plain "mode" for it (that is query or retrieve)

### Load test

**Run**:
One headless Locust execution against one target, producing one set of result files.
_Avoid_: test, session, experiment

**Run name**:
The name the load-test side gives a run; it names the run's result files.
_Avoid_: run label

**Run label**:
The free-form name the workflow app reports for the configuration being measured. Defaults to the topology plus every choice (embedding, rerank, store, generation, retrieval) that differs from its default; set explicitly to name a variant such as model-server kind or hardware tier.
_Avoid_: deployment, topology label, run name

**Target record**:
What a run observed its target reporting about itself: each topology and run label seen, and how often.

**Target**:
What a run drives: the workflow app, or a model server directly.

**STAGE row**:
A row in a run's statistics that holds one stage's timings for one endpoint.

**Baseline**:
The first run given to a comparison; every delta is relative to it.
