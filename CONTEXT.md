# query_processing

A toolkit and its use cases. A use case declares its workflow and where its components run in two files; the toolkit serves the workflow, deploys it on Domino in the topology chosen, and drives it with a Locust suite so that topologies can be compared. They are two distributions: the toolkit, `query_processing`, holds what any use case can assemble, and each use case holds its settings and the implementations that are its own. The RAG is the first use case.

## Language

### Packages

**Toolkit**:
What every use case is built from: the stage types, the components and their backends, the deploy tool and the load-test client. Its package is `query_processing`.
_Avoid_: framework, library, platform

**Use case**:
One application of the toolkit, in a package of its own that installs it: the settings of its workflow, components and topologies, and the implementations, trained models and data that are its own. RAG is the first, in `query_processing_rag_usecase`.
_Avoid_: project, app (that is what Domino publishes), workflow (a use case has one, and more besides)

### Applications

The toolkit's code is three applications over one shared domain, all in the `query_processing` distribution. A use case's distribution holds only what is its own: each application runs in the use case's directory, reads its files and loads its code. The README's "Where a file goes" places each.

**RAG system**:
What runs on Domino and what prepares a use case's data and models: the workflow app, the model server, ingestion, the corpus and the model export.
_Avoid_: the app, the service, the product (the product is all three applications)

**Deploy tool**:
What publishes the Apps of a use case's topology on Domino, from its workflow file and its deploy settings: `qp-deploy`.
_Avoid_: deployer, the CLI

**Load-test client**:
What drives a target and compares runs: the toolkit's Locust file, run from a use case's directory, and `qp-compare`.
_Avoid_: load tester, the client, Locust (that is the tool it runs on)

### Code structure

**Port**:
The abstract class one part of the workflow is used through: `BaseEmbedder`, `BaseReranker`, `BaseVectorStore` with `BaseLexicalSearch`, `BaseLLM`, `BaseRouter`. A use case can add one for a component type of its own.
_Avoid_: interface, protocol, and "ports" for what builds them (that is the composition root)

**Adapter**:
An implementation of a port, one per backend. Only ports and their implementations are adapters.
_Avoid_: adapter for code that implements no port (that is infrastructure)

**Request and answer**:
What a component takes and what it gives: two domain classes, one pair for each component type. A stage's reads fill the request, and its writes take the answer.
_Avoid_: input and output, payload, and "response" for the answer (that is what comes back on the wire)

**Infrastructure**:
Technical code that implements no port: the HTTP client pieces, the local model runtime, the corpora download, the log format.
_Avoid_: shared, utils, helpers

**Composition root**:
The one place that builds the adapters from the settings and owns what they share.
_Avoid_: factory (a factory builds one port), container, wiring

### Deployment shape

**Topology**:
Where each component runs relative to the workflow app: `local` in its process, or `served` behind a model server. The three compared are `monolith` (nothing served), `split-reranker` (the reranker served) and `split-all` (the embedder and the reranker served, each by its own App). The deploy tool applies it: the workflow app only reads each component's backend.
_Avoid_: deployment, layout, and using "topology" for a free-form name (that is a run label)

**Topology alias**:
The name a topology is chosen by, defined in the deploy settings: `monolith`, `split-reranker`, `split-all`.
_Avoid_: alias alone (that is a served model's name), profile, preset

**Component**:
A named thing a stage calls that has a backend: a model (the embedder, the reranker, the LLM, a router) or the vector store. Its type is the port it is used through, and components of one type have settings shaped the same way. Every component is called the same way: a request in, an answer out.
_Avoid_: model, service, resource, dependency

**Backend**:
How a component runs: `local` (a model loaded in the process that calls it), `served` (a model server of ours serves it), `api` (an endpoint we do not run, called over HTTP), `fake`, `none` (rerank only), and for the vector store `elasticsearch`, `chroma`, `fake`. A topology says `local` or `served` for the components it names. Generation has no `local` backend.
_Avoid_: provider, implementation, driver, external (that is `api`), and a vendor's name for a backend (a vendor is reached through an endpoint)

**Endpoint**:
Where a component on the `served` or `api` backend is called: the full route, the API format it speaks, an API key and the model name sent. A model server of ours, a gateway and a vendor's API are all endpoints.
_Avoid_: base URL (an endpoint is the full route), hosted API, external model, remote model

**API format**:
The request and response shapes an endpoint speaks, named after whoever defined them: `openai` for embeddings and generation, `cohere` for rerank. A format is a wire shape, not a vendor: any server that speaks it is called the same way.
_Avoid_: provider, vendor, protocol, contract

**Standard format**:
The one API format each component uses on the wire: OpenAI's for embedding and generation, Cohere's for rerank, and one of our own where no outside format exists, as for the router. OVMS, our model server and any gateway are called in it, and an in-process model follows the same contract. An API that speaks another shape is reached through an adapter that presents it as the standard.
_Avoid_: canonical format, native format, default format

**Runtime**:
What runs an in-process model: `openvino` on files exported beforehand, or `torch`.
_Avoid_: backend (that is the adapter kind), engine, device

**Workflow app**:
The process that serves a use case's workflow over HTTP: `qp-serve`.
_Avoid_: service, API, RAG app, server

**Model server**:
A process of ours that serves one component: a model on the /v3 routes (OpenVINO Model Server or the FastAPI model server), or a component with no outside API format on its call route (the FastAPI one only). To the workflow app it is an endpoint like any other.
_Avoid_: inference service, OVMS app (when the FastAPI one is also meant)

**/v3 routes**:
The HTTP paths a model server answers on, as OpenVINO Model Server defines them: embeddings and rerank, each in an API format, and the model probe.
_Avoid_: /v3 contract, OVMS API, model API

**Call route**:
The HTTP path a model server answers on for a component that has no outside API format: one request in, one answer out.
_Avoid_: /v3 route (those are OpenVINO Model Server's), endpoint (that is the whole of where a component is called)

**Served model**:
A model as a model server exposes it: a source id paired with an alias.

**Source id**:
The Hugging Face id of a served model's weights.
_Avoid_: model name, weights id

**Alias**:
The name clients send to a model server to select a served model.
_Avoid_: servable, model name, and "alias" for a topology's name (that is a topology alias)

**Role**:
What one Domino App is published as: the workflow app (`workflow`) or a model server (`model-server`), which is told the component it serves and its kind.
_Avoid_: mode, app type

**Model-server kind**:
What serves a component in a model-server App: `fastapi` or `ovms`, set for each App. Each kind is one registered class, which says which component types it can serve, on which route, and how it is probed, and one launch script, which holds its bootstrap and its command.
_Avoid_: server type, runtime (that is what runs an in-process model)

**App**:
What Domino publishes and serves behind its proxy, in one role; an Agent is an App with tracing.
_Avoid_: service, deployment

**App part**:
Which App of a topology it is: the workflow app (`workflow`), or the model server of one served component, named after it (`embedder`, `reranker`). A topology's App parts are the components it serves, plus the workflow app. The App is named by the use case's prefix and the part, such as `qp-rag-reranker`.
_Avoid_: role (a role says only workflow app or model server, not which component), app name

**Instance**:
What runs an App's current version, as one replica or several; an App answers at its instance's address.
_Avoid_: pod, and "instance" for a cloud VM (that is a machine)

**Replica**:
One of the identical processes of an instance, each on the App's hardware tier and all behind the instance's one address; an App's autoscaling says how many run. In the emulation each replica has a machine to itself.
_Avoid_: pod, copy, instance (the instance is all of them)

**Hardware tier**:
A named size of compute that Domino runs an App or a Workspace on. In the emulation each tier is a pool of machines of one size.
_Avoid_: machine type, instance type, flavour

**Environment**:
The Domino compute environment an App or a Workspace starts in. An environment revision is one build of it, and an App can be held to one.
_Avoid_: image, environment version (Domino says revision)

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
One step of the workflow graph: it hands the fields it reads to the component it names, and writes the fields the answer fills. Timings are reported per stage.
_Avoid_: step, node, phase

**Stage type**:
The registered behaviour of a stage that does more than call its component: `generate` renders a prompt, and `retrieve` reads different fields in each retrieval mode. A stage that only calls needs none.
_Avoid_: node type, stage adapter (a stage implements no port)

**CPU-bound stage**:
A stage whose work runs in the App's own process, on its CPUs: embed and rerank when their model is local. Retrieve and generate wait on another service.
_Avoid_: heavy stage, compute stage

**Calibration**:
Timing a CPU-bound stage under each candidate batch size and thread count, and keeping the fastest. An App does it at start-up, only for a parameter the settings leave at 0; `qp-calibrate` does it on demand for every parameter and prints the settings that pin the choice.
_Avoid_: tuning, auto-tune, finder, benchmark (that is a run)

**Edge**:
A link from one stage to the next. A conditional edge asks an edge condition and goes the way its answer names.
_Avoid_: link, transition, arrow

**Edge condition**:
A registered question about the workflow state, asked by a conditional edge. Its answer is yes or no, or one label out of a known set.
_Avoid_: condition (alone), predicate, guard, rule

**Router**:
A component that chooses, for a query, one route out of its own list. A use case can have several, each with its routes.
_Avoid_: classifier, selector, dispatcher

**Route**:
One answer a router can choose: a name, what it means, and the utterances that exemplify it. An HTTP path is not called a route on its own: those are the /v3 routes and the call route.
_Avoid_: class, label, intent, category

**Route choice**:
The route a router chose for one query, with its confidence.
_Avoid_: decision, prediction

**Utterance**:
A message that exemplifies a route. A router learns its routes from their utterances.
_Avoid_: example, sample, training query

**Metadata filter**:
The conditions a passage's metadata must meet for it to be retrieved. Each names one metadata field, and all must hold; the client can send some with a message, and a stage can add more.
_Avoid_: filters (alone), where clause, facet

**Prefilter**:
The conditions a vector store adds to every search, before a request's own: which passages a workflow can read at all. Neither a client nor the LLM can name its fields.
_Avoid_: default filter, base filter, and "prefilter" for a request's own conditions (those are its metadata filter)

**Ingestion pipeline version**:
Which run of ingestion wrote a passage, carried in its metadata. A vector store's prefilter keeps the passages of one.
_Avoid_: corpus version, index version, embedder version

**Mode**:
How far a request runs through the stages: `query` runs every stage, `retrieve` stops after rerank.

**Retrieval mode**:
How the retrieve stage finds candidates: `dense` (by vector), `lexical` (by words, ranked by the store) or `hybrid` (both, fused by reciprocal rank). A lexical workflow has no embed stage.
_Avoid_: search type, and plain "mode" for it (that is query or retrieve)

### Conversation

**Message**:
One thing one party sends: a question or an answer. On another channel it would be an email or an utterance.

**Turn**:
One message from the user and the answer the workflow gave it.
_Avoid_: exchange, round

**Conversation**:
The turns that belong together, in order.
_Avoid_: thread, session, dialogue

**Conversational history**:
The turns of the conversation before the current one. The client sends them with each request: the workflow keeps no conversation.
_Avoid_: history (alone), memory, context (that is the passages given to the LLM)

**Conversational query**:
The user's message of the current turn, as sent. It may lean on the conversational history, and it is never part of it.
_Avoid_: raw query, user query

**Standalone query**:
The conversational query made self-contained, which the stages search with. A message that leans on nothing is its own standalone query.
_Avoid_: rewritten query, condensed question

**Channel**:
How messages reach the workflow and leave it. Chat over HTTP is the only one today; email and calls would be others.
_Avoid_: interface, medium

### Load test

**Run**:
One headless Locust execution against one target, producing one set of result files.
_Avoid_: test, session, experiment

**Run name**:
The name the load-test side gives a run; it names the run's result files.
_Avoid_: run label

**Run label**:
The free-form name the workflow app reports for the configuration being measured. Defaults to each component with its backend; the deploy tool sets it to the topology alias plus each served component's model-server kind. Set explicitly to name a variant such as a hardware tier.
_Avoid_: deployment, topology label, run name

**Target record**:
What a run observed its target reporting about itself: each topology and run label seen, and how often.

**Target**:
What a run drives: the workflow app, or a model server directly.

**STAGE row**:
A row in a run's statistics that holds one stage's timings for one endpoint.

**Baseline**:
The first run given to a comparison; every delta is relative to it.
