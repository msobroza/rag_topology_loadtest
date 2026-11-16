# Components with one common API, a router, prompts and metadata filters

Design agreed with the owner on 15 and 16 November 2026. Nothing of it is built. The decisions and their reasons are in [ADR 0006](../adr/0006-a-use-case-holds-the-implementations-that-are-its-own.md), [ADR 0007](../adr/0007-an-edge-condition-can-answer-with-a-label.md) and [ADR 0008](../adr/0008-every-component-is-called-the-same-way.md), and the vocabulary is in `CONTEXT.md`. This document says what to build. It stacks on pull request 19, the toolkit `query_processing` and its RAG use case.

The code and file examples below come from the design review and from small checks run on stand-ins of the toolkit. Names of the toolkit's inner helpers are indicative. Names a user types (in a file or on a class line) are decided.

## Why

Adding a component type today means editing eleven files in six packages: the list of types, the closed union of component settings, the URL examples, the table of openers, the table of API formats, the composition root's local openers, the model server's routes, the standard formats, each model-server kind's routes, the dependency probes and the closed workflow state. ADR 0005 also kept all code out of a use case.

The owner wants a user of the toolkit to add a component by knowing what it takes and what it gives, and nothing of what is already implemented. Their words: "abstract classes should be easy", and "avoid give complexity to user. Only if he wants to redefine it".

The first use is in the RAG: a router decides whether a message leans on the conversation, a rewrite makes it self-contained and can write a metadata filter, and the prompts move to a file of their own.

## Goals

1. A user adds an implementation of a component type by writing one class on one abstract class, with no registry, opener, mixin or stage type.
2. A user adds a whole component type by writing its request, its answer and one abstract class.
3. Every component is called the same way, so a stage needs no code to call one.
4. A component type added this way can run in its own App with no code from the user.
5. The workflow file shows, for every stage and every conditional edge, the fields it reads and writes.
6. The RAG routes a message, rewrites it when it leans on the conversation, and filters retrieval on passage metadata, on every vector store.
7. A new component type edits no table in the toolkit.

## Non-goals

- The workflow app does not store conversations. The client sends the conversational history.
- A router has no confidence threshold, and answers one route, never a list.
- No router that reads the text. The toolkit's router takes a query's vector.
- No filter trees with and, or and not. A metadata filter is a flat list.
- A request that skips the rewrite gets no generated conditions.
- No model-server kind serves an LLM. This is unchanged.
- The twelve adapters of the embedder, the reranker, the LLM and the vector store are not rewritten onto `invoke`.
- A use case does not add a backend to one of those four component types. They keep their openers and their settings classes.
- No command generates utterances. The owner had them written by hand.
- `qp-ingest` does not change for the moment. It writes neither field a prefilter names, and it keeps asking for `--embedder` when the file declares several embedders.

## What would make this a failure

- A user has to import a registry, write an opener, order mixins or write a stage type to add a component.
- A prompt mentions a deploy target or carries a JSON schema.
- A class of a library sits in the domain.
- The toolkit imports or names a use case, or ships one use case's routes, prompts or models.
- The workflow file hides a field a stage touches, or lists blocks for backends that are not in use.
- A body that `/query` or `/retrieve` accepts today is refused, or a response field disappears.
- Start-up accepts a graph that then fails on a request for a missing field or an unlisted route.

## Constraints

- The owner's design rules hold: no module-level tunables, ports are abstract classes named `Base*`, the layers of the toolkit, errors that carry the offending value and the expected shape, a docstring with one example on every public function, and short functions. The root `CLAUDE.md` and the owner's global one list them.
- The toolkit ships no value of a use case. Its tests run on its fixture workflow file.
- Tests run headless, with no model weights and no network. `make check` is the gate.
- No model is downloaded on the owner's laptop. Model work is verified with fakes, or on a machine that has the weights.
- Jinja2 3.1.6 is installed but not declared: the toolkit declares it. scikit-learn 1.9.1 and joblib become dependencies of the use case, not of the toolkit.
- Every commit is authored by the owner alone, with no trailer.

## Approaches considered

| Approach | What the user writes | Why not, or why |
|---|---|---|
| Mixins the user picks for each serving mode | A class per mode, with the right mixin placed before the port | Rejected: the user must know which mixin and its order. |
| A registry and an opener function per implementation, with the served mode derived | A port, its settings, an opener under a registry's decorator | Rejected by the owner: no abstract class tells the registry's name or the opener's signature. |
| One abstract class per component type, registered on the class line, with one common API | A request, an answer, one abstract class, and one class per implementation | Chosen. The mixins stay inside the toolkit. |

The other alternatives, among them the whole router in the use case and a library's filter types, are in the three records.

## The design

### 1. Code in a use case

- `workflow.yaml` gains a top-level key, `extensions`: a list of package names. The toolkit imports every module of each package before it reads the components, so that their classes register. The use case's `__init__.py` files stay empty.
- The workflow app, the model server, the deploy tool, ingestion and every other command that reads the workflow file load the extensions.
- A use case imports the toolkit only through `query_processing.extension_api`. That module re-exports every name a user needs, and its docstring holds the table of section 2.6.
- A use case's package follows the toolkit's layers: `domain`, `config`, `adapters`, `application`, `infrastructure`, `entrypoints`. It has no `composition` package, since a class builds itself (section 2.2).
- A use case adds registrations and never replaces one of the toolkit's. A name registered twice stops the load, naming both classes.
- The toolkit holds what any use case can assemble. A use case holds its implementations, its trained models and its data.

### 2. Components

#### 2.1 A component type

A component type is one abstract class. It names its request and its answer, two domain classes, and registers under its type name:

```python
class BaseAnonymizer(BaseComponent[AnonymizeRequest, AnonymizedText], component_type="anonymizer"):
    """Masks personal data in a text."""
```

`BaseComponent` gives every component two methods:

- `ainvoke(request)`, the one method the toolkit calls. By default it runs `invoke` off the event loop.
- `invoke(request)`, the blocking method an in-process implementation writes.

The names and that default are LangChain's. Nothing subclasses LangChain's `Runnable`.

#### 2.2 An implementation

An implementation is one subclass of the component type's abstract class. It names its backend and, when it has settings, their class:

```python
class RegexAnonymizer(BaseAnonymizer, backend="local", settings=RegexAnonymizerSettings):
    def __init__(self, patterns: Mapping[str, re.Pattern[str]]) -> None: ...

    @classmethod
    def from_settings(cls, settings: RegexAnonymizerSettings,
                      resources: ComponentResources) -> "RegexAnonymizer": ...

    def invoke(self, request: AnonymizeRequest) -> AnonymizedText: ...
```

- `from_settings` receives the block of the workflow file named after the backend, already checked against the settings class. An implementation that names no settings class needs no `from_settings`.
- `resources` is a small typed object with what an implementation can need and no inner part of the toolkit: the settings of another component by its name, and for a router its routes. The builder fixes its fields.
- An implementation that awaits, such as one that calls a service, writes `ainvoke` in place of `invoke`.
- The class is refused when it is defined if it writes neither method, names no backend, or names a backend its component type already has. The message says what to write, for example `Empty still has to write: invoke(self, request) -> AnonymizedText`.
- A use case therefore has one implementation for each backend of a component type.

The toolkit's composition finds the class, checks its block, gathers the resources and calls `from_settings`. This bends the rule that only the composition reads the settings, as ADR 0008 records.

#### 2.3 Settings

- A component's block in `workflow.yaml` has the keys every component has (`backend`, `type` when its name is not its type, `http`), the keys of its component type, and one block per backend in use, named after the backend.
- A component type that has keys of its own declares a settings class for them. The router does: `embedder` and `routes`.
- A backend's block is checked against the settings class its implementation names. An unknown key is refused, in the workflow file and in the `overrides` of `deploy.yaml` alike.
- The workflow file states only the backend in use. The RAG's file loses its `chroma` block.

#### 2.4 The `served` backend and the model server

- Every component type made this way has a `served` backend with no code from the user. The toolkit builds the class that calls a model server of ours, and reads the URL from the component's `served` block.
- `qp-model-server --serve <component>` serves any such component. It opens the component in process and answers `POST /call/<component type>`.
- A request and an answer never go on the wire as they are. An API format translates in both directions: `request_to_wire`, `request_from_wire`, `answer_to_wire`, `answer_from_wire`. The toolkit builds a format that mirrors the request and the answer unless one is registered for the component type. An API format is an adapter and lives in `adapters/`.
- A use case writes a `served` class only to redefine the built one.
- The `fastapi` model-server kind serves these component types. The `ovms` kind does not, and the deploy tool refuses such a row before anything is published.
- A topology alias can name such a component as `local` or `served`, and an `apps` block can be keyed by its name.

#### 2.5 The four existing component types

Each port's abstract class gains `ainvoke`, written once from its current methods. No adapter changes for it. Ingestion and the model server keep the current methods.

With several embedder components, the fake vector store can no longer be seeded with "the" embedder. It is seeded with the one its searches use: the component called by the stage that writes the embedding the `retrieve` stage reads. In the RAG that stage is `embed`, which calls `embedder`. A graph with no such stage, or with several that call different embedders, keeps today's rule, the only embedder declared.

The four types keep their openers in the composition and their settings classes. Only what names them in a table moves to a registration, and the class line of section 2.2 is not applied to their adapters. The one change to an existing port is the LLM's, for structured output (section 5).

| Component type | Request attributes | Answer attributes | Written from |
|---|---|---|---|
| embedder | `text` | `embedding` | `embed_query`, which applies the query prefix |
| reranker | `query`, `passages`, `top_k` | `passages` | `rerank` |
| llm | `messages`, `output_schema` | `text` | `generate` |
| vector store | `embedding`, `text`, `metadata_filter`, `top_k`, `retrieval_mode` | `passages` | `query` and `lexical_query` |
| router (new) | `embedding` | `name`, `confidence` | section 8 |

The tables that name the four types are replaced by what the classes register, so that a fifth type edits none of them. The builder removes each table listed under "Why".

#### 2.6 Where a user looks

The docstring of `extension_api` and the README carry this table:

| To add | Subclass | Write |
|---|---|---|
| An implementation of a component type | its abstract class, such as `BaseRouter` | `invoke`, and `from_settings` when it has settings |
| A whole component type | `BaseComponent[Request, Answer]` | its request and its answer |
| A stage type | `BaseStage` | what the stage does beyond calling its component |
| An edge condition | a function under the toolkit's decorator | the function |
| An output schema | a pydantic model under the toolkit's decorator | its fields |

Two spellings fill the same registries: a keyword on the class line for a component, a decorator for the other entries.

#### 2.7 Tests the toolkit ships

The toolkit ships, in an importable package and not under `tests/`, two test classes: `ComponentContract`, which fits any component type, and `RouterContract` for the router. A user subclasses one and says how to make one component and one request:

```python
class TestRegexAnonymizer(ComponentContract):
    def make_component(self) -> BaseAnonymizer: ...
    def make_request(self) -> AnonymizeRequest: ...
```

It checks that the answer has the component type's shape, that it survives the API format, and that the served class answers as the in-process one does through an in-process model server. `RouterContract` adds that the route chosen is one of the router's routes and that the confidence is between 0 and 1.

### 3. Stages, fields and edges

#### 3.1 `reads` and `writes`

Every stage of `workflow.yaml` states `reads` and `writes`, both lists of state fields. A stage hands its fields to the component it calls by three rules:

1. A key of the stage that names a request attribute fills it, such as `top_k: 10`.
2. `reads` fill the remaining request attributes, in order.
3. `writes` take the answer's attributes, in order, and can stop early.

```yaml
embed_for_route: {call: router_embedder, reads: [conversational_query], writes: [routing_embedding]}
route:           {call: router, reads: [routing_embedding], writes: [route]}
rerank:          {call: reranker, top_k: 10, reads: [standalone_query, candidates], writes: [reranked]}
```

A stage is of the stage type its `type` names, else of the registered stage type of its own name, else it is a plain call, which does only the above. The `embed` and `rerank` stage types are removed. Two stage types remain:

- **`generate`** renders a prompt and calls the LLM (sections 4 and 5). Its `reads` are any number of fields, each the prompt variable of the same name. Its `writes` are one field, the text, or the fields of its output schema, by name.
- **`retrieve`** reads different fields in each retrieval mode, then the metadata filter: `[query_embedding, metadata_filter]` for `dense`, `[standalone_query, metadata_filter]` for `lexical`, and `[query_embedding, standalone_query, metadata_filter]` for `hybrid`. It keeps `retrieval_mode`, `top_k` and its fusion setting.

#### 3.2 The state

- The workflow state is built from the file. A field exists because the request or a stage writes it, and its type is the type of what writes it. Two writers that disagree on a field's type are refused.
- Before the first stage the workflow app writes five fields: `conversational_query` (the body's `question`), `conversational_history`, `mode`, `standalone_query` (the message itself, until a stage rewrites it) and `metadata_filter` (the client's conditions, possibly none).
- The response reads `answer`, `reranked` and `metadata_filter`. These names are fixed in the toolkit, as the names of the wire are.
- `metadata_filter` accumulates: a stage that writes it adds conditions and never removes the client's. Every other field is replaced by its last writer.
- `timings_ms` works as today.

#### 3.3 Conditional edges

A conditional edge states `reads` as a stage does, and its edge condition answers yes or no, or a label:

```yaml
conditional_edges:
  - source: START
    path: has_history
    reads: [conversational_history]
    path_map: {true: embed_for_route, false: embed}
  - source: route
    path: label
    reads: [route]
    path_map: {self_contained: embed, leans_on_history: rewrite}
  - source: rerank
    path: is_query_mode
    reads: [mode]
    path_map: {true: generate, false: END}
```

The toolkit has three edge conditions: `is_query_mode`, `has_history`, and `label`, which answers with the field it reads. An edge condition is a function of the values it reads, no longer of the whole state.

#### 3.4 Checks at start-up

Start-up refuses, naming the stage or the edge, the value found and the shape expected:

- a stage whose `reads` and keys do not fill its component's request exactly once, or whose types differ. The message shows both orders, for example `stage route: router takes (embedding: list[float]) and gives (name: str, confidence: float); reads lists 2 fields`;
- a field read that some path does not write before, conditional edges and cycles included;
- a field written that no stage, no edge and no response reads, as a likely typo;
- a `path_map` whose labels differ from the labels of the field's writer. For a router these are its routes. On LangGraph 1.2.21 a label missing from `path_map` raises a `KeyError` only when a request takes it;
- a `generate` stage whose prompt uses a variable the stage does not read, or whose output schema's fields differ from its `writes`;
- a stage that calls a component the file does not declare, and a prompt, a route or an output schema that is named and not defined.

### 4. Prompts

- A use case has a `prompts.yaml` beside its workflow file. A command finds it as `QP_PROMPTS_FILE` names it, else as `prompts.yaml` in the working directory. It feeds a top-level setting, `prompts`. A `prompts` key written in the workflow file is refused.
- A prompt is a list of messages with `role` (`system`, `user` or `assistant`) and `content`, as OpenAI writes them. Each `content` is a Jinja2 template, rendered on its own. The structure of the list is never rendered.
- An entry can be `placeholder: <field>` in place of a message. The messages of that field are spliced there, with only their `role` and `content`.
- The variables of a prompt are the fields its stage reads. A `generate` stage with `metadata_fields_of` also gets `metadata_field_info` (section 7).
- Templates render in Jinja2's sandbox, with undefined variables as errors and with `trim_blocks` and `lstrip_blocks`. Jinja2 is used directly, behind one module.
- A stage refers to a prompt by `prompt: <name>`. The `system_prompt` and `user_prompt` settings of `generate` are removed.
- A variable or an override replaces a whole prompt, as JSON.

```yaml
rewrite_as_standalone:
  - role: system
    content: |-
      Rewrite the last user message so that it needs no conversation to be understood.
      Add a condition on a metadata field only when the conversation names one of its values.
      {% for field in metadata_field_info %}
      - {{ field.name }}: {{ field.description }} Values: {{ field.allowed_values | join(", ") }}
      {% endfor %}
      {% for message in conversational_history if message.metadata_filter %}
      An earlier message was restricted to: {{ message.metadata_filter | tojson }}
      {% endfor %}
  - placeholder: conversational_history
  - role: user
    content: "{{ conversational_query }}"
```

The RAG's `answer_from_context` prompt must render, character for character, what today's `system_prompt` and `user_prompt` give. A check during the review rendered it identically from a loop over `reranked`.

### 5. Structured output

- A `generate` stage can name `output_schema`, a registered pydantic model. Its fields are the stage's `writes`, by name. Without it the stage writes one field, the text.
- The toolkit registers one, `standalone_query_with_filter`: a `standalone_query` and a `metadata_filter`.
- `metadata_fields_of: <vector store>` on the stage narrows the schema's conditions to that store's declared fields and their allowed values.
- How the schema reaches the endpoint is a setting of the LLM component, in its `api` and `served` blocks, and never a matter of the prompt:

```yaml
llm:
  backend: api
  api:
    url: https://api.openai.com/v1/chat/completions
    api_format: openai
    model: gpt-4o-mini
    # structured_output: json_schema   # the default; format_instructions for an endpoint that imposes no schema
```

- **`json_schema`** sends OpenAI's `response_format` with the schema. The OVMS, vLLM and Ollama documentation say they accept that shape.
- **`format_instructions`** appends the schema as an instruction to the messages and parses the answer. The text and the parser are LangChain core's `PydanticOutputParser`. It is for an endpoint that does not impose a schema.
- Each method is one registered class. Under both, the stage validates the answer against the model. An answer that fails is asked again within the component's `retries`, then the request fails and shows the text.
- The fake LLM answers a valid instance of the schema.
- This is the one change to an existing port. `BaseLLM.generate` takes an optional output schema and still returns text, which under a schema is the JSON object alone. The `api` adapter, the OpenAI chat format and the fake LLM change with it, and the method class is applied in the `api` adapter. The port's `ainvoke` passes the request's `output_schema` through.

### 6. The conversation on the wire

- The bodies of `/query` and `/retrieve` keep `question`. Two fields are new and optional: `conversational_history` and `metadata_filter`.
- `conversational_history` is a list of messages with `role` (`user` or `assistant`) and `content`. A user message can carry `metadata_filter`, the conditions that applied to it. That is information for the prompts, which read it with Jinja. It is checked as the request's filter is, and it is never applied to retrieval by itself.
- The response keeps every field it has and gains `metadata_filter`: the conditions applied, the client's and the generated ones.
- The response headers are unchanged.
- No limit on the number of messages was decided. The builder adds none.

```json
{"question": "Et pour un véhicule loué ?",
 "conversational_history": [
   {"role": "user", "content": "Que dit l'AMF sur la franchise ?",
    "metadata_filter": [{"attribute": "source", "comparator": "in", "value": ["AMF"]}]},
   {"role": "assistant", "content": "..."}]}
```

### 7. The metadata filter

- A metadata filter is a flat list of conditions that must all hold. A condition has an `attribute`, a `comparator` and a `value`. The comparators are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in` and `nin`. An empty list filters nothing.
- The filter and its conditions are domain classes of ours. No library's filter type is used.
- A vector store declares what a filter can name, under `metadata_field_info`: for each field a `name`, a `type` (string, integer, float or boolean), a `description`, and optionally its `allowed_values`.
- A condition is checked against that declaration. A client's condition that names an unknown field, a value outside the allowed ones or a comparator its type does not have is answered with a 422 that names it.
- `BaseVectorStore.query` and `BaseLexicalSearch.lexical_query` take the filter, and a hybrid query passes it to both.
- One translator per store turns the filter into the store's own, behind one port in `adapters/`:
  - **Elasticsearch:** filter clauses on `metadata.<attribute>`, with the `.keyword` sub-field for a string.
  - **Chroma:** a `where` with its own operators.
  - **In memory:** a predicate over a passage's metadata.

#### The prefilter

- A vector store can state a `prefilter`: conditions, in the same shape, that it adds to every search before the request's own. It says which passages a workflow can read at all.
- The same translators serve it. On Elasticsearch the conditions go inside the `knn` clause, which the server applies during the vector search, and in the filter of a text search. On Chroma they go in `where`, which is applied before its vector search too.
- A prefilter's fields are not in `metadata_field_info`, so neither a client nor the LLM can name them. The response does not report them.
- This work changes nothing in ingestion. Whatever fills a store writes the fields its prefilter names, and `qp-ingest` writes none of them for the moment.
- A store whose passages lack a field returns nothing under a condition on it. That holds for every index `qp-ingest` filled.
- A filter is cheap when it keeps most of an index, and costly when it keeps little: the search explores more of the graph. A store should therefore hold one live ingestion pipeline version.
- The store keeps its `index` setting (`collection` on Chroma), which can name an index or an alias. One index for each ingestion pipeline version is the faster way to keep versions apart, and it needs no condition on the version.

### 8. The router

The router is the toolkit's fifth component type. Only how it chooses is a use case's.

- **Domain:** `Route` (a `name`, a `description`, its `utterances`), `RouteRequest` (an `embedding`) and `RouteChoice` (a `name`, a `confidence`).
- **Port:** `BaseRouter`, registered as `router`.
- **Settings of the type:** `embedder`, the name of the embedder component whose vectors the router reads, and `routes`, the names of the routes it chooses between, two at least.
- **Routes:** a use case has a `routes.yaml`, found as `prompts.yaml` is (`QP_ROUTES_FILE`, else `routes.yaml` in the working directory). It is a catalogue: several routers can each list some of its routes. A route a router lists and the file lacks is refused.
- **Backends the toolkit gives:** `fake`, which always chooses the first of its routes, and `served`, built. The toolkit registers the router's API format by hand, as the worked example of one.
- **Stage:** a plain call. The labels checked against `path_map` are the router's `routes`.

```yaml
routes:
  - name: self_contained
    description: A message that can be understood without the conversation.
    utterances:
      - Quelle est la franchise habituelle en cas de collision ?
      - Comment faire une réclamation après un accident de la route au Québec ?
      - Que couvre la garantie de responsabilité civile ?
      - Mon assurance couvre-t-elle un véhicule de location ?
      - Quel est le montant minimal d'assurance responsabilité obligatoire au Québec ?
      - Que faire si l'autre conducteur n'est pas assuré ?
      - Comment la SAAQ indemnise-t-elle les blessures corporelles ?
      - Puis-je résilier mon contrat d'assurance automobile avant son échéance ?
  - name: leans_on_history
    description: A message that needs the earlier turns to be understood.
    utterances:
      - Et pour un véhicule loué ?
      - Est-ce que ça s'applique aussi à mon conjoint ?
      - Combien de temps ai-je pour le faire ?
      - Et si c'est l'autre conducteur qui est responsable ?
      - Peux-tu préciser le deuxième point ?
      - Quel est le montant dans ce cas ?
      - Pourquoi est-ce obligatoire ?
      - Et sinon, que se passe-t-il ?
```

This is the RAG's whole `routes.yaml`: the sixteen utterances the owner accepted in the review.

### 9. The RAG use case

#### 9.1 Its files

```
query_processing_rag_usecase/
├── workflow.yaml             changed
├── prompts.yaml              new: answer_from_context, rewrite_as_standalone
├── routes.yaml               new: two routes, eight utterances each
├── router_pipeline.joblib    new, written by `make train-router`
├── router_pipeline.json      new, its manifest
├── pyproject.toml            its packages, scikit-learn==1.9.1, joblib, the command
├── Makefile                  one target, train-router
├── .env.example              the fake backends of the two new components
├── deploy.example.yaml       Solon's model path, the optional structured_output line, the prefilter
├── src/query_processing_rag_usecase/
│   ├── config/knn_router_settings.py
│   ├── adapters/router/local/knn_router.py
│   ├── infrastructure/pipeline_artifact.py
│   └── entrypoints/train_router.py
└── tests/
```

#### 9.2 `workflow.yaml`

```yaml
extensions: [query_processing_rag_usecase]

local_models:
  runtime: openvino # openvino | torch, for every component on the local backend

components:
  embedder:
    backend: local # local | served | api | fake
    source_id: Snowflake/snowflake-arctic-embed-l-v2.0
    query_prefix: "query: "
  router_embedder:
    type: embedder
    backend: local
    source_id: OrdalieTech/Solon-embeddings-large-0.1
    query_prefix: "query : "
  router:
    backend: local # local | served | fake
    embedder: router_embedder
    routes: [self_contained, leans_on_history]
    local:
      pipeline_file: router_pipeline.joblib
      n_neighbors: 3
      metric: cosine
  reranker:
    backend: local # local | served | api | fake | none
    source_id: antoinelouis/crossencoder-camembert-base-mmarcoFR
  llm:
    backend: api # served | api | fake
    api:
      url: https://api.openai.com/v1/chat/completions
      api_format: openai
      model: gpt-4o-mini
    http:
      timeout_s: 120
  vector_store:
    backend: elasticsearch # elasticsearch | chroma | fake
    # No prefilter here: an index qp-ingest filled has neither of its fields.
    # A site whose passages carry them states it in deploy.yaml (section 9.4).
    elasticsearch:
      url: http://localhost:9200
      index: rag_passages # an index or an alias
    metadata_field_info:
      - name: source
        type: string
        description: The reference the passage comes from.
        allowed_values: [A-25, AMF, CHAD, GAA, code_securite_routiere, educaloi, infoassurances]

graph:
  stages:
    embed_for_route: {call: router_embedder, reads: [conversational_query], writes: [routing_embedding]}
    route: {call: router, reads: [routing_embedding], writes: [route]}
    rewrite:
      type: generate
      call: llm
      prompt: rewrite_as_standalone
      output_schema: standalone_query_with_filter
      metadata_fields_of: vector_store
      reads: [conversational_query, conversational_history]
      writes: [standalone_query, metadata_filter]
    embed: {call: embedder, reads: [standalone_query], writes: [query_embedding]}
    retrieve:
      call: vector_store
      retrieval_mode: dense
      top_k: 20
      reads: [query_embedding, metadata_filter]
      writes: [candidates]
    rerank: {call: reranker, top_k: 10, reads: [standalone_query, candidates], writes: [reranked]}
    generate:
      call: llm
      prompt: answer_from_context
      reads: [conversational_query, reranked]
      writes: [answer]
  edges:
    - [embed_for_route, route]
    - [rewrite, embed, retrieve, rerank]
    - [generate, END]
  conditional_edges:
    # the three of section 3.3
```

The file keeps its comments, which the example leaves out.

#### 9.3 The kNN router

```python
class KnnRouterSettings(SettingsBlock):
    pipeline_file: str
    n_neighbors: int = Field(ge=1)
    metric: str


class KnnRouter(BaseRouter, backend="local", settings=KnnRouterSettings):
    """Chooses with a trained sklearn pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def from_settings(cls, settings: KnnRouterSettings, resources: RouterResources) -> "KnnRouter":
        return cls(load_pipeline(settings, resources.embedder, resources.routes))

    def invoke(self, request: RouteRequest) -> RouteChoice:
        probabilities = self._pipeline.predict_proba([request.embedding])[0]
        best = int(probabilities.argmax())
        return RouteChoice(name=str(self._pipeline.classes_[best]),
                           confidence=float(probabilities[best]))
```

- **Training:** the command `qp-rag-train-router` embeds each utterance of the router's routes with the component `embedder` names, through its `ainvoke`, so the query prefix is applied. It fits a scikit-learn pipeline of a normalizer and a k-nearest-neighbours classifier with the settings' values, and saves it with joblib.
- **The real file is the last step of the work.** Training needs Solon's weights, which are not downloaded on the owner's laptop. It runs on a cloud machine with the owner's go-ahead at that time, and the file is committed before the work is merged. Until then the command and the router are tested on the fake embedder.
- **Manifest:** `router_pipeline.json` records the file's hash, the scikit-learn version, the routes, a hash of their utterances, the embedder's alias, query prefix and vector size, the training values and the leave-one-out accuracy. It records the embedder's alias and not its source id, which is a Dataset folder on Domino.
- **At start-up** the router refuses a file its manifest does not match, or a manifest that differs from the installed scikit-learn, the routes, the utterances, the embedder or the training values. The message names both values.
- **The file is committed** with its manifest. With sixteen utterances it is about 130 KB.
- **The corpus's 82 real questions** are never committed. The training command can take a questions file and report how many it routes as `self_contained`, as a check run by hand.
- **No default in code:** `n_neighbors` and `metric` are pinned in `workflow.yaml`.

#### 9.4 `routes.yaml`, `deploy.yaml` and the layouts

- The sixteen utterances are those of section 8. They were written from general knowledge of Quebec automobile insurance and not from the corpus, which states no licence.
- `deploy.example.yaml` gains Solon's model path under `overrides.components.router_embedder.source_id`, and a commented `llm: {api: {structured_output: format_instructions}}`.
- The RAG's prefilter is a value of the site, since which ingestion pipeline version is live differs from one site to another. `workflow.yaml` states none, and `deploy.example.yaml` shows it, commented:

```yaml
overrides:
  components:
    vector_store:
      prefilter:
        - {attribute: ingestion_pipeline_version, comparator: eq, value: "1"}
        - {attribute: active, comparator: eq, value: true}
```
- The router and its embedder stay `local` in every layout of the experiments, so the layouts keep their replica counts. No topology alias names them.

### 10. The load test

A line of the questions file can carry `conversational_history` and `metadata_filter` beside `question`, and the request sends them. The file decides the mix of first messages and follow-ups. Without a history a request never reaches the router.

### 11. What changes for what exists

- Every workflow file states `reads` and `writes`: the toolkit's fixture file and the RAG's.
- The `generate` settings `system_prompt` and `user_prompt` go, and the README's table of older names says where they went.
- The stage types `embed` and `rerank` go, and the edge condition `is_query_mode` takes the value it reads.
- The README's "Where a file goes", its section on a second use case, the root `CLAUDE.md` ("Put code in the toolkit, never in a use case") and the use case's `pyproject.toml` comment are brought in line with ADR 0006.
- `CONTEXT.md` already holds the vocabulary, a model server's call route included.
- The first run with no model weights puts the two new components on `fake` too.
- `qp-ingest` does not change. With two embedder components it already asks which one to use, so the RAG's `make ingest` names it: `--embedder embedder`.

## Testing

A good test here asserts what a process does from the outside. The feature is tested at the two seams the toolkit already has, and at one new one that is public.

- **An app through its HTTP API.** The workflow app and the model server are built from settings and driven with a test client, on the fixture workflow file with fake components. Prior art: the entry-point tests. They assert the answers of `/query` and `/retrieve` with and without a history and a filter, the path a request takes (by what the named fakes were asked, since `timings_ms` keeps a key at 0.0 for a stage that did not run), the conditions reported, each start-up refusal of section 3.4 with its message, and `POST /call/<component type>` on the model server.
- **The deploy tool against the fake Domino.** Prior art: the deploy tests and the dry run. They assert that a topology alias can serve such a component in its own App, the refusal on the `ovms` kind, and that `overrides` are checked against the settings of an extension.
- **The shipped test classes** of section 2.7, run in the toolkit on a small component type of its own, defined in its test helpers the way a use case would define one, and on the fake router.

Below the seams:

- **The filter translators** follow the store tests already there: Elasticsearch over a fake LangChain store, asserting the clauses sent; Chroma on a real embedded Chroma in a temporary folder; the in-memory store directly. Each comparator is covered on each store.
- **The structure tests** gain the rules of section 1: the toolkit still never imports a use case, and the use case imports only `extension_api`.
- **The use case** tests its own files: the workflow file loads with its extension and its graph passes the checks, each route has its utterances, and a pipeline trained on the fake embedder is accepted by its own manifest and refused after each kind of change.

Every new function gets a test, and named fakes are used, not patches.

## Order of work

A suggested order, each step leaving `make check` green:

1. The registrations replace the tables that name the four component types, with no change of behaviour.
2. The common API on the four ports, the plain call, `reads` and `writes` in the files, and the state built from the file.
3. `reads` on conditional edges, and the label answer.
4. Extensions, `extension_api`, a component type and an implementation from one class each, on the test helpers' component type.
5. The built `served` class, the API format, the model server's route and the deploy tool.
6. `prompts.yaml` and the `generate` stage.
7. The conversation on the wire and `has_history`.
8. The metadata filter: its classes, the declaration of fields, the three translators and the wire.
9. Structured output.
10. The router in the toolkit and `routes.yaml`.
11. The RAG use case: the kNN router, its training, its graph and its files.
12. The load-test lines and the documents.

## Left to the builder

- The names and placement of the toolkit's inner pieces: the registries, the resources object, the state builder and the module that holds the route path.
- How the built `served` class and the built API format are made, as long as a user's registration takes their place.
- How `metadata_filter` is made to accumulate in LangGraph. A check during the review built a state class at run time with a reducer on `timings_ms`.
- The exact wording of each refusal, as long as it carries the value found and the shape expected.

## Not verified

- No code of this design has run on the toolkit. The checks of the review ran on small stand-ins: a label answer and a conditional edge from `START` on LangGraph 1.2.21, the class-line registration with its refusals, the binding of `reads` and `writes`, a flat filter's schema, and LangChain core's instruction text and parser.
- That OVMS can fail to impose a schema comes from its documentation, not from a run.
- The real pipeline file needs Solon's weights, so it cannot be trained on the owner's laptop. It is trained on a machine that has them, or against a served Solon, and then committed.
- Whether a 2-CPU machine holds the two embedders and the reranker in memory in the monolith layout is not checked.
- What the prefilter costs on the project's Elasticsearch 9.0.0 is not measured. That a filter inside the `knn` clause is applied during the search, and that it is cheap when it keeps most of an index, come from Elasticsearch's documentation and from the installed client's code.
