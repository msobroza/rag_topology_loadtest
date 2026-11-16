# rag_topology_loadtest

The toolkit `query_processing` and its first use case, a RAG. Read `CONTEXT.md`
for the vocabulary, the README's "Where a file goes" for the layout and its
rules, and `docs/adr/0003` to `0008` for the design.

- `query_processing/`: the toolkit, one distribution holding what any use case
  can assemble (package `query_processing`, commands `qp-*`, variables `QP_*`):
  the component types with their requests and answers, the stage types, the
  edge conditions, the API formats, the backends of the embedder, the
  reranker, the LLM and the vector store, and the router's `fake` and `served`.
- `query_processing_rag_usecase/`: the RAG use case. It installs the toolkit and
  holds its settings (`workflow.yaml`, `prompts.yaml`, `routes.yaml`,
  `deploy.example.yaml`, `.env.example`, a `Makefile` of run targets), its
  tests, and the implementations, trained models and data that are its own: a
  package under `src/` (its kNN router, the router's `local` backend, and
  `qp-rag-train-router`), which `workflow.yaml` names under `extensions`.
- Put in the toolkit what any use case can assemble; a use case holds its own
  implementations, trained models and data (ADR 0006), and a stage type, an
  edge condition or an output schema only it needs. A use case imports
  the toolkit through `query_processing.extension_api` alone, its tests import
  the toolkit's test classes from `query_processing.testing`, and it adds a
  component by subclassing one abstract class (the README's "What a use case
  adds"). The toolkit never imports nor names a use case, and ships no use
  case's values: its tests run on
  `query_processing/tests/helpers/fixture_workflow.yaml`.

## Commands

Every `make` target runs the tools of `.venv/bin/` when a `.venv` exists at the
repository's root (one for the repository), and those on the PATH otherwise.

From the repository's root:

- `make install-dev`: install the toolkit with its test and gate tools and the use
  case, in one command (no package index has the toolkit).
- `make check`: the gate, every target of the Health Stack below in that order over
  the toolkit, stopping at the first failure, then the use case's tests (linted
  with the toolkit's configuration).
- `make -C query_processing check`: the same gates over the toolkit alone.

From `query_processing_rag_usecase/`, where the commands find `workflow.yaml`,
`deploy.yaml` and `.env`:

- `make help`: the run targets (`serve`, `ingest`, `train-router`, `loadtest`,
  `compare`, `deploy-dry-run`, `deploy`, ...).
- Run the workflow app with no model weights by putting every component on its
  fake backend: `QP_COMPONENTS__<NAME>__BACKEND=fake` for `EMBEDDER`,
  `ROUTER_EMBEDDER` (the router's embedder), `ROUTER`, `RERANKER`, `LLM` and
  `VECTOR_STORE`. The router's pipeline file and its manifest
  (`router_pipeline.joblib`, `router_pipeline.json`) are committed together;
  start-up refuses them after a change to an utterance, the router's training
  values, its embedder's alias or query prefix, or the scikit-learn pin. Then
  run `make train-router` on a machine that has the embedder's model (it
  refuses the fake embedder) and commit both files.

## Health Stack

- lint: make lint
- typecheck: make typecheck
- complexity: make complexity
- deadcode: make deadcode
- deps: make deps
- test: make test
