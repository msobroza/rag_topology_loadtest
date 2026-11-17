# rag_topology_loadtest

The toolkit `query_processing` and its first use case, a RAG. Read `CONTEXT.md`
for the vocabulary, the README's "Where a file goes" for the layout and its
rules, and `docs/adr/0003` to `0009` for the design.

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
- `experiment_charts/`: not the product. A distribution of its own (package
  `experiment_charts`, matplotlib and PyYAML) that draws the figures and tables
  of `experiments-explained/` from the registry of measurements in
  `experiments/`. It imports nothing from the toolkit, the fake Domino or a use
  case, and the toolkit's `pyproject.toml` names nothing of it. The records in
  `experiments/` keep the names of their day: never reword one.

## Commands

Every `make` target runs the tools of `.venv/bin/` when a `.venv` exists at the
repository's root (one for the repository), and those on the PATH otherwise.

From the repository's root:

- `make install-dev`: install the toolkit with its test and gate tools, the use
  case and the charts package, in one command (no package index has the toolkit).
- `make check`: the gate, every target of the Health Stack below in that order over
  the toolkit, stopping at the first failure, then lint, typecheck, complexity
  and deps over the use case with the toolkit's configuration and its tests,
  then the same over the charts package and its tests.
- `make -C query_processing check`: the same gates over the toolkit alone.
- `make experiment-charts`: redraw `experiments-explained/` from `experiments/`,
  after a change to the registry (the charts package's tests refuse a table
  that is out of date).

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

## Defensive patterns

Every network call has a timeout and bounded retries. Beyond that default this
project uses one further pattern:

- Backoff between the retries of an HTTP call: the wait doubles from
  `backoff_s`. Timeout, retries and backoff come from the settings (the
  workflow file's `http` block, `timeout_s`, `retries` and `backoff_s`, the
  shared one or a component's own); a new call reuses them and adds no
  constant of its own.

No circuit breaker, no rate limit, no graceful degradation: add none unless the
owner asks for one.

## The tests /verify runs

`.gstack/pregate.json` names the command for the tests a change selects. The
repository has three tests folders, so the command sorts the selected files by
the folder that holds them (`query_processing/`, `query_processing_rag_usecase/`,
`experiment_charts/`) and runs pytest inside each, where its `pyproject.toml`
is: pytest from `.venv/bin` when a `.venv` exists here, as every `make` target
does.

A green `/verify` is not a green suite. For a changed file under `src/` it selects
the tests that name the file's path, not the tests that import it: a change to
`domain/passages.py` alone ran one test file. `make check` is the gate.
