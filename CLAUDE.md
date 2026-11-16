# rag_topology_loadtest

The toolkit `query_processing` and its first use case, a RAG. Read `CONTEXT.md`
for the vocabulary, the README's "Where a file goes" for the layout and its
rules, and `docs/adr/0003` to `0005` for the design.

- `query_processing/`: the toolkit, one distribution holding all the code
  (package `query_processing`, commands `qp-*`, variables `QP_*`).
- `query_processing_rag_usecase/`: the RAG use case, settings only
  (`workflow.yaml`, `deploy.example.yaml`, `.env.example`, a `Makefile` of run
  targets, its tests). It installs the toolkit.
- Put code in the toolkit, never in a use case. The toolkit never imports nor
  names a use case, and ships no use case's values: its tests run on
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

- `make help`: the run targets (`serve`, `ingest`, `loadtest`, `compare`,
  `deploy-dry-run`, `deploy`, ...).
- Run the workflow app with no model weights by putting every component on its
  fake backend: `QP_COMPONENTS__<NAME>__BACKEND=fake` for `EMBEDDER`,
  `ROUTER_EMBEDDER`, `ROUTER`, `RERANKER`, `LLM` and `VECTOR_STORE`.

## Health Stack

- lint: make lint
- typecheck: make typecheck
- complexity: make complexity
- deadcode: make deadcode
- deps: make deps
- test: make test
