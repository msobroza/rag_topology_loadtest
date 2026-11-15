# rag_topology_loadtest

A Locust load test of one RAG workflow deployed in three topologies on Domino.
Read `CONTEXT.md` for the vocabulary and the README's "Where a file goes" for the
layout.

## Commands

Two distributions: the toolkit in `query_processing/` (all the code) and the RAG
use case in `query_processing_rag_usecase/` (settings only; it installs the
toolkit). Run the commands below from the repository's root: its `Makefile`
installs both and hands each gate to `query_processing/Makefile`, which also runs
them by itself over the toolkit alone (`make -C query_processing check`). The run
targets (`serve`, `ingest`, `loadtest`, `deploy`, ...) are in
`query_processing_rag_usecase/Makefile` only: run them from that directory, where
the commands find `workflow.yaml`, `deploy.yaml` and `.env`.

Every `make` target runs the tools of `.venv/bin/` when a `.venv` exists at the
repository's root (one for the repository), and those on the PATH otherwise.

- `make install-dev`: install the toolkit with its test and gate tools and the use
  case, in one command (no package index has the toolkit).
- `make check`: the gate, every target of the Health Stack below in that order over
  the toolkit, stopping at the first failure, then the use case's tests (linted
  with the toolkit's configuration).
- The toolkit never imports nor names a use case, and ships no use case's
  values: its tests run on `query_processing/tests/helpers/fixture_workflow.yaml`.

## Health Stack

- lint: make lint
- typecheck: make typecheck
- complexity: make complexity
- deadcode: make deadcode
- deps: make deps
- test: make test
