# rag_topology_loadtest

A Locust load test of one RAG workflow deployed in three topologies on Domino.
Read `CONTEXT.md` for the vocabulary and the README's "Where a file goes" for the
layout.

## Commands

The toolkit's distribution is in `query_processing/`. Run the commands below from
the repository's root: its `Makefile` hands each one to `query_processing/Makefile`,
which also runs them by itself (`make -C query_processing check`). The run targets
(`serve`, `ingest`, `loadtest`, `deploy`, ...) are in `query_processing/Makefile`
only: run them from `query_processing/`.

Every `make` target runs the tools of `.venv/bin/` when a `.venv` exists at the
repository's root (one for the repository), and those on the PATH otherwise.

- `make install-dev`: install the toolkit from `query_processing/` with its test and
  gate tools.
- `make check`: the gate, every target of the Health Stack below in that order,
  stopping at the first failure.

## Health Stack

- lint: make lint
- typecheck: make typecheck
- complexity: make complexity
- deadcode: make deadcode
- deps: make deps
- test: make test
