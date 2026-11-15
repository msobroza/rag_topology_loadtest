# rag_topology_loadtest

A Locust load test of one RAG workflow deployed in three topologies on Domino.
Read `CONTEXT.md` for the vocabulary and the README's "Where a file goes" for the
layout.

## Commands

Every `make` target runs the tools of `.venv/bin/` when a `.venv` exists here, and
those on the PATH otherwise.

- `make install-dev`: install the project with its test and gate tools.
- `make check`: the gate, every target of the Health Stack below in that order,
  stopping at the first failure.

## Health Stack

- lint: make lint
- typecheck: make typecheck
- complexity: make complexity
- deadcode: make deadcode
- deps: make deps
- test: make test
