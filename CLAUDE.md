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

## Defensive patterns

Every network call has a timeout and bounded retries. Beyond that default this
project uses one further pattern:

- Backoff between the retries of an HTTP call: the wait doubles from
  `http_backoff_s`. Timeout, retries and backoff come from the settings
  (`http_timeout_s`, `http_retries`, `http_backoff_s`); a new call reuses them and
  adds no constant of its own.

No circuit breaker, no rate limit, no graceful degradation: add none unless the
owner asks for one.

## The tests /verify runs

`.gstack/pregate.json` names the command for the tests a change selects: pytest
from `.venv/bin` when a `.venv` exists here, as every `make` target does.
