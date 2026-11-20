# query_processing — the toolkit's developer tasks: its install and its gates (run from
# this directory; the repository's root Makefile hands install, install-dev, check
# and each gate here). The run targets read a use case's files and write its
# output, so they are a use case's: make -C ../query_processing_rag_usecase help.
.DEFAULT_GOAL := help
.PHONY: help install install-dev check lint format typecheck complexity deadcode deps test pin-answers fake-domino

# The repository's one .venv, at its root beside this directory, is used without
# activating it; without one, commands come from the PATH as usual. The root
# Makefile passes its choice as BIN (an absolute path ending in /, or empty), which
# wins over this line. Two lines because make 3.81 (macOS) looks a recipe's command
# up on its own PATH: $(BIN) is for those commands, the export is for the scripts
# and shells the recipes start.
VENV := $(abspath $(CURDIR)/../.venv)
BIN := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)
export PATH := $(VENV)/bin:$(PATH)

PYTHON ?= $(BIN)python
BEHAVIOUR ?= starts

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# No package index has the core the toolkit depends on: its directory, beside
# this one, is named first in the same command.
install: ## Install the core and the toolkit (runtime deps only); the root's make install adds the use case
	$(PYTHON) -m pip install -e ../query_processing_core -e .

install-dev: ## Install the core and the toolkit with locust + test tooling
	$(PYTHON) -m pip install -e ../query_processing_core -e ".[dev]"

check: lint typecheck complexity deadcode deps test ## Run every quality gate, stopping at the first failure

lint: ## ruff (rules + Google docstrings), black --check, pydoclint
	$(BIN)ruff check src tests locustfile.py
	$(BIN)black --check src tests locustfile.py
	$(BIN)pydoclint src locustfile.py

format: ## Auto-format with ruff --fix + black
	$(BIN)ruff check --fix src tests locustfile.py
	$(BIN)black src tests locustfile.py

typecheck: ## mypy and ty over the source tree
	$(BIN)mypy src
	$(BIN)ty check src

complexity: ## complexipy: cognitive complexity <= 10 per function
	$(BIN)complexipy

deadcode: ## vulture: nothing unused at confidence >= 60 outside vulture_whitelist.py
	$(BIN)vulture

deps: ## deptry: no missing, unused or transitive dependency
	$(BIN)deptry .

test: ## Run the test suite (no model weights, no network; fixture workflow files only)
	$(BIN)pytest

# The test that compares the workflow app on the fixture workflow with its pinned
# answer record, run to make the record again instead: in process, every
# component on its fake backend, no socket.
pin-answers: ## Make the fixture workflow's pinned answer record again, after a change that alters the answers on purpose (commit it)
	$(BIN)pytest tests/testing/test_pinned_answer_record.py --pin-answers

fake-domino: ## Off Domino: serve a fake Apps API on :8899 (Swagger at /docs): make fake-domino BEHAVIOUR=slow
	$(BIN)qp-fake-domino --behaviour $(BEHAVIOUR)
