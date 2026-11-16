# rag_topology_loadtest — the repository's tasks. Two distributions live here: the
# toolkit in query_processing/ and the RAG use case in query_processing_rag_usecase/,
# which installs it (ADR 0005). install and install-dev install both in one command;
# check runs the toolkit's gates, then the use case's; each gate is handed to the
# toolkit's Makefile, then run over the use case's source with the toolkit's
# configuration where it can be (lint, typecheck, complexity, deps). The run targets (serve, ingest, loadtest, deploy, ...) are
# the use case's: cd query_processing_rag_usecase && make help.
.DEFAULT_GOAL := help
TOOLKIT := query_processing
USE_CASE := query_processing_rag_usecase
HANDED := deadcode test
# The gates also run over the use case's source, after the toolkit's.
SHARED := lint typecheck complexity deps
.PHONY: help install install-dev check format $(HANDED) $(SHARED)

# The repository's one .venv, in this directory, is used without activating it;
# without one, commands come from the PATH as usual. The toolkit's Makefile is
# given that choice as BIN, an absolute path (or empty, for the PATH), so a BIN
# given here on the command line reaches it unchanged.
BIN := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/,)
# The use case's source and tests run from its directory, and are linted,
# formatted and type-checked with the toolkit's configuration (its paths read
# from that directory). vulture is not run there: the toolkit's whitelist and
# paths are its own.
IN_USE_CASE := cd $(USE_CASE) &&
TOOLKIT_PYPROJECT := ../$(TOOLKIT)/pyproject.toml
TOOLKIT_CONFIG := --config $(TOOLKIT_PYPROJECT)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo "  The run targets are the use case's: make -C $(USE_CASE) help"

# No package index has the toolkit: the use case's dependency on it is met only
# when both directories are named in one command, the toolkit first.
install: ## Install the toolkit and the RAG use case (runtime deps only)
	$(BIN)python -m pip install -e ./$(TOOLKIT) -e ./$(USE_CASE)

install-dev: ## Install the toolkit with locust + test tooling, and the RAG use case
	$(BIN)python -m pip install -e "./$(TOOLKIT)[dev]" -e ./$(USE_CASE)

# The gates over the use case's source, each run after the toolkit's own.
define USE_CASE_lint
	$(IN_USE_CASE) $(BIN)ruff check $(TOOLKIT_CONFIG) src tests
	$(IN_USE_CASE) $(BIN)black --check $(TOOLKIT_CONFIG) src tests
	$(IN_USE_CASE) $(BIN)pydoclint $(TOOLKIT_CONFIG) src
endef
define USE_CASE_typecheck
	$(IN_USE_CASE) $(BIN)mypy --config-file $(TOOLKIT_PYPROJECT) src
	$(IN_USE_CASE) $(BIN)ty check src
endef
define USE_CASE_complexity
	$(IN_USE_CASE) $(BIN)complexipy src --max-complexity-allowed 10 --failed
endef
define USE_CASE_deps
	$(IN_USE_CASE) $(BIN)deptry src
endef

check: ## Every quality gate over the toolkit, then over the use case and its tests, stopping at the first failure
	$(MAKE) -C $(TOOLKIT) check BIN="$(BIN)"
	$(USE_CASE_lint)
	$(USE_CASE_typecheck)
	$(USE_CASE_complexity)
	$(USE_CASE_deps)
	$(IN_USE_CASE) $(BIN)pytest

format: ## Auto-format the toolkit and the use case with ruff --fix + black
	$(MAKE) -C $(TOOLKIT) format BIN="$(BIN)"
	$(IN_USE_CASE) $(BIN)ruff check --fix $(TOOLKIT_CONFIG) src tests
	$(IN_USE_CASE) $(BIN)black $(TOOLKIT_CONFIG) src tests

lint: ## ruff, black --check, pydoclint over the toolkit, then the use case
typecheck: ## mypy and ty over the toolkit's source tree, then the use case's
complexity: ## complexipy: cognitive complexity <= 10 per function, toolkit and use case
deps: ## deptry: no missing, unused or transitive dependency, toolkit and use case
deadcode: ## vulture: nothing unused outside the toolkit's vulture_whitelist.py
test: ## The toolkit's test suite (no model weights, no network)

$(HANDED):
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"

$(SHARED):
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"
	$(USE_CASE_$@)
