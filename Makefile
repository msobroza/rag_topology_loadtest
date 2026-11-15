# rag_topology_loadtest — the repository's tasks. Two distributions live here: the
# toolkit in query_processing/ and the RAG use case in query_processing_rag_usecase/,
# which installs it (ADR 0005). install and install-dev install both in one command;
# check runs the toolkit's gates, then the use case's tests; each gate is handed to
# the toolkit's Makefile. The run targets (serve, ingest, loadtest, deploy, ...) are
# the use case's: cd query_processing_rag_usecase && make help.
.DEFAULT_GOAL := help
TOOLKIT := query_processing
USE_CASE := query_processing_rag_usecase
HANDED := lint typecheck complexity deadcode deps test
.PHONY: help install install-dev check format $(HANDED)

# The repository's one .venv, in this directory, is used without activating it;
# without one, commands come from the PATH as usual. The toolkit's Makefile is
# given that choice as BIN, an absolute path (or empty, for the PATH), so a BIN
# given here on the command line reaches it unchanged.
BIN := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/,)
# The use case's tests run from its directory, and are linted and formatted with
# the toolkit's configuration (its paths read from that directory).
IN_USE_CASE := cd $(USE_CASE) &&
TOOLKIT_CONFIG := --config ../$(TOOLKIT)/pyproject.toml

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

check: ## Every quality gate over the toolkit, then the use case's tests and their lint, stopping at the first failure
	$(MAKE) -C $(TOOLKIT) check BIN="$(BIN)"
	$(IN_USE_CASE) $(BIN)ruff check $(TOOLKIT_CONFIG) tests
	$(IN_USE_CASE) $(BIN)black --check $(TOOLKIT_CONFIG) tests
	$(IN_USE_CASE) $(BIN)pytest

format: ## Auto-format the toolkit and the use case's tests with ruff --fix + black
	$(MAKE) -C $(TOOLKIT) format BIN="$(BIN)"
	$(IN_USE_CASE) $(BIN)ruff check --fix $(TOOLKIT_CONFIG) tests
	$(IN_USE_CASE) $(BIN)black $(TOOLKIT_CONFIG) tests

lint: ## ruff, black --check, pydoclint over the toolkit
typecheck: ## mypy and ty over the toolkit's source tree
complexity: ## complexipy: cognitive complexity <= 10 per function
deadcode: ## vulture: nothing unused outside the toolkit's vulture_whitelist.py
deps: ## deptry: no missing, unused or transitive dependency of the toolkit
test: ## The toolkit's test suite (no model weights, no network)

$(HANDED):
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"
