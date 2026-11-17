# rag_topology_loadtest — the repository's tasks. Two distributions of the product
# live here: the toolkit in query_processing/ and the RAG use case in
# query_processing_rag_usecase/, which installs it (ADR 0005). A third is not the
# product: experiment_charts/ draws experiments-explained/ from experiments/.
# install installs the first two in one command, install-dev all three; check runs
# the toolkit's gates, then the use case's, then the charts package's; each gate is
# handed to the toolkit's Makefile, then run over the source of the two others with
# the toolkit's configuration where it can be (lint, typecheck, complexity, deps).
# The run targets (serve, ingest, loadtest, deploy, ...) are the use case's:
# cd query_processing_rag_usecase && make help.
.DEFAULT_GOAL := help
TOOLKIT := query_processing
USE_CASE := query_processing_rag_usecase
CHARTS := experiment_charts
HANDED := deadcode test
# The gates also run over the use case's source and the charts package's, after
# the toolkit's.
SHARED := lint typecheck complexity deps
.PHONY: help install install-dev check format experiment-charts $(HANDED) $(SHARED)

# The repository's one .venv, in this directory, is used without activating it;
# without one, commands come from the PATH as usual. The toolkit's Makefile is
# given that choice as BIN, an absolute path (or empty, for the PATH), so a BIN
# given here on the command line reaches it unchanged.
BIN := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/,)
# The source and tests of a distribution beside the toolkit run from its
# directory, and are linted, formatted and type-checked with the toolkit's
# configuration (its paths read from that directory). vulture is not run there:
# the toolkit's whitelist and paths are its own.
TOOLKIT_PYPROJECT := ../$(TOOLKIT)/pyproject.toml
TOOLKIT_CONFIG := --config $(TOOLKIT_PYPROJECT)
# ty checks imports against the first python on the PATH unless it is told of
# one, and these directories hold no .venv of their own. Called by its path in
# the repository's .venv where the PATH leads with another python (the CI's
# image), it could not find the toolkit the use case imports. So it is told the
# python its tools come from; with the tools on the PATH it is told nothing and
# finds the active environment. The toolkit's Makefile needs no such flag: it
# puts the .venv first on the PATH of what its recipes start.
TY_CHECK := $(BIN)ty check$(if $(BIN), --python $(BIN)python,)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo "  The run targets are the use case's: make -C $(USE_CASE) help"

# No package index has the toolkit: the use case's dependency on it is met only
# when both directories are named in one command, the toolkit first.
install: ## Install the toolkit and the RAG use case (runtime deps only)
	$(BIN)python -m pip install -e ./$(TOOLKIT) -e ./$(USE_CASE)

install-dev: ## Install the toolkit with locust + test tooling, the RAG use case and the charts package
	$(BIN)python -m pip install -e "./$(TOOLKIT)[dev]" -e ./$(USE_CASE) -e ./$(CHARTS)

# The gates over a distribution beside the toolkit, $(1) its directory, each run
# after the toolkit's own, and its formatting.
define BESIDE_lint
	cd $(1) && $(BIN)ruff check $(TOOLKIT_CONFIG) src tests
	cd $(1) && $(BIN)black --check $(TOOLKIT_CONFIG) src tests
	cd $(1) && $(BIN)pydoclint $(TOOLKIT_CONFIG) src
endef
define BESIDE_typecheck
	cd $(1) && $(BIN)mypy --config-file $(TOOLKIT_PYPROJECT) src
	cd $(1) && $(TY_CHECK) src
endef
define BESIDE_complexity
	cd $(1) && $(BIN)complexipy src --max-complexity-allowed 10 --failed
endef
define BESIDE_deps
	cd $(1) && $(BIN)deptry src
endef
define BESIDE_format
	cd $(1) && $(BIN)ruff check --fix $(TOOLKIT_CONFIG) src tests
	cd $(1) && $(BIN)black $(TOOLKIT_CONFIG) src tests
endef
# Those four in the gate's order, then the distribution's own tests.
define BESIDE_check
$(call BESIDE_lint,$(1))
$(call BESIDE_typecheck,$(1))
$(call BESIDE_complexity,$(1))
$(call BESIDE_deps,$(1))
	cd $(1) && $(BIN)pytest
endef

check: ## Every quality gate over the toolkit, then over the use case and its tests, then over the charts package and its tests, stopping at the first failure
	$(MAKE) -C $(TOOLKIT) check BIN="$(BIN)"
	$(call BESIDE_check,$(USE_CASE))
	$(call BESIDE_check,$(CHARTS))

format: ## Auto-format the toolkit, the use case and the charts package with ruff --fix + black
	$(MAKE) -C $(TOOLKIT) format BIN="$(BIN)"
	$(call BESIDE_format,$(USE_CASE))
	$(call BESIDE_format,$(CHARTS))

lint: ## ruff, black --check, pydoclint over the toolkit, then the use case, then the charts package
typecheck: ## mypy and ty over the toolkit's source tree, then the use case's and the charts package's
complexity: ## complexipy: cognitive complexity <= 10 per function, in all three
deps: ## deptry: no missing, unused or transitive dependency, in all three
deadcode: ## vulture: nothing unused outside the toolkit's vulture_whitelist.py
test: ## The toolkit's test suite (no model weights, no network)

$(HANDED):
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"

$(SHARED):
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"
	$(call BESIDE_$@,$(USE_CASE))
	$(call BESIDE_$@,$(CHARTS))

# From this directory, where the registry and the folder it explains are: the
# command reads experiments/ and writes experiments-explained/ by those names.
experiment-charts: ## Redraw experiments-explained/ (figures and their tables) from the registry in experiments/
	$(BIN)python -m experiment_charts
