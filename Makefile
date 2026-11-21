# rag_topology_loadtest — the repository's tasks. Three distributions of the
# product live here: the core in query_processing_core/, what toolkits exchange
# (ADR 0014); the toolkit in query_processing/, which installs it; and the RAG use
# case in query_processing_rag_usecase/, which installs the toolkit (ADR 0005). A
# fourth is not the product: experiment_charts/ draws experiments-explained/ from
# experiments/. install installs the three in one command, install-dev all four,
# each after doclang's core, from its clone (DOCLANG_CLONE, in doclang.mk);
# doclang-wheel builds the wheel of doclang's core an App and the image install;
# check runs the core's gates, then the toolkit's, then the use case's, then the
# charts package's; each gate is run over the core's source with the toolkit's
# configuration where it can be (lint, typecheck, complexity, deps), handed to the
# toolkit's Makefile, then run over the source of the two others in the same way.
# The run targets (serve, ingest, loadtest, deploy, ...) are the use case's:
# cd query_processing_rag_usecase && make help.
.DEFAULT_GOAL := help
CORE := query_processing_core
TOOLKIT := query_processing
USE_CASE := query_processing_rag_usecase
CHARTS := experiment_charts
HANDED := deadcode test
# The gates also run over the core's source, before the toolkit's, and over the
# use case's and the charts package's, after it.
SHARED := lint typecheck complexity deps
.PHONY: help install install-dev doclang-wheel check format experiment-charts $(HANDED) $(SHARED)

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

# Where doclang's core is: its clone (DOCLANG_CLONE), as the toolkit's Makefile
# reads it too.
include doclang.mk
# Where doclang-wheel writes the wheel of doclang's core: ignored by git, so the
# private wheel never enters the repository.
WHEELS := wheels

# No package index has doclang's core, the core or the toolkit: the core's
# dependency on doclang's, the toolkit's on the core, and the use case's on the
# toolkit are met only when the directories are named in one command, in that
# order.
install: ## Install doclang's core from its clone, the core, the toolkit and the RAG use case (runtime deps only)
	$(BIN)python -m pip install -e $(DOCLANG_CORE_OR_STOP) -e ./$(CORE) -e ./$(TOOLKIT) -e ./$(USE_CASE)

install-dev: ## Install doclang's core from its clone, the core, the toolkit with locust + test tooling, the RAG use case and the charts package
	$(BIN)python -m pip install -e $(DOCLANG_CORE_OR_STOP) -e ./$(CORE) -e "./$(TOOLKIT)[dev]" -e ./$(USE_CASE) -e ./$(CHARTS)

# An App that installs at its start, and the image, have no clone: they take
# doclang's core from this wheel, built from the clone as it stands. Without its
# dependencies, which are on the index the install reads. Copying it to the
# Dataset (the deploy setting bootstrap_wheels names the directory) is the owner's.
# The earlier wheels of doclang's core go first, so that wheels/ holds the one
# built: the image copies every wheel there, and an App stops on several.
doclang-wheel: ## Build the wheel of doclang's core from its clone into wheels/ (git-ignored), replacing an earlier one, for an App's install and the image
	rm -f $(WHEELS)/doclang_domain_core-*.whl
	$(BIN)python -m pip wheel --no-deps --wheel-dir $(WHEELS) $(DOCLANG_CORE_OR_STOP)

# The gates over a distribution beside the toolkit, $(1) its directory, each run
# before the toolkit's own for the core and after it for the others, and its
# formatting.
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

check: ## Every quality gate over the core and its tests, then the toolkit, then the use case and its tests, then the charts package and its tests, stopping at the first failure
	$(call BESIDE_check,$(CORE))
	$(MAKE) -C $(TOOLKIT) check BIN="$(BIN)"
	$(call BESIDE_check,$(USE_CASE))
	$(call BESIDE_check,$(CHARTS))

format: ## Auto-format the core, the toolkit, the use case and the charts package with ruff --fix + black
	$(call BESIDE_format,$(CORE))
	$(MAKE) -C $(TOOLKIT) format BIN="$(BIN)"
	$(call BESIDE_format,$(USE_CASE))
	$(call BESIDE_format,$(CHARTS))

lint: ## ruff, black --check, pydoclint over the core, then the toolkit, the use case and the charts package
typecheck: ## mypy and ty over the core's source tree, then the toolkit's, the use case's and the charts package's
complexity: ## complexipy: cognitive complexity <= 10 per function, in all four
deps: ## deptry: no missing, unused or transitive dependency, in all four
deadcode: ## vulture: nothing unused outside the toolkit's vulture_whitelist.py
test: ## The toolkit's test suite (no model weights, no network)

$(HANDED):
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"

$(SHARED):
	$(call BESIDE_$@,$(CORE))
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"
	$(call BESIDE_$@,$(USE_CASE))
	$(call BESIDE_$@,$(CHARTS))

# From this directory, where the registry and the folder it explains are: the
# command reads experiments/ and writes experiments-explained/ by those names.
experiment-charts: ## Redraw experiments-explained/ (figures and their tables) from the registry in experiments/
	$(BIN)python -m experiment_charts
