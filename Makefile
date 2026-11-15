# rag_topology_loadtest — the repository's tasks. The toolkit is in query_processing/:
# install, install-dev, check, format and each gate are handed to its Makefile. Its
# run targets (serve, ingest, loadtest, deploy, ...) run from there:
# cd query_processing && make help.
.DEFAULT_GOAL := help
TOOLKIT := query_processing
HANDED := install install-dev check lint format typecheck complexity deadcode deps test
.PHONY: help $(HANDED)

# The repository's one .venv, in this directory, is used without activating it;
# without one, commands come from the PATH as usual. The toolkit's Makefile is
# given that choice as BIN, an absolute path (or empty, for the PATH), so a BIN
# given here on the command line reaches it unchanged.
BIN := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/,)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo "  The run targets are the toolkit's: make -C $(TOOLKIT) help"

install: ## Install the toolkit from query_processing/ (runtime deps only)
install-dev: ## Install the toolkit with locust + test tooling
check: ## Run every quality gate over the toolkit, stopping at the first failure
lint: ## ruff, black --check, pydoclint over the toolkit
format: ## Auto-format the toolkit with ruff --fix + black
typecheck: ## mypy and ty over the toolkit's source tree
complexity: ## complexipy: cognitive complexity <= 10 per function
deadcode: ## vulture: nothing unused outside the toolkit's vulture_whitelist.py
deps: ## deptry: no missing, unused or transitive dependency of the toolkit
test: ## The toolkit's test suite (no model weights, no network)

$(HANDED):
	$(MAKE) -C $(TOOLKIT) $@ BIN="$(BIN)"
