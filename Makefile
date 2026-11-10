# rag_load_test — developer tasks (run from this directory)
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test ingest serve model-server loadtest loadtest-models compare ovms-export infra-up infra-down deploy deploy-embedder deploy-reranker deploy-workflow status loadtest-domino fake-domino deploy-dry-run

# A .venv in this directory is used without activating it; without one, commands
# come from the PATH as usual. Two lines because make 3.81 (macOS) looks a recipe's
# command up on its own PATH: $(BIN) is for those commands, the export is for the
# scripts and shells the recipes start.
BIN := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/,)
export PATH := $(CURDIR)/.venv/bin:$(PATH)

PYTHON ?= $(BIN)python
RUN_NAME ?= monolith
HOST ?= http://localhost:8888
USERS ?= 20
SPAWN_RATE ?= 5
RUN_TIME ?= 2m
TARGET ?= rerank
SERVE ?= reranker
PORT ?= 8011
BEHAVIOUR ?= starts

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the service (runtime deps only)
	$(PYTHON) -m pip install -e .

install-dev: ## Install with locust + test tooling
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff + black in check mode
	$(BIN)ruff check src tests locustfile.py
	$(BIN)black --check src tests locustfile.py

format: ## Auto-format with ruff --fix + black
	$(BIN)ruff check --fix src tests locustfile.py
	$(BIN)black src tests locustfile.py

typecheck: ## Run mypy over the source tree
	$(BIN)mypy src

test: ## Run the test suite (no model weights, no network)
	$(BIN)pytest

ingest: ## Index the synthetic corpus into Elasticsearch (RAG_ES_URL)
	$(BIN)rag-ingest --synthetic 200

serve: ## Run the FastAPI service with the current RAG_* environment
	$(BIN)rag-serve

model-server: ## Serve one model with FastAPI (OVMS contract): make model-server SERVE=reranker PORT=8011
	$(BIN)rag-model-server --serve $(SERVE) --host 0.0.0.0 --port $(PORT)

loadtest-models: ## Locust straight at a model server: make loadtest-models RUN_NAME=ovms-rerank HOST=http://host:8001 TARGET=rerank|embeddings
	mkdir -p results
	RAG_LOADTEST_TARGET=$(TARGET) $(BIN)locust --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME) \
		--csv results/$(RUN_NAME) --html results/$(RUN_NAME).html --only-summary

loadtest: ## Headless locust run: make loadtest RUN_NAME=monolith HOST=http://host:8888
	mkdir -p results
	$(BIN)locust --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME) \
		--csv results/$(RUN_NAME) --html results/$(RUN_NAME).html --only-summary

compare: ## Compare all runs found under results/ into results/comparison.md
	$(BIN)rag-compare $(sort $(patsubst %_stats.csv,%,$(wildcard results/*_stats.csv))) --out results/comparison.md

ovms-export: ## Export reranker + embedder to ovms/models (needs HF access)
	./ovms/export_models.sh

deploy: ## On Domino: publish every App of RAG_DEPLOY_TOPOLOGY (settings in deploy.env)
	$(BIN)rag-deploy deploy all

deploy-embedder: ## On Domino: publish (or redeploy) the rag-embedder App only
	$(BIN)rag-deploy deploy embedder

deploy-reranker: ## On Domino: publish (or redeploy) the rag-reranker App only
	$(BIN)rag-deploy deploy reranker

deploy-workflow: ## On Domino: publish (or redeploy) the rag-workflow App only
	$(BIN)rag-deploy deploy workflow

status: ## On Domino: each App's status and URL, then the workflow App's /readyz
	$(BIN)rag-deploy status

deploy-dry-run: ## Run make deploy against an in-process fake Domino: nothing is sent, prints each App's entry script
	$(BIN)rag-deploy --dry-run deploy all

fake-domino: ## Off Domino: serve a fake Apps API on :8899 (Swagger at /docs): make fake-domino BEHAVIOUR=slow
	$(BIN)rag-fake-domino --behaviour $(BEHAVIOUR)

loadtest-domino: ## On Domino: loadtest against rag-workflow: make loadtest-domino RUN_NAME=split-all
	host=$$($(BIN)rag-deploy url workflow) && \
		RAG_LOADTEST_BEARER_TOKEN="$$(curl -s http://localhost:8899/access-token)" \
		$(MAKE) loadtest HOST="$$host"

infra-up: ## Start Elasticsearch (9200) + OVMS reranker (8001) + embedder (8002)
	docker compose --env-file ovms/versions.env up -d

infra-down: ## Stop the local containers
	docker compose --env-file ovms/versions.env down
