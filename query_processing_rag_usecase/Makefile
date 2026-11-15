# query_processing_rag_usecase — the RAG use case's run targets. Every command runs
# in this directory: it finds workflow.yaml, deploy.yaml and .env here, and what it
# writes lands here (results/, data/, models/, ovms/models/, chroma/). The install
# and the gates are the repository root's: make install-dev, make check there.
.DEFAULT_GOAL := help
.PHONY: help ingest serve model-server loadtest loadtest-models compare ovms-export openvino-export calibrate infra-up infra-down deploy deploy-embedder deploy-reranker deploy-workflow status loadtest-domino deploy-dry-run

# The toolkit, beside this directory: its Locust file, its OVMS exporter and its
# local infrastructure.
TOOLKIT := ../query_processing

# The repository's one .venv, at its root beside this directory, is used without
# activating it; without one, commands come from the PATH as usual. Two lines
# because make 3.81 (macOS) looks a recipe's command up on its own PATH: $(BIN) is
# for those commands, the export is for the scripts and shells the recipes start.
VENV := $(abspath $(CURDIR)/../.venv)
BIN := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)
export PATH := $(VENV)/bin:$(PATH)

RUN_NAME ?= monolith
HOST ?= http://localhost:8888
USERS ?= 20
SPAWN_RATE ?= 5
RUN_TIME ?= 2m
TARGET ?= rerank
SERVE ?= reranker
PORT ?= 8011
LOCUSTFILE := $(TOOLKIT)/locustfile.py
COMPOSE := docker compose -f $(TOOLKIT)/docker-compose.yml --project-directory . --env-file $(TOOLKIT)/ovms/versions.env

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

ingest: ## Index the synthetic corpus into Elasticsearch (RAG_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL)
	$(BIN)rag-ingest --synthetic 200

serve: ## Run the FastAPI service with this workflow file and the current RAG_* environment
	$(BIN)rag-serve

model-server: ## Serve one component with FastAPI (OVMS contract): make model-server SERVE=reranker PORT=8011
	$(BIN)rag-model-server --serve $(SERVE) --host 0.0.0.0 --port $(PORT)

loadtest-models: ## Locust straight at a model server: make loadtest-models RUN_NAME=ovms-rerank HOST=http://host:8001 TARGET=rerank|embeddings
	mkdir -p results
	RAG_LOADTEST_TARGET=$(TARGET) $(BIN)locust -f $(LOCUSTFILE) --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME) \
		--csv results/$(RUN_NAME) --html results/$(RUN_NAME).html --only-summary

loadtest: ## Headless locust run: make loadtest RUN_NAME=monolith HOST=http://host:8888
	mkdir -p results
	$(BIN)locust -f $(LOCUSTFILE) --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME) \
		--csv results/$(RUN_NAME) --html results/$(RUN_NAME).html --only-summary

compare: ## Compare all runs found under results/ into results/comparison.md
	$(BIN)rag-compare $(sort $(patsubst %_stats.csv,%,$(wildcard results/*_stats.csv))) --out results/comparison.md

ovms-export: ## Export each embedder and reranker of the workflow file to ovms/models (needs HF access)
	$(TOOLKIT)/ovms/export_models.sh

openvino-export: ## Convert the in-process reranker + embedder to OpenVINO under models/openvino (needs HF access)
	$(BIN)rag-export-openvino

calibrate: ## Calibrate the in-process embedder + reranker on this machine; prints the environment paths that pin them
	$(BIN)rag-calibrate

deploy: ## On Domino: publish (or redeploy) every App of the topology (settings in deploy.yaml)
	$(BIN)rag-deploy deploy all

deploy-embedder: ## On Domino: publish (or redeploy) the embedder App only
	$(BIN)rag-deploy deploy embedder

deploy-reranker: ## On Domino: publish (or redeploy) the reranker App only
	$(BIN)rag-deploy deploy reranker

deploy-workflow: ## On Domino: publish (or redeploy) the workflow App only
	$(BIN)rag-deploy deploy workflow

status: ## On Domino: each App's status and instance URL, then the workflow App's /readyz
	$(BIN)rag-deploy status

deploy-dry-run: ## Run make deploy against an in-process fake Domino: nothing is sent, prints each App's settings and entry script
	$(BIN)rag-deploy --dry-run deploy all

loadtest-domino: ## On Domino: loadtest against the workflow App: make loadtest-domino RUN_NAME=split-all
	host=$$($(BIN)rag-deploy url workflow) && \
		RAG_LOADTEST_BEARER_TOKEN="$$(curl -s http://localhost:8899/access-token)" \
		$(MAKE) loadtest HOST="$$host"

infra-up: ## Start Elasticsearch (9200) + OVMS reranker (8001) + embedder (8002), serving ovms/models from here
	$(COMPOSE) up -d

infra-down: ## Stop the local containers
	$(COMPOSE) down
