# query_processing_rag_usecase — the RAG use case's run targets. Every command runs
# in this directory: it finds workflow.yaml, deploy.yaml and .env here, and what it
# writes lands here (results/, data/, models/, ovms/models/, chroma/, the router's
# pipeline file). The install and the gates are the repository root's: make
# install-dev, make check there.
.DEFAULT_GOAL := help
.PHONY: help ingest train-router serve model-server loadtest loadtest-models compare ovms-export openvino-export calibrate infra-up infra-down deploy deploy-embedder deploy-reranker deploy-llm deploy-workflow status loadtest-domino deploy-dry-run tiers environments record-answers record-answers-domino compare-answers pin-answers

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

# The workflow file declares two embedders, so qp-ingest is told which one
# fills the store: `embedder`, whose vectors the retrieve stage searches with.
ingest: ## Index the synthetic corpus into Elasticsearch (QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL)
	$(BIN)qp-ingest --synthetic 200 --embedder embedder

# No --allow-fake-embedder: a pipeline trained on the fake embedder's vectors
# would pass for one trained on the real embedder, so the command refuses the
# router's embedder on fake; only the tests pass the flag.
train-router: ## Train the router again on the utterances of routes.yaml, after a change start-up refuses: rewrites the committed router_pipeline.joblib and its manifest, to commit together (needs the router's embedder's model, refused on fake)
	$(BIN)qp-rag-train-router

serve: ## Run the FastAPI service with this workflow file and the current QP_* environment
	$(BIN)qp-serve

model-server: ## Serve one component with FastAPI (OVMS contract): make model-server SERVE=reranker PORT=8011
	$(BIN)qp-model-server --serve $(SERVE) --host 0.0.0.0 --port $(PORT)

loadtest-models: ## Locust straight at a model server: make loadtest-models RUN_NAME=ovms-rerank HOST=http://host:8001 TARGET=rerank|embeddings
	mkdir -p results
	QP_LOADTEST_TARGET=$(TARGET) $(BIN)locust -f $(LOCUSTFILE) --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME) \
		--csv results/$(RUN_NAME) --html results/$(RUN_NAME).html --only-summary

loadtest: ## Headless locust run: make loadtest RUN_NAME=monolith HOST=http://host:8888
	mkdir -p results
	$(BIN)locust -f $(LOCUSTFILE) --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME) \
		--csv results/$(RUN_NAME) --html results/$(RUN_NAME).html --only-summary

compare: ## Compare all runs found under results/ into results/comparison.md
	$(BIN)qp-compare $(sort $(patsubst %_stats.csv,%,$(wildcard results/*_stats.csv))) --out results/comparison.md

record-answers: ## Send each load-test request once and record the answers: make record-answers RUN_NAME=monolith HOST=http://host:8888 (QP_LOADTEST_QUERY_WEIGHT=0: /retrieve alone)
	mkdir -p results
	$(BIN)qp-record-answers --host $(HOST) --out results/$(RUN_NAME)

# The answer records under results/, sorted, BASELINE's first when it names one.
ANSWER_RECORDS = $(sort $(patsubst %_answers.json,%,$(wildcard results/*_answers.json)))
ANSWER_RUNS = $(strip $(if $(BASELINE),results/$(BASELINE)) $(filter-out results/$(BASELINE),$(ANSWER_RECORDS)))
compare-answers: ## Compare every answer record under results/ with the first (BASELINE=<run> picks it) into results/answers-comparison.md
	$(BIN)qp-compare-answers $(ANSWER_RUNS) --out results/answers-comparison.md

# The test that compares the workflow app here with its pinned answer records,
# run to make them again instead: in process, every component on its fake
# backend, no socket, nothing downloaded.
pin-answers: ## Make the pinned answer records of this workflow on fake backends again, after a change that alters the answers on purpose (commit them)
	$(BIN)pytest tests/answer_records/test_pinned_answer_records.py --pin-answers

ovms-export: ## Export each embedder and reranker of the workflow file, the router's embedder included, to ovms/models (needs HF access)
	$(TOOLKIT)/ovms/export_models.sh

openvino-export: ## Convert each embedder and reranker, the router's embedder included, to OpenVINO under models/openvino (needs HF access)
	$(BIN)qp-export-openvino

calibrate: ## Calibrate each in-process embedder and reranker, the router's embedder included, on this machine; prints the environment paths that pin them
	$(BIN)qp-calibrate

deploy: ## On Domino: publish (or redeploy) every App of the topology (settings in deploy.yaml)
	$(BIN)qp-deploy deploy all

deploy-embedder: ## On Domino: publish (or redeploy) the embedder App only
	$(BIN)qp-deploy deploy embedder

deploy-reranker: ## On Domino: publish (or redeploy) the reranker App only
	$(BIN)qp-deploy deploy reranker

deploy-llm: ## On Domino: publish (or redeploy) the LLM's App only (an alias that says llm: served)
	$(BIN)qp-deploy deploy llm

deploy-workflow: ## On Domino: publish (or redeploy) the workflow App only
	$(BIN)qp-deploy deploy workflow

status: ## On Domino: each App's status and instance URL, then the workflow App's /readyz
	$(BIN)qp-deploy status

deploy-dry-run: ## Run make deploy against an in-process fake Domino: nothing is sent, prints each App's settings and entry script
	$(BIN)qp-deploy --dry-run deploy all

tiers: ## On Domino: the hardware tiers with their size, a list to move through in a terminal; make tiers ARGS="cpu-16 --details" for every field
	$(BIN)qp-deploy tiers $(ARGS)

environments: ## On Domino: the environments, a list to move through in a terminal; make environments ARGS="<name> --table" for its revisions and their build dates
	$(BIN)qp-deploy environments $(ARGS)

loadtest-domino: ## On Domino: loadtest against the workflow App, each request with the run token: make loadtest-domino RUN_NAME=split-all
	host=$$($(BIN)qp-deploy url workflow) && \
		QP_LOADTEST_KEY__SOURCE=domino $(MAKE) loadtest HOST="$$host"

record-answers-domino: ## On Domino: record the workflow App's answers, each request with the run token: make record-answers-domino RUN_NAME=split-all
	host=$$($(BIN)qp-deploy url workflow) && \
		QP_LOADTEST_KEY__SOURCE=domino $(MAKE) record-answers HOST="$$host"

infra-up: ## Start Elasticsearch (9200) + OVMS reranker (8001) + embedder (8002), serving ovms/models from here
	$(COMPOSE) up -d

infra-down: ## Stop the local containers
	$(COMPOSE) down
