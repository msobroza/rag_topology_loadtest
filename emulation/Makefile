# The emulation's tasks, apart from the project's own: run `make -C emulation <target>`
# from the repository root, or `make <target>` in this folder. See README.md here.
#   In Cloud Shell: setup, init, plan, apply, ready, ssh, tunnel-domino, tunnel-locust, destroy
#   On the Workspace machine: ingest, deploy, status, loadtest, app-machines, loadtest-direct
.DEFAULT_GOAL := help
.PHONY: help setup init plan apply ready ssh tunnel-domino tunnel-locust destroy ingest corpus deploy status loadtest app-machines loadtest-direct

TF ?= terraform -chdir=terraform
BUCKET ?=
REGION ?= europe-west1
STATE_PREFIX ?= rag-emulation
RUN_NAME ?= emulation
APP_HOST ?=
PLACEMENT ?= /etc/rag-emulation/placement.json
CORPUS ?= quebec-insurance
CORPUS_ANALYZER ?= french
CORPUS_DIR = $(abspath ../data/$(CORPUS))
# Once `make corpus` has run, Locust asks that corpus's questions, not the
# synthetic English ones.
QUESTIONS = $(if $(wildcard $(CORPUS_DIR)/questions.jsonl),RAG_LOADTEST_QUESTIONS_FILE=$(CORPUS_DIR)/questions.jsonl)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Cloud Shell, once: Terraform into ~/bin and the state bucket: make setup BUCKET=<name>
	@test -n "$(BUCKET)" || { echo "BUCKET=<state bucket name> is required" >&2; exit 1; }
	./cloudshell/setup.sh "$(BUCKET)" "$(REGION)"

init: ## terraform init on the state bucket: make init BUCKET=<name>
	@test -n "$(BUCKET)" || { echo "BUCKET=<state bucket name> is required" >&2; exit 1; }
	$(TF) init -backend-config="bucket=$(BUCKET)" -backend-config="prefix=$(STATE_PREFIX)"

plan: ## terraform plan (terraform/terraform.tfvars holds project_id and git_ref)
	$(TF) plan -out=emulation.tfplan

apply: ## Apply the saved plan
	$(TF) apply emulation.tfplan

ready: ## Each machine's startup state: running, done or failed
	@eval "$$($(TF) output -raw readiness)"

ssh: ## SSH to the Workspace machine as ubuntu, through IAP
	$$($(TF) output -raw ssh_workspace)

tunnel-domino: ## Forward localhost:8899 to the fake Domino (leave running)
	$$($(TF) output -raw tunnel_fake_domino)

tunnel-locust: ## Forward localhost:8089 to Locust's web UI (leave running)
	$$($(TF) output -raw tunnel_locust)

destroy: ## Delete everything Terraform created (the state bucket stays)
	$(TF) destroy

ingest: ## Workspace: index the synthetic English corpus into the emulation's Elasticsearch (the project's make ingest)
	$(MAKE) -C .. ingest

corpus: ## Workspace: fetch a French corpus and index it instead (one corpus per index: this or ingest): make corpus CORPUS=quebec-insurance
	cd .. && rag-fetch-corpus $(CORPUS) --out $(CORPUS_DIR)
	cd .. && RAG_ES_TEXT_ANALYZER=$(CORPUS_ANALYZER) rag-ingest --corpus $(CORPUS_DIR)/corpus.jsonl

deploy: ## Workspace: publish every App through the fake Domino (the project's make deploy)
	$(MAKE) -C .. deploy

status: ## Workspace: each App's status and URL, then the workflow App's /readyz
	$(MAKE) -C .. status

loadtest: ## Workspace: Locust through the fake Domino's proxy: make loadtest RUN_NAME=split-all
	$(QUESTIONS) $(MAKE) -C .. loadtest-domino RUN_NAME=$(RUN_NAME)

app-machines: ## Workspace: what answers /readyz on port 8888 of each pool machine
	@python3 -c 'import json, sys; [print(m) for t in json.load(open(sys.argv[1]))["tiers"] for m in t["machines"]]' $(PLACEMENT) \
		| while read -r machine; do \
			printf '%s: ' "$$machine"; curl -s --max-time 3 "http://$$machine:8888/readyz" || printf 'no App'; echo; \
		done

loadtest-direct: ## Workspace: Locust straight at the workflow App, bypassing the proxy: make loadtest-direct APP_HOST=<address> RUN_NAME=split-all-direct
	@test -n "$(APP_HOST)" || { echo "APP_HOST=<the workflow App's machine (make app-machines)> is required" >&2; exit 1; }
	$(QUESTIONS) $(MAKE) -C .. loadtest HOST=http://$(APP_HOST):8888 RUN_NAME=$(RUN_NAME)
