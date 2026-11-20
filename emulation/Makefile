# The emulation's tasks, apart from the project's own: run `make -C emulation <target>`
# from the repository root, or `make <target>` in this folder. See README.md here.
#   In Cloud Shell: setup, init, plan, apply, wheel-save, ready, ssh, tunnel-domino, tunnel-locust, destroy
#   On the Workspace machine: ingest, corpus-files, corpus, deploy, status, loadtest,
#     app-machines, loadtest-direct, and the cache bucket's corpus-save, corpus-restore,
#     models-save, results-save, cache-status
.DEFAULT_GOAL := help
.PHONY: help setup init plan apply wheel-save ready ssh tunnel-domino tunnel-locust destroy ingest corpus-files corpus corpus-save corpus-restore deploy status loadtest app-machines loadtest-direct models-save results-save cache-status

TF ?= terraform -chdir=terraform
BUCKET ?=
REGION ?= europe-west1
STATE_PREFIX ?= rag-emulation
RUN_NAME ?= emulation
APP_HOST ?=
PLACEMENT ?= /etc/rag-emulation/placement.json
CORPUS ?= quebec-insurance
CORPUS_ANALYZER ?= french
# The use case the emulation deploys: its commands and make targets run in its
# directory, where they find its files, and what they write (data/, results/) is
# there.
USE_CASE = ../query_processing_rag_usecase
CORPUS_DIR = $(abspath $(USE_CASE)/data/$(CORPUS))
# Once `make corpus` has run, Locust asks that corpus's questions, not the
# synthetic English ones.
QUESTIONS = $(if $(wildcard $(CORPUS_DIR)/questions.jsonl),QP_LOADTEST_QUESTIONS_FILE=$(CORPUS_DIR)/questions.jsonl)
# An index snapshot is named after its corpus and analyzer (index_snapshot.sh adds the embedder).
SNAPSHOT_PREFIX = $(CORPUS)-$(CORPUS_ANALYZER)
# FORCE=1: models-save replaces the models the cache bucket already has.
FORCE ?=
# The pool machines' addresses, one per line, from the emulated Domino's placement file.
POOL_MACHINES = python3 -c 'import json, sys; [print(m) for t in json.load(open(sys.argv[1]))["tiers"] for m in t["machines"]]' $(PLACEMENT)

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

# The wheel of doclang's core, which the machines install and cannot build (they have
# no clone of document_doclang): `make doclang-wheel` writes it into the repository's
# wheels/, and the machines copy every wheel under this prefix of the cache bucket at
# boot (common.sh's CACHE_WHEELS_PREFIX).
WHEELS = ../wheels
CACHE_WHEELS_PREFIX = wheels/

wheel-save: ## Where the wheel was built (make doclang-wheel at the root): copy it to the cache bucket, right after the first apply
	@ls $(WHEELS)/doclang_domain_core-*.whl >/dev/null 2>&1 \
		|| { echo "no wheel of doclang's core in $(abspath $(WHEELS)): run make doclang-wheel at the repository's root" >&2; exit 1; }
	gcloud storage cp $(WHEELS)/doclang_domain_core-*.whl "gs://$$($(TF) output -raw cache_bucket)/$(CACHE_WHEELS_PREFIX)"

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

ingest: ## Workspace: index the synthetic English corpus into the emulation's Elasticsearch (the use case's make ingest)
	$(MAKE) -C $(USE_CASE) ingest

corpus-files: ## Workspace: fetch a corpus's files (its questions.jsonl included) without indexing it: make corpus-files CORPUS=quebec-insurance
	cd $(USE_CASE) && qp-fetch-corpus $(CORPUS) --out $(CORPUS_DIR)

corpus: corpus-files ## Workspace: fetch a French corpus and index it instead (one corpus per index: this or ingest): make corpus CORPUS=quebec-insurance
	cd $(USE_CASE) && QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__TEXT_ANALYZER=$(CORPUS_ANALYZER) qp-ingest --corpus $(CORPUS_DIR)/corpus.jsonl

corpus-save: ## Workspace: snapshot the index into the cache bucket, named after corpus, analyzer and embedder
	cd $(USE_CASE) && $(CURDIR)/machine/index_snapshot.sh save "$(SNAPSHOT_PREFIX)"

corpus-restore: corpus-files ## Workspace: the index back from the cache bucket, in place of make corpus
	cd $(USE_CASE) && $(CURDIR)/machine/index_snapshot.sh restore "$(SNAPSHOT_PREFIX)"

deploy: ## Workspace: publish every App through the fake Domino (the use case's make deploy)
	$(MAKE) -C $(USE_CASE) deploy

status: ## Workspace: each App's status and URL, then the workflow App's /readyz
	$(MAKE) -C $(USE_CASE) status

loadtest: ## Workspace: Locust through the fake Domino's proxy: make loadtest RUN_NAME=split-all
	$(QUESTIONS) $(MAKE) -C $(USE_CASE) loadtest-domino RUN_NAME=$(RUN_NAME)

app-machines: ## Workspace: what answers /readyz on port 8888 of each pool machine
	@$(POOL_MACHINES) \
		| while read -r machine; do \
			printf '%s: ' "$$machine"; curl -s --max-time 3 "http://$$machine:8888/readyz" || printf 'no App'; echo; \
		done

loadtest-direct: ## Workspace: Locust straight at the workflow App, bypassing the proxy: make loadtest-direct APP_HOST=<address> RUN_NAME=split-all-direct
	@test -n "$(APP_HOST)" || { echo "APP_HOST=<the workflow App's machine (make app-machines)> is required" >&2; exit 1; }
	$(QUESTIONS) $(MAKE) -C $(USE_CASE) loadtest HOST=http://$(APP_HOST):8888 RUN_NAME=$(RUN_NAME)

# The clone has the same path on every machine, and the Workspace's ~/.ssh/config
# knows the pool machines.
models-save: ## Workspace: upload this machine's and every pool machine's exported models (FORCE=1 replaces)
	FORCE=$(FORCE) ./machine/cache_bucket.sh models-save
	@$(POOL_MACHINES) \
		| while read -r machine; do \
			echo "$$machine:"; \
			ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$$machine" "FORCE=$(FORCE) $(CURDIR)/machine/cache_bucket.sh models-save" || exit 1; \
		done

results-save: ## Workspace: upload query_processing_rag_usecase/results/ to the cache bucket as one archive and print its gs:// URL
	./machine/cache_bucket.sh results-save $(abspath $(USE_CASE)/results)

cache-status: ## Workspace: the cache bucket's objects, then the index snapshots
	./machine/cache_bucket.sh status
	./machine/index_snapshot.sh list
