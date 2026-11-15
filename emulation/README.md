# Emulation: an emulated Domino on Google Cloud

**Applied on 14 November 2026.** That day `plan` and `apply` created the machines,
the three topologies were deployed through the emulated Domino and measured on
real models, and two later sittings tried the cache bucket. "Verified on 14
November 2026" and "Not verified" at the end say what ran and what has not;
`tests/emulation/` still checks the files only by parsing them.

The emulation is temporary: it lets the owner rehearse a Domino deployment, and
compare topologies, before the real Domino is available. All of it lives in this
folder, `tests/emulation/` and the emulation's own subpackage, so that it can be
removed without touching the rest (see "Removing the emulation").

Terraform creates what already exists before anyone deploys on a real Domino (ADR
`docs/adr/0001-emulation-platform-and-app-placement.md`). Then the project's
unchanged `rag-deploy` publishes the Apps through the **emulated Domino**
(`rag-domino-emulation`), which places each App on free machines of its hardware
tier over SSH (one machine per replica, see "Replicas") and proxies every call to
it. Terraform never creates an App.

```
                    Cloud Shell (terraform, gcloud)
                              |  IAP only: SSH, ports 8899 and 8089
 +--------------------------- v ---------------------------------------------+
 | network <name>-net, subnet 10.10.0.0/24, no public address, Cloud NAT out |
 |                                                                            |
 |  <name>-workspace (n2-standard-8)        <name>-cpu-8-1..3 (n2-standard-8) |
 |   rag-domino-emulation :8899  -- ssh, :8888 -->  one App each, :8888       |
 |   Workspace for ubuntu: make,  <-- :8899 ----   localhost:8899 forwarder   |
 |   rag-deploy, rag-ingest, locust                (socat), OVMS tarball      |
 |            \                                         /                     |
 |             +------> <name>-es (n2-standard-4) <-----+                     |
 |                      Elasticsearch 9.0.0 in Docker, :9200                  |
 +----------------------------------------------------------------------------+
        |  each machine's service account: roles/storage.objectUser on it only
        v
  bucket <project id>-<name>-cache: exported models, index snapshots, results

  Spot (spot = true): the pool machines and Elasticsearch. Never the Workspace.
```

## What gets created

| Resource | Default | Why |
|---|---|---|
| APIs | Compute Engine, IAP, IAM, Secret Manager, Cloud Storage | left enabled on destroy |
| Network, subnet, router, Cloud NAT | `10.10.0.0/24` in `europe-west1` | outbound internet (apt, pip, the Hugging Face Hub, GitHub, Docker images, an LLM API) without public addresses |
| Firewall rules | IAP range `35.235.240.0/20` to ports 22, 8899, 8089; Workspace to pool on 22 and 8888; pool to Workspace on 8899; Workspace and pool to Elasticsearch on 9200 | nothing else gets in |
| Workspace machine | `n2-standard-8`, 50 GB | the emulated Domino on 8899 and the Workspace (Locust runs here) |
| Elasticsearch machine | `n2-standard-4`, 50 GB, heap 4g, Spot | single node, security off, data on its disk |
| Pool machines | tier `cpu-8`: 3 x `n2-standard-8`, 50 GB each, Spot | one App, or one replica of an App, per machine: by default the workflow, a reranker, an embedder |
| Two reserved internal addresses | Workspace and Elasticsearch | stable `deploy.yaml` |
| Three service accounts | no project role; `roles/storage.objectUser` on the cache bucket only, and for Elasticsearch's `roles/storage.legacyBucketReader` on it too | the Workspace's may read two secrets |
| Cache bucket `<project id>-<name>-cache` | private (uniform access, public access prevention), in the region | what outlives the machines (see "The cache bucket"); `destroy` deletes it |
| Secret `<name>-ssh-key` | ED25519 private key | the emulated Domino's key to the pool machines |

Every machine runs Ubuntu 22.04 LTS, **deletes itself after `max_run_hours`
(default 8)** and carries the label `rag-emulation=<name>`, as do its disk, the
addresses, the secret and the cache bucket. With `spot = true` (the default) the
pool and Elasticsearch machines are Spot VMs (see "Spot machines").

**Cost at the default sizes (approximate, europe-west1, on-demand):** an
`n2-standard-8` is about $0.43/h and an `n2-standard-4` about half that, so four
8-vCPU machines and one 4-vCPU machine come to about **$1.95/h**; disks (250 GB
balanced), Cloud NAT and its traffic add roughly $0.10/h. About **$16 for an
8-hour day**. These are on-demand prices: with `spot = true` the pool and
Elasticsearch machines are billed at Spot prices instead, which Google states are
60 to 91% below on-demand and which vary; only the Workspace machine stays at the
on-demand price. The cache bucket holds a few GB of Standard storage. Check the
current prices before relying on these numbers. Budget alerts do not cap Compute
Engine spend: the self-deletion does.

## Steps

### 1. Once, in Google Cloud Shell

Cloud Shell no longer ships Terraform. From a clone of this repository:

```bash
gcloud config set project <project id>
make -C emulation setup BUCKET=<globally unique bucket name>   # Terraform into ~/bin, private versioned state bucket
```

Optional, for the generation endpoint's API key (a Project environment variable):

```bash
printf 'RAG_GENERATION_API_KEY=%s\n' "<key>" | gcloud secrets create rag-emu-project-env --data-file=-
```

### 2. Create the emulation

```bash
cp emulation/terraform/terraform.tfvars.example emulation/terraform/terraform.tfvars
# set project_id and git_ref (a pushed branch, tag or commit: the machines clone from GitHub);
# project_env_secret = "rag-emu-project-env" if you created it
make -C emulation init BUCKET=<bucket name>
make -C emulation plan
make -C emulation apply
make -C emulation ready         # repeat until every machine says done (about 2 min)
```

`ready` reads each machine's guest attribute `rag-emulation/startup`
(`running`, `done` or `failed`). On `failed`, read the machine's log:
`gcloud compute instances get-serial-port-output <machine> --zone europe-west1-b`.

### 3. Work in the Workspace

```bash
make -C emulation ssh           # gcloud compute ssh ubuntu@<name>-workspace ... --tunnel-through-iap
```

The shell starts in the repository (`/repos/rag_topology_loadtest`, the same path
on every machine) with the project's virtualenv on `PATH`, and with
`DOMINO_API_PROXY=http://localhost:8899`, `DOMINO_PROJECT_ID` and `RAG_ES_URL`
set, as a Domino Workspace would have them (the Project environment variables are
exported too). `deploy.yaml` is already written for this emulation; edit it as on
Domino (topology, per-App blocks), or override with `RAG_DEPLOY_TOPOLOGY` and
`RAG_DEPLOY_MODEL_SERVER`.

The first sitting builds the index and exports the models, then keeps both in the
cache bucket:

```bash
make -C emulation corpus               # the French corpus: rag-fetch-corpus quebec-insurance, then rag-ingest with the french analyzer
make -C emulation ingest               # or: the synthetic English corpus (one corpus per index: pick one)
make -C emulation deploy               # rag-deploy deploy all, through the emulated Domino
make -C emulation corpus-save models-save   # once the Apps are up: the index snapshot, every machine's exported models
```

A later sitting, on fresh machines, restores the index instead of building it. The
models are already on the machines: each one restored them from the bucket at boot.

```bash
make -C emulation corpus-restore       # in place of corpus: the corpus's files, then the index from its snapshot
make -C emulation deploy
make -C emulation status               # each App's status and URL, then the workflow's /readyz
make -C emulation loadtest RUN_NAME=split-all             # through the emulated Domino's proxy
make -C emulation app-machines                            # which pool machine runs the workflow
make -C emulation loadtest-direct APP_HOST=<address> RUN_NAME=split-all-direct   # bypassing the proxy
make compare
make -C emulation results-save         # results/ as one archive in the bucket; prints its gs:// URL
make -C emulation cache-status         # the bucket's objects, then the index snapshots
```

To bring the archive to Cloud Shell:

```bash
gcloud storage cp gs://<bucket>/results/<UTC time>.tar.gz .
```

Every call to an App passes the emulated Domino's proxy, a single Python process; the run
that bypasses it shows the proxy's share of the numbers (ADR 0001). In the first
sitting the first start of each App is slow: it downloads and converts its models
(the embedder is 2.3 GB). Every App keeps its models in the machine's dataset folder
(`dataset_path`, standing in for a Domino Dataset), which `models-save` caches. Each
pool machine installs the Apps' dependencies once at boot, the same two `uv`
commands as `domino/app.sh`, so an App's own install only reads uv's cache.

From Cloud Shell, `make -C emulation tunnel-domino` forwards `localhost:8899` to
the emulated Domino (Swagger at `/docs`, via Cloud Shell's web preview) and
`make -C emulation tunnel-locust` forwards Locust's web UI on 8089 (for a
`locust` started without `--headless` in the Workspace).

### 4. Remove it

```bash
make -C emulation destroy
```

The cache bucket goes, with everything in it. The state bucket stays (delete it
with `gcloud storage rm -r gs://<bucket>` when done for good), and so does the
optional `project_env_secret`, which is yours.

## A real app behind the emulated Domino, on one machine

No cloud is needed to see the proxy at work. On Domino a proxy sits in front of
every App and Agent (Agents use the same hosting): it checks who is calling
before the request reaches the app, and tells the app who that is. The emulated
Domino (`rag-domino-emulation`, installed by the `emulation` extra) does the same on both URL shapes of an App,
`/apps/<app id>/...` and `/<owner>/<project>/r/notebookSession/<instance id>/...`,
for every method and sub-path, and forwards the request to a real app on your
machine when `--upstream` names one for that App:

```bash
make model-server SERVE=reranker PORT=8011           # a real model server
rag-domino-emulation --upstream rag-reranker=http://localhost:8011
DOMINO_API_PROXY=http://localhost:8899 DOMINO_PROJECT_ID=local make deploy-reranker
url=$(DOMINO_API_PROXY=http://localhost:8899 DOMINO_PROJECT_ID=local rag-deploy url reranker)
curl -i "$url/v3/models/crossencoder-camembert-base-mmarcoFR"                    # 401
curl -i -H "Authorization: Bearer $(curl -s localhost:8899/access-token)" \
  "$url/v3/models/crossencoder-camembert-base-mmarcoFR"                          # 200, from :8011
```

With `RAG_RERANK_API_URL=$url/v3/rerank` and `RAG_RERANK_API_KEY=domino`, a
workflow app on the same machine fetches its token from the emulated Domino and
calls the reranker through the proxy, as it would on Domino.

| The request | What the emulated Domino does |
|---|---|
| `Authorization: Bearer <the token /access-token serves>` or `X-Domino-Api-Key: fake-api-key` | the caller is `fake-owner`; the request goes on |
| no credential, or one the emulated Domino does not know | `401` with `WWW-Authenticate` (`--unauthenticated redirect`: `302` to a login URL) |
| an App or instance the emulated Domino does not have | `404` |
| an App with an upstream | forwarded (method, sub-path, query, body), the answer relayed; `502` naming the upstream when it does not answer |
| an App without an upstream | `/readyz` answers as a stand-in; any other path is a `502` saying how to name an upstream |

What the app receives, in place of the caller's own `Authorization`,
`X-Domino-Api-Key` and `domino-username`:

* `domino-username: fake-owner`;
* `Authorization: Bearer <JWT>`, the identity token of Domino's secure identity
  propagation: RS256, audience `apps`, claims `sub`, `preferred_username`,
  `email`, `given_name` and `family_name`, verifiable with the keys at
  `/auth/realms/DominoRealm/protocol/openid-connect/certs` (`--no-identity-jwt`
  leaves it out, as a Domino with that feature off does);
* for a version published with `extendedIdentityPropagationToAppsEnabled`, the
  caller's own token as `Authorization: Bearer` and no `domino-username`.

The path prefix is stripped before forwarding; `--keep-prefix` sends the full
path, which is the case `RAG_OVMS_PREFIX_PROXY=1` exists for.
`--forward-timeout` bounds a forwarded call (30 s). The fake has one user and
does not apply `visibility` or access lists: it stores and echoes them.

Domino's documentation states the proxy, the two headers, the token's algorithm,
audience and claims, the keys URL, both URL shapes, and the run token at
`/access-token`. It does not state which credentials the proxy takes from a
script, what an unauthenticated call gets back, whether the caller's own headers
are removed, or whether the prefix is stripped: there the emulated Domino follows the Domino
API's rules (bearer token or API key, `401`), and its Swagger page says which is
which.

## Replicas

On Domino an App's replicas are pods of one instance. The emulated Domino runs each
replica on a machine of its own. In `deploy.yaml`, an App block with
`autoscaling: {min_replicas: N, max_replicas: N}` gives that App N machines of its
hardware tier; without it, the App gets one. A tier needs a free machine for every
replica: an App asking for more machines than its tier has is refused, and one
that finds too few free stays `Queued` until enough are free. Add machines to the
tier (`hardware_tiers` in `terraform.tfvars`) before you deploy more replicas.

The App still has one instance and one URL. Its proxy spreads the requests over
the machines by one of two rules, which are the two that Kubernetes uses:

- `round-robin` (the default): each request goes to the next machine in turn, as
  an ingress controller does.
- `connection-random`: each caller connection gets one machine at random and keeps
  it, as a Kubernetes Service does.

Domino says that requests spread across an App's pods when session affinity is
off. It does not say by which rule. To change the rule on a running emulation, add
`--replica-balancing connection-random` to the `ExecStart` line of the
`rag-domino-emulation` unit on the Workspace, then run `sudo systemctl
daemon-reload` and `sudo systemctl restart rag-domino-emulation`. A restart makes
the emulated Domino forget every App, so deploy again.

Not emulated:

- Pods added or removed during a run: only `min_replicas` counts. `max_replicas`,
  the CPU target and session affinity are stored, never acted on.
- Session affinity: a caller is never kept on one replica by a cookie.
- One disk shared by the replicas: each machine keeps its own dataset folder.

## The cache bucket

The bucket outlives the machines, so that a later sitting starts in minutes instead
of exporting the models and embedding the corpus again. It holds:

- `models/openvino/<folder>.tar` and `models/ovms-models.tar`: the exported models,
  one uncompressed tar per folder of the dataset folder. `models-save` uploads them
  from the Workspace and, over SSH, from every pool machine; `ovms-models` only once
  it holds both OVMS configs. At boot each pool machine restores every model, the
  Workspace the OpenVINO ones (what `rag-ingest` loads); a folder already on the
  machine is left alone.
- `elasticsearch/`: Elasticsearch's index snapshots, written and read by
  Elasticsearch itself (a `gcs` repository, with the Elasticsearch machine's service
  account). A snapshot's name carries the corpus, the analyzer and the embedder
  (`quebec-insurance-french-snowflake-snowflake-arctic-embed-l-v2.0`): an index is
  only valid for the embedder and the analyzer it was built with, so a change of
  either misses the cache and `corpus-restore` says to build and save it.
- `results/<UTC time>.tar.gz`: the archives `results-save` uploads.

Models are keyed by folder name, not by how they were exported. `models-save` keeps
an object the bucket already has. After a change to the export recipe of the same
model, replace them with `make -C emulation models-save FORCE=1`, or empty the
prefix first: `gcloud storage rm -r gs://<bucket>/models`. A bucket error fails the
boot (`make ready` says `failed`); an empty bucket is the first sitting's normal
state. `make -C emulation destroy` deletes the bucket with everything in it.

Measured on 14 November 2026, in two sittings on `split-all` with the French corpus
(1,592 passages), Spot pool machines and a 4-CPU Workspace:

| | First sitting (empty bucket) | Second sitting (fresh machines) |
|---|---|---|
| Every machine ready after `apply` | 2 min | 2 min |
| The index | `corpus`: 20 min 15 s | `corpus-restore`: 3 s |
| `deploy` of the three Apps | 1 min 47 s (the embedder's export: 50 s) | 56 s, no export |
| From the end of `apply` to a workflow ready to measure | 23 min | 3 min 18 s |
| `/retrieve`, 5 users, 2 minutes | 87 requests, 0.73/s, median 6.2 s | 87 requests, 0.73/s, median 6.3 s |

`corpus-save` took 3 s and `models-save` 22 s (2.3 GB and 0.4 GB). At boot a pool
machine restored both models in about 15 s and installed the dependencies in about
20 s. Before this cache, on the same day, the first measurement came 18.5 minutes
after `apply`, and the first deploy took about 7 minutes.

## Spot machines

With `spot = true` (the default) the pool machines and the Elasticsearch machine are
Spot VMs. Google can take one back at any time; it is then deleted, as at the end of
`max_run_hours`. The Workspace machine is never Spot: it holds the results and the
emulated Domino's state. When a machine is gone:

1. `make -C emulation plan apply` creates it again.
2. A pool machine: it comes back with a new address; follow the instruction about
   `placement.json` below, then deploy the App again. Its models come back from the
   bucket at boot.
3. The Elasticsearch machine: its address is reserved, but its index is gone:
   `make -C emulation corpus-restore`.

When the zone has no Spot capacity, set `spot = false` in `terraform.tfvars`, then
`plan` and `apply`.

## The 8-hour self-deletion and Terraform's state

After `max_run_hours` Compute Engine deletes each machine; Terraform is not told.
The network, NAT, firewall, addresses, service accounts, secret and cache bucket
remain (they cost next to nothing; the NAT gateway a little). The next `plan` shows
the machines as to be created again: `make -C emulation plan apply` brings back
fresh machines with fresh disks, so the Workspace's results and `deploy.yaml` edits
are gone, and the Apps must be deployed again. The index and the models come back
from the cache bucket (`corpus-restore`; the models at boot) if they were saved.
Save `results/` before the deadline (`make -C emulation results-save`).
`destroy` works at any time, with or without the machines.

A startup script runs at every boot. A pool machine re-created alone gets a new
address: re-apply, then reboot the Workspace machine (or run `sudo
google_metadata_script_runner startup` on it) so the emulated Domino reads the new
`placement.json`.

## Variables

Only `project_id` and `git_ref` are required; `terraform/variables.tf` describes
each one and `terraform/terraform.tfvars.example` lists them with their defaults.
`hardware_tiers` maps each tier id to `{machine_type, count, default, disk_gb}`,
with exactly one default tier; the cores and memory the emulated Domino reports for a
tier come from its machine type. `spot` (default `true`) makes the pool and
Elasticsearch machines Spot VMs. `cache_bucket` names the cache bucket (default `""`:
`<project_id>-<name>-cache`). `dataset_path` (default
`/domino/datasets/local/emulation`) is the folder on the Workspace and every pool
machine that stands in for a Domino Dataset: the Apps and the Workspace's ingestion
keep their exported models in it.

## Verified on 14 November 2026

One sitting, before the cache bucket and the Spot machines existed: five
on-demand machines (a 4-CPU Workspace through `workspace_machine_type`, three
`n2-standard-8` pool machines, Elasticsearch on an `n2-standard-4`), FastAPI model
servers, the French corpus. It showed:

- `plan` (5 to add) and `apply`; every machine reported `done` about 2 minutes after
  `apply`, the Workspace and the pool machines with their clone at the commit id
  that `git_ref` named.
- The startup scripts on Ubuntu 22.04, including what had only been parsed until
  then: the `ubuntu` user and its passwordless `sudo` (the emulated Domino starts
  each App with `sudo systemd-run`), uv's release tarball, Secret Manager over the
  REST API with the metadata server's token (the SSH key), the socat forwarder,
  Docker from Ubuntu's `docker.io` package and Elasticsearch on its data folder.
- `scheduling.max_run_duration` with `instance_termination_action = "DELETE"` on
  `STANDARD` machines: the API accepted it. The deletion itself was not observed
  (see below).
- `rag-domino-emulation` with the flags the Workspace's startup script gives it:
  it read `placement.json`, placed each App on a free pool machine over SSH, and
  proxied every call. The workflow App fetched its token from `localhost:8899`
  through the forwarder and called the model servers through the proxy.
- `rag-fetch-corpus` and `rag-ingest` with the French analyzer (1,223 documents,
  1,592 passages).
- The three topologies, each deployed in turn on the same machines: `deploy` of
  `split-all` was done about 11 minutes after `apply` and the corpus ingested
  after 19, the two running side by side, then `monolith` and `split-reranker`.
- The proxy under load: six `/retrieve` runs through it (`split-all` at 5, 10, 15
  and 20 users, `monolith` and `split-reranker` at 10), with no failed request.

The two sittings of that evening, on Spot machines with the cache bucket, are
measured under "The cache bucket".

## Not verified

Not shown by the runs of 14 November 2026:

- `terraform validate` on its own, and which provider versions the ranges
  `>= 6.0, < 8.0` (`google`) and `~> 4.0` (`tls`) picked: `plan` and `apply` ran,
  but the versions were not recorded.
- The deletion at the end of `max_run_hours` of an on-demand (`STANDARD`) machine
  with the default `automatic_restart`: every machine that day was deleted by hand
  before its deadline.
- `enable-oslogin = FALSE` on the machines worked in a project that does not
  require OS Login; an organisation policy that requires it would break both the
  emulated Domino's SSH and `gcloud compute ssh ubuntu@...`.
- The cost figures are approximate and were not read from a current price list.
- The Cloud Shell setup script (pinned Terraform 1.12.2, the bucket flags): the
  notes of that day do not show it running.
- `rag-domino-emulation` handing the Apps a Project environment variable from
  `project.env`, and `/query` (generation) through the emulation: only `/retrieve`
  was measured.
- OVMS: no OVMS App was deployed, so the OVMS bundle on the pool machines, the
  OVMS export, and whether 50 GB per pool machine is enough for it are untried.
  The FastAPI model servers fit.
- Of the cache bucket and the Spot machines (the rest ran on 14 November 2026, see
  "The cache bucket"):
  - a Spot machine taken back by Google, and the recovery steps of "Spot machines";
  - the self-deletion of a Spot machine at the end of `max_run_hours` (the machines
    of both sittings were deleted by hand), and an apply with `spot = false`;
  - `models/ovms-models.tar`: no OVMS App was deployed, so no OVMS export was saved
    or restored;
  - `models-save FORCE=1`, and the boot's wait for a grant that has not applied yet
    (the first bucket call succeeded at once);
  - whether Elasticsearch's repository needs `roles/storage.legacyBucketReader`: it
    worked with both roles, and was not tried with one.

## Removing the emulation

Delete:

- `emulation/` (run `make -C emulation destroy` first if anything is still up)
- `tests/emulation/`
- `src/fake_domino/emulation/` and its tests `tests/fake_domino/test_fake_domino_emulation_*`
- the `rag-domino-emulation` script entry and the `emulation` extra in `pyproject.toml`
- the `pyjwt`, `cryptography` and `python-hcl2` dev dependencies in `pyproject.toml`
- in the main `README.md`: the paragraph pointing to this folder, and the lines marked
  TEMPORARY in its code layout
- the "Emulation" block of `CONTEXT.md`

`docs/adr/0001-emulation-platform-and-app-placement.md` and
`docs/adr/0002-replicas-on-machines-and-an-assumed-balancing-rule.md` may be kept as
history.

Then `make check` must still pass.
