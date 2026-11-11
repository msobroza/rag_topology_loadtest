# Emulation: an emulated Domino on Google Cloud

**Nothing here has been applied.** No `terraform plan`, `apply` or `validate` has
run against this configuration, and no machine has booted these scripts. The
files are checked only by parsing (`tests/emulation/`). Read the list
"Not verified" below before the first apply.

The emulation is temporary: it lets the owner rehearse a Domino deployment, and
compare topologies, before the real Domino is available. All of it lives in this
folder, `tests/emulation/` and the emulation's own subpackage, so that it can be
removed without touching the rest (see "Removing the emulation").

Terraform creates what already exists before anyone deploys on a real Domino (ADR
`docs/adr/0001-emulation-platform-and-app-placement.md`). Then the project's
unchanged `rag-deploy` publishes the Apps through the **emulated Domino**
(`rag-domino-emulation`), which places each App on a free machine of its hardware
tier over SSH and proxies every call to it. Terraform never creates an App.

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
```

## What gets created

| Resource | Default | Why |
|---|---|---|
| APIs | Compute Engine, IAP, IAM, Secret Manager | left enabled on destroy |
| Network, subnet, router, Cloud NAT | `10.10.0.0/24` in `europe-west1` | outbound internet (apt, pip, the Hugging Face Hub, GitHub, Docker images, an LLM API) without public addresses |
| Firewall rules | IAP range `35.235.240.0/20` to ports 22, 8899, 8089; Workspace to pool on 22 and 8888; pool to Workspace on 8899; Workspace and pool to Elasticsearch on 9200 | nothing else gets in |
| Workspace machine | `n2-standard-8`, 50 GB | the emulated Domino on 8899 and the Workspace (Locust runs here) |
| Elasticsearch machine | `n2-standard-4`, 50 GB, heap 4g | single node, security off, data on its disk |
| Pool machines | tier `cpu-8`: 3 x `n2-standard-8`, 50 GB each | one App per machine: the workflow, a reranker, an embedder |
| Two reserved internal addresses | Workspace and Elasticsearch | stable `deploy.yaml` |
| Three service accounts | no project role | the Workspace's may read two secrets |
| Secret `<name>-ssh-key` | ED25519 private key | the emulated Domino's key to the pool machines |

Every machine runs Ubuntu 22.04 LTS, **deletes itself after `max_run_hours`
(default 8)** and carries the label `rag-emulation=<name>`, as do its disk, the
addresses and the secret.

**Cost at the default sizes (approximate, europe-west1, on-demand):** an
`n2-standard-8` is about $0.43/h and an `n2-standard-4` about half that, so four
8-vCPU machines and one 4-vCPU machine come to about **$1.95/h**; disks (250 GB
balanced), Cloud NAT and its traffic add roughly $0.10/h. About **$16 for an
8-hour day**. Check the current prices before relying on these numbers. Budget
alerts do not cap Compute Engine spend: the self-deletion does.

## Steps

### 1. Once, in Google Cloud Shell

Cloud Shell no longer ships Terraform. From a clone of this repository:

```bash
gcloud config set project <project id>
make -C emulation setup BUCKET=<globally unique bucket name>   # Terraform into ~/bin, private versioned state bucket
```

Optional, for an LLM API key (a Project environment variable):

```bash
printf 'OPENAI_API_KEY=%s\n' "<key>" | gcloud secrets create rag-emu-project-env --data-file=-
```

### 2. Create the emulation

```bash
cp emulation/terraform/terraform.tfvars.example emulation/terraform/terraform.tfvars
# set project_id and git_ref (a pushed branch, tag or commit: the machines clone from GitHub);
# project_env_secret = "rag-emu-project-env" if you created it
make -C emulation init BUCKET=<bucket name>
make -C emulation plan
make -C emulation apply
make -C emulation ready         # repeat until every machine says done (about 5-10 min)
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

```bash
make -C emulation corpus               # the French corpus: rag-fetch-corpus quebec-insurance, then rag-ingest with the french analyzer
make -C emulation ingest               # or: the synthetic English corpus (one corpus per index: pick one)
make -C emulation deploy               # rag-deploy deploy all, through the emulated Domino
make -C emulation status               # each App's status and URL, then the workflow's /readyz
make -C emulation loadtest RUN_NAME=split-all             # through the emulated Domino's proxy
make -C emulation app-machines                            # which pool machine runs the workflow
make -C emulation loadtest-direct APP_HOST=<address> RUN_NAME=split-all-direct   # bypassing the proxy
make compare
```

Every call to an App passes the emulated Domino's proxy, a single Python process; the run
that bypasses it shows the proxy's share of the numbers (ADR 0001). The first
start of each App is slow: it installs its dependencies and downloads and converts
its models (the embedder is 2.3 GB).

From Cloud Shell, `make -C emulation tunnel-domino` forwards `localhost:8899` to
the emulated Domino (Swagger at `/docs`, via Cloud Shell's web preview) and
`make -C emulation tunnel-locust` forwards Locust's web UI on 8089 (for a
`locust` started without `--headless` in the Workspace).

### 4. Remove it

```bash
make -C emulation destroy
```

The state bucket stays (delete it with `gcloud storage rm -r gs://<bucket>` when
done for good), and so does the optional `project_env_secret`, which is yours.

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

With `RAG_OVMS_RERANK_URL=$url` and `RAG_OVMS_AUTH=domino`, a workflow app on the
same machine fetches its token from the emulated Domino and calls the reranker through the
proxy, as it would on Domino.

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

## The 8-hour self-deletion and Terraform's state

After `max_run_hours` Compute Engine deletes each machine; Terraform is not told.
The network, NAT, firewall, addresses, service accounts and secret remain (they
cost next to nothing; the NAT gateway a little). The next `plan` shows the
machines as to be created again: `make -C emulation plan apply` brings back fresh
machines with fresh disks, so the Elasticsearch index, the Workspace's results
and `deploy.yaml` edits are gone, and the Apps must be deployed again. Copy
`results/` off the Workspace before the deadline (`gcloud compute scp
--tunnel-through-iap`). `destroy` works at any time, with or without the machines.

A startup script runs at every boot. A pool machine re-created alone gets a new
address: re-apply, then reboot the Workspace machine (or run `sudo
google_metadata_script_runner startup` on it) so the emulated Domino reads the new
`placement.json`.

## Variables

Only `project_id` and `git_ref` are required; `terraform/variables.tf` describes
each one and `terraform/terraform.tfvars.example` lists them with their defaults.
`hardware_tiers` maps each tier id to `{machine_type, count, default, disk_gb}`,
with exactly one default tier; the cores and memory the emulated Domino reports for a
tier come from its machine type.

## Not verified

Nothing here has been applied. In particular:

- No `terraform init`, `validate`, `plan` or `apply` has run: the resource
  arguments were written from the `google` provider's documentation, not checked
  by Terraform. Provider range `>= 6.0, < 8.0`, `tls ~> 4.0`.
- `scheduling.max_run_duration` with `instance_termination_action = "DELETE"` on a
  `STANDARD` (not Spot) machine with the default `automatic_restart`: accepted by
  the API per its documentation, not tried.
- The startup scripts have only been parsed (`bash -n`, and `shellcheck` where
  installed) after rendering; none has run on Ubuntu 22.04. Untested in
  particular: the `ubuntu` user and passwordless `sudo`, uv's release tarball
  layout (`uv_version = 0.8.0`), `git fetch` of a commit id, Secret Manager access
  over the REST API with the metadata server's token, the socat forwarder, Docker
  from Ubuntu's `docker.io` package, Elasticsearch's start with this data folder.
- `enable-oslogin = FALSE` on the machines: an organisation policy that requires
  OS Login would break both the emulated Domino's SSH and `gcloud compute ssh ubuntu@...`.
- The cost figures are approximate and were not read from a current price list.
- The Cloud Shell setup script (pinned Terraform 1.12.2, the bucket flags) has not
  run.
- `rag-domino-emulation` (its flags, its reading of `placement.json` and
  `project.env`, its SSH placement with `systemd-run`) and `rag-fetch-corpus` are
  built elsewhere; this configuration matches the agreed contract but has not run
  against them.
- Whether Apps reach `localhost:8899` through the forwarder (it listens on
  127.0.0.1 only, not `::1`) and the emulated Domino's proxy under load.
- The time until every machine is ready, and whether 50 GB per pool machine is
  enough for every model-server choice.

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

`docs/adr/0001-emulation-platform-and-app-placement.md` may be kept as history.

Then `make check` must still pass.
