# Experiments

The registry of every measurement taken with this repository: what was asked, on which
hardware, with which settings, and what each run measured. Results here are records, not
code. The vocabulary is the one of [CONTEXT.md](../CONTEXT.md).

New to the subject? [experiments-explained/](../experiments-explained/README.md) tells
what these measurements showed, with one chart per finding and no vocabulary assumed.

## Conventions

- **A sitting** is one stretch of work on one set of machines. It has one folder,
  `<date>-<slug>/`, where the slug says what the sitting asked. Folders sort oldest first
  by name.
- **An experiment** is one configuration measured at one or more user counts. Its id is
  the run-name prefix its runs share: `v8-alone-fastapi-fp32` holds the runs
  `v8-alone-fastapi-fp32-u1` to `-u32`. A measurement that is not a Locust run (a
  calibration, a probe, a timing) is an experiment too: its id names it, it has no runs,
  and its figures sit under `measurements`.
- **A run** is one headless Locust execution. Its run name ends with `-u<users>`.

Each sitting folder holds:

| File | What it holds |
|---|---|
| `README.md` | The question, the hardware, the settings common to the sitting, one table of its experiments, what it showed, its limits. Dated. |
| `experiments.yaml` | One entry per experiment, its settings in full. A sitting with many experiments splits it by phase: `experiments-<phase>.yaml`. |
| `results.csv` | One row per row of every run's Locust statistics. |

### `experiments.yaml`

A list under `experiments:`. YAML anchors (`&name`, `*name`) hold what several entries
share; they parse to the full settings. Each entry has:

| Key | What it says |
|---|---|
| `id` | The run-name prefix, or a name for a measurement that is not a run. |
| `question` | One line. |
| `hardware` | Machine type, vCPUs and replicas of each App part, Elasticsearch and the Workspace. |
| `deploy` | What `deploy.yaml` held: `topology`, `model_server`, and per App `hardware_tier_id`, `autoscaling` and `settings`. Never a host, URL or address. |
| `load` | The Locust target, the route, the user counts, the duration, which passages. |
| `calibration` | What calibration chose and its trials, when the notes give them. |
| `runs` | The run names. Each is in `results.csv`. |
| `measurements` | Figures that are not Locust statistics. |
| `void` | Why a run, or the whole experiment, does not count. |
| `notes` | What else the reader must know. |

A setting the sources do not establish is written `unknown`.

### `results.csv`

Columns: `run, experiment, users, type, name, requests, failures, requests_per_s, p50_ms,
p95_ms, p99_ms, max_ms`. One row per row of each run's `<run name>_stats.csv`: the request
rows (`POST`), the STAGE rows, and the `Aggregated` row (its `type` is empty). Rows are
sorted by experiment, then by user count, then in the file's order. Requests per second
keep 3 decimals; times are in milliseconds, rounded to integers. `p50_ms`, `p95_ms` and
`p99_ms` are Locust's own percentile columns (`50%`, `95%`, `99%`), which Locust rounds to
two significant figures. The `Aggregated` row sums the request rows and the STAGE rows,
so its counts are not a count of requests. Read the `POST` row for the route.

The file is derived from the `*_stats.csv` files Locust wrote on the Workspace, with a
throw-away script that is not kept in the repository.

## Sittings

| Date | Folder | Question | App machines | Experiments | Runs |
|---|---|---|---|---|---|
| 14 November 2026 | [2026-11-14-baseline-three-topologies-fastapi-8-vcpus](2026-11-14-baseline-three-topologies-fastapi-8-vcpus/README.md) | Which topology serves `/retrieve` best with FastAPI model servers? | 3 x 8 vCPUs | 5 | 8 |
| 14 November 2026 | [2026-11-14-cache-bucket-cold-and-warm-sittings](2026-11-14-cache-bucket-cold-and-warm-sittings/README.md) | Does a sitting restored from the cache bucket measure what a sitting built from scratch measures? | 3 x 8 vCPUs, Spot | 2 | 2 |
| 15 November 2026 | [2026-11-15-ovms-against-fastapi-equal-hardware](2026-11-15-ovms-against-fastapi-equal-hardware/README.md) | Does OVMS serve the reranker better than the FastAPI model server, on equal hardware? | 4 x 2 vCPUs, Spot | 22 | 57 |
| 15 November 2026 | [2026-11-15-reranker-scaled-up-8-and-16-vcpus](2026-11-15-reranker-scaled-up-8-and-16-vcpus/README.md) | Does one bigger reranker machine favour OVMS over the FastAPI model server? | 2 x 2 vCPUs and 1 x 8, then 16 vCPUs | 41 | 84 |
| 15 November 2026 | [2026-11-15-scale-out-against-scale-up-8-vcpus](2026-11-15-scale-out-against-scale-up-8-vcpus/README.md) | Do four machines of 2 vCPUs serve what one machine of 8 vCPUs serves? | 6 x 2 vCPUs and 1 x 8 vCPUs, Spot | 14 | 46 |

Every sitting ran on the Google Cloud emulation of Domino (`emulation/`), with the French
corpus (1,223 documents, 1,592 passages), the reranker
`antoinelouis/crossencoder-camembert-base-mmarcoFR` and the embedder
`Snowflake/snowflake-arctic-embed-l-v2.0`. Runs last 2 minutes. The 14 November
sittings and the 15 November sittings ran on different machine sizes and code: compare
figures within a sitting.

## Adding a sitting or an experiment

1. Create `<date>-<slug>/` with the three files above, or add to a sitting's files.
2. Take every figure from a statistics file or from the notes of the session that ran it.
3. Check that each `experiments*.yaml` parses, that every run it names is in
   `results.csv`, and that every run in `results.csv` belongs to exactly one experiment.
4. Add the sitting to the table above.
