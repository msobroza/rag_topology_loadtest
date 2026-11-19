# A Hardware Tier by Name, a Revision by Number, and Two Listing Commands: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `deploy.yaml` names a hardware tier by its name and an environment revision by its number, `qp-deploy` finds the ids in Domino, and two commands show what Domino has.

**Architecture:** `deploy/domino_lookups/` asks Domino (the tiers, the environments, the revisions behind one abstract class with a documented and an internal source, and `BaseDominoIds`, which fills an App's ids or, in a dry run, asks nothing). `deploy/domino_listings/` shows the answers as an aligned table, as YAML, or as an interactive list drawn by Textual. `deploy()` finds every id before it publishes the first App. The fake Domino gains every route the lookups call.

**Tech Stack:** Python 3.11, pydantic and pydantic-settings, httpx, PyYAML, FastAPI (the fake Domino), Textual 8 (an optional extra), pytest with pytest-asyncio (`asyncio_mode = "auto"`).

**Spec:** `docs/specs/2026-11-18-tier-names-and-revision-numbers-in-the-deploy-file.md`. Read it with this plan: the plan argues from it.

## Global Constraints

- Base commit: `d37e46b` (`main` after pull request 33). Branch: `claude/domino-api-hardware-list-05d24b`.
- A deploy file that works today works unchanged and makes no new call to Domino.
- Nothing is published before every name and number of the Apps being deployed is found.
- A dry run asks Domino nothing. The listing commands send nothing but GET.
- The internal route is called only for a revision number the documented API lacks, or for an environment named or opened in a listing.
- The product imports `fake_domino` only from `deploy/qp_deploy.py`; `fake_domino` and `tests/fake_domino/` import nothing from `query_processing`; `deploy/` imports only `deploy` and `domain`.
- Textual is imported by one module, `deploy/domino_listings/opening_list.py`, which nothing imports at module level. The toolkit's core `dependencies` do not name Textual.
- Files hold at most 500 lines, function bodies at most 25, nesting at most 2, a directory at most 12 modules. `tests/helpers/` (21) and `tests/fake_domino/` (15) are at their ceilings: no new module in either.
- No two modules of the product share a file name, and the README's tree lists every module: a task that adds a module adds its line to the tree in `README.md`.
- Every public function has a Google docstring with `Args`, `Returns`, `Raises` where they apply and one `Example`, as the files beside it do. Errors carry the offending value.
- The lookups call Domino through the `httpx.Client` `qp-deploy` already opens, so they add no timeout, retry or page-size constant.
- Tables and details hold no colour and no cursor movement.
- Commits are authored by the configured git identity alone: no `Co-Authored-By` trailer, no `--author`, no signing, no push. A commit message is one full sentence saying what is now true, as `git log` shows.
- The worktree has no `.venv`: run `pytest` with the tools directory the orchestrator gives, each run with its own `--basetemp`. A task's agent does not run `mypy` or `make check`; the orchestrator runs the gate once at the end.
- Never touch `experiments/`, `emulation/terraform/` or the emulation's deploy file.

## Review Focus

Inputs the spec implies and a person will meet; each has its test in the task named.

1. A revision written as `v7`, as `"7"` in quotes or as `true`: refused when the file is read, with the value (Task 5).
2. A tier name that differs from Domino's by capitals or by a space at the end: refused as unknown, and the message lists the real name (Task 2).
3. An environment opened in the interactive list while the internal route fails: the item shows what the route answered, and the list stays up (Task 8).
4. A list whose pages do not add up: a tier list with `totalCount` missing or too large stops at the first empty page (Task 2), and an internal route that still answers several pages when one was asked is told as a failure, never shown as the whole list (Task 4).
5. An answer with status 200 that is not JSON, such as a proxy's sign-in page: a `DominoDeployError` naming the call, no traceback (Task 2).

## File Structure

All paths are relative to `query_processing/` unless they start with `README.md`, `docs/` or `query_processing_rag_usecase/`. Run tests from `query_processing/`.

| File | Responsibility |
|---|---|
| `src/fake_domino/api_shapes.py`, `state.py`, `stand_ins.py`, `server.py` (modify) | the routes the lookups call, their shapes, the fake's tiers and environments |
| `src/fake_domino/emulation/tiers.py` (modify) | the emulated Domino's list of tiers |
| `src/query_processing/deploy/apps_client.py` (modify) | `send_to_domino`, `domino_json`, `describe_failure`, shared by every call |
| `src/query_processing/deploy/domino_lookups/paged_list.py` (create) | read a documented list page after page |
| `.../domino_lookups/hardware_tiers.py` (create) | `HardwareTier`, list the tiers, find one by its name |
| `.../domino_lookups/environments.py` (create) | `DominoEnvironment`, `EnvironmentRevision`, list the environments |
| `.../domino_lookups/revision_sources.py` (create) | `BaseRevisionSource`, the documented source, find a revision by its number |
| `.../domino_lookups/internal_revisions.py` (create) | the internal route's source |
| `.../domino_lookups/base_domino_ids.py`, `domino_ids_asked.py`, `domino_ids_not_asked.py` (create) | fill an App's ids: asked of Domino, or not asked (dry run) |
| `src/query_processing/deploy/deploy_refusals.py` (create) | `refuse_secrets` and `refuse_placeholders`, moved out of the settings |
| `src/query_processing/deploy/deploy_settings.py`, `publish.py`, `qp_deploy.py` (modify) | the two keys; the ids found before publishing; the two commands |
| `.../domino_listings/aligned_table.py`, `listing_names.py` (create) | columns padded to the widest value; narrowing by names |
| `.../domino_listings/tiers_listing.py`, `environments_listing.py` (create) | the tables and the details of each command |
| `.../domino_listings/listing_looks.py`, `listing_commands.py` (create) | which look is shown; the two commands |
| `.../domino_listings/opening_list.py` (create) | the interactive list, the only module importing Textual |
| `pyproject.toml`, `query_processing_rag_usecase/Makefile`, `query_processing_rag_usecase/deploy.example.yaml`, `README.md` (modify) | the extra, the two targets, the two keys, what the user reads |

Tests: `tests/fake_domino/test_fake_domino_stand_ins.py` and `test_fake_domino_emulation_placement.py` (modify); `tests/deploy/domino_lookups/` and `tests/deploy/domino_listings/` (create, one test file a module); `tests/deploy/test_deploy_settings_two_ways.py` (create); `tests/deploy/domino_deploy/test_domino_deploy_domino_ids.py` (create) and `test_domino_deploy_dry_run.py` (modify); `tests/deploy/test_qp_deploy_listings.py` (create).

Order: 1, then 2, 3, 4 in that order beside 5; then 6, 7, 8, 9.

Shared by the tests below: `BUILT_7 = "2026-11-12T09:41:00Z"`, `BUILT_6 = "2026-11-03T16:02:00Z"`, `WHERE = "App rag-reranker (apps.reranker or defaults in deploy.yaml)"`, and `http_on(apps) = fake_domino_http(create_fake_domino(apps))`.

---

### Task 1: The fake Domino serves what the lookups call

**Files:**
- Modify: `src/fake_domino/api_shapes.py`, `state.py`, `stand_ins.py`, `server.py`, `emulation/tiers.py`
- Test: `tests/fake_domino/test_fake_domino_stand_ins.py`, `tests/fake_domino/test_fake_domino_emulation_placement.py`
- Modify: `README.md` (the tree's lines for the three core modules; the `forbidden` row of the behaviours table)

**Interfaces:**
- Produces, in `fake_domino.api_shapes`, with names and required flags as the reference gives them:
  - `HardwareTierGpuConfigurationV1` (`numberOfGpus: int`, `gpuKey: str`); `HardwareTierV1.gpuConfiguration: HardwareTierGpuConfigurationV1 | None = None`; `PaginatedHardwareTierEnvelopeV1` (`hardwareTiers`, `metadata: PaginatedMetadataV1`)
  - `EnvironmentOwnerV1` (`id`, `username`, `ownerType: str`); `EnvironmentRevisionV1` (`id: str`, `number: int`, `availableTools: list[dict[str, object]]`, `status: Literal["queued", "starting", "pulling", "building", "pushing", "succeeded", "failed", "killed"] | None = None`)
  - `EnvironmentV1` (`id`, `name`, `visibility: Literal["global", "private", "organization"]`, `rebuildOnBaseChanges: Literal["never", "followActive"]`, `supportedClusters: list[str]`, `archived: bool`, `owner: EnvironmentOwnerV1 | None = None`, and `latestRevision`, `selectedRevision`, `restrictedRevision`, each `EnvironmentRevisionV1 | None = None`)
  - `EnvironmentEnvelopeV1` (`environment`, `metadata: MetadataV1`); `PaginatedEnvironmentEnvelopeV1` (`environments`, `metadata: PaginatedMetadataV1`)
  - the published internal shape: `RevisionSummary` (`id: str`, `number: int`, `created: int`, `active: bool | None = None`), `PaginatedRevisionInfo` (`totalPages`, `currentPage`, `pageSize`, all `int`), `PaginatedRevisionData` (`revisions`, `pageInfo`)
- Produces, in `fake_domino.state`:
  - `@dataclass(frozen=True) class FakeEnvironment`: `environment: EnvironmentV1`, `revisions: tuple[RevisionSummary, ...]` (every revision, newest first)
  - `FakeDominoApps(..., tiers: Sequence[HardwareTierV1] = (), environments: Sequence[FakeEnvironment] = ())`, with attributes `tiers: list[HardwareTierV1]`, `environments: dict[str, FakeEnvironment]` keyed by id, and `revisions_route_status: int = 200`, which a test sets to make the internal route's stand-in fail
- Produces, in `fake_domino.stand_ins`:
  - `ENVIRONMENTS_PATH = "/api/environments/v1/environments"`, `ENVIRONMENTS_LIST_PATH = "/api/environments/beta/environments"`, `REVISIONS_ROUTE = "/v4/environments/{environmentId}/paginatedRevisionData"`
  - `fake_tier(tier_id: str, name: str, cores: float, memory_gib: float, *, gpus: int = 0, is_default: bool = False) -> HardwareTierV1` (`tier_envelope` is rebuilt on it)
  - `fake_environment(environment_id: str, *revisions: tuple[int, str, str], name: str = "", active: int | None = None, restricted: int | None = None) -> FakeEnvironment`: each revision is `(number, id, built)`, `built` an ISO date in UTC; the latest is the highest number; `active=None` makes the latest the active one; `name=""` takes the id
  - the routes: `GET /api/hardwaretiers/v1/hardwaretiers` and `GET /api/environments/beta/environments` (both with `offset: int = 0`, `limit: int = 10`), `GET /api/environments/v1/environments/{environmentId}`, and `GET /v4/environments/{environmentId}/paginatedRevisionData` (`page: int = 0`, `pageSize: int = 5`)

- [ ] **Step 1: Write the failing tests** in `tests/fake_domino/test_fake_domino_stand_ins.py`

```python
TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"
ENVIRONMENT_PATH = "/api/environments/v1/environments"
ENVIRONMENTS_LIST = "/api/environments/beta/environments"
ELEVEN_TIERS = [fake_tier(f"tier-{n}", f"cpu-{n}", n, 4 * n) for n in range(1, 12)]
ENV_1 = fake_environment("env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6), name="qp-rag", active=6)


def test_the_tiers_are_listed_by_pages_of_ten_with_their_total() -> None:
    http = http_to(FakeDominoApps(tiers=ELEVEN_TIERS))
    first, second = http.get(TIERS_PATH).json(), http.get(f"{TIERS_PATH}?offset=10").json()
    assert [t["name"] for t in first["hardwareTiers"]] == [f"cpu-{n}" for n in range(1, 11)]
    assert first["metadata"] == {
        "notices": [], "requestId": "fake-request", "limit": 10, "offset": 0, "totalCount": 11,
    }
    assert [t["id"] for t in second["hardwareTiers"]] == ["tier-11"]


def test_an_environment_gives_its_three_revisions() -> None:
    body = http_to(FakeDominoApps(environments=[ENV_1])).get(f"{ENVIRONMENT_PATH}/env-1").json()
    environment = body["environment"]
    assert (environment["name"], environment["owner"]["username"]) == ("qp-rag", "fake-owner")
    assert environment["latestRevision"] == {
        "id": "rev-7", "number": 7, "availableTools": [], "status": "succeeded",
    }
    assert environment["selectedRevision"]["number"] == 6
    assert environment["restrictedRevision"] is None


def test_every_revision_is_listed_by_the_internal_route_newest_first() -> None:
    body = http_to(FakeDominoApps(environments=[ENV_1])).get("/v4/environments/env-1/paginatedRevisionData").json()
    assert body["revisions"] == [
        {"id": "rev-7", "number": 7, "created": 1794476460000, "active": False},
        {"id": "rev-6", "number": 6, "created": 1793721720000, "active": True},
    ]
    assert body["pageInfo"] == {"totalPages": 1, "currentPage": 0, "pageSize": 5}
```

The two `created` values are `BUILT_7` and `BUILT_6` in milliseconds since 1970, UTC.

And, each one test:

| Test | Set-up | Expected |
|---|---|---|
| `test_a_listed_tier_gives_its_size_and_its_gpus` | `fake_tier("gpu-k8s", "gpu-1", 8, 32, gpus=1)` and a tier without | memory `{"value": 32.0, "units": "GiB"}`, `gpuConfiguration.numberOfGpus` 1, and none for the other |
| `test_a_fake_started_with_nothing_lists_nothing` | `FakeDominoApps()` | no tier with `totalCount` 0; no environment with `totalCount` 0 |
| `test_the_environments_are_listed_by_pages` | eleven environments | ten then one, `totalCount` 11 |
| `test_an_unknown_environment_is_404` (parametrized: the documented call, the internal route) | `env-9` | 404 |
| `test_the_internal_route_pages_by_five_and_by_the_size_asked` | six revisions | no parameter: five revisions, `totalPages` 2; `pageSize=10`: six, `totalPages` 1 |
| `test_the_internal_route_answers_the_status_a_test_sets` | `apps.revisions_route_status = 404` | 404 |
| `test_forbidden_answers_the_documented_lookups_403` (parametrized: `TIERS_PATH`, `ENVIRONMENTS_LIST`, `f"{ENVIRONMENT_PATH}/env-1"`) | behaviour `forbidden` | 403, `errors` `["Forbidden by the fake Domino (behaviour: forbidden)"]` |

Add the four paths to the mapping of `test_the_openapi_document_shows_the_documented_contract`. `test_any_hardware_tier_is_described_with_a_fixed_number_of_cores` stays as it is: one tier by its id is not forbidden.

In `tests/fake_domino/test_fake_domino_emulation_placement.py`:

```python
def test_the_emulated_domino_lists_the_tiers_of_its_placement_file() -> None:
    body = http_to(placed_apps(FakeAppLauncher())).get(TIERS_PATH).json()
    listed = {tier["name"]: (tier["id"], tier["resources"]["cores"]) for tier in body["hardwareTiers"]}
    assert listed == {"cpu-8": ("cpu-8", 8.0), "cpu-16": ("cpu-16", 16.0)}  # CPU_8 and CPU_16 of the support file
    assert body["metadata"]["totalCount"] == 2
```

- [ ] **Step 2: Run them, expect failures**

Run: `pytest tests/fake_domino/test_fake_domino_stand_ins.py tests/fake_domino/test_fake_domino_emulation_placement.py -q`
Expected: `ImportError: cannot import name 'fake_tier'`.

- [ ] **Step 3: Add the shapes, the state and the routes**

`fake_environment` gives each of the three revisions `availableTools=[]` and `status="succeeded"`, and the environment `visibility="private"`, `rebuildOnBaseChanges="never"`, `supportedClusters=[]`, `archived=False`, and the owner `fake-owner` (`FAKE_OWNER`, of type `Individual`). The documented environment routes have a router of their own, tagged `"Environments (Domino 6.2 API)"`; the internal route goes on `_stand_ins_router`. In `server.py`, `_record_and_forbid` forbids a path that starts with `APPS_API_PATH`, equals `HARDWARE_TIERS_PATH` or `ENVIRONMENTS_LIST_PATH`, or starts with `ENVIRONMENTS_PATH`. The `forbidden` behaviour's description becomes `"every Apps API call, the list of hardware tiers and the environments answer 403 with a FailureEnvelopeV1"`. In `emulation/tiers.py`, a route `GET ""` on `_placed_tiers_router` lists the placement file's tiers in one page, each as `tier_envelope` describes it.

- [ ] **Step 4: Run them, expect a pass**

Run: `pytest tests/fake_domino tests/structure -q`

- [ ] **Step 5: README, then commit**

Commit: `The fake Domino lists its hardware tiers and its environments, gives an environment with its three revisions, and stands in for the route that lists every revision`.

---

### Task 2: Domino's hardware tiers are read, and one is found by its name

**Files:**
- Create: `src/query_processing/deploy/domino_lookups/__init__.py`, `paged_list.py`, `hardware_tiers.py`
- Modify: `src/query_processing/deploy/apps_client.py`
- Test: `tests/deploy/domino_lookups/test_paged_list.py`, `tests/deploy/domino_lookups/test_hardware_tiers.py`
- Modify: `README.md` (the tree)

**Interfaces:**
- Consumes: Task 1's `fake_tier`, `FakeDominoApps(tiers=...)`.
- Produces, in `deploy.apps_client` (the Apps client then calls them with `APPS_API`, its messages unchanged):
  - `APPS_API = "Domino Apps API"`, `DOMINO_API = "Domino API"`
  - `send_to_domino(http: httpx.Client, method: str, path: str, *, api: str, json: object = None, params: Mapping[str, str] | None = None) -> httpx.Response`: raises `DominoDeployError` `"{api} {method} {path} could not be reached at {base_url}: {describe_error(exc)}"`
  - `describe_failure(api: str, response: httpx.Response) -> str`: `"{api} {method} {path} returned HTTP {status}: {detail}"`, the detail as today
  - `domino_json(api: str, response: httpx.Response) -> Mapping[str, object]`: a status other than 200 raises `describe_failure`; a 200 that is not JSON raises `"{api} {method} {path} answered HTTP 200 with a body that is not JSON: {first 200 characters}"`
- Produces, in `deploy.domino_lookups.paged_list`: `read_pages(http: httpx.Client, path: str, key: str, *, needs: str) -> list[Mapping[str, object]]`. It asks `offset` 0 with no `limit`, then `offset` plus the items received, until it has read `metadata.totalCount` items or a page comes back empty. On 401 or 403 it raises `describe_failure` followed by `"; {needs}"`.
- Produces, in `deploy.domino_lookups.hardware_tiers`:
  - `HARDWARE_TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"`
  - `@dataclass(frozen=True) class HardwareTier`: `id: str`, `name: str`, `cores: float`, `memory: str` (`"64 GiB"`), `gpus: int`, `default: bool = False`, `fields: Mapping[str, object] = field(default_factory=dict, compare=False)` (the tier as Domino gave it); method `size() -> str`: `"16 cores, 64 GiB"`, with `", 1 GPU"` or `", 2 GPUs"`. Numbers print with `:g`.
  - `list_hardware_tiers(http: httpx.Client) -> list[HardwareTier]`
  - `tier_named(tiers: Sequence[HardwareTier], name: str, where: str) -> HardwareTier`

- [ ] **Step 1: Write the failing tests**

```python
SMALL, LARGE = fake_tier("small-k8s", "cpu-8", 8, 32, is_default=True), fake_tier("large-k8s", "cpu-16", 16, 64)


def test_the_tiers_are_read_with_their_size_and_every_field() -> None:
    small, gpu = list_hardware_tiers(http_on(FakeDominoApps(tiers=[SMALL, fake_tier("gpu-k8s", "gpu-1", 0.5, 2, gpus=2)])))
    assert small == HardwareTier("small-k8s", "cpu-8", 8.0, "32 GiB", 0, default=True)
    assert gpu.size() == "0.5 cores, 2 GiB, 2 GPUs"
    assert small.fields["centsPerMinute"] == 0.0  # a field no attribute holds


def test_a_tier_on_the_second_page_is_read() -> None:
    apps = FakeDominoApps(tiers=[fake_tier(f"tier-{n}", f"cpu-{n}", n, n) for n in range(1, 12)])
    assert len(list_hardware_tiers(http_on(apps))) == 11
    assert apps.received == [("GET", HARDWARE_TIERS_PATH)] * 2


@pytest.mark.parametrize("name", ["cpu-61", "CPU-8", "cpu-8 "])
def test_a_name_no_tier_has_is_refused_with_the_tiers_domino_has(name: str) -> None:
    tiers = list_hardware_tiers(http_on(FakeDominoApps(tiers=[SMALL, LARGE])))
    with pytest.raises(DominoDeployError) as refusal:
        tier_named(tiers, name, WHERE)
    assert str(refusal.value) == (
        f"hardware_tier {name!r} of {WHERE} names no hardware tier of Domino, which has: "
        "cpu-8 (id small-k8s, 8 cores, 32 GiB); cpu-16 (id large-k8s, 16 cores, 64 GiB)"
    )
```

And, each one test with the message asserted whole:

| Test | Set-up | Expected |
|---|---|---|
| `test_a_tier_is_found_by_its_exact_name` | `tier_named(tiers, "cpu-8", WHERE)` | id `small-k8s` |
| `test_no_tier_at_all_is_said` | `tier_named([], "cpu-8", WHERE)` | `hardware_tier 'cpu-8' of {WHERE} names no hardware tier of Domino, which has none` |
| `test_a_name_two_tiers_share_is_refused_with_their_ids` | two tiers named `cpu-16`, ids `large-k8s`, `large-k8s-2` | `hardware_tier 'cpu-16' of {WHERE} is the name of 2 hardware tiers of Domino (ids large-k8s, large-k8s-2): write the one meant as hardware_tier_id` |
| `test_a_forbidden_list_names_the_permission` | behaviour `forbidden` | `Domino API GET /api/hardwaretiers/v1/hardwaretiers returned HTTP 403: Forbidden by the fake Domino (behaviour: forbidden); listing the hardware tiers needs the permission ViewHardwareTiers: ask for it, or write the tier's id as hardware_tier_id, which needs no such call` |
| `test_reading_stops_at_an_empty_page` (in `test_paged_list.py`, parametrized: `totalCount` 3, no `totalCount`) | an `httpx.MockTransport` answering one item at offset 0 and none after | one item, after two requests |
| `test_an_answer_that_is_not_json_is_refused` (in `test_paged_list.py`) | an `httpx.MockTransport` answering 200 with `<html>Sign in</html>` | `Domino API GET /api/x answered HTTP 200 with a body that is not JSON: <html>Sign in</html>` |

- [ ] **Step 2: Run them, expect failures**

Run: `pytest tests/deploy/domino_lookups -q`
Expected: `No module named 'query_processing.deploy.domino_lookups'`.

- [ ] **Step 3: Extract the three helpers in `apps_client.py`, then write the two modules**

`DominoAppsClient._send` and `_json` become one-line calls to `send_to_domino` and `domino_json` with `APPS_API`.

- [ ] **Step 4: Run them, expect a pass**

Run: `pytest tests/deploy -q` (the Apps client's tests included).

- [ ] **Step 5: README tree lines, then commit**

Commit: `Domino's hardware tiers are read page after page with their size, and one is found by its exact name or refused with the tiers Domino has`.

---

### Task 3: The environments are read, and a revision is found by its number through the documented call

**Files:**
- Create: `src/query_processing/deploy/domino_lookups/environments.py`, `revision_sources.py`
- Test: `tests/deploy/domino_lookups/test_environments.py`, `tests/deploy/domino_lookups/test_revision_sources.py`
- Modify: `README.md` (the tree)

**Interfaces:**
- Consumes: Task 1's `fake_environment`; Task 2's `read_pages`, `send_to_domino`, `domino_json`, `describe_failure`, `DOMINO_API`.
- Produces, in `deploy.domino_lookups.environments`:
  - `ENVIRONMENTS_PATH = "/api/environments/v1/environments"`, `ENVIRONMENTS_LIST_PATH = "/api/environments/beta/environments"`
  - `@dataclass(frozen=True) class EnvironmentRevision`: `id: str`, `number: int`, `built: datetime | None = None` (UTC), `status: str = ""`
  - `@dataclass(frozen=True) class DominoEnvironment`: `id: str`, `name: str`, `latest`, `active`, `restricted` (each `EnvironmentRevision | None`), `fields: Mapping[str, object]` (`compare=False`, the environment as Domino gave it); methods `three() -> list[EnvironmentRevision]` (the present ones, in that order, each id once) and `marks(revision_id: str) -> list[str]` (of `"active"`, `"latest"`, `"restricted"`, in that order)
  - `environment_from(fields: Mapping[str, object]) -> DominoEnvironment`
  - `list_environments(http: httpx.Client) -> list[DominoEnvironment]`
- Produces, in `deploy.domino_lookups.revision_sources`:
  - `class BaseRevisionSource(ABC)`: `revisions(self, environment_id: str) -> list[EnvironmentRevision]` (every revision the source knows; an empty list when it has none or could not be asked) and `describe(self, environment_id: str) -> str` (what it holds or what it answered, for a refusal)
  - `class DocumentedRevisions(BaseRevisionSource)`: `__init__(self, http: httpx.Client)`; one call an environment, remembered; `environment(self, environment_id: str) -> DominoEnvironment`
  - `revision_numbered(sources: Sequence[BaseRevisionSource], environment_id: str, number: int, where: str) -> EnvironmentRevision`: asks the sources in order, each only if those before it lack the number

- [ ] **Step 1: Write the failing tests**

```python
ENV_1 = fake_environment(
    "env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6), (3, "rev-3", BUILT_6), name="qp-rag", active=6, restricted=3
)


def documented(*environments: FakeEnvironment) -> tuple[DocumentedRevisions, FakeDominoApps]:
    apps = FakeDominoApps(environments=environments)
    return DocumentedRevisions(http_on(apps)), apps


@pytest.mark.parametrize(("number", "revision_id"), [(7, "rev-7"), (6, "rev-6"), (3, "rev-3")])
def test_the_latest_the_active_and_the_restricted_revisions_are_found(number: int, revision_id: str) -> None:
    source, _ = documented(ENV_1)
    found = revision_numbered([source], "env-1", number, WHERE)
    assert (found.id, found.number, found.status) == (revision_id, number, "succeeded")


def test_an_environment_is_asked_once_however_many_numbers_are_looked_up() -> None:
    source, apps = documented(ENV_1)
    for number in (7, 6):
        revision_numbered([source], "env-1", number, WHERE)
    assert apps.received == [("GET", f"{ENVIRONMENTS_PATH}/env-1")]


def test_a_later_source_is_asked_only_for_a_number_the_first_lacks() -> None:
    source, _ = documented(ENV_1)
    older = RevisionsOf(EnvironmentRevision("rev-5", 5))
    assert revision_numbered([source, older], "env-1", 7, WHERE).id == "rev-7"
    assert older.calls == 0
    assert revision_numbered([source, older], "env-1", 5, WHERE).id == "rev-5"


def test_a_number_no_source_has_is_refused_with_what_each_source_holds() -> None:
    source, _ = documented(ENV_1)
    with pytest.raises(DominoDeployError) as refusal:
        revision_numbered([source, RevisionsOf()], "env-1", 5, WHERE)
    assert str(refusal.value) == (
        f"environment_revision 5 of {WHERE} is no revision Domino gives for environment 'env-1': "
        "the documented API gives its latest (7), active (6) and restricted (3) revisions; "
        "the named fake gives 0 revisions. Write the revision's id as environment_revision_id"
    )


def test_the_environments_are_listed_with_their_three_revisions_and_every_field() -> None:
    [environment] = list_environments(http_on(FakeDominoApps(environments=[ENV_1])))
    assert (environment.id, environment.name) == ("env-1", "qp-rag")
    assert [r.number for r in environment.three()] == [7, 6, 3]
    assert environment.marks("rev-6") == ["active"]
    assert environment.fields["visibility"] == "private"
```

`RevisionsOf(*revisions)` is a `BaseRevisionSource` written in `test_revision_sources.py`: `revisions` returns what it was given and adds one to `calls`; `describe` returns `"the named fake gives N revisions"`.

And:

| Test | Set-up | Expected |
|---|---|---|
| `test_an_environment_never_built_says_none_three_times` | `fake_environment("env-1")`, number 1 | the refusal holds `latest (none), active (none) and restricted (none)` |
| `test_one_revision_that_is_latest_and_active_is_given_once_with_both_marks` | `fake_environment("env-1", (7, "rev-7", BUILT_7))` | `three()` has one revision; `marks("rev-7") == ["active", "latest"]` |
| `test_an_unknown_environment_is_refused_with_its_id` | `"env-9"` | `Domino has no environment 'env-9' (GET /api/environments/v1/environments/env-9 returned HTTP 404): check environment_id in deploy.yaml` |
| `test_a_forbidden_environment_names_the_permission` | behaviour `forbidden` | `Domino API GET /api/environments/v1/environments/env-1 returned HTTP 403: Forbidden by the fake Domino (behaviour: forbidden); reading an environment needs the permission ViewEnvironment: ask for it, or write the revision's id as environment_revision_id, which needs no such call` |
| `test_eleven_environments_are_read_on_two_pages` | eleven environments | eleven, two GETs of `ENVIRONMENTS_LIST_PATH` |

- [ ] **Step 2: Run them, expect failures**

Run: `pytest tests/deploy/domino_lookups -q`
Expected: `ImportError`.

- [ ] **Step 3: Write the two modules**

`DocumentedRevisions.revisions` is its environment's `three()`. Its `describe` says `the documented API gives its latest (N), active (N) and restricted (N) revisions`, with `none` for an absent one. `revision_numbered` joins the sources' `describe` texts with `"; "`. `list_environments` reads with `read_pages(..., needs="listing the environments needs no permission in Domino's reference: check the sign-in")`.

- [ ] **Step 4: Run them, expect a pass**

Run: `pytest tests/deploy/domino_lookups -q`

- [ ] **Step 5: README tree lines, then commit**

Commit: `The environments are read with their three revisions, and a revision is found by its number through sources asked in order, the documented call first`.

---

### Task 4: The internal route gives every revision of an environment

**Files:**
- Create: `src/query_processing/deploy/domino_lookups/internal_revisions.py`
- Test: `tests/deploy/domino_lookups/test_internal_revisions.py`
- Modify: `README.md` (the tree)

**Interfaces:**
- Consumes: Task 1's stand-in and `revisions_route_status`; Task 3's `BaseRevisionSource`, `EnvironmentRevision`; Task 2's `send_to_domino`, `DOMINO_API`.
- Produces: `REVISIONS_ROUTE = "/v4/environments/{environment_id}/paginatedRevisionData"` and `class InternalRevisions(BaseRevisionSource)`, `__init__(self, http: httpx.Client)`. One environment is asked once and remembered. `revisions` gives every revision, newest first, each with `built` from `created` (milliseconds since 1970, UTC); it gives an empty list when the route fails. `describe` then says what the route answered.

- [ ] **Step 1: Write the failing tests**

```python
SIX = [(n, f"rev-{n}", BUILT_6) for n in range(6, 0, -1)]


def internal(*environments: FakeEnvironment) -> tuple[InternalRevisions, FakeDominoApps]:
    apps = FakeDominoApps(environments=environments)
    return InternalRevisions(http_on(apps)), apps


def test_every_revision_is_given_with_the_date_it_was_built() -> None:
    source, _ = internal(fake_environment("env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6)))
    assert source.revisions("env-1") == [
        EnvironmentRevision("rev-7", 7, datetime(2026, 11, 12, 9, 41, tzinfo=UTC)),
        EnvironmentRevision("rev-6", 6, datetime(2026, 11, 3, 16, 2, tzinfo=UTC)),
    ]
    assert source.describe("env-1") == "the internal route GET /v4/environments/env-1/paginatedRevisionData lists 2 revisions (7 to 6)"


def test_more_revisions_than_a_page_are_read_in_one_second_call() -> None:
    source, apps = internal(fake_environment("env-1", *SIX))
    assert [r.number for r in source.revisions("env-1")] == [6, 5, 4, 3, 2, 1]
    assert [r.number for r in source.revisions("env-1")] == [6, 5, 4, 3, 2, 1]  # remembered
    assert apps.received == [("GET", "/v4/environments/env-1/paginatedRevisionData")] * 2
```

And, each with `revisions(...) == []` and `describe` asserted whole (`ROUTE` is `GET /v4/environments/env-1/paginatedRevisionData`):

| Test | Set-up | `describe` |
|---|---|---|
| `test_a_route_that_answers_another_status_is_told` | `apps.revisions_route_status = 404` | `the internal route {ROUTE} returned HTTP 404, so the revisions it would list are not known` |
| `test_a_route_that_cannot_be_reached_is_told` | an `httpx.MockTransport` raising `httpx.ConnectError("refused")` | starts `the internal route {ROUTE} could not be reached` |
| `test_an_answer_of_another_shape_is_told` (parametrized: `{}`, `{"revisions": "x", "pageInfo": {}}`, a body that is not JSON) | an `httpx.MockTransport` | `the internal route {ROUTE} answered in a shape other than the one Domino publishes (revisions with id, number and created; pageInfo with pageSize and totalPages)` |
| `test_a_route_that_still_answers_several_pages_is_told` | an `httpx.MockTransport` answering `totalPages` 2 whatever `pageSize` is asked | `the internal route {ROUTE} answered 2 pages when one page of 10 was asked, so its list is not complete` |

- [ ] **Step 2: Run them, expect failures**

Run: `pytest tests/deploy/domino_lookups/test_internal_revisions.py -q`

- [ ] **Step 3: Write the module**

The route is read without assuming how its pages are numbered: one call with no parameter gives `pageInfo.pageSize` and `pageInfo.totalPages`; when `totalPages` is above 1, a second call asks `pageSize` equal to `pageSize * totalPages`, and its answer must hold one page. The module's docstring says the path is not in Domino's reference, that its shape is the `PaginatedRevisionData` of the published 6.3.1 spec, and that the path, the parameter name and the unit of `created` are not verified on a live Domino.

- [ ] **Step 4: Run them, expect a pass**

Run: `pytest tests/deploy/domino_lookups -q`

- [ ] **Step 5: README tree line, then commit**

Commit: `Every revision of an environment is read with its build date through the route Domino's own pages use, and a route that fails is told, never raised`.

---

### Task 5: The deploy file takes the two keys

**Files:**
- Create: `src/query_processing/deploy/deploy_refusals.py`
- Modify: `src/query_processing/deploy/deploy_settings.py`, `src/query_processing/deploy/publish.py` (imports and `_refuse_a_default_revision_alone`), `tests/deploy/test_deploy_settings.py` and `query_processing_rag_usecase/tests/test_deploy_dry_run.py` (imports only)
- Test: `tests/deploy/test_deploy_settings_two_ways.py`
- Modify: `README.md` (the tree)

**Interfaces:**
- Produces, in `deploy.deploy_refusals`: `refuse_secrets` and `refuse_placeholders`, moved as they are with `_values_used`, `_PLACEHOLDER` and `_secrets_in`. `_values_used` also yields `"hardware_tier"`.
- Produces, in `deploy.deploy_settings`: `AppBlock.hardware_tier: str | None = None`; `AppBlock.environment_revision: int | None = Field(default=None, strict=True, ge=1)`; `AppSettings.hardware_tier: str = ""` and `AppSettings.environment_revision: int | None = None`, after `settings`.

- [ ] **Step 1: Move the refusals, run the suite**

Move the five names to `deploy_refusals.py` and repoint the three importers. Run `pytest tests/deploy -q` and, from `query_processing_rag_usecase/`, `pytest tests/test_deploy_dry_run.py -q`. Expected: the same tests pass as before the move.

- [ ] **Step 2: Write the failing tests** in `tests/deploy/test_deploy_settings_two_ways.py`, with `make_settings` of `domino_deploy_helpers`

```python
TIER_WAYS = [({"hardware_tier": "cpu-16"}, ("cpu-16", "")), ({"hardware_tier_id": "large-k8s"}, ("", "large-k8s"))]
REVISION_WAYS = [({"environment_revision": 7}, (7, "")), ({"environment_revision_id": "rev-7"}, (None, "rev-7"))]


def tier_of(settings: DeploySettings, part: str) -> tuple[str, str]:
    app = settings.for_app(part)
    return app.hardware_tier, app.hardware_tier_id


def revision_of(settings: DeploySettings, part: str) -> tuple[int | None, str]:
    app = settings.for_app(part)
    return app.environment_revision, app.environment_revision_id


@pytest.mark.parametrize(("own", "expected"), TIER_WAYS)
@pytest.mark.parametrize("default", [{"hardware_tier": "cpu-8"}, {"hardware_tier_id": "small-k8s"}])
def test_an_apps_own_tier_replaces_the_default_whichever_way_each_is_written(
    own: dict[str, str], expected: tuple[str, str], default: dict[str, str]
) -> None:
    settings = make_settings(defaults={"environment_id": "env-1", **default}, apps={"reranker": own})
    assert tier_of(settings, "reranker") == expected
    assert tier_of(settings, "embedder") == (default.get("hardware_tier", ""), default.get("hardware_tier_id", ""))


@pytest.mark.parametrize(("own", "expected"), REVISION_WAYS)
def test_the_default_revision_does_not_follow_an_app_to_its_own_environment(
    own: dict[str, object], expected: tuple[int | None, str]
) -> None:
    settings = make_settings(
        defaults={"environment_id": "env-1", **own}, apps={"reranker": {"environment_id": "env-2"}}
    )
    assert revision_of(settings, "embedder") == expected
    assert revision_of(settings, "reranker") == (None, "")
```

And:

| Test | Deploy settings | Expected |
|---|---|---|
| `test_an_apps_own_revision_replaces_the_default_either_way` (parametrized over `REVISION_WAYS` twice) | default one way, `apps.reranker` the other | the App's, and none of the default's |
| `test_both_keys_of_the_tier_in_one_block_are_refused` (parametrized: `defaults`, `apps.reranker`) | `hardware_tier: cpu-16` and `hardware_tier_id: large-k8s` | `ValidationError` holding the block's name and `hardware_tier 'cpu-16' and hardware_tier_id 'large-k8s' are both set: they are two ways to write one setting, keep one` |
| `test_both_keys_of_the_revision_in_one_block_are_refused` (same two blocks) | `environment_revision: 7` and `environment_revision_id: rev-7` | `environment_revision 7 and environment_revision_id 'rev-7' are both set: they are two ways to write one setting, keep one` |
| `test_an_empty_id_stands_beside_the_other_key` | `defaults: {hardware_tier_id: "", hardware_tier: cpu-8, environment_revision_id: "", environment_revision: 7}` | accepted: `("cpu-8", "")` and `(7, "")` |
| `test_an_apps_empty_tier_does_not_replace_the_default` | default `hardware_tier: cpu-8`, `apps.reranker: {hardware_tier: "", hardware_tier_id: ""}` | `("cpu-8", "")` |
| `test_a_revision_that_is_no_whole_number_from_one_is_refused` (parametrized: `0`, `-1`, `7.5`, `"7"`, `"v7"`, `True`) | `defaults.environment_revision` | `ValidationError` naming `defaults.environment_revision` and the value |
| `test_a_placeholder_left_in_the_tier_name_is_refused` | `apps.reranker.hardware_tier: "<a tier with 16 cores>"` | `refuse_placeholders` raises `hardware_tier in apps.reranker of deploy.yaml still holds the example placeholder '<a tier with 16 cores>': replace it with a real value` |
| `test_a_default_revision_number_without_a_default_environment_is_refused` | `defaults: {environment_revision: 7}`, through `deploy(...)` on the fake | `DominoDeployError` starting `defaults.environment_revision 7 in deploy.yaml has no defaults.environment_id beside it` |

- [ ] **Step 3: Run them, expect failures**

Run: `pytest tests/deploy/test_deploy_settings_two_ways.py -q`
Expected: `Extra inputs are not permitted` for `hardware_tier`.

- [ ] **Step 4: Add the keys and the rules**

The one-way-per-block rule is one `model_validator(mode="after")` on `AppBlock`, comparing values that are not empty. In `for_app`, the tier comes from the App's block when it sets either key, else from `defaults`. `_environment_of` returns the environment id, the revision id and the revision number, on today's rule. `_refuse_a_default_revision_alone` names the key that is set, with the tail of today's message.

- [ ] **Step 5: Run, expect a pass**

Run: `pytest tests/deploy tests/structure -q`
Expected: all pass, `deploy_settings.py` under 500 lines.

- [ ] **Step 6: README tree line, then commit**

Commit: `A block of the deploy file writes its hardware tier by name or by id and its environment revision by number or by id, one way a block`.

---

### Task 6: `qp-deploy` finds the ids before it publishes, and a dry run asks nothing

**Files:**
- Create: `src/query_processing/deploy/domino_lookups/base_domino_ids.py`, `domino_ids_asked.py`, `domino_ids_not_asked.py`
- Modify: `src/query_processing/deploy/publish.py`, `src/query_processing/deploy/qp_deploy.py`
- Test: `tests/deploy/domino_deploy/test_domino_deploy_domino_ids.py`, `tests/deploy/domino_deploy/test_domino_deploy_dry_run.py`
- Modify: `README.md` (the tree)

**Interfaces:**
- Consumes: `list_hardware_tiers`, `tier_named`, `HardwareTier.size` (Task 2); `BaseRevisionSource`, `DocumentedRevisions`, `revision_numbered` (Task 3); `InternalRevisions` (Task 4); `AppSettings.hardware_tier`, `.environment_revision` (Task 5).
- Produces:
  - `base_domino_ids.py`: `@dataclass(frozen=True) class AppWithIds` (`app: AppSettings`, with `hardware_tier_id` and `environment_revision_id` filled; `found: tuple[str, ...]`, the lines to print); `class BaseDominoIds(ABC)` with `find(self, app: AppSettings) -> AppWithIds`; `where_set(app: AppSettings) -> str`, giving `"App rag-reranker (apps.reranker or defaults in deploy.yaml)"`
  - `domino_ids_asked.py`: `class DominoIdsAsked(BaseDominoIds)`, `__init__(self, http: httpx.Client, sources: Sequence[BaseRevisionSource] | None = None)`; `None` is `[DocumentedRevisions(http), InternalRevisions(http)]`; the tiers are listed once, at the first tier named
  - `domino_ids_not_asked.py`: `class DominoIdsNotAsked(BaseDominoIds)`: the ids become `"(tier cpu-16: its id is asked of Domino at deploy)"` and `"(revision 7: its id is asked of Domino at deploy)"`, and `found` is empty
  - `publish.deploy(..., ids: BaseDominoIds | None = None)` and `publish.publish_app(..., ids: BaseDominoIds | None = None)`; `None` is `DominoIdsAsked(client.http)`

- [ ] **Step 1: Write the failing tests** in `test_domino_deploy_domino_ids.py`

```python
TIERS = [fake_tier("small-k8s", "cpu-8", 8, 32), fake_tier("large-k8s", "cpu-16", 16, 64)]
ENV_1 = fake_environment("env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6), (5, "rev-5", BUILT_6))
TIERS_CALL = ("GET", "/api/hardwaretiers/v1/hardwaretiers")
ENVIRONMENT_CALL = ("GET", "/api/environments/v1/environments/env-1")
INTERNAL_CALL = ("GET", "/v4/environments/env-1/paginatedRevisionData")
LOOKUPS = {TIERS_CALL, ENVIRONMENT_CALL, INTERNAL_CALL}
NAMED = {"environment_id": "env-1", "environment_revision": 7, "hardware_tier": "cpu-8"}


def deployed(target: str = "all", **given: object) -> FakeDominoApps:
    apps = FakeDominoApps(tiers=TIERS, environments=[ENV_1])
    deploy(make_client(apps), make_settings(**given), target, launch=LAUNCH, sleep=no_sleep)
    return apps


def test_a_tier_and_a_revision_by_name_are_sent_as_their_ids(capsys: pytest.CaptureFixture[str]) -> None:
    apps = deployed(defaults=NAMED, apps={"reranker": {"hardware_tier": "cpu-16"}})
    sent = {c.name: (c.version.hardwareTierId, c.version.environmentRevisionId) for c in apps.creates}
    assert sent == {
        "rag-embedder": ("small-k8s", "rev-7"),
        "rag-reranker": ("large-k8s", "rev-7"),
        "rag-workflow": ("small-k8s", "rev-7"),
    }
    assert capsys.readouterr().out.startswith(
        "rag-embedder: hardware tier cpu-8 (id small-k8s): 8 cores, 32 GiB\n"
        "rag-embedder: environment revision 7 (id rev-7)\n"
        "rag-reranker: hardware tier cpu-16 (id large-k8s): 16 cores, 64 GiB\n"
        "rag-reranker: environment revision 7 (id rev-7)\n"
        "rag-workflow: hardware tier cpu-8 (id small-k8s): 8 cores, 32 GiB\n"
        "rag-workflow: environment revision 7 (id rev-7)\n"
    )
    assert [call for call in apps.received if call in LOOKUPS] == [TIERS_CALL, ENVIRONMENT_CALL]


def test_an_older_revision_is_found_through_the_internal_route() -> None:
    apps = deployed(defaults={"environment_id": "env-1", "environment_revision": 5})
    assert {c.version.environmentRevisionId for c in apps.creates} == {"rev-5"}
    assert [call for call in apps.received if call in LOOKUPS] == [ENVIRONMENT_CALL, INTERNAL_CALL]


@pytest.mark.parametrize(
    ("by_name", "by_id"),
    [
        ({"hardware_tier": "cpu-8"}, {"hardware_tier_id": "small-k8s"}),
        ({"environment_revision": 7}, {"environment_revision_id": "rev-7"}),
    ],
)
def test_a_setting_written_either_way_publishes_the_same_versions(
    by_name: dict[str, object], by_id: dict[str, object]
) -> None:
    named = deployed(defaults={"environment_id": "env-1", **by_name})
    with_ids = deployed(defaults={"environment_id": "env-1", **by_id})
    assert named.creates == with_ids.creates


@pytest.mark.parametrize(
    ("wrong", "message"),
    [
        ({"hardware_tier": "cpu-61"}, "hardware_tier 'cpu-61' of App rag-workflow"),
        ({"environment_revision": 9}, "environment_revision 9 of App rag-workflow"),
    ],
)
def test_a_wrong_value_of_the_last_app_publishes_nothing(wrong: dict[str, object], message: str) -> None:
    apps = FakeDominoApps(tiers=TIERS, environments=[ENV_1])
    settings = make_settings(defaults=NAMED, apps={"workflow": wrong})
    with pytest.raises(DominoDeployError, match=message):
        deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    assert apps.creates == []
```

And:

| Test | Set-up | Expected |
|---|---|---|
| `test_a_number_the_route_cannot_give_is_refused_with_what_the_route_answered` | revision 5, `apps.revisions_route_status = 404` | `DominoDeployError` holding `latest (7), active (7) and restricted (none)` and `returned HTTP 404`; no App created |
| `test_a_deploy_file_with_ids_only_asks_domino_nothing_new` | ids for the tier and the revision | no call of `LOOKUPS` in `apps.received` |
| `test_a_deploy_that_names_no_revision_number_never_calls_the_internal_route` | `hardware_tier` only | `INTERNAL_CALL` and `ENVIRONMENT_CALL` not in `apps.received` |
| `test_deploying_one_app_looks_up_only_what_that_app_names` | target `embedder`, a wrong tier name in `apps.reranker` | only `rag-embedder` created, no call of `LOOKUPS` |
| `test_the_ids_are_found_through_the_fake_given_to_main` | `run_main(["deploy", "all"], create_fake_domino(apps))`, the deploy file naming `hardware_tier: cpu-8` | exit 0, `hardwareTierId` `small-k8s` in `apps.creates` |

In `test_domino_deploy_dry_run.py`:

```python
def test_dry_run_shows_a_name_and_a_number_as_written_and_asks_domino_nothing(run_main: RunMain) -> None:
    named = "  environment_revision: 7\n  hardware_tier: cpu-16\n"
    Path("deploy.yaml").write_text(ENV_1 + named + WORKFLOW_SCALES)
    code, out, _ = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 0
    assert out.endswith(
        dry_run_report(
            tier="(tier cpu-16: its id is asked of Domino at deploy)",
            revision="(revision 7: its id is asked of Domino at deploy)",
        )
    )
```

The dry run's fake has no tier and no environment, so exit code 0 also shows that nothing was looked up.

- [ ] **Step 2: Run them, expect failures**

Run: `pytest tests/deploy/domino_deploy/test_domino_deploy_domino_ids.py tests/deploy/domino_deploy/test_domino_deploy_dry_run.py -q`
Expected: the named tier is sent as no tier.

- [ ] **Step 3: Write the three modules and wire them**

`DominoIdsAsked.find` fills each id with `dataclasses.replace` and adds one line an id found: `"{app.name}: hardware tier {name} (id {id}): {size}"` and `"{app.name}: environment revision {number} (id {id})"`. In `deploy()`, after `_check_deploy_settings` and before the loop that publishes: build `ids` once, call `find` for every App of `parts`, print every line. `publish_app` then calls `ids.find(settings.for_app(part)).app`, which makes no new call. In `qp_deploy._run_command`, `deploy(...)` gets `ids=DominoIdsNotAsked()` in a dry run and `None` otherwise; `dry_run_report` does not change.

- [ ] **Step 4: Run, expect a pass**

Run: `pytest tests/deploy tests/structure -q`
Expected: all pass, `publish.py` under 500 lines.

- [ ] **Step 5: README tree lines, then commit**

Commit: `qp-deploy finds the id of every tier and revision the deploy file names before it publishes an App, and a dry run shows them as written`.

---

### Task 7: The two listing commands, as tables and as details

**Files:**
- Create: `src/query_processing/deploy/domino_listings/__init__.py`, `aligned_table.py`, `listing_names.py`, `tiers_listing.py`, `environments_listing.py`, `listing_looks.py`, `listing_commands.py`
- Modify: `src/query_processing/deploy/qp_deploy.py`, `query_processing_rag_usecase/Makefile`
- Test: `tests/deploy/domino_listings/test_aligned_table.py`, `test_listing_names.py`, `test_tiers_listing.py`, `test_environments_listing.py`, `test_listing_looks.py`; `tests/deploy/test_qp_deploy_listings.py`
- Modify: `README.md` (the tree)

**Interfaces:**
- Consumes: `HardwareTier`, `list_hardware_tiers` (Task 2); `DominoEnvironment`, `EnvironmentRevision`, `list_environments` (Task 3); `BaseRevisionSource`, `InternalRevisions` (Tasks 3 and 4).
- Produces:
  - `aligned_table.aligned_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, right: Collection[int] = ()) -> str`: columns padded to their widest value, two spaces between them, the columns of `right` aligned right, no space at the end of a line, one line a row after the headers
  - `listing_names.only_named(items: Sequence[T], names: Sequence[str], kind: str) -> list[T]` (`T` has a `name`): every item when `names` is empty, else the items of those names in Domino's order; a name no item has raises `DominoDeployError` `"no {kind} of Domino is named {name!r}; it has: {names joined by ', '}"`
  - `tiers_listing.tiers_table(tiers: Sequence[HardwareTier]) -> str` and `tiers_details(tiers: Sequence[HardwareTier]) -> str`
  - `environments_listing.environments_table(environments: Sequence[DominoEnvironment]) -> str`; `shown_revisions(environment: DominoEnvironment, internal: BaseRevisionSource) -> tuple[list[EnvironmentRevision], str]` (every revision of the internal source, each with the status the documented three give it; when the internal source has none, the documented three and the source's `describe` as the note); `revisions_table(environment, revisions, note) -> str`; `environments_details(environments, revisions: Mapping[str, tuple[list[EnvironmentRevision], str]]) -> str`
  - `listing_looks.Look = Literal["list", "table", "details"]` and `chosen_look(*, details: bool, table: bool, terminal: bool) -> Look`: `"details"` when asked, else `"table"` when asked or when `terminal` is false, else `"list"`
  - `listing_commands.show_tiers(http: httpx.Client, names: Sequence[str], look: Look) -> None` and `show_environments(http: httpx.Client, names: Sequence[str], look: Look) -> None`: they print; in this task the look `"list"` prints the table (Task 8 replaces that)
  - `qp-deploy tiers [NAME ...] [--table | --details]` and `qp-deploy environments [NAME ...] [--table | --details]`

- [ ] **Step 1: Write the failing tests**

```python
TIERS = [
    HardwareTier("small-k8s", "cpu-8", 8, "32 GiB", 0, default=True),
    HardwareTier("large-k8s", "cpu-16", 16, "64 GiB", 0),
    HardwareTier("gpu-k8s", "gpu-1", 8, "32 GiB", 1),
]


def test_the_tiers_table_aligns_its_columns_and_marks_the_default() -> None:
    assert tiers_table(TIERS) == (
        "NAME    CORES  MEMORY  GPUS  DEFAULT  ID\n"
        "cpu-8       8  32 GiB     -  yes      small-k8s\n"
        "cpu-16     16  64 GiB     -           large-k8s\n"
        "gpu-1       8  32 GiB     1           gpu-k8s\n"
    )


def test_the_details_give_every_field_domino_gave_the_name_first() -> None:
    odd = HardwareTier("x", "odd", 1, "1 GiB", 0, fields={"id": "x", "name": "odd", "aFieldOfAnotherVersion": [1, 2]})
    [shown] = yaml.safe_load(tiers_details([odd]))
    assert list(shown) == ["name", "id", "aFieldOfAnotherVersion"]
    assert shown["aFieldOfAnotherVersion"] == [1, 2]


def test_the_revisions_table_marks_each_revision_and_dates_it_in_utc() -> None:
    [environment] = list_environments(http_on(FakeDominoApps(environments=[ENV_1])))
    revisions, note = shown_revisions(environment, InternalRevisions(http_on(FakeDominoApps(environments=[ENV_1]))))
    assert note == ""
    assert revisions_table(environment, revisions, note) == (
        "qp-rag   id env-1   private, owner fake-owner\n"
        "\n"
        "REVISION  BUILT (UTC)       STATUS     ID\n"
        "7 latest  2026-11-12 09:41  succeeded  rev-7\n"
        "6 active  2026-11-03 16:02  succeeded  rev-6\n"
        "5         2026-11-03 16:02             rev-5\n"
    )
```

Here `ENV_1 = fake_environment("env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6), (5, "rev-5", BUILT_6), name="qp-rag", active=6)`.

And:

| Test | Set-up | Expected |
|---|---|---|
| `test_the_environments_table_gives_the_active_and_latest_numbers` | `ENV_1` and `fake_environment("env-2", name="empty")` | `"NAME    VISIBILITY  OWNER       ACTIVE  LATEST  ID\nqp-rag  private     fake-owner       6       7  env-1\nempty   private     fake-owner       -       -  env-2\n"` |
| `test_a_route_that_fails_leaves_the_three_documented_revisions_and_a_note` | `revisions_route_status = 404` | the table holds revisions 7 and 6 with `-` under `BUILT (UTC)`, then a blank line and the note `the internal route GET /v4/environments/env-1/paginatedRevisionData returned HTTP 404, so the revisions it would list are not known` |
| `test_the_environment_details_hold_its_revisions_when_given` | `environments_details([environment], {"env-1": (revisions, "")})` | parsed with `yaml.safe_load`: `name` first; `revisions[0] == {"number": 7, "id": "rev-7", "built": "2026-11-12 09:41 UTC", "marks": ["latest"], "status": "succeeded"}` |
| `test_only_named_keeps_dominos_order_and_refuses_an_unknown_name` | names `["cpu-16", "cpu-8"]`, then `["cpu-61"]` | `cpu-8` then `cpu-16`; `no hardware tier of Domino is named 'cpu-61'; it has: cpu-8, cpu-16, gpu-1` |
| `test_two_items_of_one_name_are_both_kept` | two environments named `qp-rag` | both |
| `test_the_look` (parametrized) | `details`, `table`, `terminal` | `details` wins; `table` or no terminal gives `"table"`; a terminal with neither gives `"list"` |

`test_aligned_table.py` holds one test of `aligned_table` on a literal: a column aligned right, an empty cell in the middle, and no space at the end of a line.

In `tests/deploy/test_qp_deploy_listings.py`, with `run_main` (`pytest_plugins = ["domino_deploy_fixtures"]`) and a fake holding the three tiers of the table above (built with `fake_tier`, `cpu-8` the default) and `ENV_1`:

| Test | Command | Expected |
|---|---|---|
| `test_tiers_prints_its_table_where_there_is_no_terminal` | `["tiers"]` | exit 0, the tiers table above |
| `test_tiers_named_with_details_prints_yaml_of_that_tier_alone` | `["tiers", "cpu-16", "--details"]` | one item, `id` `large-k8s`, `resources.cores` 16.0 |
| `test_environments_prints_its_table` | `["environments"]` | the environments table; `INTERNAL_CALL` not in `apps.received` |
| `test_a_named_environment_prints_its_revisions` | `["environments", "qp-rag", "--table"]` | the revisions table; the internal route called once, for `env-1` |
| `test_details_alone_never_call_the_internal_route` | `["environments", "--details"]` | YAML with `latestRevision`, no `revisions` key; `INTERNAL_CALL` not received |
| `test_an_unknown_name_is_refused_with_the_names_domino_has` | `["tiers", "cpu-61"]` | exit 2, `qp-deploy: error: no hardware tier of Domino is named 'cpu-61'; it has: cpu-8, cpu-16, gpu-1` |
| `test_the_two_options_exclude_each_other` | `["tiers", "--table", "--details"]` | exit 2, argparse's `not allowed with argument` |
| `test_a_listing_is_refused_in_a_dry_run` (parametrized: `tiers`, `environments`) | `["--dry-run", <command>]` | exit 2, `qp-deploy: error: qp-deploy <command> asks Domino what it has, and a dry run asks Domino nothing: run it without --dry-run` |
| `test_a_listing_sends_nothing_but_get` | both commands, every look but the list | every method in `apps.received` is `GET` |
| `test_a_listing_needs_no_valid_app_block` | a deploy file with `defaults: {environment_id: "<environment id>"}` | `["tiers"]` exits 0 |

- [ ] **Step 2: Run them, expect failures**

Run: `pytest tests/deploy/domino_listings tests/deploy/test_qp_deploy_listings.py -q`

- [ ] **Step 3: Write the modules, the two sub-commands and the two targets**

The details are `yaml.safe_dump(items, sort_keys=False, allow_unicode=True)`, each item the `fields` Domino gave with `name` moved first. A revision's cell under `REVISION` is its number then its marks, one space apart. `BUILT (UTC)` is `%Y-%m-%d %H:%M`, or `-` when not known; `GPUS`, `ACTIVE` and `LATEST` show `-` for none. In `qp_deploy.py`, the two sub-commands share one parser set-up (`names` with `nargs="*"`, and a mutually exclusive group of `--table` and `--details`), `terminal` is `sys.stdin.isatty() and sys.stdout.isatty()`, and the commands run on `client.http`. The `Makefile` gains the two targets of the spec's section "The two listing commands", word for word.

- [ ] **Step 4: Run, expect a pass**

Run: `pytest tests/deploy tests/structure -q`, and from `query_processing_rag_usecase/`: `pytest tests -q`.

- [ ] **Step 5: README tree lines, then commit**

Commit: `qp-deploy tiers and qp-deploy environments show what Domino has as an aligned table, and with --details every field Domino gave`.

---

### Task 8: In a terminal, the list that opens

**Files:**
- Create: `src/query_processing/deploy/domino_listings/opening_list.py`
- Modify: `src/query_processing/deploy/domino_listings/listing_commands.py`, `pyproject.toml`
- Test: `tests/deploy/domino_listings/test_opening_list.py`, `tests/deploy/domino_listings/test_listing_commands.py`
- Modify: `README.md` (the tree; the install line of the extra)

**Interfaces:**
- Consumes: Task 7's `show_tiers`, `show_environments`, `shown_revisions`, `Look`.
- Produces:
  - `pyproject.toml`: the optional extra `terminal = ["textual>=8.2"]`, and the same line in `dev`
  - `opening_list.py`, the only module that imports Textual:
    - `@dataclass(frozen=True) class OpeningItem`: `label: str`, `lines: Callable[[], Sequence[str]]` (called once, when the item is first opened), `starts_open: bool = False`
    - `class OpeningList(App[None])`: `__init__(self, title: str, items: Sequence[OpeningItem])`; one `Tree` whose root is hidden, one node an item; an `Input` that `/` shows and focuses
    - `run_opening_list(title: str, items: Sequence[OpeningItem]) -> None`
  - `listing_commands.py`: `tier_items(tiers: Sequence[HardwareTier], opened: Collection[str]) -> list[OpeningItem]`, `environment_items(environments: Sequence[DominoEnvironment], internal: BaseRevisionSource, opened: Collection[str]) -> list[OpeningItem]`, and `list_is_installed() -> bool` (`importlib.util.find_spec("textual") is not None`)

- [ ] **Step 1: Write the failing tests** in `test_opening_list.py` (async tests need no mark: `asyncio_mode = "auto"`)

```python
def items(calls: list[str]) -> list[OpeningItem]:
    def lines_of(name: str) -> Callable[[], Sequence[str]]:
        def lines() -> Sequence[str]:
            calls.append(name)
            return [f"{name} line 1", f"{name} line 2"]

        return lines

    return [OpeningItem(name, lines_of(name)) for name in ("cpu-8", "cpu-16", "gpu-1")]


def shown(app: OpeningList) -> list[tuple[str, list[str]]]:
    """Each visible item with the lines under it when it is open."""
    tree = app.query_one(Tree)
    return [
        (str(node.label), [str(child.label) for child in node.children] if node.is_expanded else [])
        for node in tree.root.children
    ]


async def test_space_opens_the_item_under_the_cursor_and_closes_it_again() -> None:
    calls: list[str] = []
    app = OpeningList("Hardware tiers", items(calls))
    async with app.run_test() as pilot:
        await pilot.press("down", "space")
        assert shown(app)[1] == ("cpu-16", ["cpu-16 line 1", "cpu-16 line 2"])
        await pilot.press("space", "space")
        assert shown(app)[1][1] == ["cpu-16 line 1", "cpu-16 line 2"]
    assert calls == ["cpu-16"]  # asked once, at the first opening


async def test_slash_narrows_the_list_to_the_names_holding_the_text() -> None:
    app = OpeningList("Hardware tiers", items([]))
    async with app.run_test() as pilot:
        await pilot.press("/", "c", "p", "u")
        assert [label for label, _ in shown(app)] == ["cpu-8", "cpu-16"]
        await pilot.press("escape")
        assert [label for label, _ in shown(app)] == ["cpu-8", "cpu-16", "gpu-1"]


async def test_q_quits() -> None:
    app = OpeningList("Hardware tiers", items([]))
    async with app.run_test() as pilot:
        await pilot.press("q")
    assert app.return_code == 0
```

The first key press of the first test assumes the cursor starts on the first item: if Textual starts it elsewhere, move it there when the list is mounted, and keep the test.

And:

| Test | Set-up | Expected |
|---|---|---|
| `test_an_item_that_starts_open_shows_its_lines_at_once` | `OpeningItem("cpu-8", lines, starts_open=True)` | its lines shown before any key; `lines` called once |
| `test_an_item_whose_lines_fail_shows_the_failure_and_the_list_stays_up` | `lines` raises `DominoDeployError("the route answered HTTP 404")` | the item's one line is `the route answered HTTP 404`; a following `down` still moves the cursor |
| `test_the_line_of_keys_is_shown` | any | the screen's text holds `up/down move`, `space open/close`, `/ filter` and `q quit` |

In `test_listing_commands.py`:

| Test | Set-up | Expected |
|---|---|---|
| `test_a_tiers_item_is_its_name_size_and_id_and_opens_on_every_field` | `tier_items(TIERS, opened=())` | label `cpu-16   16 cores, 64 GiB   id large-k8s`; `lines()` is the tier's details as YAML lines |
| `test_an_environment_asks_its_revisions_only_when_opened` | `environment_items([environment], internal, opened=())` on the fake | label `qp-rag   id env-1`; no internal call until `lines()` is called; then `id env-1`, one line a revision (`v7  latest  built 2026-11-12 09:41  id rev-7` first), then the documented fields as YAML lines; one internal call |
| `test_an_environment_whose_route_fails_shows_the_three_and_the_note` | `revisions_route_status = 404` | the lines hold `v7`, `v6` and the note of Task 4 |
| `test_a_named_item_starts_open` | `tier_items(TIERS, opened=["cpu-16"])` | `starts_open` true for `cpu-16` alone |
| `test_the_list_look_runs_the_list_when_textual_is_there` | `show_tiers(http, [], "list")` with `run_opening_list` replaced by a named fake recording its arguments | called once with title `Hardware tiers` and three items; nothing printed |
| `test_the_list_look_without_the_extra_prints_the_table_and_how_to_install_it` | `list_is_installed` giving false | the table on the output; on the error output `qp-deploy: the interactive list needs the toolkit's "terminal" extra (pip install -e "query_processing[terminal]" from the repository's root); here is the table` |
| `test_the_core_dependencies_do_not_name_textual` | `tomllib` on `pyproject.toml` | no `textual` in `project.dependencies`; `textual` in the `terminal` and `dev` extras |
| `test_only_the_opening_list_imports_textual` | every module under `src/query_processing` read with `ast` | `opening_list.py` alone imports `textual` |

- [ ] **Step 2: Run them, expect failures**

Run: `pytest tests/deploy/domino_listings -q`
Expected: `No module named 'query_processing.deploy.domino_listings.opening_list'` (install the extra first if `textual` is missing: `pip install "textual>=8.2"` into the tools' environment).

- [ ] **Step 3: Write the list and wire the look**

The `Tree` gives `up`, `down` and `space` (its `toggle_node`); `OpeningList` adds the bindings `/` (show and focus the `Input`), `escape` (clear and hide it, back to the tree) and `q` (quit). An item's lines are added as leaves on `Tree.NodeExpanded`, the first time only; `lines` raising `DominoDeployError` adds its message as the one leaf. Narrowing rebuilds the root's children from the items whose label holds the text typed, whatever the capitals. `listing_commands` imports `opening_list` inside the function that runs the list, so that the module loads without Textual.

- [ ] **Step 4: Run, expect a pass**

Run: `pytest tests/deploy tests/structure -q`

- [ ] **Step 5: README, then commit**

Commit: `In a terminal qp-deploy tiers and qp-deploy environments are a list whose items open with the space bar, drawn by Textual from an optional extra`.

---

### Task 9: The example file and the README tell it all

**Files:**
- Modify: `query_processing_rag_usecase/deploy.example.yaml`, `README.md`
- Test: `query_processing_rag_usecase/tests/test_deploy_dry_run.py`

**Interfaces:**
- Consumes: the behaviour of Tasks 5 to 8. Produces no code.

- [ ] **Step 1: Run the use case's tests on the example as it is**

From `query_processing_rag_usecase/`: `pytest tests/test_deploy_dry_run.py -q`. Expected: pass. Note what the tests quote of the example (`SERVED_LLM` quotes `#   hardware_tier_id: real`).

- [ ] **Step 2: Edit the example**

Under `defaults`, the comments and the two lines `environment_revision_id: ""` and `hardware_tier_id: ""` are replaced by the block of the spec's section "`deploy.example.yaml`, the lines that change", word for word. The comment above `apps` and the two commented tier lines of `embedder` and `reranker` change as that section says. The commented LLM block keeps `hardware_tier_id: real`, which the test quotes.

- [ ] **Step 3: Edit the README**

In "Deploying on Domino 6.2": the sample deploy file shows `hardware_tier: cpu-16` in one App and keeps one `hardware_tier_id`; the paragraph "Environment revision." says the number, the order of the two sources and that the documented API holds three revisions; a paragraph "Hardware tier." says the name, the printed line and the refusals; a paragraph "What Domino has." tells the two commands with one table and one list sketch from the spec, their three looks, and the `terminal` extra. The fake's description names the new routes, the internal route's stand-in among the stand-ins. The list "not verified on a live Domino" gains the four lines of the spec's "Where the code goes", and the two commands of the spec's "The one check on a live Domino".

- [ ] **Step 4: Run, then commit**

Run `pytest tests -q` in `query_processing_rag_usecase/` and `pytest tests/structure -q` in `query_processing/`. Expected: pass.
Commit: `The example deploy file and the README name a hardware tier by its name and an environment revision by its number, and tell the two listing commands`.

---

## At the end

The orchestrator runs `make check` from the repository's root once (with `textual` in the tools' environment), then one review of the whole branch against the spec's "What would make this a failure" and this plan's Review Focus.
