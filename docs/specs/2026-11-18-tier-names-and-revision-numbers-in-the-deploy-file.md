# A hardware tier by its name and an environment revision by its number, in the deploy file

Design written on 18 November 2026 from the owner's request of that day, and extended the same day with the two listing commands. Nothing of it is built, and nothing of it was tried on a live Domino. Names of modules, classes and functions below are indicative; names a user types (in `deploy.yaml` or on a command line) are the design.

## Why

`deploy.yaml` takes two ids that a person copies by hand from Domino's pages: `hardware_tier_id` and `environment_revision_id`. Neither says what it is: a revision id is 24 hexadecimal characters, and the size of a tier is nowhere in the file.

The owner wants to write what a person knows, the tier's name and the revision's number ("v7"), and have `qp-deploy` ask Domino for the ids and for the tier's size. Their words: "I would like from number to id env and I would like to obtain hardware/name and characteristics from API to avoid hard code".

To know which name and which number to write, the owner also wants to ask Domino from the command line: "we should be able via CLI to list env availables names with a toogle, obtain the environment_ids and available revisions and all infos (date env build etc), the same for ressources and hardwares".

## Goals

1. An App block of `deploy.yaml` names its hardware tier by the tier's name in Domino, and `qp-deploy` sends that tier's id.
2. `qp-deploy` prints the cores, the memory and the GPUs of each tier named, as Domino gives them.
3. An App block names its environment revision by its number, and `qp-deploy` sends that revision's id.
4. Any revision number can be found: through Domino's documented API when it holds the number, and through an internal route of Domino otherwise.
5. A name or a number Domino does not have is refused before any App is published, with what Domino does have.
6. A deploy file that works today works unchanged and makes no new call to Domino.
7. `qp-deploy tiers` shows Domino's hardware tiers with their name, id, cores, memory and GPUs, and on request every other field Domino holds for a tier.
8. `qp-deploy environments` shows the environments the person can see with their name and id, and for an environment named or opened, every revision with its number, its id and the date it was built.
9. In a terminal both commands are a list a person moves through and opens item by item; anywhere else they print a table that reads the same in a file.

## Non-goals

- The environment itself is not named by its name in `deploy.yaml`: `environment_id` stays, and `qp-deploy environments` gives it.
- No word such as `latest` or `active` in place of a revision number.
- The size of a tier written as an id is not printed at a deploy (it would be a new call for today's files).
- The image, the Dockerfile and the scripts of a revision are not shown: they need one more internal call for each revision.
- The listing commands write nothing: no command picks a value and writes it into `deploy.yaml`.
- No JSON output, and no colour in the table or the details.
- No search on Domino's side: names on the command line are exact, and the list's filter narrows what was already read.
- The thread budget of a running App (`infrastructure/local_models/cpu_count.py`) does not change.
- No file remembers an id once found: Domino is asked at each deploy.
- The build status of a revision is not checked at a deploy: Domino refuses a revision it cannot start.
- The route `/v4/projects/{projectId}/hardwareTiers`, which Domino's Python library calls, is not used.

## What would make this a failure

- A deploy file that works today is refused, or makes a call it did not make.
- A mistyped tier name publishes the App on the project's default tier.
- An App is published before a wrong name or number of a later App is found.
- A dry run asks Domino something.
- The internal route is called for a number the documented API holds, or a deploy that names no revision number fails because of it.
- `qp-deploy` sends an id it did not read from Domino or from the deploy file.
- A listing command changes something in Domino, or its details hide a field Domino gave behind a list of fields written by hand.
- The interactive list starts where no person is there to quit it (a pipe, a script, a test), and the command hangs.
- An App installs the library that draws the list.
- Code of the emulated Domino is imported by the product.

## What Domino offers

Read in the Domino 6.2 Platform API reference, and in the published specs of 6.3.1 and of Domino Cloud 7.0.0, on 18 November 2026.

| Call | What it gives | Permission |
|---|---|---|
| `GET /api/hardwaretiers/v1/hardwaretiers` | every tier, by pages | `ViewHardwareTiers` |
| `GET /api/environments/beta/environments` | every environment the person can see, by pages | none |
| `GET /api/environments/v1/environments/{environmentId}` | one environment with three of its revisions | `ViewEnvironment` |

**The tiers.** The query parameters are `offset` (default 0), `limit` (default 10) and `includeArchived` (default false). The answer is a `PaginatedHardwareTierEnvelopeV1`: `hardwareTiers`, a list of `HardwareTierV1`, and `metadata` with `offset`, `limit` and `totalCount`. A deploy reads of a tier `id`, `name`, `resources.cores`, `resources.memory` (`value` and `units`) and `gpuConfiguration.numberOfGpus` (the block is optional). A tier holds more, which `qp-deploy tiers --details` prints as given: its flags, its node pool, its price, its dates of creation and update, its capacity.

**The environments.** The list is a beta call "with known limitations", with `offset` and `limit` (default 10). Its answer is a `PaginatedEnvironmentEnvelopeV1`: `environments`, a list of `EnvironmentV1`, and `metadata`. The call for one environment answers an `EnvironmentEnvelopeV1`: `environment` and `metadata`. An `EnvironmentV1` carries its `id`, `name`, `visibility`, `owner`, `archived` and, each optional, `latestRevision`, `selectedRevision` (the active one) and `restrictedRevision`. Each is an `EnvironmentRevisionV1` with `id`, `number` and `status`.

**What is missing.** No version of the documented API lists an environment's revisions, finds one by its number, or tells the date a revision was built. So the documented API turns a number into an id for three revisions at most.

**The internal route.** Domino's own pages list every revision of an environment. The published 6.3.1 spec holds the shape of that list among its components, though not its path:

- `PaginatedRevisionData`: `revisions`, a list of `RevisionSummary`, and `pageInfo` with `totalPages`, `currentPage` and `pageSize`.
- `RevisionSummary`: `id`, `number`, `created` (an epoch) and `active`.

The path is not published. From memory it is `GET /v4/environments/{environmentId}/paginatedRevisionData`, with `page` and `pageSize`; the name of the shape agrees with it. This design assumes that path, that `created` counts milliseconds, and that the route answers through `DOMINO_API_PROXY`. See "The one check on a live Domino".

## Approaches

For the tier there is one sensible way, the documented list call, since it gives the name, the id and the size in one answer.

For the revision number, three ways were considered:

1. **The documented API alone.** Built and tested against the reference. A number that is no longer the latest, the active or the restricted revision cannot be found, which is the day a pin matters most: a newer revision was built.
2. **The internal route alone.** Any number is found. Every deploy that names a number then depends on a route Domino does not document.
3. **Both, the documented API first.** The documented call answers when it holds the number; the internal route is asked only for the others. A change of the internal route in a later Domino then breaks only the pins of older revisions.

The owner chose both ("you can do both ?"). That the documented API is asked first is this document's call, for the reason given in 3.

For the listings, two commands of `qp-deploy`, which already holds the connection to Domino and the settings. Three looks were shown to the owner:

1. **Tables only.** Aligned columns with marks, no interaction, no new library.
2. **Tables and a live toggle.** The tables, and in a terminal an interactive list whose items open and close. One new library, in an optional extra.
3. **A guided pick that writes.** The second, and a command that asks which environment, revision and tier each App takes and writes them into `deploy.yaml`.

The owner chose the second.

## The deploy file

An App block (`defaults`, or one entry of `apps`) gains two keys. Each is a second way to write a setting the block already has.

| Setting | By what a person knows | By its id (as today) |
|---|---|---|
| The hardware tier | `hardware_tier: cpu-16` | `hardware_tier_id: <id>` |
| The environment revision | `environment_revision: 7` | `environment_revision_id: <id>` |

The rules:

- **An empty value is not set**, as today for the two id keys: `hardware_tier_id: ""` or `hardware_tier: ""` in a block is the same as the key left out. A file copied from today's example, which holds both id keys as `""`, therefore takes a new key beside them. An App's `""` does not replace the default.
- **One way per block.** A block that sets both keys of one setting, each to a value that is not empty, is refused when the file is read, naming the block and both keys.
- **The App's own wins.** An App's own tier, written either way, replaces the default tier, written either way. The same holds for the revision.
- **The revision goes with its environment**, as today: the default revision, written either way, goes only with the default environment, and an App that sets its own `environment_id` does not take it. A default revision without a default `environment_id` is refused, and so is an App with a revision but no environment.
- **`hardware_tier`** is the tier's name exactly as Domino shows it, capitals included. A value still holding an `<angle-bracket>` placeholder is refused, as for every other key.
- **`environment_revision`** is a whole number from 1 up, written bare (`7`, not `"7"` nor `v7`). Anything else is refused when the file is read.
- A key left out means what it means today: the project's default tier, the revision Domino chooses.

### `deploy.example.yaml`, the lines that change

The example leads with the two new keys. Under `defaults`, these lines take the place of the comments and the two lines `environment_revision_id: ""` and `hardware_tier_id: ""`:

```yaml
  # Optional: holds the Apps to one build of that environment, by its revision
  # number as Domino shows it (v7 is 7): qp-deploy asks Domino for that
  # revision's id and sends it as environmentRevisionId. Left out, Domino
  # chooses. It goes only with the environment above: an App that sets its own
  # environment_id does not take it, and without environment_id it is refused.
  # The id itself can be written instead, as environment_revision_id.
  # `make environments` shows the environments and their revisions.
  # environment_revision: 7
  # Optional: the hardware tier, by its name in Domino: qp-deploy asks Domino
  # for its id and prints its cores and memory. Left out, the project's
  # default. The id itself can be written instead, as hardware_tier_id.
  # `make tiers` shows the tiers and their sizes.
  # hardware_tier: <tier name>
```

In the comment above `apps`, `environment_revision_id` and `hardware_tier_id` become "`environment_revision` or `environment_revision_id`" and "`hardware_tier` or `hardware_tier_id`". In the Apps' blocks:

```yaml
  embedder:
    kind: app
    # hardware_tier: <a tier with 8 cores>
  reranker:
    kind: app
    # hardware_tier: <a tier with 16 cores>
```

## What `qp-deploy deploy` does

After the checks it makes today and before it publishes the first App, `qp-deploy` finds the id behind every name and number that the Apps being deployed use. A wrong value therefore stops the deploy with nothing published.

1. **Tiers.** When at least one of those Apps names a tier, `qp-deploy` lists Domino's tiers once, page after page from `offset` 0 until it has read `totalCount` tiers or a page comes back empty, and looks each name up in that list.
2. **Revisions.** For each environment of which a revision number is named, `qp-deploy` asks Domino once and looks each number up, as "Finding a revision" says.
3. **What it prints**, one line per App and setting found, before publishing:

```
qp-rag-reranker: hardware tier cpu-16 (id large-k8s): 16 cores, 64 GiB
qp-rag-reranker: environment revision 7 (id 62313cfd7a0af0281c01a6a6)
```

   A tier with GPUs adds `, 1 GPU`. The memory is printed with the unit Domino gives.
4. **What it sends.** The App's version carries `hardwareTierId` and `environmentRevisionId` as today, with the ids found.

An App whose block gives ids is published as today, with no lookup.

`qp-deploy status` and `qp-deploy url` look nothing up.

### Finding a revision

The number is looked for in two places, in this order, and the first that holds it gives the id:

1. **The documented call.** The environment's `latestRevision`, `selectedRevision` and `restrictedRevision`.
2. **The internal route**, asked only when none of those three has the number: every revision of the environment.

The two are implementations of one abstract class, asked in order, as the CPU counters of `cpu_count.py` are. The internal one has a module to itself, whose docstring says, as `endpoint_url.py` does for its path, that the route is not in Domino's reference and where its shape comes from.

The internal route is read without assuming how its pages are numbered: one call with no parameter tells `pageSize` and `totalPages`, and when there is more than one page, a second call asks one page of `pageSize` times `totalPages`.

### What is refused, and what the message says

Each is a `DominoDeployError`, shown by `qp-deploy` without a traceback, before anything is published.

| Case | The message gives |
|---|---|
| No tier has the name | the name, the App, that its block or `defaults` sets it, and every tier Domino listed: name, id, cores, memory |
| Two tiers or more have the name | the name and the ids of those tiers, and says to write `hardware_tier_id` |
| The tiers call answers 401 or 403 | Domino's answer, the permission `ViewHardwareTiers`, and says `hardware_tier_id` needs no such call |
| The environment call answers 404 | the environment id |
| The environment call answers 401 or 403 | Domino's answer and the permission `ViewEnvironment` |
| No source has the number | the number, the environment id, the numbers of the latest, the active and the restricted revisions, what the internal route gave, and says `environment_revision_id` takes the id itself |
| The internal route fails (no answer, a status other than 200, an answer of another shape) | the same as the line above, and the route with what it answered |

A failure of the documented environment call is final: the internal route is not asked in its place.

The calls go through the HTTP client `qp-deploy` already opens on `DOMINO_API_PROXY`, so they take its timeout and its retries (`http_timeout_s`, `http_retries`) and add no constant.

## The two listing commands

Both only read, and both take names to narrow what they show. Each has three looks, chosen by the owner on 18 November ("Add more friendly UI in CLI", then the look "tables and a live toggle"):

| Look | When | What it is |
|---|---|---|
| The list that opens | in a terminal, by default | an interactive list: the arrows move, the space bar opens or closes an item |
| The table | where there is no terminal (a pipe, a file, a script), or with `--table` | plain text in aligned columns, with marks |
| The details | with `--details` | every field Domino gave, as YAML |

### `qp-deploy tiers [NAME ...] [--table | --details]`

The table, in Domino's order, with a mark on the default tier:

```
$ qp-deploy tiers --table
NAME    CORES  MEMORY  GPUS  DEFAULT  ID
cpu-8       8  32 GiB     -  yes      small-k8s
cpu-16     16  64 GiB     -           large-k8s
gpu-1       8  32 GiB     1           gpu-k8s
```

With `--details`, every field Domino gave for each tier, as YAML, the name first:

```
$ qp-deploy tiers cpu-16 --details
- name: cpu-16
  id: large-k8s
  resources:
    cores: 16.0
    memory: {value: 64.0, units: GiB}
    allowSharedMemoryToExceedDefault: false
  flags: {isDefault: false, isVisible: true, isGlobal: true, isArchived: false}
  nodePool: default
  centsPerMinute: 0.0
  creationTime: '2026-03-12T02:13:44.467Z'
  updateTime: '2026-03-12T02:13:44.467Z'
```

No field is picked by hand: what Domino answers is what is printed, so a field of another Domino version shows too.

### `qp-deploy environments [NAME ...] [--table | --details]`

The table, one line an environment, with the numbers of its active and latest revisions:

```
$ qp-deploy environments --table
NAME                                     VISIBILITY  OWNER  ACTIVE  LATEST  ID
Domino Standard Environment Py3.10 R4.5  global      admin       3       3  6231328...01a69c
qp-rag                                   private     alice       7       7  6231328...01a6a0
```

For an environment **named on the command line**, the table is its revisions, newest first, through the internal route. The dates are in UTC, and the marks say which revision is active, latest or restricted:

```
$ qp-deploy environments qp-rag --table
qp-rag   id 623132867a0af0281c01a69c   private, owner alice

REVISION   BUILT (UTC)       STATUS     ID
7  active  2026-11-12 09:41  succeeded  62313cfd7a0af0281c01a6a6
6          2026-11-03 16:02             6229a1b07a0af0281c01a511
```

An id is printed whole; it is shortened above only to fit the page. The status is known for the three revisions the documented API gives, and left empty for the others.

With `--details`, every field of the documented answer for each environment, the name first, its three revisions included:

```
$ qp-deploy environments --details
- name: qp-rag
  id: 623132867a0af0281c01a69c
  visibility: private
  owner: {id: 6231327c7a0af0281c01a69b, username: alice, ownerType: Individual}
  archived: false
  latestRevision: {id: 62313cfd7a0af0281c01a6a6, number: 7, status: succeeded}
  selectedRevision: {id: 62313cfd7a0af0281c01a6a6, number: 7, status: succeeded}
```

For an environment named on the command line, `--details` adds every revision, newest first:

```
$ qp-deploy environments qp-rag --details
- name: qp-rag
  id: 623132867a0af0281c01a69c
  ...
  revisions:
  - {number: 7, id: 62313cfd7a0af0281c01a6a6, built: '2026-11-12 09:41 UTC', active: true}
  - {number: 6, id: 6229a1b07a0af0281c01a511, built: '2026-11-03 16:02 UTC', active: false}
```

### In a terminal: the list that opens

When both the input and the output are a terminal, each command opens an interactive list in place of the table:

```
$ qp-deploy environments

  up/down move   space open/close   / filter   q quit

  > Domino Standard Environment Py3.10 R4.5
  v qp-rag             id 623132867a0af0281c01a69c
      v7  active  built 2026-11-12 09:41  id 62313cfd7a0af0281c01a6a6
      v6          built 2026-11-03 16:02  id 6229a1b07a0af0281c01a511
  > team-gpu-env
```

- The arrows move the cursor, the space bar opens or closes the item under it, `/` narrows the list to the names holding the text typed, and `q` quits.
- An open tier shows every field Domino gave for it. An open environment shows its id, its documented fields and every revision with its date and marks.
- An environment's revisions are asked of Domino when it is first opened, so the internal route is called only for what the person looks at.
- Names given on the command line narrow the list, and those items start open.
- It writes nothing: neither to Domino nor to `deploy.yaml`.

The list is drawn by the Textual library, behind one class of the toolkit's own. Textual sits in an optional extra of the toolkit (proposed: `terminal`), which the `dev` extra includes, so that an App, which installs the toolkit at start-up, does not install it. In a terminal without the extra, the command prints the table and one line on the error output saying how to install it.

### The rules of both commands

- A name given that Domino does not have is refused, with the names Domino has.
- Two environments or two tiers can share a name: each is shown.
- The revisions are asked only for environments named or opened, one internal call each, so that no look makes a call for every environment a person can see.
- When the internal route fails, the environment's documented fields are still shown, with one line saying what the route answered in place of the revisions.
- The table and the details hold no colour and no cursor movement, so they read the same in a file.
- With `--dry-run` both are refused: they ask Domino, and a dry run asks nothing.
- They read `deploy.yaml` only for how to reach Domino (`http_timeout_s`, `http_retries`): no App block needs to be valid.

The use case's `Makefile` gains two targets:

```make
tiers: ## On Domino: the hardware tiers with their size, a list to move through in a terminal; make tiers ARGS="cpu-16 --details" for every field
	$(BIN)qp-deploy tiers $(ARGS)

environments: ## On Domino: the environments, a list to move through in a terminal; make environments ARGS="<name> --table" for its revisions and their build dates
	$(BIN)qp-deploy environments $(ARGS)
```

## What a dry run shows

A dry run asks Domino nothing, so it cannot know an id. It shows the name or the number as written:

```
qp-rag-reranker
  kind: app (configurationType STANDARD)
  environmentId: env-123
  environmentRevisionId: (revision 7: its id is asked of Domino at deploy)
  hardwareTierId: (tier cpu-16: its id is asked of Domino at deploy)
```

The rules of "The deploy file" are checked in a dry run as in a deploy, since they need no call.

## The fake Domino

The fake is transcribed from the reference, and gains what `qp-deploy` now calls:

- `GET /api/hardwaretiers/v1/hardwaretiers`: the tiers of its state, by pages, with `offset`, `limit` (default 10) and `metadata.totalCount`, in the reference's shape. The `forbidden` behaviour answers it 403.
- `GET /api/environments/beta/environments`: the environments of its state, by pages, in the reference's shape.
- `GET /api/environments/v1/environments/{environmentId}`: an environment of its state with its three revisions, in the reference's shape; 404 for an id it does not have. The `forbidden` behaviour answers it 403.
- `GET /v4/environments/{environmentId}/paginatedRevisionData`: every revision of an environment of its state, in the published shape, among the stand-ins that are not Domino's documented API.

Its state gains tiers and environments that a test sets, each environment with all its revisions and their dates. A fake started with none lists no tier and has no environment. The call for one tier by its id keeps answering for any id, as today.

**The emulated Domino** serves the list of tiers from its placement file, in `fake_domino/emulation/tiers.py`, where a tier's name is its id: `hardware_tier: cpu-8` then finds `cpu-8`, and `qp-deploy tiers` lists the pools with their real sizes. It has no environments, so `qp-deploy environments` lists none there and a deploy file for it names no revision number. Giving it environments is a design of its own, since it changes the emulation's Terraform. Nothing of the emulation is imported by the product, as today.

The deploy file the emulation writes (`emulation/terraform/scripts/workspace.sh.tftpl`, with `hardware_tier_id`) stays on ids, so a sitting runs as it does today, with no new call. No record under `experiments/` is touched.

## Where the code goes

- **`deploy/deploy_settings.py`**: the two keys and their rules. The file holds 498 lines of the 500 the code-shape test allows, so something leaves it; what leaves is the builder's call.
- **Two sub-packages of `deploy/`**, since `deploy/` holds 11 modules and the directory rule allows 12. Their tests mirror them.
  - Proposed `deploy/domino_lookups/`, for what asks Domino: a module for the tiers, one for the documented revisions with the abstract class, one for the internal route, one for the environments list, and the ones that find an App's ids.
  - Proposed `deploy/domino_listings/`, for what shows the answers: the tables, the details, and the interactive list, the only module that imports Textual.
- **`pyproject.toml`** of the toolkit: the optional extra holding Textual, also in `dev`.
- **`deploy/publish.py`**: the step that finds the ids before the first App is published. The file holds 475 lines.
- **`deploy/qp_deploy.py`**: the two commands on the command line.
- **`fake_domino/`**: the routes beside the stand-ins, the shapes in `api_shapes.py`, the tiers and environments in the state.
- **`query_processing_rag_usecase/Makefile`**: the two targets above.
- **`README.md`**: the "Deploying on Domino 6.2" section (the two keys, the order of the two sources, what is printed, the two commands), the fake's description, the layout list, and the list "not verified on a live Domino", which gains:
  - the tiers call answers for the person who deploys, who needs `ViewHardwareTiers`;
  - the environment calls answer through `DOMINO_API_PROXY`;
  - the internal route: its path, its two parameters, that it answers through `DOMINO_API_PROXY`, and that `created` counts milliseconds;
  - the interactive list draws and takes its keys in the terminal of a Domino Workspace.

## Tests

Each line is one behaviour a test proves, on the fake Domino unless said.

- A setting written by name or number and the same setting written by id publish the same version: one parametrized test for the tier, one for the revision.
- Both keys of one setting in one block are refused, in `defaults` and in an App's block.
- An App's own tier, either way, replaces the default's, either way; the same for the revision; the default revision does not follow an App to its own environment.
- A block holding an id key as `""` beside the other key of the same setting is accepted, and an App's `""` does not replace the default.
- A placeholder in `hardware_tier` is refused; `environment_revision` of 0, of a negative number or of a text is refused.
- A tier is found on the second page of the list.
- An unknown tier name, a name two tiers share, and a 403 each give their message, and the fake received no App.
- A number is found as the latest, as the active and as the restricted revision, and the internal route was not called.
- A number none of the three has is found by the internal route, on one page and on several.
- A number no source has, and an internal route that fails, each give their message with the three numbers.
- A deploy that names no revision number never calls the internal route.
- A deploy file with ids only makes the calls it makes today and no other.
- A deploy of one App looks up only what that App uses.
- The dry-run report shows the two lines above and the fake received no lookup.
- Outside a terminal `qp-deploy tiers` prints its table, columns aligned and the default tier marked; with names, only those; with `--details`, every field the fake gave, a field the shapes do not name included.
- Outside a terminal `qp-deploy environments` prints its table; for a named environment, its revisions with their dates, newest first, each marked active, latest or restricted, and the internal route is called for that environment alone; with `--details`, the same as YAML.
- A name neither command finds is refused with the names Domino has; two environments of one name are both shown.
- An internal route that fails leaves the documented fields shown, with what the route answered.
- Both commands are refused in a dry run, and neither sends anything but GET.
- The interactive list, driven without a terminal by Textual's test pilot: the arrows move, the space bar opens and closes an item, `/` narrows the names, `q` quits; an environment's revisions are asked when it is first opened and not again; a named item starts open.
- The list starts only when input and output are both a terminal: with either one redirected, and with `--table`, the table is printed and the command ends.
- In a terminal without the extra, the table is printed, with the line saying how to install it.
- The toolkit's core dependencies do not name Textual.
- The fake's routes answer in the reference's shapes and the published internal shape, and its Swagger lists them.
- The emulated Domino lists the tiers of its placement file.
- The use case's example file still passes its dry-run test.

`make check` is green at the end.

## The one check on a live Domino

Only the owner can reach a Domino. Since the shape of the internal route is published, nothing waits for this check: the route is built on the path assumed above and listed in the README as not verified. One command from a Workspace then tells whether the assumption holds; it prints the status and the field names, not the values:

```bash
curl -s -o /tmp/revisions.json -w "HTTP %{http_code}\n" "$DOMINO_API_PROXY/v4/environments/<environment id>/paginatedRevisionData"
```

```bash
python3 -c 'import json; shape = lambda v: {k: shape(x) for k, x in v.items()} if isinstance(v, dict) else [shape(v[0])] if isinstance(v, list) and v else type(v).__name__; print(json.dumps(shape(json.load(open("/tmp/revisions.json"))), indent=1))'
```

Expected: `HTTP 200`, then `revisions` with `id`, `number`, `created`, and `pageInfo` with `totalPages`, `currentPage`, `pageSize`. If the status is another, the browser's network panel on the Revisions tab of an environment's page shows the real request, and the path is corrected in the one module that holds it. Until then a number the documented API lacks is refused with what the route answered, and the listing prints the same in place of the revisions.

## Order of work

1. The fake Domino: the documented routes, the internal route's stand-in, their shapes and the state; the emulated Domino's list of tiers.
2. The deploy file: the two keys, their rules.
3. The tier by its name: the lookup, the step before publishing, the printed line, the dry-run line.
4. The revision by its number: the documented source, then the internal one; the printed line, the dry-run line.
5. The two listing commands as tables and details, and the `Makefile`'s targets.
6. The interactive list, with its extra.
7. The example file and the README.

## Left to the builder

- The names of the sub-package, the modules, the classes and the functions.
- How the names and numbers reach the dry-run report.
- Where the helper that calls Domino and reads its failures lives, now that calls other than the Apps API's use it, and the wording "Domino API" in their messages.
- What leaves `deploy_settings.py` to keep it under 500 lines.
- How the fake's state is given its tiers and environments.
- The exact layout of the YAML the details print, beyond "the name first, then every field as Domino gave it".
- The wording and spacing of the list's lines and of its line of keys, beyond the sketch above.

## Calls made in this document, for the owner to check

- The documented API is asked before the internal route.
- The internal route is built now, on a path remembered and a shape published, and listed as not verified; it no longer waits for the check.
- A tier or a revision written as an id is not looked up, so its size is not printed.
- A tier name is matched exactly, and a name two tiers share is refused.
- A dry run shows the name and the number, looks nothing up, and refuses the two listing commands.
- An id key left as `""` is not set, so it can stand beside the new key.
- The interactive list is the default in a terminal; `--table` gives the table there, and `--details` every field Domino gave, as YAML.
- Every revision with its date is asked only for environments named on the command line or opened in the list.
- The list is drawn by Textual, in an optional extra named `terminal`; without it a terminal gets the table and a line saying how to install it.
- The list moves, opens, filters and quits, and nothing else: it copies nothing and writes nothing.
- The tables are plain text, with no colour.
- The code is in two sub-packages, `deploy/domino_lookups/` and `deploy/domino_listings/`.
- A revision's image, Dockerfile and scripts are left out; they are one more internal call a revision, to add if wanted.
- The emulated Domino lists tiers and has no environments, and the deploy file it writes stays on ids; giving it environments is a separate design.
- The tiers route of Domino's Python library is left out; if the tiers call answers 403 on the owner's Domino, it is the next thing to add.
