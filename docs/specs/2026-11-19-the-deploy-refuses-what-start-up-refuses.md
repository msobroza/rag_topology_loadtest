# The deploy refuses what an App's start-up would refuse

Design written on 19 November 2026, after two reviewers of the orchestrator work found the gap, and approved by the owner the same day (approach A). Built the same day (ticket 13); "What was built" at the end says what differs. Names of functions below are indicative; one rule of the README changes.

## Why

An App reads the workflow file with the overrides of `deploy.yaml`. When it starts, it refuses two things before it opens anything: a name nobody registered (`check_registered_names`: a backend, an API format, a structured output method, the orchestrator), and, for the workflow App, a graph that cannot run or that the chosen orchestrator does not run (`check_workflow_graph`).

`qp-deploy` checks neither. It holds an override to the workflow settings' keys, types and ranges (`deploy/overrides.py`), then publishes. Seen on 19 November 2026, on fake backends, with `make deploy-dry-run` and with the reviewers' script:

| In `deploy.yaml` | `qp-deploy` | The App at start-up |
| --- | --- | --- |
| `overrides: {orchestrator: pydantic}` | exit 0, the entry script exports `QP_ORCHESTRATOR=pydantic` | `QP_ORCHESTRATOR: unknown orchestrator 'pydantic' (expected one of: 'langgraph', 'pydantic_graph')` |
| `overrides.components.reranker.backend: bogus` | exit 0, exports the value | refused, in every App |
| `orchestrator: pydantic_graph` with a second plain edge out of `rerank` | 3 Apps published | `orchestrator 'pydantic_graph': 'rerank' has more than one way out at once; ...` |

So a mistake a laptop could catch in a second is found on Domino, after every App was published and one of them, or all of them, stopped. The unknown backend is older than the orchestrator; the orchestrator made the gap one key wider.

## Goals

1. `qp-deploy deploy`, the dry run included, refuses before anything is published what `check_registered_names` and `check_workflow_graph` would refuse of the settings an App will read.
2. The refusal carries the line start-up prints, unchanged, after the two files it comes from.
3. The check is the App's own: no name is read a second way, and a new registry or a new check of start-up is covered with no change to the deploy tool.
4. A deploy file that deploys today, and whose Apps start, deploys unchanged.

## Non-goals

- What only an open component can tell (a model that is not there, a store that does not answer) stays start-up's and `/readyz`'s.
- `deploy/registered_backends.py`, which reads the backends from the openers' module names for a topology alias, is not replaced here, though the door below would allow it.
- No new flag and no new setting.

## Approaches

**A. One module of `deploy/` may import the two checks (chosen).** The deploy tool already builds the settings an App will read: the workflow file with the overrides over it, and nothing of this machine's variables or `.env` (`workflow_components.py`, `_file_with_overrides`). The two checks are called on that object, in process. About fifteen lines, the messages are start-up's by construction, and goal 3 holds. Its cost is one clause of README rule 1: the deploy package imports `deploy` and `domain`, with two named exceptions today (`qp_deploy.py` for the fake Domino, `workflow_components.py` for `config`); the second is widened by two functions.

**B. A child process asks the workflow App's own command.** A new `--check` flag on `qp-serve` and `qp-model-server` stops after the two checks, and `qp-deploy` runs it with the App's exports. No import rule changes. Rejected as the recommendation: a flag on two commands, an environment to rebuild by hand, and the child would read this machine's `.env`, which the App does not have; three more places where the check and the App can differ.

**C. Read the names from the source's layout**, as `registered_backends.py` does for backends. Rejected: an API format's module is not named after its registered name (`openai_chat_format.py` registers `openai`), each registry would need a reader of its own, and no reading of names can ask an orchestrator about a graph.

What rule 1 protects, and what A costs it: the deploy tool was kept from the composition so that publishing never builds an adapter or loads a model library. Importing the two checks builds nothing. Measured here: `qp-deploy` loads in 0.36 s, and the composition's checks in 0.55 s with LangGraph and pydantic-graph and without torch, sentence-transformers, OpenVINO or a store's client, which the adapters import only when a component opens.

## The design (approach A)

- **`deploy/workflow_components.py`**, the one module of the deploy tool that reads the workflow settings, is also the one that asks about them: `refuse_what_start_up_refuses(overrides, path)` builds the workflow file with the overrides, as `refuse_override_values` does today (one function now builds it for both), calls `check_registered_names` then `check_workflow_graph`, the order of `qp-serve`, and raises their `ValueError` as a `DominoDeployError`:

  `the workflow file 'workflow.yaml' with the overrides of deploy.yaml makes settings an App refuses at start-up: QP_ORCHESTRATOR: unknown orchestrator 'pydantic' (expected one of: 'langgraph', 'pydantic_graph')`

  The check has no module of its own for two reasons: `deploy/` holds 12 modules, the most README rule 9 allows; and the settings object then never leaves the one module allowed to import `config`, so no second module needs that exception to name its type.
- **`deploy/overrides.py`**, `check_overrides`: after its three checks, the new one. It no longer returns early when `deploy.yaml` has no overrides, since the workflow file alone can name an orchestrator nobody registered or hold a graph its orchestrator refuses. `publish.deploy` calls it where it does today, before the topology is resolved and before Domino is asked anything, so a dry run makes it too.
- **README rule 1**, the row of `deploy`: `workflow_components.py` alone may also import `config`, as today, and the composition's two start-up checks, to ask of the settings an App will read what its start-up would refuse. `tests/structure/test_layering.py` adds `composition` to that module's line in its table of exceptions.
- **The documents** made true again: the README's "Try the deploy without Domino" (every check of a real deploy but the two that ask Domino, now also what start-up refuses of the settings) and the comment above `overrides:` in the use case's `deploy.example.yaml`, which today says these are not checked.

What is checked is the workflow file with the overrides. The deploy tool's own exports (the backend and URL of each component the topology alias names) are not in it: the alias's backends are already held to the registered ones by `registered_backends.py`, and a URL changes neither check.

## Tests

Each line is one behaviour, at the deploy tool's seam (`deploy`, as the dry run calls it, on the fake Domino):

- An override naming an orchestrator nobody registered is refused with start-up's line, and nothing is published.
- The same for a backend, an API format and a structured output method nobody registered.
- A workflow file whose graph the chosen orchestrator does not run is refused with the orchestrator's line, with an override that chooses the orchestrator and with the orchestrator written in the file.
- A workflow file with a mistake of that kind and no override at all is refused.
- A deploy file that deploys today still deploys: the existing tests of the deploy tool pass unchanged.
- The message is, to the character, what `qp-serve` prints for the same settings, after the two files' names.
- Only `deploy/workflow_components.py` imports the composition, and of it only the two checks.

## Calls made in this document, for the owner to check

- README rule 1 widens the exception of `workflow_components.py` by the composition's two start-up checks; no other module of `deploy/` imports the composition.
- Both checks run for every target, the graph's too when only a model server is redeployed: the files describe one topology, and its workflow App would not start.
- The workflow file is checked even when `deploy.yaml` has no overrides.
- The settings checked are the workflow file with the overrides, without the deploy tool's own exports.
- `registered_backends.py` stays as it is.

## What was built

As designed, in `deploy/workflow_components.py` (`refuse_what_start_up_refuses`) and `deploy/overrides.py`, with three differences:

- The workflow file with the overrides is now read with the prompts file too, found as the routes file already was (the one `QP_PROMPTS_FILE` names, else `prompts.yaml` here, where the App runs): start-up's graph check refuses a stage that names a prompt the prompts file does not hold, so without it every workflow file with a generate stage was refused.
- The one function that builds the settings raises the refusal `refuse_override_values` gave before. Since the settings are now built when `deploy.yaml` has no overrides as well, a workflow file the workflow settings refuse on their own (one with no `graph`, say) is refused with that line, before start-up's two checks are asked.
- Three tests of the topology alias wrote a workflow file with components and no graph, which an App refuses at start-up; they now write the tests' workflow file with a second reranker beside its components, and keep their expectations. The emulation's test of its generated `deploy.yaml` named the use case's workflow and routes files but left the tests' prompts file, which lacks the use case's prompts; it now names the use case's prompts file too.
