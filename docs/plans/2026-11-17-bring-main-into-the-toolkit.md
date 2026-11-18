# Bring `main` into the toolkit's layout

A record of the plan written on 17 November 2026 and of what became of it. The plan was approved that day with each of its seven decisions taken as recommended, and its last stage ended the same day: pull request 29, merged as `0488bda`.

## Where things stood

- The toolkit's work was on `claude/use-case-components` at `1a279f0`. It contained the commits of pull request 19, which conflicted with `main`.
- `main` was at `dcb6e9b`. Since that work's base (`de3bba8`, pull request 16) it had gained nine pull requests, all written in the layout from before the toolkit: one package `src/rag_load_test/`, commands `rag-*`, variables `RAG_*`.
- Two of the nine (17 and 18) had already been carried to the toolkit's layout on a local branch (`1c87755`, `530ca47`).
- The merges were simulated with `git merge-tree`, which writes nothing to the checkout:

| Merge simulated | Conflicted paths |
|---|---|
| `main` into the branch, directly | 103 |
| Stage A: the earlier port into the branch | 3 files, 7 hunks |
| Stage B: the rest of `main` into that | 56 (27 content, 24 changed on `main` where the toolkit replaced the module, 5 new files in a moved folder) |

So the merge was done in two stages.

## What `main` had gained, and where each piece landed

| Pull request | What it holds | Where it landed |
|---|---|---|
| 17, export precision | The weight format of an OpenVINO or OVMS export, as a setting | Stage A |
| 18, OVMS stream calibration | Calibration of OVMS streams | Stage A |
| 22, test command | `.gstack/pregate.json`, a ruff exclude, lines of `CLAUDE.md` | Stage B1, by hand: its runner assumed one tests folder, and the repository has three |
| 20, experiments registry | `experiments/`: the sittings' records, `docs/model-servers.md` | Stage B1, as it is, at the root |
| 21, experiments explained | `experiments-explained/`, the package `experiment_charts`, `make experiment-charts`, its tests | Stage B1: the folder as it is, the package in a distribution of its own (decision 2) |
| 23, 25, 26, three sittings | Their records, two more charts | Stage B1, as they are |
| 27, the LLM served by OVMS | See below | Stage B2: said again in the toolkit's concepts |

### Pull request 27, piece by piece

It changed modules the toolkit had replaced (main's roles, `rag_settings.py`, `llm_factory.py`, `publish.py`), so a merge could not carry it. Its 72 files were read against the toolkit's code.

| # | Piece on `main` | In the toolkit |
|---|---|---|
| 1 | A cap on an answer's length, `RAG_GENERATION_MAX_TOKENS` | `max_tokens` on a generate stage (decision 5) |
| 2 | Token counts on what the LLM returns | Two optional counts on `GenerateAnswer`; the OpenAI format's `usage` carries over |
| 3 | Token counts in the response and in the request's log line | A second measured field beside `timings_ms`: `tokens`, by stage (decision 6) |
| 4 | Load test: a request timeout, tokens in a run's target record, a tokens table in the comparison | Carried with `RAG_` to `QP_`; the tokens are kept per stage |
| 5 | The LLM in an App of its own on OVMS: `generator_server`, a Role, `make deploy-generator` | A served LLM: the `ovms` kind gains the chat route, a topology alias row says `llm: served`, the App's block says `server: ovms` (decision 3) |
| 6 | Launching it: the entry script, `rag-served-models`, the model's name, two OVMS options | The model and its served name are the component's `source_id` and `alias`; the command is in `domino/model_servers/ovms.sh`; the two options are fields of the OVMS settings (decision 7) |
| 7 | The emulated Domino's forward timeout, a Terraform variable | Carried in stage B1 |
| 8 | OVMS 2026.3.1 to 2026.4.1 | Carried in stage B1 |

### Where pull request 27 met what the toolkit had built

- **The request to the endpoint:** `main` added `max_tokens` where the toolkit had added `response_format`. Both hold, as two parameters.
- **What `generate` returns:** `main` returned the text with its counts; the toolkit asks again when an answer does not match its schema. Both hold: an answer asked again reports the sum of its attempts.
- **The load test's checks:** the toolkit had removed the stage names from the client, and `main` read the `generate` stage's time. The RAG has two stages that call the LLM, `rewrite` and `generate`, so the load test reads every stage that reports tokens, each on its own.
- **Retries:** `main` set them to 0 for a generation run. In the toolkit that is the LLM's `http.retries`, which also bounds the schema re-asks.

## The seven decisions, as taken

1. **One pull request**, from the toolkit's branch: the merge with `main` was resolved once, against the final code. Pull request 19 showed as merged when its commits reached `main` through pull request 29.
2. **The charts package lives in a folder of its own at the root**, with its `pyproject.toml`, `src/` and `tests/`. It imports neither the product nor the fake Domino, and its own test checks it. The toolkit stays free of this repository's findings.
3. **A deploy file says "the LLM is served by OVMS" the toolkit's own way:** an alias row with `llm: served` and `apps.llm.server: ovms`. An LLM App left on the `fastapi` kind is refused. ADR 0009 records it.
4. **"LLM", not "generator"**, the glossary's word. Main's documents were reworded as they were carried; the records under `experiments/` keep the names of their day.
5. **The answer cap is on the stage**, not on the component: on the component it would also cap the rewrite, whose answer must be whole to match its schema.
6. **The response reports tokens as a map by stage**, like `timings_ms`, with the sum of the attempts for an answer asked again.
7. **The two OVMS options of an LLM** are fields of the OVMS settings, beside the stream settings that pull request 18's port had put there.

## The stages, and what each became

| Stage | What | Commits |
|---|---|---|
| A | The earlier port of pull requests 17 and 18 merged into the branch: three files by hand | `867bcac` |
| B1 | The merge commit with `main`: records and documents as they are, the charts package in its place, pieces 7 and 8, and the toolkit's side kept wherever pull request 27 had changed a replaced module. Its message lists what was left to say again | `c4acd86` |
| B2 | Pull request 27 said again, one ticket a piece, each starting from main's tests carried to the toolkit | `bc90678`, `0129eea`, `0033d3c`, `f654da5`, `97c2123`, `9a4ed77`, `926960c` |
| C | A review of the whole merge by two readers, one on the workflow app's side and one on the deploy tool's, then the fixes of what they proved | `e0f3614`, `95efe7b`, `bde22c6`, `20b4130`, `d25ebfe` |
| D | Pull request 29 from the branch, with a fix its first run of the checks asked for and an on-demand check that compares a target's answers before and after a refactor | `16d6cca`, `8c0c229`, `c3caf5b`, `115f7a5`; merged as `0488bda` |

Between B1 and the end of B2 the branch lacked behaviour `main` had, and was not merged in that interval.

What the review of stage C found and fixed: a stage that ran more than once in a request reported its last run only; a target record written by main's load test was no longer read; the example deploy file held three commented `llm:` keys of one mapping, of which YAML kept the last; main's test of an LLM App's entry script had not been carried; and the tokens table added up every stage of a request under headers that named one. That last one became a decision of its own: the table has one row per run and stage.

## What the plan did not settle, and still is not

- **Whether OVMS follows `response_format`.** The rewrite sends a schema that way by default. If OVMS ignores it, the LLM's `structured_output: format_instructions` is the setting made for that case. Only a machine can tell.
- **An LLM App on OVMS has never run on a machine**, in either layout. That is a sitting to propose.
- **The pooling the export script gives the router's embedder.** `ovms/export_models.sh` applies the RAG embedder's pooling to every embedder it exports. Whether that is right for the router's needs a run on a machine.
