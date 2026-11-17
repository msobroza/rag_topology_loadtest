# The deploy tool resolves the topology, and the workflow app only reads each component's backend

A topology says where each component runs: `local` in the workflow app's process, or `served` behind a model server. The chosen topology and the topology aliases (`monolith`, `split-reranker`, `split-all`) live in the deploy settings, because where a component runs is a deployment matter and differs per site. An App starts from the repository, where the git-ignored deploy settings file does not exist, so the workflow app cannot look an alias up. The deploy tool therefore turns the alias into each component's backend and endpoint URL and writes them into each App's launch script, with the alias as a label for the response header.

Amended by [ADR 0009](0009-a-served-llm-is-asked-for-as-any-served-component.md), where the `ovms` kind serves an LLM, asked for as any served component is.

## Considered options

- **Aliases in the workflow file, resolved by the workflow app.** One variable would switch the topology on a laptop, and the definitions would be versioned with the graph. Rejected: it keeps a placement rule in every process, and the owner wants the topology with the other deploy settings.
- **The workflow app reads the deploy file.** Rejected: the file is not in the repository an App starts from, which is why an App's settings already travel in its launch script.
- **A closed set in the domain code**, as before (`domain/topology.py`). Rejected: it could not name a placement for a new component, such as a served LLM.

## Consequences

- On a laptop a topology variable changes nothing. A component is moved by setting its backend and URL directly.
- A row of a topology alias says only `local` or `served`. How a component is served (model-server kind, environment, hardware tier) is in its App's block of the deploy settings.
- The workflow app reports the alias only when the deploy tool gave one. Its default run label lists each component with its backend.
- The deploy tool reads the workflow file too, to know the components, so the `deploy` package may import the workflow settings.
- A topology alias that serves a component no model-server kind can serve (an LLM, today) is refused before anything is published.
