# The workflow graph runs on an orchestrator a setting chooses, LangGraph by default

ADR 0004 made the graph a file, spelled with LangGraph's names, and LangGraph ran it. The file now runs on an orchestrator: the library the workflow file's `orchestrator` key names, `langgraph` (the default) or `pydantic_graph` (Pydantic AI's graph library). The orchestrator is a port with one adapter per library, shaped as a stage type is: start-up asks its class, before any component opens, to refuse a graph it would not run as LangGraph does; the class is then built from the opened graph, and runs each request. The opened graph is what the application builds with no library: the description's three keys, each stage now what it runs, already timed, and each conditional edge with what it asks of the state.

Designed in [the spec of 19 November 2026](../specs/2026-11-19-the-graph-on-an-orchestrator-a-setting-chooses.md), whose closing section says what the build changed; the README's "Where a file goes" and "The workflow file" describe it as it is.

## Considered options

- **The toolkit schedules the stages itself, and a library runs one step at a time.** Rejected: the toolkit would write a scheduler of its own, and the library would add nothing.
- **The file restricted to what every library runs.** Rejected: LangGraph runs fan-out, joins at two depths and loops today, and the toolkit tests them.
- **A list of graph features in the toolkit, which each orchestrator declares and the toolkit's check compares.** Rejected: the list, its detection and its refusal are three pieces in three places, and a library whose limit is not on the list means editing all of them. A library's limits are its adapter's own code.
- **A second class for the built graph, returned by the orchestrator.** Rejected: a stage type is already checked as a class, built with what it runs, and run by one method; the orchestrator follows it with one class.
- **Pydantic AI's first graph API (`BaseNode`)**, which reads the edges from the return annotations of node classes. Rejected: a graph read from a file would need classes made at run time. Its builder API (`GraphBuilder`) takes steps and edges as values.
- **`pydantic-ai` as the dependency.** Rejected: its agents are not used, and `pydantic-graph` is the graph library alone.
- **A settings block, with `backend` as its key.** Rejected: neither library has a setting, so the value is a name; and a Backend is how a component runs, which an orchestrator is not.
- **The orchestrator in the default run label.** Rejected: the RAG's default label would grow from 99 to 127 of the 128 characters allowed, and two components on `served` would take it past the limit.

## Consequences

- The file's names (`stages`, `source`, `path`, `path_map`, `START`, `END`) are the toolkit's. They came from LangGraph, and they stay.
- A graph that runs today runs the same on LangGraph. Its check refuses only a stage name LangGraph does not take for a node, in LangGraph's own words: such a name failed before too, but at the build, with the components open.
- pydantic-graph refuses a stage or `START` with more than one way out at once, since the library ends the run when the first branch reaches `END` and cancels the others (seen on 2.70.0). It refuses a loop, since nothing stops one that never ends. It also refuses a stage named as a node its adapter adds for a conditional edge, and what its library refuses to build: a stage with no way out, nothing that leads to `END`, a stage no edge from `START` reaches.
- Each check asks its library where it can, on a graph linked with stand-in stages, so a rule of a library is not written a second time in its adapter.
- How the workflow state takes a stage's update is now stated in the toolkit: a field with a reducer is reduced, any other is replaced. LangGraph keeps its own reducers; a library without them uses that rule.
- `application/` and `domain/` name no library; each library is imported by its adapter alone.
- A third library is one directory with one class that registers itself. The checks, the builder, the setting and the tests find it through the registry.
- Tracing does not change. On Domino, a request on pydantic-graph has no span per stage.
- A run on another orchestrator is a variant, named with `run_label` as a hardware tier is.
- An edge condition is still used at most once per source: each library names the branch, or its step, after the edge condition.
