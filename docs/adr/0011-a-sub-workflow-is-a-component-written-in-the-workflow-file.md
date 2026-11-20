# A sub-workflow is a component, written in the workflow file and called as a client calls a workflow app

A use case had one workflow, and nothing let a stage of it call another. A stage can call a sub-workflow: a component of type `workflow`, whose block in the workflow file holds its graph and whose stages call the file's other components. A stage calls it as a client calls a workflow app, with a message, the conversational history and a metadata filter, and takes its answer and its passages. Its backend is `local`, where its graph runs in its caller's process, or `served`, where it runs behind a workflow app of its own. It is a component because ADR 0008 already calls every component with a request and takes an answer, and because where a component runs is what a topology says and what the toolkit compares.

Decided with the owner on 20 November 2026, before any of it is built. Its design document follows in `docs/specs/`.

## Considered options

- **An LLM that picks, at run time, among workflows offered to it as tools.** Rejected for now: a request would no longer take the same path each time, so runs would not compare, and the graph would no longer be all in a file (ADR 0004). The graph names the sub-workflow a stage calls, and choosing among several is a conditional edge on a route choice. An LLM's choice can be built on this, as a design of its own.
- **A complete workflow file for each sub-workflow.** Rejected by the owner: the workflows of a use case share its components. Each shared component would be written twice, with nothing saying the two are one model, and one process would hold two settings objects over registries that take one `local` implementation of a component type.
- **A file of its own that holds only the graph.** Left for later: once the workflow file grows too long, its reader can put such a file's content where the block's `graph` key is, and nothing else changes.
- **Each workflow declaring its own request and answer.** Rejected: the request, the response and the HTTP routes are fixed today and would have to be made generic first. With the one shape, a sub-workflow is served, load-tested and held to a pinned answer record as any workflow is.
- **Only in its caller's process, or only as an App of its own.** Rejected: which of the two is a row of the topology, as for any component. A workflow app that another use case deployed is not called yet: that would be the `api` backend.
- **Each inner stage in the response**, named after the calling stage and its own. Rejected for now: every component is timed as one stage, and the inner stages show in the trace (ADR 0013). It can be added without changing what is reported.

## Consequences

- A use case has a main workflow, the one its clients call, and any number of sub-workflows. CONTEXT.md no longer says that a use case has one workflow.
- The workflows of a file share its components, its prompts and its orchestrator. A graph the chosen orchestrator refuses is refused whichever workflow it is.
- A sub-workflow may call another. Start-up refuses a workflow that reaches itself, directly or through others: nothing would end it.
- The calling stage has a `mode` key, `query` (the default) or `retrieve`, for how far the sub-workflow runs.
- The response reports the calling stage alone: its time covers the whole sub-workflow, and its token counts are the sub-workflow's sum.
- On `served`, the deploy tool publishes one more App part of the topology, named after the sub-workflow, an Agent by default as the main workflow's App is. It runs `qp-serve` on the same workflow file, told which sub-workflow to serve. The role `workflow` no longer means the one workflow app of a topology.
- A call to a served sub-workflow carries the request id and the trace context, and its caller's `/readyz` asks the sub-workflow's.
- `local` is built first: with the graph in the same file it needs one settings object and one composition root, as today.
- The pinned answer records of today do not change. A fixture workflow with a sub-workflow gets a pinned answer record of its own, the same on every orchestrator and on `local` and `served`.
