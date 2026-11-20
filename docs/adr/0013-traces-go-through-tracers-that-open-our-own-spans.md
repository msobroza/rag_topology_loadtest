# Traces go through tracers that open our own spans, and the trace server is an App outside any topology

Tracing was Domino's alone: the setting `domino.tracing` wrapped each request with Domino's library, which records a span per stage on LangGraph and none on pydantic-graph (ADR 0010). The owner wants the traces in Arize Phoenix as well, on a server the deploy tool publishes or one that already runs, and room for other kinds. A workflow app now has tracers, a port with one adapter per kind, and several can be on at once. A tracer is told when a request starts and ends and when each stage starts and ends, with what the stage read and wrote. `domino` wraps the request as before and leaves the stages to Domino's library. `otlp` opens our own spans and sends them in OpenTelemetry's format, which Phoenix receives. Phoenix is the trace server: on Domino, an App the deploy tool publishes on its own, outside any topology.

Decided with the owner on 20 November 2026, before any of it is built. Its design document follows in `docs/specs/`.

## Considered options

- **The ready-made instrumentation of LangChain** for the `otlp` spans. Rejected: it records LangGraph alone, since none exists for pydantic-graph, and it names and fills the spans as that library decides. ADR 0010 says the graph does not change with its orchestrator, and neither does its trace. Our spans open where each stage is already timed.
- **A backend named `phoenix`.** Rejected: a format is a wire shape, not a vendor. Any server that receives OpenTelemetry's format is sent the same spans. Their attributes are named as OpenInference names them, which is what Phoenix reads to show an LLM's or a retrieval's span as such.
- **Phoenix in place of Domino's tracing, or one tracer at a time.** Rejected by the owner: both are wanted, and the deploy tool turns Domino's on for every Agent.
- **A launch script the owner publishes by hand.** Rejected by the owner: the deploy tool publishes the App.
- **The trace server as an App part of a topology.** Rejected: it holds the traces of every topology and must stay up when one is deployed again.
- **Spans from the model servers.** Rejected: a model server's call is inside its stage's span, and OVMS and vLLM run none of our code. Only workflow apps send spans.
- **A request that fails when its trace cannot be sent.** Rejected by the owner: the request is answered, the spans are dropped and a log line says so. Start-up does not ask the trace server.
- **A tracer a use case adds**, as it adds a component. Not yet: tracers are the toolkit's, as orchestrators are (ADR 0010).

## Consequences

- A `tracers` block replaces the setting `domino.tracing`. The deploy tool writes the `domino` tracer for an Agent, as it set `QP_DOMINO__TRACING=1`.
- An `otlp` trace has one span for the request and one for each stage, with the fields it reads and writes, and the token counts on an LLM's stage. A sub-workflow's spans sit under the stage that called it (ADR 0011), and a served one joins its caller's trace through the `traceparent` header. `content: false` leaves the message, the passages and the answer out.
- On Domino's tracer a request on pydantic-graph still has no span per stage. On `otlp` it has one on every orchestrator.
- The deploy tool has a third role, `trace-server`, with one launch script that starts Phoenix, a `trace_server` block in the deploy file and a command of its own, `qp-deploy trace-server`. A deploy of a topology leaves the trace server running and writes its URL into each `otlp` tracer left without one. A server that runs elsewhere is named in the deploy file's overrides, with its key.
- An `otlp` tracer's key comes from a key source (ADR 0012): the run token, for the trace server App.
- Behind Domino's proxy Phoenix's own log-in stays off, since the proxy already uses the `Authorization` header. It keeps its traces in SQLite on a Dataset, on one replica, with its own retention in days. Both were read in the two products' documents and neither was run, and whether an App can write to a Dataset on the owner's Domino is not checked either.
- Another kind of tracer is one adapter class, registered under its backend's name with its settings block.
