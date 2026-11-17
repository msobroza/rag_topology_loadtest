# The workflow graph is a file, with LangGraph's names and edge conditions that return a bool

So that a use case can change its graph without code, and a new step is one registered stage type, the graph is declared in `workflow.yaml`: the stages, each calling a named component, then `edges` and `conditional_edges` spelled with LangGraph's own argument names (`source`, `path`, `path_map`, `START`, `END`). A conditional edge names a registered edge condition: a function of the workflow state that returns a bool.

Amended by [ADR 0007](0007-an-edge-condition-can-answer-with-a-label.md), where an edge condition can also answer with a label, and by [ADR 0008](0008-every-component-is-called-the-same-way.md), where the file states each stage's fields.

## Considered options

- **A graph built in Python for each use case.** Rejected: a use case holds settings only (ADR 0005).
- **A `path` that returns a label**, LangGraph's own form for more than two ways. Rejected: a bool keeps an edge condition small. More than two ways is several conditional edges on one source, each with its own edge condition (run on LangGraph 1.2.20: with one true only that branch runs, with none true the graph ends).
- **Branching on a state field named in the file, or expressions in the file.** Rejected: the owner chose registered functions, and expressions would be a second language inside the YAML.
- **`Command(goto=...)` returned by a stage.** Rejected: the links would be in Python, and the file would no longer show how the stages are linked.

## Consequences

- The author keeps the edge conditions of one source mutually exclusive: when two are true, both branches run in the same step.
- One edge condition cannot be used twice on the same source, because LangGraph names a branch after its function. The loader refuses it with its own message.
- Each stage type declares the state fields it reads and writes, and start-up refuses a graph in which a stage reads a field nothing before it writes. Lexical retrieval is therefore a file without the embed stage, no longer a rule in the code.
- Timings are keyed by the stage names of the file, and the load-test client takes its rows from the response.
- The file says `stages` where LangGraph says nodes: the glossary keeps Stage.
