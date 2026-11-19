# The workflow graph on an orchestrator a setting chooses: LangGraph or pydantic-graph

Design written on 19 November 2026 from the owner's request of that day, and revised the same day at their request: simpler, on SOLID principles, easy to extend, and shaped as the toolkit's other abstractions are. Nothing of it is built. Two spikes on the laptop ran pydantic-graph 2.70.0 and LangGraph 1.2.20 on stand-in stages, the second on the port as this document shapes it ("What the two libraries do"); nothing ran on Domino or on a real model. Names of modules, classes and functions below are indicative; what a user writes (the `orchestrator` key and its variable) is the design. [ADR 0010](../adr/0010-the-graph-runs-on-an-orchestrator-a-setting-chooses.md) records the decision.

## Why

The workflow graph is a file (ADR 0004), and LangGraph runs it. The owner wants the same file to run on another library: "use an adapter to use the same yaml for the graph but change the orchestrator framework (instead of langraph use others). One that I would like to implement concrete class would be pydanticAI".

LangGraph is already confined. Three places of `application/` name it:

- `workflow_builder.build_workflow` builds the `StateGraph`: a node per stage, an edge per pair, a conditional edge per entry of `conditional_edges`.
- `workflow.CompiledWorkflow` holds a `CompiledStateGraph` and calls its `ainvoke`.
- `state_fields.state_type` builds the `TypedDict` LangGraph takes, with the reducers of `metadata_filter`, `timings_ms` and `tokens`.

Everything else knows no library: the description, the stages and stage types, the edge conditions, `timed`, `asked_of_state`, `check_graph`. So the change is one port, with those three pieces behind it.

## What a user writes

Nothing changes in the `graph` block, and the RAG use case's `workflow.yaml` is not edited: the toolkit's default is LangGraph. Another library is one value, set as any value of the workflow file is:

```bash
QP_ORCHESTRATOR=pydantic_graph make serve
```

```yaml
# or as a top-level key of a workflow file
orchestrator: pydantic_graph # langgraph (the default) | pydantic_graph
```

```yaml
# or for the Apps of a deploy, in deploy.yaml, with the run label that names the variant
overrides:
  orchestrator: pydantic_graph
  run_label: monolith+orchestrator=pydantic_graph
```

## Goals

1. The RAG use case's `workflow.yaml` runs unchanged on LangGraph and on pydantic-graph, Pydantic AI's graph library.
2. One value chooses the library: `orchestrator`, `langgraph` by default.
3. For the same request and the same components, both libraries give the same result: the answer, the passages, the metadata filter, `timings_ms` with the same keys in the same order, the same `tokens`.
4. A graph a library would not run as LangGraph does is refused at start-up, before any component opens, by that library's own adapter, naming the stages.
5. `application/` and `domain/` import neither library; each is imported by its adapter alone.
6. A third library is one more directory of `adapters/orchestrator/`. No check, no builder, no setting and no test is edited for it.

## Non-goals

- The graph's language does not change: no key added to the `graph` block, none renamed.
- Tracing does not change (see "What stays as it is").
- The run label, the deploy tool and the load-test client do not change.
- No settings block for an orchestrator: neither library needs a setting today.
- What pydantic-graph offers beyond what the file can say is not used: joins with reducers, `map`, streaming steps.
- Pydantic AI's `Agent` is not part of this. It would be a backend of the LLM, a design of its own.
- A use case cannot add an orchestrator: `extension_api` does not expose the port.
- One orchestrator for the workflow app: none per stage, none per request.

## What would make this a failure

- A graph that runs today runs differently on LangGraph after the change, or is refused.
- The two libraries give different results for one request, on a graph start-up accepts for both.
- A graph pydantic-graph would cut short, or run without end, is accepted on it.
- A refusal comes after a component opened, or as a traceback instead of qp-serve's one line.
- A module of `application/` or `domain/` imports `langgraph` or `pydantic_graph`.
- Adding a third library means editing a file outside its own directory, the dependency list and the README apart.
- A stage's error reaches qp-serve wrapped in an `ExceptionGroup`, so its response changes.

## What the two libraries do

**pydantic-graph 2.70.0.** Its builder API, documented at `pydantic.dev/docs/ai/graph/builder/` (read on 19 November 2026), builds a graph at run time:

- `g.step(function, node_id=name)` makes a step without the decorator;
- `g.add_edge(a, b)` links two nodes, and `g.start_node` and `g.end_node` are the graph's ends;
- `g.decision().branch(g.match(TypeExpression[object], matches=predicate).to(node))` routes on the value the step before it returned. Branches are tried in order and the first match is taken.

A step gets `ctx.state`, one mutable object for the whole run. The package depends on anyio, logfire-api, pydantic 2.12 or later and typing-inspection. Next to the toolkit's dependencies it adds 1.2 MB: itself and logfire-api.

**The first spike** built the RAG graph's shape by hand, with stand-in stages, and tried the shapes `check_graph` accepts beyond it:

| Tried | pydantic-graph 2.70.0 | LangGraph 1.2.20 |
|---|---|---|
| A stage that raises `LookupError` | `LookupError` reaches the caller, not wrapped | the same |
| Two plain edges out of `START`, to `a` (fast) and `b` (0.2 s), both to `END` | both start; the run ends when `a` reaches `END`, `b` is cancelled and its write lost: `{x: 1}` | both run: `{x: 1, y: 2}` |
| Two branches that join at two depths (the graph of `test_stage_that_runs_more_than_once.py`) | `c` ran twice, as on LangGraph; after the row above, a matter of timing | `c` runs twice |
| A loop that stops at its second call | ran twice | ran twice |
| A loop that never stops | 3,639 runs in 1 s, and no limit | `GraphRecursionError` at its default limit, 10,007 steps |
| 200 concurrent runs of one built graph | each its own result | what the workflow app does today |
| Wall time of a run, 7 no-op stages and 3 branches, 1,000 runs | 1.5 ms | 2.2 ms |

The times come from one run of stand-ins on the laptop. They say only that a library's own cost is a few milliseconds, against stages that take tens of milliseconds to seconds.

**The second spike** wrote the port as "The design" has it and read the `graph` block of the use case's `workflow.yaml` with the toolkit's `GraphDescription`:

| Tried | Result |
|---|---|
| The RAG graph: a first message; a message with history on each route; `query` and `retrieve` | equal last states on both libraries, the metadata filter and the timings added up the same |
| `check_graph_runs` of the pydantic-graph class, on the RAG graph | accepted |
| The same, on: two plain edges out of `START`; a plain edge and a conditional edge out of one stage; two conditional edges on one source; a loop | each refused, naming the stages |
| `check_graph_runs` of the LangGraph class, on all five | accepted |

Two things follow:

- **One way out at a time runs the same on both.** A stage or `START` with one plain edge or one conditional edge out, and paths that meet when only one of them is taken. The RAG graph is only that.
- **Two shapes do not.** More than one way out at once: pydantic-graph ends the run at the first `END` and cancels the rest, where LangGraph waits for every branch. And a loop: it runs on pydantic-graph, but nothing stops one that never ends.

## Approaches

1. **Each library builds its own graph from the same opened stages, and says itself which graphs it does not run as LangGraph does.** The library really runs the graph. What it cannot run is known in one place, its adapter.
2. **The toolkit schedules the stages and asks a library only to run a step.** One behaviour for every library, but the toolkit writes a scheduler of its own (the owner's rule: no hand-written algorithm a library provides), and the library adds nothing.
3. **The file restricted to what every library runs.** The simplest, but it takes from LangGraph what the toolkit accepts and tests today.

This design takes the first.

Within it, two ways to say what a library does not run were considered:

- **A list of graph features in the toolkit** (`fan_out`, `several_conditions`, `cycle`), which each orchestrator declares and the toolkit's check compares. The list, its detection and its refusal are three pieces in three places, and a library whose limit is none of the three means editing all of them.
- **A check in each adapter.** A library's limits are its adapter's own code, and the toolkit's checks do not know them.

This design takes the second.

## The design

### The orchestrator is shaped as a stage type is

The toolkit already has an abstraction with this life: a stage type (`BaseStage`). Its class is found in a registry by a name the file gives. Start-up asks the class, before any component opens, whether it can do what the file asks (`check_fields`). Then the class is built with what it runs, and has one method that runs. The orchestrator is the same, one level up:

```python
# adapters/orchestrator/base_orchestrator.py: imports the domain only
class BaseOrchestrator(abc.ABC):
    """A library that runs the workflow graph: built once, run for every request.

    For every graph its check accepts, an orchestrator gives the last state
    LangGraph gives.
    """

    @classmethod
    @abc.abstractmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        """Refuse a graph this library would not run as LangGraph does.

        Raises:
            ValueError: Naming the stages, what the library would do, and what
                was expected.
        """

    @abc.abstractmethod
    def __init__(self, opened: OpenedGraph) -> None:
        """Build the library's graph from the opened stages and edges."""

    @abc.abstractmethod
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run one request from its first state to its last."""


ORCHESTRATORS: Registry[type[BaseOrchestrator]] = Registry("orchestrator")
```

| The orchestrator | What it follows in the toolkit |
|---|---|
| `adapters/orchestrator/base_orchestrator.py`, an ABC named `BaseOrchestrator` | every port: `adapters/<x>/base_<x>.py`, `Base*` |
| one directory per library, one implementation in each | one directory per Backend value |
| a class asked before anything opens, then built with what it runs, then one method that runs | `BaseStage`: `check_fields`, `__init__`, `run` |
| `ainvoke` | a component's `ainvoke`, and what `CompiledWorkflow` calls today |
| `ORCHESTRATORS`, each class registering itself in its own module | README rule 6 |
| an unknown name refused by `check_registered_names`, listing the registered ones | an unknown backend or API format |
| the class's `ValueError` raised as a `WorkflowGraphError` that names it | `open_stage`: `stage 'rerank': ...` |
| the composition finds the class from the settings | README rule 5 |
| `OpenedGraph` in the domain beside `GraphDescription`, with its three keys | a port imports the domain only |

### The opened graph: what an orchestrator is given

`build_workflow` keeps doing what needs no library. It opens each stage, wraps it in `timed`, makes each conditional edge's `asked_of_state`, and reads the field types. It hands that over as the description, opened: the same three keys as the `graph` block, each stage now what it runs and each conditional edge now with what it asks.

```python
# domain/opened_graph.py: no library, no I/O
StageRun = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
EdgeAsk = Callable[[dict[str, Any]], bool | str]
Reducer = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class OpenedConditionalEdge:
    edge: ConditionalEdge  # its source, its edge condition (path) and its rows (path_map)
    ask: EdgeAsk  # the edge condition asked of the state; an answer with no row is refused


@dataclass(frozen=True)
class OpenedGraph:
    stages: Mapping[str, StageRun]  # each stage, timed, by name, in the file's order
    edges: Sequence[tuple[str, str]]  # the description's edge_pairs()
    conditional_edges: Sequence[OpenedConditionalEdge]
    field_types: Mapping[str, Any]  # every field of the workflow state, with its type
    reducers: Mapping[str, Reducer]  # the fields that add up: metadata_filter, timings_ms, tokens

    def merged(self, values: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
        """Return the state after a stage's update: a field with a reducer reduced, any other replaced."""
```

- A conditional edge keeps the description's own `ConditionalEdge`: its source, the name of its edge condition and its rows are not copied into a second shape.
- `merged` states once, in the toolkit, how the workflow state takes an update. Until now that rule was LangGraph's channels. LangGraph keeps its own reducers and does not call it; a library without reducers does.
- The reducers are today's functions (`add_conditions`, `sum_timings_by_stage`, `sum_tokens_by_stage`), handed over as values, so the domain imports nothing of the application.

`build_workflow` then ends with one line: `CompiledWorkflow(orchestrator(opened), stage_names=list(graph.stages))`, where `orchestrator` is the class it was given. `CompiledWorkflow` calls `ainvoke` as it does today.

### The LangGraph adapter

`adapters/orchestrator/langgraph/langgraph_orchestrator.py` holds what leaves `application/`: the `StateGraph` wiring of `build_workflow`, the `TypedDict` of `state_type`, `langgraph_name`, `path_map` and the `LangGraphNode` protocol. Its behaviour does not change.

```python
@ORCHESTRATORS.register("langgraph")
class LangGraphOrchestrator(BaseOrchestrator):
    @classmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        """LangGraph is the reference: it runs every graph start-up accepts."""

    def __init__(self, opened: OpenedGraph) -> None:
        builder = StateGraph(state_type(opened.field_types, opened.reducers))
        for name, run in opened.stages.items():
            builder.add_node(name, run)
        for source, target in opened.edges:
            builder.add_edge(langgraph_name(source), langgraph_name(target))
        for conditional in opened.conditional_edges:
            source = langgraph_name(conditional.edge.source)
            builder.add_conditional_edges(source, conditional.ask, path_map(conditional.edge))
        self.graph = builder.compile()

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return await self.graph.ainvoke(state)
```

A branch keeps the edge condition's name, since `asked_of_state` already gives the function that name. The tests that pass LangChain callbacks reach LangGraph's compiled graph as `graph`, on this class.

### The pydantic-graph adapter

`adapters/orchestrator/pydantic_graph/pydantic_graph_orchestrator.py`, as the second spike ran it:

```python
@dataclass
class _RunState:
    values: dict[str, Any]  # the workflow state of one request


@ORCHESTRATORS.register("pydantic_graph")
class PydanticGraphOrchestrator(BaseOrchestrator):
    @classmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        ways_out = Counter(source for source, _ in graph.edge_pairs())
        ways_out.update(edge.source for edge in graph.conditional_edges)
        several = sorted(source for source, count in ways_out.items() if count > 1)
        if several:
            raise ValueError(...)  # names them; see below
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for source, target in graph.links():
            sorter.add(target, source)
        try:
            sorter.prepare()
        except CycleError as exc:
            raise ValueError(...) from exc  # names the stages of the loop

    def __init__(self, opened: OpenedGraph) -> None:
        g = GraphBuilder(state_type=_RunState, output_type=object)
        nodes = {START: g.start_node, END: g.end_node}
        for name, run in opened.stages.items():
            nodes[name] = g.step(_stage_step(run, opened), node_id=name)
        for source, target in opened.edges:
            g.add_edge(nodes[source], nodes[target])
        for conditional in opened.conditional_edges:
            edge = conditional.edge
            asked = g.step(_ask_step(conditional.ask), node_id=f"{edge.source}:{edge.path}")
            decision = g.decision()
            for answer, target in edge.path_map.items():
                decision = decision.branch(
                    g.match(TypeExpression[object], matches=_is(answer)).to(nodes[target])
                )
            g.add(g.edge_from(nodes[edge.source]).to(asked), g.edge_from(asked).to(decision))
        self._graph = g.build()

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        run = _RunState(dict(state))
        await self._graph.run(state=run)
        return run.values
```

- **Its two rules.** At most one way out of a stage or `START` at a time: one plain edge, or one conditional edge. And no loop. The loop is found by the standard library's `graphlib`, not by a search written here.
- **A stage's step** hands the stage a copy of the values, as LangGraph hands a node the state, then keeps `opened.merged(values, update)`.
- **A conditional edge** is a step that asks the edge condition and returns its answer, then a decision with one branch for each row. The step is named `<source>:<edge condition>` (`START:has_history`, `route:label`), which is unique in the graph because an edge condition is used once per source.
- **`_is(answer)`** matches only an answer of the row's own type and value, so `1` does not take the `True` row, as `answer_with_a_row` already holds.

What a refusal says, through qp-serve:

```text
qp-serve: error: orchestrator 'pydantic_graph': 'START' has more than one way out at
once; pydantic-graph ends a run when the first reaches END and cancels the others
(expected one plain edge or one conditional edge out of each stage, or the
orchestrator 'langgraph')
```

### How start-up uses it

The order of today, with one line more in each step:

1. `check_registered_names(settings)` also refuses an unknown orchestrator, as it refuses an unknown backend: `QP_ORCHESTRATOR: unknown orchestrator 'pydantic' (expected one of: 'langgraph', 'pydantic_graph')`.
2. `check_workflow_graph(settings)` runs `check_graph` as today, then asks the chosen class its `check_graph_runs`. The orchestrator is asked only about a graph the toolkit already accepts, and its `ValueError` is raised as a `WorkflowGraphError` that names it. `check_graph` itself does not change.
3. qp-serve's lifespan hands the class to `build_workflow`, and its `rag_startup` log line names the orchestrator.

The composition finds the class (`ORCHESTRATORS.get(settings.orchestrator)`), since only the composition goes from the settings to an adapter. The application receives the class and builds it with the opened graph.

### The setting

One field of `WorkflowSettings`: `orchestrator: str = "langgraph"`. It is a top-level key of the workflow file, the variable `QP_ORCHESTRATOR`, and a key the deploy file's `overrides` accept with no change to the deploy tool, which checks them against `WorkflowSettings`.

It is a name and not a block, because neither library has a setting. It is not called a backend: a Backend is how a component runs (CONTEXT.md), and an orchestrator is no component.

### What stays as it is

- **Tracing.** `traced` keeps `autolog_frameworks=["langchain"]` for every orchestrator. LangChain is not the orchestrator's alone: the Elasticsearch store and the format-instructions method import it too, so the list does not follow the library chosen. Under pydantic-graph the autolog finds no LangGraph step, so a traced request has no span per stage; the response's `timings_ms` still times each one. Bringing pydantic-graph's own spans into Domino's traces is a later design.
- **The run label.** It does not name the orchestrator. The RAG's default label is 99 characters, and `+orchestrator=pydantic_graph` would bring it to 127 of the 128 allowed: with two components on `served`, start-up would refuse a label it built itself. A run on another library is a variant, named with `run_label` as a hardware tier is.
- **`check_graph` and its rules**, the stage types, the edge conditions, the components, the deploy tool, the load-test client.

### The dependency

`pydantic-graph>=2.70` becomes a core dependency of the toolkit, which raises its floor for pydantic from 2.7 to 2.12 and adds 1.2 MB to every install.

An extra would save that. But every App on `pydantic_graph` would then have to name the extra in `QP_BOOTSTRAP_EXTRAS`, the adapter would import its library late, and an App started without the extra would fail only at start-up, on Domino.

## The five principles, in this design

- **One reason to change.** The application decides what runs: it opens the stages, times them and asks the edge conditions. An orchestrator only links them and runs them. What a library cannot run is in that library's adapter, not in the toolkit's checks.
- **Open to a new library, closed to edits.** A new library is a directory with one class that registers itself. The check, the builder, the setting and the tests find it through the registry.
- **Any orchestrator in place of another.** The port states the promise: for every graph its check accepts, the last state LangGraph gives. One test holds every registered orchestrator to it.
- **A small interface.** Three members. The opened graph holds what a library uses and nothing of the components, the settings or the stage types.
- **Depend on the port.** The application imports `BaseOrchestrator` alone, the composition names the class, and each library is imported by one module.

## Extending it

- **A third library**: a directory `adapters/orchestrator/<name>/` with one class, the three members and `@ORCHESTRATORS.register("<name>")`; its dependency in `pyproject.toml`; a line in the README. `QP_ORCHESTRATOR=<name>` then works, an unknown name lists it, and the tests below run on it.
- **Loops on pydantic-graph**: its adapter stops a run after a number of stages. That number is then the first setting of an orchestrator, and `orchestrator` grows a block at that point, not before.
- **An orchestrator of a use case**: `extension_api` exposes `BaseOrchestrator`, `ORCHESTRATORS` and `OpenedGraph`. The registry already takes a use case's entries (ADR 0006).

## Where the code goes

New:

- **`domain/opened_graph.py`**: `OpenedGraph`, `OpenedConditionalEdge`. `domain/` holds 9 modules.
- **`adapters/orchestrator/`**: `base_orchestrator.py`, `langgraph/langgraph_orchestrator.py`, `pydantic_graph/pydantic_graph_orchestrator.py`. `test_layering.py`'s pattern for a port already matches `adapters/orchestrator/base_orchestrator.py`, so the application may import it with no change to the table.

Changed:

- **`application/workflow_builder.py`** (333 lines): builds the opened graph, then the class it is given from it; loses `LangGraphNode`, `langgraph_name` and `path_map` to the LangGraph adapter.
- **`application/workflow.py`**: `CompiledWorkflow` holds a `BaseOrchestrator`.
- **`application/state_fields.py`**: gives the state's field types and reducers as two mappings; the `TypedDict` goes to the LangGraph adapter.
- **`domain/graph_description.py`**: `links()` becomes a method of the description, beside `edge_pairs()`. It is `edge_check.links` today, which `edge_check.py` and `graph_check.py` call once each; the pydantic-graph adapter needs it and may not import the application.
- **`application/edge_check.py`**: the rule "one edge condition once per source" stays. Its message gives LangGraph as the reason; the reason is now every library's.
- **`config/workflow_settings.py`** (488 lines): the `orchestrator` field.
- **`composition/registered_names.py`**: the unknown orchestrator.
- **`composition/composition_root.py`** (496 lines): the class from the settings, and the orchestrator's check in `check_workflow_graph`. Something leaves the file.
- **`entrypoints/qp_serve.py`** (476 lines): the class handed to `build_workflow`, and the log line.
- **`pyproject.toml`** of the toolkit: the dependency.
- **`tests/structure/test_layering.py`**: `langgraph` is imported only under `adapters/orchestrator/langgraph/`, and `pydantic_graph` only under `adapters/orchestrator/pydantic_graph/`.
- **`README.md`**: rule 4 (`adapters/` holds a package per component and `orchestrator/`), rule 6 (one more registry, `ORCHESTRATORS`), the tree, the workflow file's keys and variables, "What start-up refuses in a graph" (what the chosen orchestrator does not run), and how a run on another library is labelled.
- **`CONTEXT.md`**: a new term, under "Code structure":

  > **Orchestrator**:
  > The library that runs the workflow graph along its edges: LangGraph by default, or pydantic-graph. The workflow file's `orchestrator` key chooses it; the graph does not change with it.
  > _Avoid_: framework, engine, runtime (that is `local_models.runtime`), executor

- **`docs/adr/0004-...`**: "Amended by ADR 0010" beside its two other amendments, once this is built.

Not changed: `application/graph_check.py` beyond the one call of `links`, the deploy tool, the load-test client, the use case.

## Non-regression

The toolkit already has the check that a refactor changed nothing a caller sees: the answer records (`qp-record-answers`, then `qp-compare-answers`). Its tests already record the workflow app in process on fake backends and compare the records, for two topology aliases and for a served reranker. This change uses the same check at three levels.

- **Before and after, in the gate.** Before anything moves, what the workflow app answers on fake backends is recorded and committed: one record for the toolkit's fixture workflow, and for the RAG use case's own workflow as many as its three paths need (a first message, a message the router calls self-contained, a rewritten one). A test of each distribution records the app again in process and compares it with the committed record. From then on `make check` is the before-and-after check of every step, and the move of step 2 passes it with the records unchanged.
- **Every orchestrator, in the gate.** The same test runs under every registered orchestrator. So pydantic-graph is compared with what LangGraph answered before the change, not only with LangGraph as it is after it, and a third library is held to the same records with no test edited.
- **Real models, on demand.** On a deployment, the RAG is recorded under each orchestrator and the records are compared: the passages, their order, and their scores within the tolerance; the LLM's text by ROUGE-L. It downloads nothing: the corpus and the questions are the synthetic ones, and the models are where the deployment has them. It needs machines, so it runs only on the owner's go-ahead, and its sitting is registered in `experiments/`.

An answer record holds no timing, no token count and no request id, so it is the same from run to run. What it leaves out is held by the tests below: `timings_ms` with the same keys in the same order, and the same `tokens`. When a change alters the answers on purpose, one documented command makes a record again, and the failing test names it.

## Tests

Each line is one behaviour a test proves. Those that say "every orchestrator" are parametrized over `ORCHESTRATORS.names()`, so a new library is tested with no test edited.

- The workflow app on fake backends compares as identical with the answer record committed before the change, on the toolkit's fixture workflow and on the use case's own workflow, under every orchestrator.
- Every orchestrator, on each graph of one set (a chain, a yes-or-no branch, a label branch, a conditional edge at `START`, two plain edges out of `START`, two conditional edges on one source, a loop), either refuses the graph or gives the result LangGraph gives.
- Every orchestrator runs the fixture workflow, and the run tests of `test_workflow.py` and `test_workflow_builder.py` that it accepts.
- The use case's tests of its conversation paths pass on every orchestrator.
- For every orchestrator, a stage's `LookupError` reaches `workflow.run` as itself, an answer with no row fails with a `WorkflowGraphError` naming the edge, and 200 concurrent runs of one built graph each give their own result.
- `pydantic_graph` refuses two plain edges out of `START`, a plain edge and a conditional edge out of one stage, two conditional edges on one source, and a loop; each message names the stages.
- `langgraph` refuses none of them, and `test_stage_that_runs_more_than_once.py` and the tracer-name tests keep running on it as they are.
- The orchestrator's refusal comes from `check_workflow_graph`, before any component opens, as a `WorkflowGraphError` that names the orchestrator.
- `OpenedGraph.merged` reduces a field that has a reducer, replaces one that has none, and keeps a field the update does not hold.
- Without the key the orchestrator is `langgraph`; `QP_ORCHESTRATOR=pydantic_graph` gives `pydantic_graph`; an unknown name is refused by `check_registered_names`, with its variable and the registered names.
- The pydantic-graph adapter names the step of a conditional edge `<source>:<edge condition>`, and an answer of `1` does not take the `True` row.
- Each library is imported only by its adapter.
- The move of step 2 below leaves the count of tests the same before and after.

`make check` is green at the end.

## Order of work

1. The answers on fake backends are recorded and committed, with the test that compares the app with them. Nothing of the application or the adapters changes.
2. The opened graph and the port; today's wiring moves into the LangGraph adapter and nothing else changes. The tests count the same before and after, and the committed records are not touched.
3. The `orchestrator` field, its refusal when unknown, the orchestrator's check in `check_workflow_graph`, the class handed to `build_workflow`, the log line.
4. The pydantic-graph adapter, its dependency, and the tests over every orchestrator, the committed records among them.
5. The README, `CONTEXT.md`, and the amendment line of ADR 0004.
6. On the owner's go-ahead, the check on real models, registered in `experiments/`.

A measure of the two libraries' speed on the RAG graph is no part of this design. If the owner wants one, it is a sitting registered in `experiments/`.

## Left to the builder

- The names of the modules, classes and functions.
- What leaves `composition_root.py`, and how `workflow_settings.py` and `qp_serve.py` stay under the 500 lines the code-shape test allows.
- How the existing tests are handed an orchestrator: a fixture, or a helper.
- The wording of the refusals, beyond what they name.

## Not verified

- What pydantic-graph does with a request that is cancelled, such as a client that hangs up.
- The spikes' results on another version of pydantic-graph. The dependency's floor is the version tried.
- The spikes ran stand-in stages. Real components run in the same stages, so no difference is expected.
- That nothing of MLflow's LangChain autolog fails on a request with no LangGraph step. Tracing was not run.
- That an answer record made on the laptop compares as identical in CI. The fake backends are deterministic and scores are compared within a tolerance, but no record is committed yet.
- Both orchestrators on real models. That is the on-demand check of "Non-regression".

## Calls made in this document, for the owner to check

- The library is `pydantic-graph`, Pydantic AI's graph library, without Pydantic AI's agents.
- The value is `orchestrator` (`QP_ORCHESTRATOR`), a name and not a block, and not called a backend.
- LangGraph stays the default and runs every graph it runs today.
- A library's limits are its adapter's own code. The toolkit keeps no list of graph features.
- pydantic-graph refuses more than one way out at once, and a loop.
- The port is one class shaped as `BaseStage` is, with no second class for the built graph.
- The opened graph and `links()` are in the domain.
- `pydantic-graph` is a core dependency, not an extra.
- Tracing and the run label do not change. On Domino, a request on pydantic-graph has no span per stage.
- The before-and-after check is an answer record on fake backends, committed before the change and compared in the gate, in the toolkit and in the use case.
- The check on real models is on demand, and waits for the owner's go-ahead.
- A use case cannot add an orchestrator through `extension_api`.
- ADR 0010 is written with this document.
