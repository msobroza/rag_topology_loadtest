# An edge condition can answer with a label, and a conditional edge states the field it reads

ADR 0004 kept an edge condition to a yes or a no, and gave more than two ways to several conditional edges on one source. A router's routes are data: they come from the use case's `routes.yaml`, so a route added there should not need a function added in code. An edge condition can therefore answer with a label, and `path_map` then has one row for each label. A conditional edge also states `reads`, as every stage does, so the file shows which field decides the way.

## Considered options

- **One yes-or-no edge condition for each route**, as ADR 0004 had it. Rejected: a new route would need a new registered function, and the author would have to keep the conditions of one source mutually exclusive.
- **The field named in the code of each edge condition**, as `is_query_mode` read `mode`. Rejected: every stage now names its fields in the file, and a field renamed there would send every request the same way without an error.

## Consequences

- This takes back part of what ADR 0004 rejected: a conditional edge now names a state field in the file. What is asked of the field is still a registered function, and the file still holds no expression.
- Start-up refuses a `path_map` whose labels differ from the routes of the router that writes the field. On LangGraph 1.2.21 a label missing from `path_map` raises a `KeyError` only when a request takes it.
- The toolkit has one edge condition that answers with the field it reads, `label`, and `has_history`, which sends a first message past the router and the rewrite.
- A yes-or-no edge condition works as before, with `reads` added in the file: `is_query_mode` reads `mode`.
- A conditional edge can start at `START`.
