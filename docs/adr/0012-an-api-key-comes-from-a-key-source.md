# An API key comes from a key source, and a key that is refused is asked for again

An endpoint's `api_key` was one string read at start-up: empty, a fixed key, or the value `domino`, which stood for Domino's run token and alone was asked for again when it expired. A fixed key that was rotated failed every call until its App restarted, and the Elasticsearch key and the load-test client's token were read once too. An API key now comes from a key source, a port with three adapters: `env` (an environment variable the block names), `domino` (the run token) and `vault` (the internal Vault). Each endpoint names its source in a `key` block, and sends no key without one. One rule covers them all: on a 401 the caller asks its source for the key again and retries, inside the retries its `http` block already bounds.

Decided with the owner on 20 November 2026, before any of it is built. Its design document follows in `docs/specs/`.

## Considered options

- **Every source rotating without a restart.** Rejected: a process does not see its environment change, so a key from `env` changes only when its App restarts. Its retries run out, and the error names the variable and says to restart the App.
- **`api_key` and its value `domino` kept beside the block.** Rejected: two ways to say one thing. Start-up refuses the old variable in one line that names the block.
- **The key's value as a setting**, as `QP_COMPONENTS__LLM__API__API_KEY` held it. Rejected: `env` names the variable to read, so the settings hold no key.
- **Reading the Vault again on a timer.** Rejected: a stored key says nothing of when it was rotated, so a refused call is the sign. A key is read at first use and on a 401.
- **A 403 treated as a refused key.** Rejected: it stays a failure at once, as it was. It says the key was read and is not allowed.
- **Retries of its own for a rotated key.** Rejected: every timeout, retry and backoff comes from the `http` block (CLAUDE.md's defensive patterns), and a rotation uses the same retries.

## Consequences

- It covers every key sent at run time: a component's endpoint on `api`, the vector store, an `otlp` tracer (ADR 0013) and the load-test client. A component on `served` keeps the run token on Domino, with no block.
- It breaks the settings of today. The `.env` files, Domino's project environment variables and the deploy file's refusal of a value ending in `_API_KEY` change with it.
- Elasticsearch takes its key once, when the store opens, so a rotated key means opening the store again: a ticket of its own.
- The load-test client takes the `domino` source, so a run longer than a run token lives no longer needs a Domino API key.
- `vault` is the last ticket. What the internal Vault speaks and how an App logs in to it were not known when this was decided. Its address and its log-in will sit in one shared `vault` block.
