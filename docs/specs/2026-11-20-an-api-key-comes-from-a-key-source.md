# An API key comes from a key source

Triage: `ready-for-agent`.

Spec written on 20 November 2026 from the design session of that day with the owner, which [ADR 0012](../adr/0012-an-api-key-comes-from-a-key-source.md) records. It was built on 20 November 2026, by the commits `8236b33..RANGE_END`; "What was built", at the end, says what differs from what follows, which is left as it was written. It is the first of five pieces of work (key sources, sub-workflows, log files, traces, App directories); the others have their own specs. It uses the words of `CONTEXT.md`: API key, key source, run token, endpoint, component, backend, App, Workspace, project environment variable.

## Problem Statement

The keys our Apps send get rotated, and today a rotation breaks a running App.

- An endpoint's key is one string with three meanings: empty sends no key, the value `domino` stands for Domino's run token, and anything else is a fixed key. Only the run token is ever asked for again.
- A fixed key is read once, at start-up. Once it is rotated, every call that carries it is refused, at once and without a retry, until somebody restarts the App. The error does not say that a restart is what is needed.
- The Elasticsearch key is handed to the store once, when it opens, and is never read again.
- The load-test client reads a run token that the person pasted into a variable. The token lives about five minutes, so a longer run fails halfway unless the person switches to a Domino API key, as the README advises.
- The key's own value sits in the settings, under a variable whose name the toolkit imposes, such as `QP_COMPONENTS__LLM__API__API_KEY`.
- Some keys are to be kept in the company's Vault, and nothing can read one from there.

## Solution

An API key comes from a **key source**, and every place that sends a key says which one.

- Three sources are named: `env` (an environment variable), `domino` (Domino's run token) and `vault` (the internal Vault). This spec builds `env` and `domino`, and leaves `vault` to a spec of its own.
- An endpoint names its source in a `key` block of its settings. With no block, no key is sent. The block holds a source's name and what the source needs to find the key, never the key.
- One rule covers every source. When a call is refused for its key (HTTP 401), the caller tells the source so, the source asks for the key again, and the call is made again, inside the retries its `http` block already bounds.
- A key from `env` cannot change while its App runs. Its retries run out, and the error names the variable and says to restart the App.
- The rule reaches every key sent at run time: the endpoints of components, the Elasticsearch store and the load-test client.

## User Stories

Naming a key's source

1. As the owner of a use case, I want each endpoint to name where its API key comes from in a `key` block, so that the workflow file says how the App is let in without holding any key.
2. As the owner of a use case, I want the `env` source to name the environment variable it reads, so that I can use a variable my platform already gives, such as `OPENAI_API_KEY`.
3. As the owner of a use case, I want two components to name the same variable, so that one key shared by two endpoints is set once.
4. As the owner of a use case, I want an endpoint without a `key` block to send no key, so that a server on my laptop is called with nothing more to write.
5. As the owner of a use case, I want to name the `domino` source on an `api` endpoint, so that an endpoint that is another team's Domino App is called with the run token.
6. As the owner of a use case, I want the Elasticsearch store to name its key source with the same block, so that there is one way to say where a key comes from.
7. As the owner of a use case, I want basic authentication written in the Elasticsearch URL to keep working, so that a store without API keys needs no change.
8. As the owner of a use case on a laptop, I want the `env` source to find its variable in the `.env` file my settings are read from, so that a laptop run needs no `export`.
9. As the owner of a use case, I want to set a `key` block by the variable its path names, such as `QP_COMPONENTS__LLM__API__KEY__SOURCE`, so that it is overridden as every other setting is.

A key that is rotated

10. As a client of the workflow app, I want my request answered when the run token expired between two calls, so that a token's short life never shows.
11. As a client of the workflow app, I want a call refused for a rotated key to be made again with the key its source now gives, so that a rotation costs me no failed request.
12. As the person who rotates a key kept in a project environment variable, I want the error of a refused call to name the variable and to say that the App must be restarted, so that I know what to do without reading the code.
13. As a client of the workflow app, I want a 403 to fail at once as it does today, so that a key that is known and not allowed is not retried for nothing.
14. As a client of the workflow app, I want Elasticsearch's refusal of a rotated key to open the store again with the new key and search again, so that a rotation of the store's key costs me no failed request.
15. As the person who deploys, I want the run token to be held for the time the `domino` block says and asked for again after it, as today, so that a call seldom meets an expired token at all.
16. As a reviewer, I want the retry of a refused key to use the retries and the backoff of the `http` block, so that no retry constant is added.
17. As a reviewer, I want asking a source over the network to take its timeout, retries and backoff from the shared `http` block, so that this network call is bounded like every other.

What start-up refuses

18. As the person who deploys, I want start-up to refuse an `env` source whose variable is unset or empty, naming the variable and the component, so that a missing key is found before the first request.
19. As the person who deploys, I want start-up to refuse the old variable, such as `QP_COMPONENTS__LLM__API__API_KEY`, in one line that names the `key` block to write, so that an old `.env` file does not silently send no key.
20. As the owner of a use case, I want start-up to refuse the old `api_key` key in a workflow file with that same line, so that an old file tells me how to mend it.
21. As the owner of a use case, I want start-up to refuse a source name nobody registered, naming the known ones, so that a typo, or `vault` before it is built, is found at once.
22. As the owner of a use case, I want start-up to refuse a key of the block its source does not read, such as `variable` beside `domino`, so that a block never holds a line that does nothing.
23. As a person who reads an error or a log, I want no error and no log line ever to hold a key's value, so that a key does not leak through a refusal.

On Domino

24. As the person who deploys, I want a component on `served` to keep the run token on Domino with no block to write, so that the topologies of today deploy unchanged.
25. As the owner of a use case, I want a component type of my own on `served` to be called with the run token on Domino the same way, so that my component needs no key code.
26. As the person who deploys, I want the deploy tool to refuse, in the deploy file, a value for a variable that an `env` source names, so that no key is written into an App's launch script.
27. As the person who deploys, I want the deploy file's overrides to be able to set a `key` block, so that a site can choose another source or another variable than the workflow file's.
28. As the person who deploys, I want the deploy tool to refuse, before anything is published, what an App's start-up would refuse about a key, so that a wrong source name is found on the laptop.
29. As the person who deploys, I want `qp-deploy status` to ask for its run token through the `domino` source, so that there is one piece of code that asks Domino for a token.
30. As the person who watches an App's readiness, I want a readiness probe to make one attempt, as today, and to make the source forget a refused key, so that the next probe asks for a new one.

The load-test client

31. As a person who runs a load test from a Workspace, I want the load-test client to take the `domino` source, so that a run longer than a run token lives goes to its end without a pasted token.
32. As a person who runs a load test, I want to name an `env` source for a fixed key, so that a target outside Domino is driven the same way.
33. As a person who runs a load test, I want `DOMINO_API_KEY` to keep working as it does, so that a run that uses a Domino API key needs no change.
34. As a person who reads a run's results, I want a 401 during a Locust run to be counted as the failure it is and not sent again, so that a retry does not bend the timings.
35. As a person who records answers, I want `qp-record-answers` to ask its source again on a 401 and to send the request again within its own retries, so that a recording does not stop on an expired token.

The toolkit's own code

36. As a developer of the toolkit, I want a key source to be one abstract class with one adapter per source, registered under its name, so that `vault` or any other source is one more class and nothing else changes.
37. As a developer of the toolkit, I want one key source to serve code that awaits and code that does not, so that the workflow app, the load-test client, the deploy tool and, later, a trace exporter share it.
38. As a developer of the toolkit, I want a named fake key source for the tests, so that no test patches an attribute to stand in for one.
39. As a developer of the toolkit, I want the endpoint's domain class to hold no key and no `domino` constant, so that a key is known only to its source and to the client that sends it.
40. As a reviewer, I want the pinned answer records to stay as they are on every orchestrator, so that I know the change altered no answer.
41. As a reader of the README, the `.env.example`, the workflow file and the deploy example, I want each to describe the `key` block and no longer the old variable, so that the documents agree with the code.

## Implementation Decisions

### The port and its adapters

- A key source is a port, `BaseKeySource`, among the adapters, as every port is, with one adapter per source in its own directory. Its methods do not await.
- It does three things:
  - It gives the key it holds, and asks its source only when it holds none or when the time it may keep one is over.
  - It is told that a given key was refused, and forgets that key only if it is still the one it holds. Two calls refused at the same moment with the same old key then cost one new ask, and a later refusal of the old key does not throw away the new one.
  - It says, for an error, where its key comes from and how the key is changed, without the key.
- Code that awaits calls a source off the event loop, and only when the source must ask: a key already held is given without leaving the loop.
- The `env` adapter reads its variable where the settings are read: the process's environment first, then the `.env` file of the directory. It reads when it is opened and holds the key for the life of the process. Told that its key was refused, it changes nothing.
- The `domino` adapter asks the token URL of the `domino` block, holds the token for `token_ttl_s`, and asks again after that or once told the token was refused. It is what the workflow app does today, moved behind the port.
- Asking a source over the network takes its timeout, retries and backoff from the shared `http` block. Today a token is asked for once.
- The sources are registered by name, as orchestrators are, and start-up's check of registered names reads that registry. `vault` is not registered by this spec, so it is refused as an unknown name until its adapter exists.
- A use case cannot add a key source: the port is not part of `extension_api`.

### The `key` block

- The block is the contract with the workflow file. It holds `source` and the keys that source reads: `variable` for `env`, none for `domino`.

  ```yaml
  llm:
    backend: api
    api:
      url: https://api.openai.com/v1/chat/completions
      api_format: openai
      model: gpt-4o-mini
      key: {source: env, variable: OPENAI_API_KEY}
  ```

- A key of the block that its source does not read is refused when the settings are read.
- With no block, no key is sent. That is the default, so an example file writes a block only where a key is sent.
- The block sits in the `api` block of every component that has one, in the `elasticsearch` block of the vector store, and in the load-test client's settings. A `served` block has none.
- The settings `api_key` of an endpoint and of the Elasticsearch block are removed, and with them the value `domino` as a key. Nothing reads the variables `QP_COMPONENTS__<NAME>__API__API_KEY` and `QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY` any more.
- The `domino` block keeps `token_url` and `token_ttl_s`: they are the `domino` source's settings.
- The RAG use case's LLM names `env` with the variable `OPENAI_API_KEY`.

### What start-up refuses

- The old variable, set in the environment or in `.env`, and the old `api_key` key in a workflow file: one line that names the component and the `key` block to write in its place.
- A `source` nobody registered, with the variable it is read from and the known names, as an unknown backend is refused today. Every `key` block is checked, whatever its component's backend.
- An `env` source whose variable is unset or empty, when the App opens the source: the line names the variable and the component. Reading the settings alone does not check it, so a command that opens no endpoint needs no key.

### The rule in the HTTP client

- The endpoint client is given a key source, or none, by the composition root. The endpoint's domain class no longer carries a key.
- With a source, a 401 is among the answers that are tried again. Before the next attempt the client tells the source which key was refused, and the attempt asks the source for its key. This counts within the `http` block's retries, with its backoff, as the run token's 401 does today.
- Without a source, a 401 fails at once. A 403 fails at once, with or without a source. A 429 and a 5xx are tried again as today.
- When the retries run out on a 401, the error carries what the source says of itself. For `env` that is the variable's name and that the App must be restarted after the variable is changed.
- A readiness probe keeps its one attempt. A refused key is still forgotten, so the next probe asks again.
- The header stays `Authorization: Bearer <key>`.
- No error and no log line holds a key's value.

### Who gets which source

- The composition root builds each source once and shares it among the endpoints that name the same block. A process therefore has one `domino` source, where today each endpoint keeps its own copy of the token.
- A component on `served`, a registered component type's included, gets the process's `domino` source when the workflow app runs on Domino (`qp-serve --domino`), and none elsewhere, as today.
- The call to OVMS on 127.0.0.1 during calibration and a model server's own calls carry no key, as today.

### The Elasticsearch store

- The store is opened with the key its source gives, and without one when it has no `key` block.
- When Elasticsearch refuses the key (its 401), the store tells its source, opens its client again with the key the source then gives, and makes the call once more. One further attempt is the bound: the vector store has no `http` block, and none is added.
- Basic authentication in the URL is left as it is.

### The load-test client

- Its settings take the same `key` block, read from `QP_LOADTEST_KEY__SOURCE` and `QP_LOADTEST_KEY__VARIABLE`.
- `QP_LOADTEST_BEARER_TOKEN` is no longer a setting of its own. Set without a `key` block, it is refused in one line that names the block.
- `DOMINO_API_KEY` and its header are unchanged. A `key` block wins over it, as the bearer token did.
- In a Locust run a 401 is recorded as a failure and is not sent again. The source is told, so the next request carries a new key.
- `qp-record-answers` tells its source on a 401 and sends the request again within its own retries.

### The deploy tool

- `qp-deploy status` asks for its run token through the `domino` source.
- The deploy file refuses, in any `settings` list, a value for a variable that an `env` source of the workflow file names: the message says to set it as a Domino project or user environment variable.
- The rule that refuses a name ending in `_API_KEY` in those lists stays, without its exception for an empty value or for `domino`.
- The overrides may set a `key` block, which holds no key, and it is checked against the workflow settings as every override is.
- What start-up refuses about a key's source is refused by the deploy before anything is published, through the check the deploy already shares with start-up.

### The documents

- The README, `.env.example`, the comments of the workflow file and of the deploy example, and the emulation's example of its secret describe the `key` block and name no old variable.
- `CONTEXT.md` already holds key source and run token. `CLAUDE.md` names ADR 0012 once this is built.
- The records under `experiments/` keep the names of their day: none is reworded.
- A closing section, "What was built", is added to this spec after the build, saying what differs.

## Testing Decisions

A good test here shows what a person or another program would see: the `Authorization` header a call carried, the answer a client of the workflow app got, the line start-up printed. It does not show which method called which. What stands in for the outside is a named fake or the HTTP transport stand-in the tests already have, never a patched attribute. Where one thing can be said two ways, in the file or by the variable its path names, one parametrized test covers both.

The tests sit at four seams, all of which exist today.

1. **The workflow app seen from outside**, the main one. `test_qp_serve_served` already builds the app with one HTTP transport that answers in place of every endpoint and of Domino's token URL, and records what each call carried. It is where these are shown:
   - Each source sends its key: `env` from the environment and from `.env`, `domino` from the token URL, and no block sends no header.
   - A 401 is followed by a new ask and a second call that carries the new key, and the client of the app gets its answer.
   - With `env`, the retries run out and the error names the variable and the restart.
   - A 403 fails at once, and a 401 without a source fails at once.
   - The run token is held for `token_ttl_s`, counted by how many times the transport served a token.
   - Two endpoints that name the same source share one ask.
   - Start-up refuses the old variable, the old key, an unknown source, a key the source does not read, and an `env` variable that is unset.
   - No refusal's text holds the key.
   - A test at the entry point shows that the source named in the settings is the one that ran.
2. **The deploy tool's command on the fake Domino**, as `test_domino_deploy_overrides` and `test_domino_deploy_dry_run` do today: a variable an `env` source names is refused in a `settings` list, a `key` block in the overrides reaches the launch script, an unknown source is refused before anything is published, and `status` sends the run token.
3. **The load-test client's commands**, as `test_loadtest_settings`, `test_locustfile` and `test_qp_record_answers` do today: the `key` block is read, `QP_LOADTEST_BEARER_TOKEN` alone is refused, a 401 in a Locust run is one failure and the next request carries a new key, and `qp-record-answers` sends again after a 401.
4. **The Elasticsearch store with its fake LangChain store**, as `test_elasticsearch_store` does today: a refused key opens the client again with the new key, the call is made once more, and a second refusal is the store's error. Elasticsearch's client does not go through the transport of seam 1, so this cannot be shown higher.

Three more things hold for every ticket.

- The tests of the run token class and the endpoint client's test of a 401 describe classes that change shape. Their cases move up to seam 1, and the count of cases before and after is stated in the ticket.
- The pinned answer records do not change, on any registered orchestrator. No ticket makes one again.
- `make check` is the gate. Its structure tests hold the layers' imports and each directory to twelve modules, the new port's directory included.

The toolkit ships no test class for key sources, since a use case cannot add one.

## Out of Scope

- **The `vault` adapter.** What the internal Vault speaks and how an App logs in to it are not known. Its adapter, its block keys and the shared `vault` block get a spec of their own once the owner has those two facts. ADR 0012 already says how it will behave: a key read at first use and on a 401, on no timer.
- **The key of a tracer.** It takes the same block, in the spec of the traces (ADR 0013). The port does not await so that a trace exporter can use it.
- **A password in the Elasticsearch URL.** It is not a key source; rotating it means a restart, as today.
- **A 403 as a refused key, a timer that reads a source again, and any circuit breaker, rate limit or graceful degradation.** None is added.
- **The deploy tool's own calls to Domino.** They go through Domino's API proxy and carry no key.
- **Who may call our Apps.** Domino's proxy decides it, and no App checks a key.
- **A key source of a use case's own.**
- **The other four pieces of work:** sub-workflows, log files, traces and App directories.

## Further Notes

- **Order.** This is the first piece because the next ones lean on it: a served sub-workflow is called with the process's `domino` source, and an `otlp` tracer names a key source.
- **A possible cut into tickets,** for the owner to make: the port with `env` and `domino`, the block on `api` endpoints, the rule and the refusals, with the documents; then the Elasticsearch store; then the load-test client; then the deploy tool. The first carries the non-regression criterion that the others repeat.
- **Decided here and not put to the owner in these words.** The design session settled the block, the rule and the scope. These follow from it, and the owner can strike any of them:
  1. `QP_LOADTEST_BEARER_TOKEN` goes, as `api_key` does.
  2. A use case cannot add a key source.
  3. Start-up refuses an `env` variable that is unset.
  4. The `env` source also reads the `.env` file.
  5. A source's methods do not await.
  6. A source forgets a refused key only if it still holds it.
  7. Asking a source over the network is retried, where a token is asked for once today.
  8. Elasticsearch gets one further attempt.
- **Not checked.** How long Domino's run token lives is not in the 6.3 documents; the 4.x ones said five minutes, and `token_ttl_s` keeps its 240 seconds.
- **The break.** Every `.env` file, every Domino project environment variable and the emulation's secret that names an old variable must be changed when this is deployed. Start-up says so in one line, in place of sending no key.

## What was built

What differs from the spec:

1. A key of the block that its source does not read is refused at start-up, by the check of registered names, not when the settings are read: the settings take any keys beside `source`, since they may not import the registry of sources.
2. A `key` block given above the workflow file (the environment, `.env`, a deploy override) that names `source` replaces the file's block whole; one that sets another key alone changes that key. The spec did not say how the places combine; merged key by key, a site could not change a source.
3. Every `key` block, the load-test client's included, is checked before the settings library validates it, and a `variable` or a `source` that is not shaped like a name (letters, digits and underscores, not starting with a digit) is refused without being quoted, so a key typed in the wrong place is never printed.
4. The Elasticsearch store keeps the clients a refused key replaced until the store closes, so a call in flight on one gets its answer; and it builds a new client only if its source then gives another key. With a source that keeps its key, such as `env`, a refused call fails at once with the source's words.
5. The key sources of the components a command calls open before any component, so a missing key is refused before a local model loads.
6. The `env` source of the workflow app reads the `.env` file the settings were built with, not always the directory's.
7. `QP_LOADTEST_BEARER_TOKEN` beside a `key` block is ignored; alone, it is refused.
8. The load-test client asks its source under its own timeout, retries and backoff (`QP_LOADTEST_REQUEST_TIMEOUT_S`, `_RETRIES`, `_BACKOFF_S`), since it reads no `http` block.
9. `qp-deploy status` asks through a second function of `composition`, `run_token_source`, with the `http` block's default backoff and the `domino` block's default time; its line for a token that could not be had now quotes the token URL's answer.
10. The port gained three things the spec did not name: how a source is opened from its block (`from_block`), what a source reads of its block (`reads`), and the variables a block names (`variables`), for the deploy tool's rule.
11. One module of `loadtest` (`target_client.py`) and one of `deploy` (`status.py`) may import, of `composition`, the module that opens a key source for a command (`standalone_key_source.py`), and of the adapters the key source port, and nothing wider; the layering test writes each exception by module.
12. The log line of a request that failed on a dependency carries the error's detail, with the key masked.
13. The Locust file's tests moved to a folder of their own, `tests/loadtest/locustfile/`: seam 3 names `test_locustfile`, now there.
14. In a Locust run, a request whose source cannot give its key is not sent and is recorded as a failed request under its own name, so the statistics `qp-compare` reads count it.
15. A setting the load test refuses, its `key` block's included, ends Locust in one line, `locustfile.py: <the refusal>`, with no traceback, and `qp-record-answers` in one line with the status of a usage error.

Kept as the spec says, though a reviewer questioned it: a refused `env` key is tried again within its retries, so the error comes once they run out.

Not built: the `vault` source, which waits for a spec of its own. Not run: the three checks of ticket 06 on a deployed app, which wait for the owner's machines.

The points decided without the owner's words, under "Further Notes": each was built as written, and the owner has still to keep or strike it.

1. `QP_LOADTEST_BEARER_TOKEN` goes: built as written; refused alone, ignored beside a `key` block (7).
2. A use case cannot add a key source: built as written; the port is not in `extension_api`.
3. Start-up refuses an `env` variable that is unset: built as written, and empty too.
4. The `env` source also reads the `.env` file: built as written, the one the settings were built with (6).
5. A source's methods do not await: built as written.
6. A source forgets a refused key only if it still holds it: built as written.
7. Asking a source over the network is retried: built as written, under the `http` block, or the command's own settings where there is none (8, 9).
8. Elasticsearch gets one further attempt: built as written, made only when the source gives another key (4).

Four more, decided in the build, for the owner too:

9. The rule of places: a `key` block given above the workflow file with a `source` replaces the file's whole (2).
10. The shape of a `variable` and of a `source`: letters, digits and underscores, not starting with a digit (3).
11. The import table's two exceptions, `loadtest/target_client.py` and `deploy/status.py` (11).
12. The detail of the error in the log line of a failed request, with the key masked (12).
