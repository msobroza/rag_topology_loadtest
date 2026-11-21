# A core holds what toolkits exchange, and the workflow handles the vector-store record

Triage: `ready-for-agent`.

Spec written on 20 November 2026 from the design session of that day with the owner, which [ADR 0014](../adr/0014-a-core-holds-what-toolkits-exchange-and-the-workflow-handles-the-vector-store-record.md) records. Nothing of it is built. It uses the words of `CONTEXT.md`: core, toolkit, use case, vector-store record, chunk, metadata filter, prefilter, answer record, pinned answer record, workflow app, App, Dataset, cache bucket.

## Problem Statement

The toolkit and what feeds it call one thing by two names, and what they would share has no home.

- The toolkit's unit of retrieval is a class of its own, `Passage`: an id, a text and a flat map of metadata. `document_doclang`, which builds chunks from documents, stores each as a `VectorStoreRecord` with the same content. A record that ingestion writes is not the class the workflow reads, and one side says "record" where the other says "passage".
- A toolkit for ingestion and one for evalops are planned. What they would exchange with query processing lives in the toolkit's own domain, behind an install that brings FastAPI, LangGraph and a model library.
- The response body's model lives in the `qp-serve` entrypoint. The load-test client, which may not import an entrypoint, restates the shape by hand.
- The synthetic handbook writes each window's index under the metadata key `chunk`, which doclang's record keeps for the chunk itself. Reading such a record as a chunk fails with a validation error, not with "this record holds no chunk".

## Solution

A third distribution, the **core**, holds what one toolkit writes and another reads, and the workflow handles ingestion's own **vector-store record**.

- The core, `query_processing_core`, holds four things: the vector-store record with its scores, the metadata filter, the request body with its conversational history, and the response body. It needs Pydantic and `doclang-domain-core` only.
- The workflow retrieves, reranks and answers on doclang's `VectorStoreRecord`. `Passage` goes, and the word with it: `records` in code, "vector-store record" in prose.
- A record made from a chunk carries the chunk; a record made from plain text carries none. The workflow handles both, and the response returns each record as it is.
- Each vector store gives a record the id the store holds it under, so an index ingested before this change still answers.
- `document_doclang` stays private. Each place that installs the toolkit is handed its core: the laptop from the clone beside this repository, CI from a clone made with a read-only key, an App and an emulation machine from a wheel kept where the exported models are.
- The change is built in four steps, and each is held to "the same answers".

## User Stories

The core

1. As the author of a future toolkit, I want to install the core without the query processing toolkit, so that I read and write what toolkits exchange without FastAPI, LangGraph or a model library.
2. As the author of evalops, I want the request body and the response body of a workflow in the core, so that I send and read what a workflow app speaks from one definition.
3. As the author of an ingestion toolkit, I want the record the workflow reads to be the class ingestion writes, so that no conversion sits between them.
4. As a reviewer, I want one rule to say what belongs in the core, an object one toolkit writes and another reads, so that a proposed addition can be judged.
5. As a reviewer, I want a structure test to hold the core to the standard library, Pydantic and doclang's core, and to importing no toolkit, so that the core stays data.
6. As a reviewer, I want one module of this repository to name doclang, so that a change in doclang's layout is mended in one place.
7. As a maintainer of the toolkit, I want every package of the toolkit to be allowed to import the core, so that each moved object is used where it was.
8. As the owner of a use case, I want to keep importing the toolkit through `extension_api` alone, so that my package's one-import rule and its layering test stand.
9. As a maintainer of the toolkit, I want the workflow's result, the components with their requests and answers, the API formats, the registries and the graph to stay in the toolkit, so that the core holds only what crosses to another toolkit.
10. As a reader of the README, I want "Where a file goes" to show the core, its line in the imports table and its modules in the tree, so that the layout is described as it is.
11. As a maintainer, I want `make check` to run the core's gates before the toolkit's, with the toolkit's configuration, so that the core is held as the use case is.

The vector-store record in the workflow

12. As a client of the workflow app, I want each record of a response to be the vector-store record as its store holds it, with its retrieval score and its rerank score, so that what I receive is what ingestion wrote.
13. As a client of the workflow app, I want a record made from a chunk to arrive with its chunk, so that I can show its pages and its section without a second call.
14. As the owner of a use case with a plain-text corpus, I want a record with no chunk to be retrieved, reranked and given to the LLM, so that the handbook and a JSONL corpus need no DocLang document.
15. As a developer who asks a handbook record for its chunk, I want doclang's own refusal, which says the record holds none, so that I do not read a validation error about the digit `0`.
16. As the owner of a use case, I want a metadata filter on `sequence` to fit a handbook record and a doclang-made record alike, so that one filter serves both.
17. As the person who ingested an index before this change, I want its records found and answered as before, so that I need not ingest again.
18. As a maintainer of a vector store's adapter, I want one contract, a record read has the id its store holds it under, so that every store is written to the same rule.
19. As the owner of a use case, I want the prefilter and a request's metadata filter to act on a record's flat metadata as they do today, so that no filter of mine changes.
20. As the author of a prompt, I want each record of the reranked list to give its `text` as before, so that my prompt changes only the name of its loop variable.
21. As a client of the workflow app, I want a hybrid retrieval to fuse its two rankings by the records' ids as before, so that the same request finds the same records.
22. As the person who runs a model server of the reranker, I want its route to answer in Cohere's format as before, so that no caller of that App changes.

One word

23. As a reader of the code, the README and the glossary, I want one word for the thing, vector-store record, so that I never wonder whether a passage and a record differ.
24. As a client of the workflow app, I want the response's list under `records`, so that the wire says what the glossary says.
25. As the owner of a use case, I want the reranker's calibration setting under a name that says records, and start-up to refuse the old name in one line that gives the new one, so that an old file tells me how to mend it.
26. As a reader of `/readyz`, I want the store's detail to count records, so that the probe uses the glossary's word.
27. As a person who runs `qp-ingest`, I want its summary line to count `records`, so that a script reads one word everywhere.
28. As the owner of a use case, I want my prompt's sentence to the LLM and my metadata fields' descriptions left as I wrote them, so that what a model reads, and so what it answers, does not change with our vocabulary.
29. As the owner of an index named by my workflow file, I want its name kept, so that the indexes and the snapshots stored under it are still found.
30. As the owner of the experiments, I want every record under `experiments/` left in the words of its day, so that a sitting still reads as it ran.
31. As a reviewer, I want a structure test to fail on the word "passage" outside the places it lists, each with its reason, so that the word does not come back.

The same answers

32. As the owner, I want the move of the first three modules to the core to leave every pinned answer record equal and the count of tests equal, so that the move is shown to change nothing.
33. As the owner, I want the class swapped while the response keeps its shape, with every pinned answer record still equal, so that the swap and the new shape are never one change with two possible causes.
34. As the owner, I want the pinned answer records made again once, when the response changes shape, and a test written first to hold the new ones to the old through a stated mapping, so that the same ids, text hashes, scores and answers are shown under the new shape.
35. As the person who compares a deployment, I want `qp-compare-answers` to read an answer record of the old shape through that same mapping, so that a branch can be held to main and to the records of 17 November.
36. As the owner, I want the comparison of main and the branch on real models to run on one index and only on my word, so that it is exact to the last digit and costs machines only when I say so.
37. As a reviewer, I want every ticket to state what it leaves equal, so that non-regression is a criterion from the first ticket.

Getting doclang's core

38. As the owner on my laptop, I want `make install-dev` to install doclang's core from the clone beside this repository, then the core, the toolkit, the use case and the charts package, so that one command still sets a machine up.
39. As the owner working in a worktree, I want the clone found beside the repository's main checkout, so that a worktree installs as the main checkout does.
40. As the owner, I want one variable to name another place for the clone, and the install to stop in one line that names it when no clone is there, so that a missing clone is said, not guessed.
41. As the owner, I want the laptop to install the clone as it stands, so that a change I make in doclang is seen here without a new pin.
42. As a reader of this public repository, I want the core's manifest to name doclang's core by name and version alone, so that no private address is written here.
43. As the owner, I want CI to clone `document_doclang` with a read-only key held as a secret and to run the same gate, so that the checks on GitHub still mean what `make check` means.
44. As the person who deploys, I want an App that installs at its start to take doclang's core from a wheel in a directory on the Dataset, named by one deploy setting, so that its start holds no credential.
45. As the person who deploys, I want a start whose wheel is not found to stop in one line that names the setting and the directory, so that a missing wheel is not read as a network failure.
46. As the person who deploys, I want every model-server kind that installs the toolkit to get the wheel the same way, so that there is one rule for every App.
47. As the owner, I want a make target to build the wheel from the clone, so that building it is one command.
48. As the person who runs the emulation, I want a machine to take the wheel from the cache bucket, in the emulation's own files, so that the emulation still leaves in one piece.
49. As the owner, I want the wheel never to enter this repository, so that private code stays private.

## Implementation Decisions

### The core distribution

- A third distribution at the repository's root, `query_processing_core`, with a package of the same name, on the repository's Python floor and built as the toolkit is. Its dependencies are Pydantic and `doclang-domain-core`, the latter by name and version alone.
- The rule of what it holds: an object one toolkit writes and another reads. It has no port, no settings and no I/O, and it imports no toolkit and no use case.
- One of its modules names doclang and hands on `VectorStoreRecord`; no other module of the repository imports doclang. A structure test holds both this and the core's allowed imports.
- The toolkit depends on the core. Every package of the toolkit may import it, and the README's imports table says so. A use case still imports through `extension_api`, which hands on the core's names a use case needs, the metadata condition among them.
- The core is laid out by the glossary's areas, as the toolkit's domain is: the retrieval objects together, the conversation and the request at its top. Rule 9 (twelve modules a directory) and rule 3 (one name, one module) apply to it.

### What moves, and what stays

- Three modules move whole in the first step, each with its tests: the metadata filter, the conversational history and the request body.
- The response body leaves the `qp-serve` entrypoint for the core in the third step, in its new shape, and takes with it the mode and the name of the total timing, the two names the load-test client already shares with the workflow. It cannot move sooner: until then it holds the toolkit's scored passage, and the core may not import the toolkit.
- The vector-store record's module is new. It holds doclang's `VectorStoreRecord`, handed on, and `ScoredVectorStoreRecord`, a subclass with `retrieval_score` and `rerank_score`. A record is frozen and refuses unknown keys, so the scores cannot sit beside it any other way.
- The workflow's result stays in the toolkit: it is what the application hands the entrypoint. So do the function that keeps the best-scored records, rank fusion, the retrieval mode, the synthetic corpus, the errors, the components with their requests and answers, the endpoints, the registries, the graph, the prompts, the roles, the run label and the runtime. The module that held the passages is renamed after what it keeps.
- The type of a metadata value stays defined in the core's metadata filter. A test holds it equal to doclang's scalar type.

### The vector-store record in the workflow

- Every place that took or gave a `Passage` or a `ScoredPassage` takes or gives the record: the vector store ports, the rerankers, the rerank and search requests and answers, rank fusion, the stages, the workflow state, ingestion, the fake store's seeding and the calibration's sample.
- A record's id is `metadata["id"]`; it has no id field. Code that builds a record by hand, the handbook's chunking and the model server's rerank route among them, puts the id there.
- A record with no chunk is valid everywhere in the workflow. Nothing in the toolkit reads the chunk; a stage or a client that wants it calls doclang's `from_vector_store`.
- The handbook's window index moves from the metadata key `chunk` to `sequence`. A record built by hand uses doclang's key where it says the same thing.

### The stores

- Each store's adapter gives a record the id the store holds it under, on every read, whatever its stored metadata holds. This is the adapter's contract, stated on the port.
- On a write, the record's id is the store's id for it, and its metadata is stored as it is.
- An index or a collection ingested before this change is read without being ingested again. Its handbook records keep the key `chunk` until they are.

### The response and the names

- The response returns the scored records as they are, under `records`: a text, the flat metadata with the id and, when there is one, the chunk, and the two scores.
- `records` replaces `passages` in code: the attributes of the rerank and search requests and answers, the reranker's calibration setting, the summary line of `qp-ingest` and the stores' `/readyz` detail. Prose says "vector-store record".
- Start-up refuses the calibration setting's old name, in the workflow file or as a variable, in one line that gives the new one. The README's section on moving from the old names gains the line.
- Four things keep the word, each listed with its reason by the structure test: a prompt's sentence to the LLM and a metadata field's description, which a model reads; an index's or a collection's name, which stored data is found by; the emulation's snapshot metadata, which the cache bucket holds; and the dated records (`docs/`, `experiments/`), with the charts and the tour that tell them.
- A use case's prompt changes only its loop variable.

### The answer records

- An answer record keeps, for each vector-store record of an answer, its id, its text's hash, its metadata and its scores, read from the new shape.
- One mapping says how the old shape and the new one meet: the id read from the metadata, `passages` read as `records`, and the handbook's `chunk` read as `sequence`. It is one function of the load-test client.
- `qp-compare-answers` reads a record of the old shape through that mapping, for as long as the registry holds such records.
- The three pinned answer records, the toolkit's and the use case's two, are made again once, in the step that changes the response.

### Getting doclang's core

- **The laptop.** The root Makefile's install targets install doclang's core from the clone, as an editable install of its working tree, before the core and the toolkit. The clone is `document_doclang` beside the repository's main checkout, found from git so that a worktree resolves it; one variable, `DOCLANG_CLONE`, names another place. With no clone there, the target stops in one line that names the variable.
- **CI.** One more step clones `document_doclang` with a read-only deploy key held as a secret, beside the checkout and never inside it, since a test reads the repository's untracked files. The install line then names it as the laptop's does. The key and the secret are the owner's to create.
- **An App.** The start script's one install function takes the core's directory with the toolkit's and the use case's, and finds doclang's core among the wheels of a directory. One deploy setting names that directory, a path on the Dataset, and the deploy tool hands it to every App as a variable beside the one that asks for the install. An App whose Environment already holds everything installs nothing, as today.
- **The wheel.** A make target builds it from the clone as it stands. It is written outside the repository's tracked files and is ignored by git. Copying it to the Dataset is the owner's, as copying the models is.
- **The emulation.** A machine copies the wheel from the cache bucket and names its directory with the same variable. This lives in the emulation's own files and leaves with them.
- Nothing is pinned, by the owner's choice: the laptop holds what is checked out, CI the default branch's tip, an App the wheel as last built.

### The documents

- The README describes the code as it is once built: "Where a file goes" and the tree, the install, the HTTP API, the answer records, the deploy on Domino, the variables and the old names.
- `CONTEXT.md` already holds the terms. `CLAUDE.md` names the core, ADR 0014 and what an install needs.
- The use case's `.env.example`, its workflow file's comments and its deploy example say records, and name the wheels' directory.
- A closing section, "What was built", is added to this spec after the build, saying what differs.

### The order

1. **The core is created and three modules move**: the metadata filter, the conversational history and the request body. Nothing else changes. doclang is not yet needed.
2. **The record replaces `Passage` inside the workflow**, with the stores' contract, while the response keeps today's shape through one mapping in the entrypoint. doclang's core is needed from here on, on the laptop, in CI and at an App's start.
3. **The response returns the records as they are, and the names change**: the response body, now the core's, the key, the attributes, the setting, the handbook's key and the word everywhere else. The entrypoint's mapping goes.
4. **On the owner's word**, answers are recorded from main and from the branch on real models, on one index, and compared.

## Testing Decisions

A good test here shows what a person or another program would see: the body a client of the workflow app got, the line an install or a start printed, the file a command wrote. It does not show which class called which. What stands in for the outside is a named fake or the stand-in the tests already have, never a patched attribute.

The tests sit at four seams, all of which exist today.

1. **The workflow app on fake backends, held to the pinned answer records**, the main one, as `test_pinned_answer_record` and the use case's `test_pinned_answer_records` do today, on every registered orchestrator. It is where these are shown:
   - Steps 1 and 2 leave the three records equal, with no record made again.
   - Step 3's new records equal the old ones through the mapping: the same ids, text hashes, scores and answers. That test is written before the records are made again.
   - The response holds `records`, each with its id in its metadata; `test_qp_serve` and its siblings show the shape, a record with a chunk returned whole, and a record without one.
   - Start-up refuses the calibration setting's old name, in the file and as a variable, in one parametrized test.
2. **The load-test client's commands**, as `test_qp_record_answers` and the two `test_answer_records_*` modules do today: a record of the new shape is written and read, a record of the old shape is read through the mapping, and an old record compares alike with a new one of the same answers.
3. **Each vector store with its stand-in**, as `test_elasticsearch_store`, `test_chroma_store` and `test_in_memory_vector_store` do today: a record read has the id its store holds it under, when the stored metadata holds an id and when, as in an index ingested before, it holds none. The fake backends of seam 1 never reach Elasticsearch's client, so this cannot be shown higher.
4. **The structure tests**, as `test_layering`, `test_placement`, `test_directory_size_rule` and `test_old_names` do today: the core's allowed imports, the one module that names doclang, the toolkit's line in the imports table, the tree, twelve modules a directory, and the word "passage" nowhere outside the listed places.

The install is shown where scripts are shown today, as `test_makefiles`, `test_domino_app_sh` and `test_emulation_machine_scripts` do: the install targets name the clone and the core in order, the start script's install takes the wheels' directory, and each stops in its one line when what it needs is missing. The deploy setting that names the directory is shown by the deploy tool's command on the fake Domino, as `test_domino_deploy_dry_run` does: it reaches every App's launch script.

Four more things hold for every ticket.

- A ticket that moves a module states the count of tests before and after, and they are equal.
- A ticket states which pinned answer records it leaves equal. Only the ticket of step 3 makes them again.
- `make check` is the gate, with the structure tests run again after the commit.
- The core's own tests are the moved ones, a test of the scored record (a record made by doclang from a chunk goes to JSON and back and gives the same chunk), and its structure test. No test of the core starts the toolkit.

The toolkit ships no new test class: a use case cannot add a kind of record.

## Out of Scope

- **An ingestion toolkit.** Whether it is doclang's ingestion package grown or a distribution here, and where it gets its embedder and its vector store, are its own design. The core holds no port.
- **An evalops toolkit**, and whether it takes over the answer records from the load-test client.
- **Records made from chunks by this repository.** `qp-ingest` and the fake store keep the handbook's word windows; nothing here chunks a DocLang document, and the toolkit does not depend on doclang's ingestion package.
- **How Elasticsearch maps a record's chunk.** No record ingested here holds one. It belongs to the first ingestion that writes them.
- **`Chunk` in the workflow, and any stage that reads a chunk.**
- **Publishing doclang's core, or making its repository public.** The owner said no.
- **A pinned commit of doclang.** The owner chose the clone as it stands.
- **Ingesting the existing indexes again**, and renaming an index, a collection or the emulation's snapshot metadata.
- **The other design of 20 November**, on the branch that holds ADRs 0011 to 0013. Its new lines say "passage"; whichever branch lands second sweeps them.
- **Any circuit breaker, rate limit or graceful degradation.** None is added.

## Further Notes

- **A possible cut into tickets,** for the owner to make: the core with the three moved modules; doclang's core on the laptop and in CI; the record in the workflow with the stores' contract; the response's shape, the answer records' mapping and the pinned records made again; the word, with its structure test; the wheel for an App and for the emulation; the documents; and last, on demand, the comparison on real models. The first carries the non-regression criterion that the others repeat.
- **What must be in place before step 2 is merged or deployed.** The deploy key and its secret, or the checks on GitHub fail. The wheel on the Dataset and the deploy setting that names its directory, or an App that installs at its start does not start.
- **Decided here and not put to the owner in these words.** The design session settled the core, the record, the names and the delivery. These follow from it, and the owner can strike any of them:
  1. The scored class is named `ScoredVectorStoreRecord`.
  2. Every package of the toolkit may import the core.
  3. One module of the repository names doclang, and a structure test holds it.
  4. Start-up refuses the calibration setting's old name in one line.
  5. A metadata field's description keeps its wording, as the prompt's sentence does.
  6. An index's name, a collection's name and the emulation's snapshot metadata keep the word.
  7. The charts package and the tour keep the word, since they tell measurements of their day.
  8. `qp-compare-answers` reads the old shape with no end date.
  9. The clone is named by `DOCLANG_CLONE`, and found beside the main checkout when it is unset.
  10. One deploy setting names the wheels' directory, and an App whose wheel is missing stops in one line.
  11. CI clones the default branch of `document_doclang`.
- **Not checked.** Nothing was run against doclang's core from this repository: that it imports on this repository's Python with the toolkit's Pydantic, that an editable install of its namespace package sits beside the others, and that the start script's installer finds a local wheel with no network are all to be shown by the first ticket that needs each.
- **The break.** A client of the workflow app reads `records` where it read `passages`, and a record's id in its metadata. An `.env` file or a deploy file that names the old calibration setting is refused at start-up. A deploy file names the wheels' directory. An answer record written before is still read.
- **What this repository now says in public.** It is public and names `document_doclang` and `doclang-domain-core`, which are not. A reader without the clone cannot install the toolkit.

## What was built

Built on 20 and 21 November 2026 on the branch `claude/package-core-domain-objects-b2dda8`: eleven tickets, each by an agent with a clean context, then a review of the whole branch by two reviewers who had built none of it, one reading the code by area and one using the toolkit from the README alone, then four fix tickets from what they found. Steps 1 to 3 of "The order" are built. Step 4, the comparison of main and the branch on real models, is not: it waits for the owner's word.

### What differs from the plan

- **The response body moved in the third step**, as "What moves, and what stays" now says: until then it held the toolkit's scored passage, and the core may not import the toolkit.
- **The core is installed in more places than were listed**: the image's Dockerfile and the toolkit's own Makefile name it too. One fragment, `doclang.mk`, says how the clone is found, and both Makefiles read it.
- **The deploy setting is `bootstrap_wheels`, the variable `QP_BOOTSTRAP_WHEELS`.** A deploy refuses, before anything is published, a file without the setting, a relative path, a directory that is not there and one without exactly one wheel of doclang's core. A dry run, which also runs off Domino, refuses the first two alone. The start script stops in one line for the same, before anything is installed, in every model-server kind.
- **A record with no id is refused in words**, by one refusal, before any store writes. The plan did not ask for it.
- **An answer record keeps a hash in place of a chunk's texts.** The plan kept a vector-store record's metadata whole; a record made from a chunk holds the chunk's text there, and an answer record of a private corpus must not hold the corpus.
- **Eight things keep the word "passage", not four.** Beyond the four of "The response and the names": the E5 models' own prefix, the old shape's key in the load-test client, the calibration setting's old name where it is refused and told, and the glossary's two "Avoid" lines. The structure test lists each as it is written, in named files, and six folders whole.
- **Two guides under `docs/` were reworded**, the model servers' and the server settings': only `docs/adr/`, `docs/specs/` and `docs/plans/` are dated records.
- **Start-up's refusal of the calibration setting's old name is code of its own**, on the local reranker's settings: the refusal of an unknown key did not give the new name. The deploy tool gives the new name too.
- **The structure tests hold the core** to rules 3 and 9, the README's tree, the code shape and its manifest, from the toolkit's tests: the core's own may not import the toolkit.
- **The handbook's chunking puts a record's id last**, so a document's own `id` cannot replace it, and the in-memory store can be given the rows it holds from the start, so that a test shows which id it reads.
- **The CI step** stops in one line when the secret is missing, as on a pull request from a fork, and pins GitHub's host key.
- **The emulation**: the wheel is built and saved from the laptop, with the cache bucket's name given, since Cloud Shell has no clone; its removal list names the entries of the word's test that go with it.

The eleven points "decided here and not put to the owner in these words" were built as written.

### What it was held to

- At the base, the toolkit collected 3272 tests, the use case 127 and the charts package 72. At the end, the core collects 48, the toolkit 3380, the use case 130 and the charts package 72, and `make check` passes over all four, the strict type checker included.
- Steps 1 and 2 left the three pinned answer records untouched and equal on every registered orchestrator.
- In step 3 they were made again once. Before that, the new answers were shown equal to the old records through the mapping; after it, the old files and the new, read by the load-test client's own reader, were equal on every field: 36 answers, 320 vector-store records.
- A chunk doclang built goes through `/query` and `/retrieve` and is rebuilt equal from the body.

### Not shown, since each needs a real machine

`make install-dev` installing doclang's core beside the others; `make doclang-wheel` building a wheel; an installer finding that wheel with no network, and which it takes if an index offered the same name; the image's build; the CI step on GitHub, which also needs the deploy key and its secret; the emulation's boot; Elasticsearch; real models.

### Left to the owner

Choices the plan did not settle, found by the review:

1. A metadata filter on `id` acts on the stored metadata, while a record read shows the id its store holds it under: a row ingested before cannot be found by id.
2. A document's own metadata keys `id` and `sequence` are replaced silently by the handbook's chunking; a key `chunk` is kept as it is and would be read as a chunk.
3. A use case's tests import the core directly; its layering test does not see it, and its manifest does not list the core.
4. The E5 prefix keeps the word, a thing beyond the plan's four.
5. The mapping reads a whole number under `chunk` as `sequence` in the new shape too, so an answer record does not say which key the app returned; a response with both lists is read by `records` and drops the other.
6. No committed test holds the app to a genuine record of the old form any more.
7. `metadata.id` is not in the wire's schema: the response's model accepts a record with no id, as doclang's record does.
8. Ids that differ only by type are one id: a metadata id `7` and an id `"7"` both read as `'7'`.
9. An index ingested before maps `metadata.chunk` as a number: a doclang-made record cannot be written into it.
10. An answer record says whether two chunks are the same, no longer what a chunk held.
11. `qp-deploy` asks every App to install at its start, as before: a site on the prebuilt image must still put the wheel on the Dataset.
12. An OVMS App with the installer, without the project and without a wheel now stops; without the installer it serves the export as exported.
13. The wheels' directory is added to what the index offers; it does not replace the index.
14. The core's manifest caps doclang's core below 0.2, while nothing is pinned.
15. CI takes the tip of doclang's default branch; the wheel carries what is uncommitted in the clone.
16. The emulation's `wheel-save` does not clear earlier wheels from the bucket.
