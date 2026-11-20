# A core distribution holds what toolkits exchange, and the workflow handles ingestion's vector-store record

The toolkit's unit of retrieval was `Passage`, a class of its own: an id, a text and a flat map of metadata. `document_doclang`, a repository of its own, builds chunks from documents and stores each as a `VectorStoreRecord` with the same content, and a toolkit for ingestion and one for evalops are planned beside query processing's. Two decisions follow. A third distribution, the core (`query_processing_core`), holds an object when one toolkit writes it and another reads it: the vector-store record, the metadata filter, the request body with its conversational history, and the response body. It needs Pydantic and `doclang-domain-core` only, has no port, no settings and no I/O, and imports no toolkit. And the workflow handles doclang's `VectorStoreRecord`, with its two scores on a subclass: `Passage` goes, and the word with it.

Amends [ADR 0005](0005-toolkit-and-use-case-are-two-distributions.md): the product is three distributions. Decided on 20 November 2026, before the build: the README describes the code as it is.

## Considered options

- **`Chunk` in the workflow.** Rejected by the owner: a stage reads a text and filters on flat metadata, which is the record. A chunk needs a document version, pages and blocks, which a plain-text corpus has not; a record made from a chunk carries it whole for whoever wants it.
- **`Passage` kept, as a class of the toolkit's own.** Rejected: two classes and two words for one thing, one on each side of the store.
- **"Passage" kept on the wire alone**, the response in its shape of before. Rejected by the owner: one word everywhere. It would have left the pinned answer records untouched.
- **The shared objects in `doclang-domain-core`.** Rejected: it is the model of documents and their chunks, and a request, a metadata filter and a response say nothing of a DocLang document.
- **doclang's core moved into this repository.** Not taken: it has a repository, gates and a Python floor of its own. It would have spared the delivery named below.
- **`doclang-domain-core` on a public index, or its repository made public.** Rejected by the owner: it stays private.
- **A commit of the clone pinned in this repository.** Rejected by the owner: the laptop installs the clone as it stands.
- **The record defined here, in doclang's shape, with no dependency.** Not taken: the class is doclang's own, so a change to it is made once.
- **Every record made from a chunk.** Rejected for now: the synthetic handbook and a JSONL corpus are plain text, and the toolkit would need doclang's ingestion package to chunk them.
- **The ports of the embedder and the vector store in the core.** Rejected: they bring settings and adapters with them, and the core is data.

## Consequences

- `document_doclang` stays private while this repository is public, so no index and no URL gives the package. The core's `pyproject.toml` names it by name and version only, and each place that installs the toolkit is handed it. The laptop installs the working tree of the clone beside this repository, so the gate follows what is checked out there. CI clones it with a read-only key. A Domino App and an emulation machine install a wheel built from the clone and kept where the exported models are, the Dataset and the cache bucket, so an App's start holds no credential, which is what ADR 0005 guarded. A reader of this repository without the clone cannot install the toolkit.
- A record has no id field: its id is `metadata["id"]`. Each store's adapter gives a record the id the store holds it under, on every read, so an index ingested before this decision still answers.
- The response returns the scored records as they are, under `records`. When a record holds a chunk, the client receives it.
- A record with no chunk is valid in the workflow, and `from_vector_store` refuses it. The handbook's window index moves from the metadata key `chunk`, which doclang keeps for the chunk itself, to `sequence`, doclang's key for the same thing.
- The pinned answer records are made again once. A test holds the new ones to the old through a stated mapping (the id read from the metadata, the renamed keys), and `qp-compare-answers` reads a record of the old shape through the same mapping, so a branch can be held to main on real models.
- `RagResult` stays in the toolkit: it is what the application hands the entrypoint, and no other toolkit sees it. So do the components with their requests and answers, the API formats, the registries and the graph.
- An ingestion toolkit cannot embed or upsert through the core alone. Its own design settles where the embedder and the vector store come from.
- A use case still imports the toolkit through `extension_api` alone, which hands on the core's names it needs.
- The prompt's sentence to the LLM keeps "passages": it is a use case's wording to a model, not the glossary's. The records in `experiments/` keep the word of their day.
