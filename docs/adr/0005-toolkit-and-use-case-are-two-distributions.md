# The toolkit and a use case are two distributions in one repository, and a use case holds settings only

The RAG became the first use case of a toolkit, `query_processing`. A use case is a package of its own that installs the toolkit, `query_processing_rag_usecase` for the RAG, and it holds settings only: the workflow file, the deploy settings example, the environment example, the load-test questions and a Makefile. Every line of code stays in the toolkit, the four RAG stage types included, so that a second use case is a package of files.

## Considered options

- **One package.** Rejected: the toolkit's settings would keep the RAG's models, prompts and sizes as defaults, and the owner wants the use case installable on its own.
- **The use case in its own repository.** Rejected for now: a Domino App mounts one repository and installs from it when it starts, so a second repository means fetching the toolkit, with credentials, at every start, and a setting and its value would change in two pull requests. Moving the directory out later is a move, not a redesign.
- **Code in the use case** (its stage types, ingestion, routes). Rejected: the registries would have to accept registrations from another package, and embed, retrieve, rerank and generate are stages any use case can assemble.

## Consequences

- The toolkit never imports a use case. A structure test checks it.
- The toolkit ships no RAG defaults. The use case's workflow file is the one place that pins the models, the prompts and the sizes, the toolkit's tests run on fixture workflow files, and the use case has tests of its own.
- A new stage type is a change to the toolkit, even when one use case alone needs it.
- A command finds the use case by location: `workflow.yaml` and `deploy.yaml` in the working directory, or the files `QP_WORKFLOW_FILE` and `QP_DEPLOY_FILE` name.
- Apps are named by the use case's prefix (`qp-rag-workflow`), so two use cases can share a Domino project.
