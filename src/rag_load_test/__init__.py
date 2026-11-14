"""rag_load_test: one RAG workflow, three Domino topologies, one Locust suite.

Three applications over one shared ``domain`` (value objects, errors, prompts,
the topology rule, the corpus; no I/O). The RAG system is split by layer:
``application`` (the LangGraph workflow, ingest), ``config`` (settings, Domino
rules), ``adapters`` (the ports and their implementations, one package per
component), ``infrastructure`` (technical code that implements no port),
``composition`` (the composition root, which builds the adapters from the
settings) and ``entrypoints`` (FastAPI apps and CLIs). ``deploy`` is rag-deploy
and ``loadtest`` the Locust request plans and the run comparison. The README's
"Where a file goes" says what each package may import.
"""

__version__ = "0.1.0"
