"""rag_load_test: one RAG workflow, three Domino topologies, one Locust suite.

Layers: ``domain`` (value objects, ports, errors, prompts, topology rule, corpus;
no I/O), ``application`` (the LangGraph workflow, ingest), ``config`` (settings,
Domino rules), ``adapters`` (local models, Elasticsearch, OpenAI, OVMS, fakes and
``rag_ports``, the composition root), ``entrypoints`` (FastAPI apps and CLIs),
``loadtest`` (Locust request plans + run comparison) and ``deploy`` (rag-deploy
and a fake Domino Apps API). Domain imports no other layer; application and config
import only domain.
"""

__version__ = "0.1.0"
