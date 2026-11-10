"""rag_load_test: one RAG workflow, three Domino topologies, one Locust suite.

Modules: ``models`` (ports, value objects, errors), ``settings``, ``adapters``
(in-process models + Elasticsearch + OpenAI), ``ovms`` (OpenVINO Model Server HTTP),
``fakes``, ``workflow`` (LangGraph graph + wiring), ``api`` (FastAPI + CLI),
``corpus`` (synthetic corpus + ingestion CLI), ``loadtest`` (Locust helpers +
run comparison).
"""

__version__ = "0.1.0"
