"""One open port of each kind: everything a workflow run needs.

``adapters/rag_ports.py`` builds it from the settings; the workflow and the
ingestion use it. The ports themselves are the ``base_*.py`` modules of the
adapters' component packages, the only part of the adapters the application
imports.

Example:
    >>> deps = await ports.workflow_dependencies()
    >>> deps.reranker
    <CrossEncoderReranker ...>
"""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.llm.base_llm import BaseLLM
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.vector_store.base_vector_store import (
    BaseLexicalSearch,
    BaseVectorStore,
)


@dataclass
class RagDependencies:
    """One open port of each kind: everything a workflow run needs.

    ``lexical_search`` is None when the store ranks no text; a lexical or
    hybrid workflow then refuses to be built.
    """

    embedder: BaseEmbedder
    vector_store: BaseVectorStore
    reranker: BaseReranker
    llm: BaseLLM
    lexical_search: BaseLexicalSearch | None = None
