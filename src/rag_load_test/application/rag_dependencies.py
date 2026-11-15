"""One open port of each kind: everything a workflow run needs.

``composition/composition_root.py`` builds it from the settings; the workflow app
builds its graph over its components, and the ingestion uses it. The ports
themselves are the ``base_*.py`` modules of the adapters' component packages,
the only part of the adapters the application imports.

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

    ``lexical_search`` is the store again when it ranks text, else None. The
    retrieve stage asks the store itself, and refuses a lexical or hybrid
    retrieval over a store that ranks no text when the graph is built.
    """

    embedder: BaseEmbedder
    vector_store: BaseVectorStore
    reranker: BaseReranker
    llm: BaseLLM
    lexical_search: BaseLexicalSearch | None = None

    def components(self) -> dict[str, object]:
        """Return the four components by the names a stage's ``call`` gives them.

        Returns:
            The embedder, reranker, LLM and vector store, by their fixed names.

        Example:
            >>> build_workflow(settings.workflow_graph(), deps.components())
        """
        return {
            "embedder": self.embedder,
            "reranker": self.reranker,
            "llm": self.llm,
            "vector_store": self.vector_store,
        }
