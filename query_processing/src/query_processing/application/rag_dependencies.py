"""The components a workflow calls, open, by the names its stages call them.

``composition/composition_root.py`` opens them from the settings; the workflow
app builds its graph over them and probes them for ``/readyz``. The ports
themselves are the ``base_*.py`` modules of the adapters' component packages,
the only part of the adapters the application imports.

Example:
    >>> deps = await composition.workflow_dependencies()
    >>> deps.components()["reranker"]
    <CrossEncoderReranker ...>
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.llm.base_llm import BaseLLM
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.vector_store.base_vector_store import BaseVectorStore

# A port's readiness probe: whether it can serve, and why.
ReadyProbe = Callable[[], Awaitable[tuple[bool, str]]]
# The ports that answer a readiness probe: BaseLLM has none.
_PROBED = (BaseVectorStore, BaseEmbedder, BaseReranker)
# The ports that name the model they run.
_NAMED = (BaseEmbedder, BaseLLM)


@dataclass(frozen=True)
class RagDependencies:
    """The components a workflow calls, open, by name: everything a run needs.

    Example:
        >>> RagDependencies({"llm": FakeLLM()}).model_names()
        {'llm': 'fake-llm'}
    """

    by_name: Mapping[str, object]

    def components(self) -> dict[str, object]:
        """Return the components by the names a stage's ``call`` gives them.

        Returns:
            Each open component, by name.

        Example:
            >>> build_workflow(settings.graph, deps.components(), context, orchestrator)
        """
        return dict(self.by_name)

    def ready_probes(self) -> list[tuple[str, ReadyProbe]]:
        """Return the readiness probe of every component that has one.

        Returns:
            The name and probe of each vector store, embedder and reranker.

        Example:
            >>> [name for name, _ in deps.ready_probes()]
            ['embedder', 'vector_store', 'reranker']
        """
        return [
            (name, port.ready)
            for name, port in self.by_name.items()
            if isinstance(port, _PROBED)
        ]

    def model_names(self) -> dict[str, str]:
        """Return the model each embedder and LLM runs, by component name.

        Returns:
            The model names, as the components report them.

        Example:
            >>> deps.model_names()["llm"]
            'gpt-4o-mini'
        """
        return {
            name: port.model_name
            for name, port in self.by_name.items()
            if isinstance(port, _NAMED)
        }
