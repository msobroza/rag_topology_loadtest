"""What a workflow graph is made of: its stages, its edges and its conditional edges.

The shape is the ``graph`` block of the workflow file, in LangGraph's own names
(ADR 0004). Stages are keyed by name; each is an instance of a stage type, which
defaults to the stage's name, and names the component it calls with ``call``;
every other key is a setting of the stage, which its stage type validates. Plain
edges are chains of stage names, ``START`` and ``END`` reserved. A conditional
edge has a ``source``, a ``path`` that names a registered edge condition, and a
``path_map`` from true and false to a stage or ``END``.

These types are in the domain because the settings, which read the description,
and the application, which builds the graph from it, may both import the
domain and neither may import the other.

Example:
    >>> graph = GraphDescription.model_validate({
    ...     "stages": {"retrieve": {"call": "vector_store", "top_k": 20}},
    ...     "edges": [["START", "retrieve", "END"]],
    ... })
    >>> graph.edge_pairs()
    [('START', 'retrieve'), ('retrieve', 'END')]
"""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

# The reserved names a chain of plain edges starts from and ends at.
START = "START"
END = "END"


class StageDescription(BaseModel):
    """One stage: its stage type, the component it calls and its own settings.

    Every key besides ``type`` and ``call`` is kept as a setting of the stage;
    its stage type validates them when the graph is built, and refuses a key it
    does not know.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str | None = None  # None = the stage's own name
    call: str

    def type_name(self, stage: str) -> str:
        """Return the name of the stage type this stage is an instance of.

        Args:
            stage: The stage's name, which the type defaults to.

        Returns:
            ``type`` when it is given, else ``stage``.

        Example:
            >>> StageDescription(call="embedder").type_name("embed")
            'embed'
        """
        return self.type or stage

    @property
    def stage_settings(self) -> dict[str, Any]:
        """The stage's own settings: every key besides ``type`` and ``call``.

        Example:
            >>> StageDescription.model_validate(
            ...     {"call": "reranker", "top_k": 10}
            ... ).stage_settings
            {'top_k': 10}
        """
        return dict(self.model_extra or {})


class ConditionalEdge(BaseModel):
    """An edge that asks an edge condition and goes one way on true, another on false.

    ``path`` names a registered edge condition; ``path_map`` maps its answer to
    a stage or ``END``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    path: str
    path_map: dict[bool, str]


# A chain names at least one edge, so at least two names.
EdgeChain = Annotated[list[str], Field(min_length=2)]


class GraphDescription(BaseModel):
    """A workflow graph: stages by name, chains of plain edges, conditional edges.

    Example:
        >>> GraphDescription.model_validate({
        ...     "stages": {"rerank": {"call": "reranker", "top_k": 10}},
        ...     "edges": [["START", "rerank"]],
        ...     "conditional_edges": [{
        ...         "source": "rerank", "path": "is_query_mode",
        ...         "path_map": {True: "END", False: "END"},
        ...     }],
        ... })
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stages: dict[str, StageDescription]
    edges: list[EdgeChain] = Field(default_factory=list)
    conditional_edges: list[ConditionalEdge] = Field(default_factory=list)

    def edge_pairs(self) -> list[tuple[str, str]]:
        """Expand every chain into its edges, one per pair of neighbours.

        Returns:
            ``(source, target)`` for each edge, in the order the chains give them.

        Example:
            >>> GraphDescription(stages={}, edges=[["START", "a", "b"]]).edge_pairs()
            [('START', 'a'), ('a', 'b')]
        """
        return [pair for chain in self.edges for pair in pairwise(chain)]
