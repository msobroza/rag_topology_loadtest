"""What a workflow graph is made of: its stages, its edges and its conditional edges.

The shape is the ``graph`` block of the workflow file, in LangGraph's own names
(ADR 0004). Stages are keyed by name; each names the component it calls with
``call``, and states the workflow state fields it reads and writes with
``reads`` and ``writes``. A stage is an instance of the stage type ``type``
names, else of the stage type of its own name, else a plain call. Every other
key is a setting of the stage, which its stage type, or for a plain call its
component's request, takes. Plain
edges are chains of stage names, ``START`` and ``END`` reserved. A conditional
edge has a ``source``, a ``path`` that names a registered edge condition, the
``reads`` whose values that edge condition is asked about, and a ``path_map``
from each answer, true and false or each label, to a stage or ``END``.

These types are in the domain because the settings, which read the description,
and the application, which builds the graph from it, may both import the
domain and neither may import the other.

Example:
    >>> graph = GraphDescription.model_validate({
    ...     "stages": {"retrieve": {"call": "vector_store", "top_k": 20,
    ...                             "reads": ["query_embedding"],
    ...                             "writes": ["candidates"]}},
    ...     "edges": [["START", "retrieve", "END"]],
    ... })
    >>> graph.edge_pairs()
    [('START', 'retrieve'), ('retrieve', 'END')]
"""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The reserved names a chain of plain edges starts from and ends at.
START = "START"
END = "END"


class StageDescription(BaseModel):
    """One stage: its stage type, the component it calls, its fields and settings.

    Every key besides ``type``, ``call``, ``reads`` and ``writes`` is kept as a
    setting of the stage; its stage type validates them when the graph is
    built, and refuses a key it does not know.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str | None = None  # None = the stage's own name
    call: str
    # The workflow state fields the stage reads and writes, in the order its
    # component's request and answer, or its stage type, take them.
    reads: list[str]
    writes: list[str]

    def type_name(self, stage: str) -> str:
        """Return the name of the stage type this stage is an instance of.

        Args:
            stage: The stage's name, which the type defaults to.

        Returns:
            ``type`` when it is given, else ``stage``.

        Example:
            >>> StageDescription(call="llm", reads=["a"], writes=["b"]).type_name(
            ...     "generate")
            'generate'
        """
        return self.type or stage

    @property
    def stage_settings(self) -> dict[str, Any]:
        """The stage's own settings: every key besides ``type``, ``call`` and the lists.

        Example:
            >>> StageDescription.model_validate(
            ...     {"call": "reranker", "top_k": 10,
            ...      "reads": ["standalone_query", "candidates"],
            ...      "writes": ["reranked"]}
            ... ).stage_settings
            {'top_k': 10}
        """
        return dict(self.model_extra or {})


# The answers of a yes-or-no edge condition as text, as an environment variable
# gives them; any other key of a path_map is a label.
_ANSWERS_AS_TEXT = {"true": True, "false": False}


class ConditionalEdge(BaseModel):
    """An edge that asks an edge condition, and goes where its answer's row leads.

    ``path`` names a registered edge condition, asked about the values of the
    fields ``reads`` names, in order; ``path_map`` maps each answer, true and
    false or a label, to a stage or ``END`` (ADR 0007).

    Example:
        >>> ConditionalEdge(source="route", path="label", reads=["route"],
        ...                 path_map={"self_contained": "embed"}).path_map
        {'self_contained': 'embed'}
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    path: str
    reads: list[str]
    path_map: dict[bool | str, str]

    @field_validator("path_map", mode="before")
    @classmethod
    def _answers_as_booleans(cls, path_map: object) -> object:
        # "true" and "false" are yes and no, not labels: the union would keep
        # them as the strings they are.
        if not isinstance(path_map, dict):
            return path_map
        return {_answer(key): to for key, to in path_map.items()}


def _answer(key: object) -> object:
    # "True" or "false" from an environment variable is a yes or a no too.
    if isinstance(key, str):
        return _ANSWERS_AS_TEXT.get(key.lower(), key)
    return key


# A chain names at least one edge, so at least two names.
EdgeChain = Annotated[list[str], Field(min_length=2)]


class GraphDescription(BaseModel):
    """A workflow graph: stages by name, chains of plain edges, conditional edges.

    Example:
        >>> GraphDescription.model_validate({
        ...     "stages": {"rerank": {"call": "reranker", "top_k": 10,
        ...                           "reads": ["standalone_query", "candidates"],
        ...                           "writes": ["reranked"]}},
        ...     "edges": [["START", "rerank"]],
        ...     "conditional_edges": [{
        ...         "source": "rerank", "path": "is_query_mode", "reads": ["mode"],
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

    def links(self) -> list[tuple[str, str]]:
        """Return every way from one stage to the next: each plain edge, then each row.

        Returns:
            ``(source, target)`` for each plain edge, then for each row of each
            conditional edge's ``path_map``.

        Example:
            >>> graph.links()
            [('START', 'embed'), ..., ('rerank', 'generate'), ('rerank', 'END')]
        """
        answers = [
            (edge.source, target)
            for edge in self.conditional_edges
            for target in edge.path_map.values()
        ]
        return [*self.edge_pairs(), *answers]
