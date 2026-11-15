"""The base of every stage type, and ``STAGE_TYPES``, the registry they register in.

A stage is one step of the workflow graph. It is an instance of a stage type,
built with the component it calls and its own settings. A stage type declares
what a check needs to know about it without knowing the stage: the type of
component it calls (a port), its settings (a model that refuses an unknown key),
and the workflow state fields it reads and writes.

Example:
    >>> @STAGE_TYPES.register("rerank")
    ... class RerankStage(BaseStage[BaseReranker, RerankSettings]): ...
    >>> STAGE_TYPES.get("rerank")
    <class 'RerankStage'>
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from ...domain.registry import Registry
from ..workflow_state import WorkflowState


class StageSettings(BaseModel):
    """The base of a stage type's own settings: an unknown key is refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


ComponentT = TypeVar("ComponentT")
SettingsT = TypeVar("SettingsT", bound=StageSettings)


class BaseStage(abc.ABC, Generic[ComponentT, SettingsT]):
    """A stage type: what one stage does with the component it calls.

    A stage type sets ``component_type`` and ``settings_type`` on its class, and
    ``reads`` and ``writes`` there too, unless its settings change them: then on
    the instance (the retrieve stage reads what its retrieval mode needs).

    Args:
        component: The component the stage calls, an instance of
            ``component_type``.
        settings: The stage's own settings, an instance of ``settings_type``.
    """

    component_type: ClassVar[type[Any]]  # the port of the component it calls
    settings_type: ClassVar[type[StageSettings]]
    reads: frozenset[str]  # the workflow state fields it reads
    writes: frozenset[str]  # and those it writes

    def __init__(self, component: ComponentT, settings: SettingsT) -> None:
        self.component, self.settings = component, settings

    @abc.abstractmethod
    async def run(self, state: WorkflowState) -> WorkflowState:
        """Run the stage once.

        Args:
            state: The workflow state so far.

        Returns:
            The fields the stage writes, the update LangGraph merges into the
            state.
        """


STAGE_TYPES: Registry[type[BaseStage[Any, Any]]] = Registry("stage type")
