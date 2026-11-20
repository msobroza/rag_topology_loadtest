"""The base of every stage type, and ``STAGE_TYPES``, the registry they register in.

A stage is one step of the workflow graph. A stage that only calls its
component is a plain call and needs no stage type; a stage type is for one that
does more. A stage of a stage type is built with the component it calls, its
own settings and the workflow state fields the workflow file says it reads and
writes. A stage type declares what a check needs to know about it without
building a stage: the type of component it calls (a port), its settings (a
model that refuses an unknown key), and which lists of fields it can use.

Example:
    >>> @STAGE_TYPES.register("retrieve")
    ... class RetrieveStage(BaseStage[BaseVectorStore, RetrieveSettings]): ...
    >>> STAGE_TYPES.get("retrieve")
    <class 'RetrieveStage'>
"""

from __future__ import annotations

import abc
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, ClassVar, Generic, Literal, NamedTuple, TypeVar

from pydantic import BaseModel, ConfigDict
from query_processing_core.retrieval.metadata_filter import MetadataFieldInfo

from ...domain.prompts import Prompt
from ...domain.registry import Registry
from ..workflow_state import WorkflowState

# A dotted module path in front of a class name, as str() of a generic shows it.
_MODULE_PATH = re.compile(r"\b(?:\w+\.)+(\w+)")


class StageSettings(BaseModel):
    """The base of a stage type's own settings: an unknown key is refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StageFields(NamedTuple):
    """The workflow state fields a stage states it reads and writes, in order.

    Example:
        >>> StageFields(("standalone_query", "candidates"), ("reranked",)).reads
        ('standalone_query', 'candidates')
    """

    reads: tuple[str, ...]
    writes: tuple[str, ...]


class FieldTypes(NamedTuple):
    """The type of each field a stage reads and writes, by the field's name.

    A field's type is that of the request or answer attribute it is bound to,
    or ``Any`` for one that is not checked. ``labels`` gives, for a field
    written that takes one of a few labels, such as a router's route, those
    labels: a label edge that reads the field has one row of ``path_map`` for
    each (ADR 0007).

    Example:
        >>> FieldTypes({"standalone_query": str}, {"query_embedding": list[float]})
        FieldTypes(reads={'standalone_query': <class 'str'>}, writes={...}, labels={})
        >>> FieldTypes({}, {"route": str}, {"route": ("self_contained",)}).labels
        {'route': ('self_contained',)}
    """

    reads: dict[str, Any]
    writes: dict[str, Any]
    # A NamedTuple's default, shared and never mutated: no field declares labels.
    labels: dict[str, tuple[str, ...]] = {}  # noqa: RUF012


@dataclass(frozen=True)
class StageContext:
    """What every stage is given besides its component, settings and fields.

    The composition builds it from the settings: the prompts of the prompts
    file, by name, which a stage names in its settings; and the metadata
    fields each vector store declares, by the store's component name, which a
    ``generate`` stage's ``metadata_fields_of`` names.

    Example:
        >>> StageContext(settings.prompts).prompts["answer_from_context"][0].role
        'system'
        >>> StageContext(metadata_field_info={"vector_store": [topic]})
    """

    prompts: Mapping[str, Prompt] = dataclass_field(default_factory=dict)
    metadata_field_info: Mapping[str, Sequence[MetadataFieldInfo]] = dataclass_field(
        default_factory=dict
    )


ComponentT = TypeVar("ComponentT")
SettingsT = TypeVar("SettingsT", bound=StageSettings)


class BaseStage(abc.ABC, Generic[ComponentT, SettingsT]):
    """A stage type: what one stage does with the component it calls.

    A stage type sets ``component_type`` and ``settings_type`` on its class, and
    says in ``check_fields`` which lists of fields it can use. The start-up
    checks ask those of the class, the settings and the lists, so they need no
    component and build no stage.

    Args:
        component: The component the stage calls, an instance of
            ``component_type``.
        settings: The stage's own settings, an instance of ``settings_type``.
        fields: The fields the stage reads and writes, as its stage states them.
        context: What every stage is given, such as the prompts.

    Raises:
        ValueError: The stage type cannot use those fields with those settings.
    """

    component_type: ClassVar[type[Any]]  # the port of the component it calls
    settings_type: ClassVar[type[StageSettings]]

    def __init__(
        self,
        component: ComponentT,
        settings: SettingsT,
        fields: StageFields,
        context: StageContext,
    ) -> None:
        self.field_types = type(self).check_fields(settings, fields, context)
        self.component, self.settings, self.fields = component, settings, fields

    @classmethod
    @abc.abstractmethod
    def check_fields(
        cls, settings: SettingsT, fields: StageFields, context: StageContext
    ) -> FieldTypes:
        """Return the type of each field a stage of this type reads and writes.

        Args:
            settings: The stage's own settings, which may change what it reads.
            fields: The fields the stage states it reads and writes, in order.
            context: What every stage is given, such as the prompts its
                settings may name.

        Returns:
            The type of each field read and written.

        Raises:
            ValueError: The stage type cannot use those lists with those
                settings and that context; the message shows what it expects.

        Example:
            >>> RetrieveStage.check_fields(
            ...     RetrieveSettings(retrieval_mode="dense", top_k=20),
            ...     StageFields(("query_embedding",), ("candidates",)),
            ...     StageContext()).reads
            {'query_embedding': list[float]}
        """

    @abc.abstractmethod
    async def run(self, state: WorkflowState) -> WorkflowState:
        """Run the stage once.

        Args:
            state: The workflow state so far.

        Returns:
            The fields the stage writes, the update the orchestrator merges
            into the state.
        """

    def read_fields(self, state: WorkflowState) -> list[Any]:
        """Return the values of the fields the stage reads, in its ``reads`` order.

        Args:
            state: The workflow state so far.

        Returns:
            One value per field of ``reads``.

        Example:
            >>> question, passages = stage.read_fields(state)
        """
        return [state[field] for field in self.fields.reads]

    def write_field(self, value: object) -> WorkflowState:
        """Return the update that writes ``value`` to the stage's one written field.

        Args:
            value: What the stage produced.

        Returns:
            The update the orchestrator merges into the state.

        Example:
            >>> stage.write_field("the answer")
            {'answer': 'the answer'}
        """
        (written,) = self.fields.writes  # check_fields allows one field only
        return {written: value}


def field_count(count: int) -> str:
    """Return a number of fields as a refusal says it.

    Args:
        count: How many fields.

    Returns:
        ``1 field`` or ``<count> fields``.

    Example:
        >>> field_count(2)
        '2 fields'
    """
    return f"{count} field" if count == 1 else f"{count} fields"


def fields_in_order(
    which: Literal["reads", "writes"],
    stated: tuple[str, ...],
    expected: Mapping[str, Any],
    expecting: str,
) -> dict[str, Any]:
    """Pair the fields a stage states with what its stage type expects, in order.

    Args:
        which: ``reads`` or ``writes``, as the workflow file names the list.
        stated: The fields the stage states, in order.
        expected: What the stage type takes or gives in their place, each with
            its type, in order.
        expecting: Who expects them, the start of the refusal's message.

    Returns:
        Each stated field with the type of what it stands for.

    Raises:
        ValueError: The stage states more or fewer fields than expected.

    Example:
        >>> fields_in_order("writes", ("found",), {"candidates": list[str]},
        ...                 "the retrieve stage")
        {'found': list[str]}
    """
    if len(stated) != len(expected):
        count = field_count(len(expected))
        wanted = ", ".join(f"{name}: {type_name(t)}" for name, t in expected.items())
        raise ValueError(
            f"{expecting} {which} {count}, for {wanted} (got {which} {list(stated)})"
        )
    return dict(zip(stated, expected.values(), strict=True))


def type_name(annotation: object) -> str:
    """Return a type as a workflow file's author reads it, without module paths.

    Args:
        annotation: A class or a generic alias.

    Returns:
        Its name, such as ``str`` or ``list[ScoredVectorStoreRecord]``.

    Example:
        >>> type_name(list[ScoredVectorStoreRecord])
        'list[ScoredVectorStoreRecord]'
    """
    if isinstance(annotation, type) and not getattr(annotation, "__args__", None):
        return annotation.__name__
    return _MODULE_PATH.sub(r"\1", str(annotation))


StageTypes = Registry[type[BaseStage[Any, Any]]]

STAGE_TYPES: StageTypes = Registry("stage type")
