"""How a stage is bound: its stage type, or for a plain call, its fields and its keys.

A stage is of the stage type its ``type`` names, else of the registered stage
type of its own name, else, when it states ``reads`` and ``writes``, a plain
call. A plain call hands its fields to the component it calls by three rules
(spec 3.1):

1. A key of the stage that names a request attribute fills it, such as
   ``top_k: 10``.
2. ``reads`` fill the remaining request attributes, in order.
3. ``writes`` take the answer's attributes, in order, and can stop early.

Both the start-up checks and the builder resolve a stage here, so they never
disagree on what a stage is.

Example:
    >>> binding = bind_call(BaseReranker, {"top_k": 10},
    ...                     StageFields(("standalone_query", "candidates"),
    ...                                 ("reranked",)))
    >>> binding.from_reads
    {'query': 'standalone_query', 'passages': 'candidates'}
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, NamedTuple

from pydantic import BaseModel

from ..domain.base_component import BaseComponent
from ..domain.graph_description import StageDescription
from .stages.base_stage import BaseStage, FieldTypes, StageFields, StageTypes

# A dotted module path in front of a class name, as str() of a generic shows it.
_MODULE_PATH = re.compile(r"\b(?:\w+\.)+(\w+)")


class CallBinding(NamedTuple):
    """Where each request attribute of a plain call comes from, and each answer goes.

    ``from_keys`` are request attributes filled by the stage's keys,
    ``from_reads`` maps a request attribute to the state field it is read from,
    and ``to_writes`` maps a state field to the answer attribute written to it.

    Example:
        >>> bind_call(BaseReranker, {"top_k": 10}, StageFields(
        ...     ("standalone_query", "candidates"), ("reranked",))).to_writes
        {'reranked': 'passages'}
    """

    from_keys: dict[str, object]
    from_reads: dict[str, str]
    to_writes: dict[str, str]
    field_types: FieldTypes


def stage_fields(description: StageDescription) -> StageFields:
    """Return the fields a stage states, none for a list it does not state.

    Args:
        description: The stage as the workflow file gives it.

    Returns:
        Its ``reads`` and ``writes``, in the file's order.

    Example:
        >>> stage_fields(StageDescription(call="llm", reads=["a"], writes=["b"]))
        StageFields(reads=('a',), writes=('b',))
    """
    return StageFields(tuple(description.reads or ()), tuple(description.writes or ()))


def stage_type_of(
    name: str, description: StageDescription, stage_types: StageTypes
) -> type[BaseStage[Any, Any]] | None:
    """Return a stage's stage type, or None for a plain call.

    Args:
        name: The stage's name, a stage type's name when ``type`` is not given.
        description: The stage as the workflow file gives it.
        stage_types: Where a stage type is found.

    Returns:
        The stage type ``type`` names, else the one of the stage's name, else
        None when the stage states ``reads`` and ``writes``.

    Raises:
        ValueError: ``type`` names no stage type, or the stage names none, its
            name is none and it does not state both lists.

    Example:
        >>> stage_type_of("embed_twice", StageDescription(
        ...     call="embedder", reads=["a"], writes=["b"]), STAGE_TYPES) is None
        True
    """
    if description.type is not None:
        return stage_types.get(description.type)
    if name in stage_types.names():
        return stage_types.get(name)
    if description.reads is None or description.writes is None:
        known = ", ".join(repr(known_name) for known_name in stage_types.names())
        raise ValueError(
            f"unknown stage type {name!r} (expected one of: {known}, or a plain "
            "call that states both reads and writes)"
        )
    return None


def bind_call(
    component_type: type[BaseComponent[Any, Any]],
    stage_keys: Mapping[str, object],
    fields: StageFields,
) -> CallBinding:
    """Bind a plain call's keys and fields to its component's request and answer.

    Args:
        component_type: The component type of the component the stage calls.
        stage_keys: The stage's own keys, besides ``type``, ``call`` and the lists.
        fields: The fields the stage reads and writes, in order.

    Returns:
        Where each request attribute comes from and each answer attribute goes.

    Raises:
        ValueError: A key names no request attribute, ``reads`` does not fill
            the attributes the keys leave exactly, or ``writes`` lists more
            fields than the answer has attributes, or one field twice. The
            message shows what the
            component takes and gives.

    Example:
        >>> bind_call(BaseEcho, {}, StageFields(("message",), ("shout",))).to_writes
        {'shout': 'text'}
    """
    takes = _attributes(component_type, component_type.request_type)
    gives = _attributes(component_type, component_type.answer_type)
    left = [attribute for attribute in takes if attribute not in stage_keys]
    problem = _binding_problem(takes, gives, stage_keys, left, fields)
    if problem:
        raise ValueError(f"{_signature(component_type, takes, gives)}; {problem}")
    from_reads = dict(zip(left, fields.reads, strict=True))
    to_writes = dict(zip(fields.writes, gives, strict=False))
    return CallBinding(
        from_keys=dict(stage_keys),
        from_reads=from_reads,
        to_writes=to_writes,
        field_types=FieldTypes(
            reads={field: takes[attr] for attr, field in from_reads.items()},
            writes={field: gives[attr] for field, attr in to_writes.items()},
        ),
    )


def _binding_problem(
    takes: Mapping[str, Any],
    gives: Mapping[str, Any],
    stage_keys: Mapping[str, object],
    left: list[str],
    fields: StageFields,
) -> str | None:
    # The first rule of the three the stage breaks, if any.
    unknown = [key for key in stage_keys if key not in takes]
    if unknown:
        return f"key {unknown[0]!r} names no attribute of the request"
    if len(fields.reads) != len(left):
        filled = [key for key in takes if key in stage_keys]
        after = f" after its keys fill {', '.join(filled)}" if filled else ""
        return f"reads lists {_count(fields.reads)}{after}"
    return _writes_problem(gives, fields.writes)


def _writes_problem(gives: Mapping[str, Any], writes: tuple[str, ...]) -> str | None:
    if len(writes) > len(gives):
        return f"writes lists {_count(writes)}"
    if len(set(writes)) != len(writes):
        # Two answer attributes into one field: the first would be lost.
        return f"writes lists a field twice: {list(writes)}"
    return None


def _attributes(
    component_type: type[BaseComponent[Any, Any]], model: type[Any]
) -> dict[str, Any]:
    # A request's or an answer's attributes, in order, each with its type.
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ValueError(
            f"{_type_name(component_type)} takes or gives {_type_name(model)}, "
            "which has no attributes to bind (expected a pydantic model for a "
            "plain call)"
        )
    return {name: info.annotation for name, info in model.model_fields.items()}


def _signature(
    component_type: type[BaseComponent[Any, Any]],
    takes: Mapping[str, Any],
    gives: Mapping[str, Any],
) -> str:
    # How the component type is called: "echo takes (text: str) and gives (...)".
    def listed(attributes: Mapping[str, Any]) -> str:
        return ", ".join(f"{a}: {_type_name(t)}" for a, t in attributes.items())

    name = getattr(component_type, "component_type_name", component_type.__name__)
    return f"{name} takes ({listed(takes)}) and gives ({listed(gives)})"


def _type_name(annotation: object) -> str:
    # str, list[float], list[ScoredPassage]: a type as a workflow file's author
    # reads it, without module paths.
    if isinstance(annotation, type) and not getattr(annotation, "__args__", None):
        return annotation.__name__
    return _MODULE_PATH.sub(r"\1", str(annotation))


def _count(names: tuple[str, ...]) -> str:
    return f"{len(names)} field" if len(names) == 1 else f"{len(names)} fields"
