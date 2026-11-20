"""How a stage is bound: its stage type, or for a plain call, its fields and its keys.

A stage is of the stage type its ``type`` names, else of the registered stage
type of its own name, else a plain call. A plain call hands its fields to the
component it calls by three rules (spec 3.1):

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
    {'query': 'standalone_query', 'records': 'candidates'}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo

from ..domain.components.base_component import BaseComponent
from ..domain.graph_description import StageDescription
from .stages.base_stage import (
    BaseStage,
    FieldTypes,
    StageFields,
    StageTypes,
    field_count,
    type_name,
)


class CallBinding(NamedTuple):
    """Where each request attribute of a plain call comes from, and each answer goes.

    ``from_keys`` are request attributes filled by the stage's keys,
    ``from_reads`` maps a request attribute to the state field it is read from,
    and ``to_writes`` maps a state field to the answer attribute written to it.

    Example:
        >>> bind_call(BaseReranker, {"top_k": 10}, StageFields(
        ...     ("standalone_query", "candidates"), ("reranked",))).to_writes
        {'reranked': 'records'}
    """

    from_keys: dict[str, object]
    from_reads: dict[str, str]
    to_writes: dict[str, str]
    field_types: FieldTypes


def stage_fields(description: StageDescription) -> StageFields:
    """Return the fields a stage states it reads and writes.

    Args:
        description: The stage as the workflow file gives it.

    Returns:
        Its ``reads`` and ``writes``, in the file's order.

    Example:
        >>> stage_fields(StageDescription(call="llm", reads=["a"], writes=["b"]))
        StageFields(reads=('a',), writes=('b',))
    """
    return StageFields(tuple(description.reads), tuple(description.writes))


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
        None for a plain call.

    Raises:
        ValueError: ``type`` names no stage type; the registry names the known
            ones.

    Example:
        >>> stage_type_of("embed", StageDescription(
        ...     call="embedder", reads=["a"], writes=["b"]), STAGE_TYPES) is None
        True
    """
    if description.type is not None:
        return stage_types.get(description.type)
    if name in stage_types.names():
        return stage_types.get(name)
    return None


def bind_call(
    component_type: type[BaseComponent[Any, Any]],
    stage_keys: Mapping[str, object],
    fields: StageFields,
    answer_labels: Mapping[str, tuple[str, ...]] | None = None,
) -> CallBinding:
    """Bind a plain call's keys and fields to its component's request and answer.

    Args:
        component_type: The component type of the component the stage calls.
        stage_keys: The stage's own keys, besides ``type``, ``call`` and the lists.
        fields: The fields the stage reads and writes, in order.
        answer_labels: The labels an answer attribute takes, by its name, as
            the component type declares them (``answer_labels``); a field the
            attribute is written to takes them. None for none.

    Returns:
        Where each request attribute comes from and each answer attribute goes,
        and the labels of each field written that takes some.

    Raises:
        ValueError: A key names no request attribute or does not fit its
            type, ``reads`` does not fill
            the attributes the keys leave exactly, or ``writes`` lists more
            fields than the answer has attributes, or one field twice. The
            message shows what the component takes and gives.

    Example:
        >>> bind_call(BaseEcho, {}, StageFields(("message",), ("shout",))).to_writes
        {'shout': 'text'}
    """
    takes = _attributes(component_type, component_type.request_type)
    gives = _attributes(component_type, component_type.answer_type)
    left = [attribute for attribute in takes if attribute not in stage_keys]
    problem = _keys_problem(component_type.request_type, stage_keys)
    problem = problem or _binding_problem(takes, gives, stage_keys, left, fields)
    if problem:
        raise ValueError(f"{_signature(component_type, takes, gives)}; {problem}")
    from_reads = dict(zip(left, fields.reads, strict=True))
    to_writes = dict(zip(fields.writes, gives, strict=False))
    labels = answer_labels or {}
    return CallBinding(
        from_keys=dict(stage_keys),
        from_reads=from_reads,
        to_writes=to_writes,
        field_types=FieldTypes(
            reads={field: takes[attr] for attr, field in from_reads.items()},
            writes={field: gives[attr] for field, attr in to_writes.items()},
            labels={f: labels[a] for f, a in to_writes.items() if a in labels},
        ),
    )


def _binding_problem(
    takes: Mapping[str, Any],
    gives: Mapping[str, Any],
    stage_keys: Mapping[str, object],
    left: list[str],
    fields: StageFields,
) -> str | None:
    # The second and third rules, once the keys fit (_keys_problem).
    if len(fields.reads) != len(left):
        filled = [key for key in takes if key in stage_keys]
        after = f" after its keys fill {', '.join(filled)}" if filled else ""
        return f"reads lists {field_count(len(fields.reads))}{after}"
    return _writes_problem(gives, fields.writes)


def _keys_problem(
    request_type: type[Any], stage_keys: Mapping[str, object]
) -> str | None:
    # The first rule: each key names a request attribute, and fits it.
    attributes: dict[str, FieldInfo] = request_type.model_fields
    unknown = [key for key in stage_keys if key not in attributes]
    if unknown:
        return f"key {unknown[0]!r} names no attribute of the request"
    for key, value in stage_keys.items():
        refusal = _misfit(attributes[key], value)
        if refusal:
            expected = type_name(attributes[key].annotation)
            return f"key {key!r} is {value!r} (expected {expected}: {refusal})"
    return None


def _misfit(attribute: FieldInfo, value: object) -> str | None:
    # Validated as the request will validate it, its constraints included, so
    # "10" from a QP_* variable fits an int, and "ten" or 0 for a top_k of at
    # least 1 is refused at start-up, not on the first request.
    # A one-attribute model of the request's own attribute, built at run time.
    annotation: Any = attribute.annotation  # never None on a model's attribute
    key_model = create_model("StageKey", value=(annotation, attribute))
    try:
        key_model.model_validate({"value": value})
    except ValidationError as exc:
        return str(exc.errors()[0]["msg"])
    return None


def _writes_problem(gives: Mapping[str, Any], writes: tuple[str, ...]) -> str | None:
    if len(writes) > len(gives):
        return f"writes lists {field_count(len(writes))}"
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
            f"{type_name(component_type)} takes or gives {type_name(model)}, "
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
        return ", ".join(f"{a}: {type_name(t)}" for a, t in attributes.items())

    name = getattr(component_type, "component_type_name", component_type.__name__)
    return f"{name} takes ({listed(takes)}) and gives ({listed(gives)})"
