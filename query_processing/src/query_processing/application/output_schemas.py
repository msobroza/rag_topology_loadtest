"""``OUTPUT_SCHEMAS``: the pydantic models a ``generate`` stage can ask the LLM for.

A stage names one by ``output_schema``; the LLM's answer is then one instance
of it, and the stage writes each of its fields to the field of ``writes`` of
the same name (spec 5). A use case registers its own schema with the
decorator, in a module of its extension package. The toolkit registers one,
``standalone_query_with_filter``. ``narrowed_to`` limits a schema's conditions
to the metadata fields a vector store declares (spec 5 and 7).

Example:
    >>> @OUTPUT_SCHEMAS.register("verdict")
    ... class Verdict(BaseModel):
    ...     model_config = ConfigDict(extra="forbid")
    ...     ok: bool
    >>> OUTPUT_SCHEMAS.get("verdict")
    <class 'Verdict'>
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    conlist,
    create_model,
)
from pydantic.fields import FieldInfo

from ..domain.metadata_filter import (
    COMPARATORS_BY_TYPE,
    LIST_COMPARATORS,
    Comparator,
    MetadataCondition,
    MetadataFieldInfo,
    MetadataFieldType,
)
from ..domain.registry import Registry

OUTPUT_SCHEMAS: Registry[type[BaseModel]] = Registry("output schema")

# Every model of a narrowed schema: an endpoint's strict mode wants each key
# required and no other key (spec 5), so no field has a default.
_STRICT = ConfigDict(extra="forbid", frozen=True)
# The type of a value of each type of metadata field, never coerced from
# another ("2020" is no integer), as check_conditions refuses it; a strict
# float still takes an integer, as check_conditions does.
_VALUE_TYPES: dict[MetadataFieldType, Any] = {
    "string": StrictStr,
    "integer": StrictInt,
    "float": StrictFloat,
    "boolean": StrictBool,
}
_NOT_A_NAME = re.compile(r"\W")
# The attributes of a field that make it required: pydantic reads a default
# of ... as none.
_NO_DEFAULT: dict[str, Any] = {"default": ..., "default_factory": None}


@OUTPUT_SCHEMAS.register("standalone_query_with_filter")
class StandaloneQueryWithFilter(BaseModel):
    """The user's message made self-contained, and the conditions it names.

    A rewrite stage writes both in one LLM call: ``standalone_query`` is what
    is searched, and ``metadata_filter`` is added to the client's conditions.

    Example:
        >>> StandaloneQueryWithFilter(standalone_query="AMF on the deductible?",
        ...     metadata_filter=[MetadataCondition(attribute="source",
        ...                                        comparator="in", value=["AMF"])])
    """

    model_config = ConfigDict(extra="forbid")

    standalone_query: str
    metadata_filter: list[MetadataCondition]


def narrowed_to(
    schema: type[BaseModel], fields: Sequence[MetadataFieldInfo]
) -> type[BaseModel]:
    """Return ``schema`` with its conditions limited to ``fields``.

    Each field of ``schema`` that is a ``list[MetadataCondition]`` holds, in
    the schema returned, conditions of one declared field each: its name, a
    comparator its type has and a value of its type, one of its
    ``allowed_values`` when it declares them, in a list for ``in`` and
    ``nin``. Every model forbids extra keys and has no default, so an
    endpoint's strict mode accepts the schema. The other fields keep their
    types, and every field keeps its alias, constraints and description.

    Args:
        schema: An output schema, such as ``StandaloneQueryWithFilter``.
        fields: The metadata fields a vector store declares.

    Returns:
        The narrowed schema, under the name of ``schema``.

    Raises:
        ValueError: ``fields`` is empty: a condition could name nothing.

    Example:
        >>> source = MetadataFieldInfo(name="source", type="string",
        ...                            description="d", allowed_values=["AMF"])
        >>> narrowed_to(StandaloneQueryWithFilter, [source]).model_validate(
        ...     {"standalone_query": "q", "metadata_filter": [
        ...         {"attribute": "source", "comparator": "eq", "value": "SAAQ"}]})
        Traceback (most recent call last):
        pydantic_core._pydantic_core.ValidationError: ...
    """
    if not fields:
        raise ValueError(
            f"cannot narrow the output schema {schema.__name__} to no metadata "
            "field (expected at least one declared field, under the vector "
            "store's metadata_field_info)"
        )
    condition = _condition_union(fields)
    narrowed: dict[str, Any] = {
        name: (_narrowed_type(info.annotation, condition), _required_copy(info))
        for name, info in schema.model_fields.items()
    }
    return create_model(schema.__name__, __config__=_STRICT, **narrowed)


def condition_fields(schema: type[BaseModel]) -> list[str]:
    """Return the fields of ``schema`` that hold conditions of a metadata filter.

    Args:
        schema: An output schema.

    Returns:
        The names of its fields typed ``list[MetadataCondition]``, in order.

    Example:
        >>> condition_fields(StandaloneQueryWithFilter)
        ['metadata_filter']
    """
    return [
        name
        for name, info in schema.model_fields.items()
        if _holds_conditions(info.annotation)
    ]


def _holds_conditions(annotation: Any) -> bool:
    is_list = get_origin(annotation) is list
    return is_list and get_args(annotation) == (MetadataCondition,)


def _narrowed_type(annotation: Any, condition: Any) -> Any:
    # A list of conditions holds the narrowed ones; another type is kept.
    return list[condition] if _holds_conditions(annotation) else annotation


def _required_copy(info: FieldInfo) -> FieldInfo:
    # The field as the schema states it (alias, constraints, description), but
    # required: the type create_model is given replaces its annotation.
    kept = copy.copy(info)
    # By setattr: only pydantic reads these, and vulture would call them unused.
    for attribute, value in _NO_DEFAULT.items():
        setattr(kept, attribute, value)
    return kept


def _condition_union(fields: Sequence[MetadataFieldInfo]) -> Any:
    # One model per declared field and shape of value: one value, or a list.
    models = [model for field in fields for model in _condition_models(field)]
    return Union[tuple(models)]  # noqa: UP007 - a union built from a tuple


def _condition_models(field: MetadataFieldInfo) -> list[type[BaseModel]]:
    comparators = COMPARATORS_BY_TYPE[field.type]
    one = sorted(comparators - LIST_COMPARATORS)
    many = sorted(comparators & LIST_COMPARATORS)
    value = _value_type(field)
    # An empty list is refused, as check_conditions refuses it: a store may
    # fail on it mid-search.
    values = conlist(value, min_length=1)
    shapes = [(one, value, ""), (many, values, "List")]
    return [
        _condition_model(field.name, kept, value_type, suffix)
        for kept, value_type, suffix in shapes
        if kept
    ]


def _condition_model(
    attribute: str, comparators: list[Comparator], value_type: Any, suffix: str
) -> type[BaseModel]:
    title = _NOT_A_NAME.sub("_", attribute).title().replace("_", "")
    return create_model(
        f"{title}{suffix}Condition",
        __config__=_STRICT,
        attribute=(_literal_of([attribute]), ...),
        comparator=(_literal_of(comparators), ...),
        value=(value_type, ...),
    )


def _value_type(field: MetadataFieldInfo) -> Any:
    # A literal of the allowed values when the field declares them.
    if field.allowed_values is None:
        return _VALUE_TYPES[field.type]
    return _literal_of(field.allowed_values)


def _literal_of(values: Sequence[object]) -> Any:
    # A Literal of values known only at run time: a type to pydantic, which a
    # type checker cannot read in a type expression.
    return Literal.__getitem__(tuple(values))
