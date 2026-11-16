"""A metadata filter: a flat list of conditions on a passage's metadata, all to hold.

A condition names an ``attribute`` of the metadata, a ``comparator`` and a
``value``. A vector store declares the fields a filter can name, each with its
type and, optionally, the values it allows (``MetadataFieldInfo``);
``check_conditions`` refuses a condition that does not fit them. Each store
turns the conditions into its own filter, through its translator in
``adapters/vector_store``: no library's filter type is used here.

Example:
    >>> topic = MetadataFieldInfo(name="topic", type="string", description="d")
    >>> check_conditions(
    ...     [MetadataCondition(attribute="topic", comparator="eq", value="pay")],
    ...     [topic])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

# A value of a passage's metadata, and of a condition on it.
MetadataValue = str | int | float | bool
Comparator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]
MetadataFieldType = Literal["string", "integer", "float", "boolean"]

# The comparators that take a list of values; every other one takes one value.
LIST_COMPARATORS: frozenset[Comparator] = frozenset({"in", "nin"})
_EQUALITY: frozenset[Comparator] = frozenset({"eq", "ne", "in", "nin"})
_ORDER: frozenset[Comparator] = frozenset({"gt", "gte", "lt", "lte"})
# The comparators each type of field has: an order only for a number.
COMPARATORS_BY_TYPE: dict[MetadataFieldType, frozenset[Comparator]] = {
    "string": _EQUALITY,
    "boolean": frozenset({"eq", "ne"}),
    "integer": _EQUALITY | _ORDER,
    "float": _EQUALITY | _ORDER,
}
# The Python types a value of each type of field may have: an integer is a float.
_PYTHON_TYPES: dict[MetadataFieldType, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "float": (int, float),
    "boolean": (bool,),
}


class MetadataCondition(BaseModel):
    """One condition of a metadata filter: ``attribute`` compared to ``value``.

    ``in`` and ``nin`` take a list of values, the other comparators one value;
    ``check_conditions`` refuses one that does not, so the condition is built
    as sent and refused with its field's declaration in hand.

    Example:
        >>> MetadataCondition(attribute="source", comparator="in", value=["AMF"])
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    attribute: str
    comparator: Comparator
    value: MetadataValue | list[MetadataValue]


class MetadataFieldInfo(BaseModel):
    """A metadata field a filter can name: its type, what it means, what it allows.

    Example:
        >>> MetadataFieldInfo(name="topic", type="string",
        ...                   description="what the passage is about")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: MetadataFieldType
    description: str
    allowed_values: list[MetadataValue] | None = None


def check_conditions(
    conditions: Sequence[MetadataCondition], fields: Sequence[MetadataFieldInfo]
) -> None:
    """Refuse a condition that does not fit the field it names.

    Args:
        conditions: The filter's conditions.
        fields: The fields a filter can name, as a vector store declares them.

    Raises:
        ValueError: A condition names no declared field, has a comparator its
            field's type does not have, takes one value where its comparator
            takes a list (or the reverse), or has a value of another type or
            outside the allowed ones. The message names the condition.

    Example:
        >>> check_conditions(
        ...     [MetadataCondition(attribute="sorce", comparator="eq", value="x")],
        ...     [MetadataFieldInfo(name="source", type="string", description="d")])
        Traceback (most recent call last):
        ValueError: metadata_filter[0] ... there is no field 'sorce' ...
    """
    by_name = {field.name: field for field in fields}
    for index, condition in enumerate(conditions):
        problem = _problem(condition, by_name)
        if problem:
            raise ValueError(
                f"metadata_filter[{index}] {condition.model_dump()} does not fit "
                f"the fields declared: {problem}"
            )


def check_condition_shapes(conditions: Sequence[MetadataCondition], label: str) -> None:
    """Refuse a condition whose value does not have its comparator's shape.

    For conditions on fields no store declares, such as a vector store's
    prefilter: only the shape is checked, not a field's type nor its values.

    Args:
        conditions: The conditions.
        label: What holds them, named in the error, such as ``"prefilter"``.

    Raises:
        ValueError: ``in`` or ``nin`` has one value or an empty list, or another
            comparator has a list. The message names the condition.

    Example:
        >>> check_condition_shapes(
        ...     [MetadataCondition(attribute="active", comparator="in", value=[])],
        ...     "prefilter")
        Traceback (most recent call last):
        ValueError: prefilter[0] ... 'in' takes a list of at least one value ...
    """
    for index, condition in enumerate(conditions):
        problem = _shape_problem(condition)
        if problem:
            raise ValueError(
                f"{label}[{index}] {condition.model_dump()} does not have the "
                f"shape of a condition: {problem}"
            )


def _problem(
    condition: MetadataCondition, by_name: dict[str, MetadataFieldInfo]
) -> str:
    field = by_name.get(condition.attribute)
    if field is None:
        declared = ", ".join(map(repr, by_name)) or "none"
        return (
            f"there is no field {condition.attribute!r} "
            f"(expected one of the fields declared: {declared})"
        )
    return _comparator_problem(condition, field) or _value_problem(condition, field)


def _comparator_problem(condition: MetadataCondition, field: MetadataFieldInfo) -> str:
    comparator = condition.comparator
    allowed = COMPARATORS_BY_TYPE[field.type]
    if comparator not in allowed:
        return (
            f"{comparator!r} is not a comparator of a {field.type} "
            f"(expected one of: {', '.join(sorted(allowed))})"
        )
    return _shape_problem(condition)


def _shape_problem(condition: MetadataCondition) -> str:
    comparator, value = condition.comparator, condition.value
    if comparator not in LIST_COMPARATORS:
        if isinstance(value, list):
            return f"{comparator!r} takes one value (got the list {value!r})"
        return ""
    if not isinstance(value, list):
        return f"{comparator!r} takes a list of values (got {value!r})"
    # An empty list keeps nothing (in) or everything (nin), never what was meant,
    # and a store may refuse it mid-search (Chroma raises a ValueError): it is
    # refused here, where the condition is read.
    if not value:
        return f"{comparator!r} takes a list of at least one value (got [])"
    return ""


def _value_problem(condition: MetadataCondition, field: MetadataFieldInfo) -> str:
    values = condition.value if isinstance(condition.value, list) else [condition.value]
    for value in values:
        if not _is_of_type(value, field.type):
            return f"{value!r} is not of type {field.type}"
        if field.allowed_values is not None and value not in field.allowed_values:
            return f"{value!r} is not one of {field.allowed_values!r}"
    return ""


def _is_of_type(value: MetadataValue, field_type: MetadataFieldType) -> bool:
    # bool is a subclass of int in Python: a boolean is never a number here.
    if isinstance(value, bool):
        return field_type == "boolean"
    return isinstance(value, _PYTHON_TYPES[field_type])
