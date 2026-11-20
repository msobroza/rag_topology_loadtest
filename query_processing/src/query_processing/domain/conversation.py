"""The conversational history: the turns a client sends with its request.

The workflow keeps no conversation: each request carries the turns before it,
as messages with a role and a content, in the order they were said (spec 6).
A user message can carry the conditions that applied to it, for the prompts to
read; they are checked as the request's are and never applied to retrieval by
themselves. The roles are not required to alternate: two user messages in a
row are kept as sent.

Example:
    >>> HistoryMessage(role="user", content="What about remote work?").role
    'user'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from query_processing_core.retrieval.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
    check_conditions,
)

# Who sent a message of the history: the system's instructions are the
# prompt's, never a client's.
ConversationRole = Literal["user", "assistant"]


class HistoryMessage(BaseModel):
    """One message of the conversational history: its sender, text and conditions.

    Example:
        >>> HistoryMessage(role="assistant", content="Two days a week.").content
        'Two days a week.'
    """

    model_config = ConfigDict(frozen=True)

    role: ConversationRole
    # An empty message says nothing a prompt could use: refused, as a 422.
    content: str = Field(min_length=1)
    metadata_filter: list[MetadataCondition] = Field(default_factory=list)


def check_history_conditions(
    conversational_history: Sequence[HistoryMessage],
    fields: Sequence[MetadataFieldInfo],
) -> None:
    """Refuse a history message's condition that does not fit the field it names.

    The conditions are checked as the request's are, though only the prompts
    read them: they are never applied to retrieval.

    Args:
        conversational_history: The earlier messages, in order.
        fields: The fields a filter can name, as the vector stores declare them.

    Raises:
        ValueError: A condition does not fit; the message names the history
            message and the condition, by their places.

    Example:
        >>> check_history_conditions(
        ...     [HistoryMessage(role="user", content="a", metadata_filter=[
        ...         MetadataCondition(attribute="sorce", comparator="eq", value="x")])],
        ...     [MetadataFieldInfo(name="source", type="string", description="d")])
        Traceback (most recent call last):
        ValueError: conversational_history[0].metadata_filter[0] ... 'sorce' ...
    """
    for index, message in enumerate(conversational_history):
        label = f"conversational_history[{index}].metadata_filter"
        check_conditions(message.metadata_filter, fields, label)
