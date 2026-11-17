"""A prompt: a list of messages with roles, as OpenAI writes them, and placeholders.

A use case writes its prompts in ``prompts.yaml``, beside its workflow file,
each under its name (spec 4). Each message's ``content`` is a Jinja2 template,
rendered on its own; an entry ``placeholder: <field>`` stands for the messages
of that workflow state field, spliced in its place. The structure of the list is
never rendered. These are the shapes only: ``application.prompt_rendering``
renders them.

Example:
    >>> prompt = [PromptMessage(role="system", content="Cite [n]."),
    ...           PromptPlaceholder(placeholder="conversational_history"),
    ...           PromptMessage(role="user", content="{{ conversational_query }}")]
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PromptMessage(BaseModel):
    """One message of a prompt: its role, and its content, a Jinja2 template.

    Example:
        >>> PromptMessage(role="user", content="{{ conversational_query }}").role
        'user'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class PromptPlaceholder(BaseModel):
    """The place of a field's messages in a prompt, by the field's name.

    Example:
        >>> PromptPlaceholder(placeholder="conversational_history").placeholder
        'conversational_history'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    placeholder: str


Prompt = list[PromptMessage | PromptPlaceholder]
