"""A prompt rendered into chat messages: the one module that uses Jinja2 (spec 4).

Each message's ``content`` is a Jinja2 template, rendered on its own in Jinja2's
sandbox, so a template reads the values it is given and calls nothing else; the
structure of the prompt is never rendered. An undefined variable is an error,
not an empty text. Blocks drop the newline after them and the spaces before
them (``trim_blocks``, ``lstrip_blocks``), so a loop can sit on lines of its
own; a content's last newline is kept, as it is written. A placeholder entry
is replaced by the messages of the field it names, each with only its role and
its content.

Example:
    >>> render_prompt([PromptMessage(role="user", content="Q: {{ q }}")], {"q": "a?"})
    [{'role': 'user', 'content': 'Q: a?'}]
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any, cast

from jinja2 import StrictUndefined, Template, TemplateError, meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

from ..domain.prompts import Prompt, PromptMessage, PromptPlaceholder

_ENVIRONMENT = ImmutableSandboxedEnvironment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    autoescape=False,  # chat text, not HTML
)


def prompt_variables(prompt: Prompt) -> frozenset[str]:
    """Return the variables a prompt uses: its templates' and its placeholders'.

    Args:
        prompt: The prompt's messages and placeholders.

    Returns:
        The name of every variable a content reads and every field a
        placeholder splices.

    Raises:
        ValueError: A content is no template Jinja2 can read; the message names
            the message, by its place and role, and Jinja2's reason.

    Example:
        >>> prompt_variables([PromptMessage(role="user", content="{{ q }}")])
        frozenset({'q'})
    """
    variables: set[str] = set()
    for place, entry in enumerate(prompt, start=1):
        if isinstance(entry, PromptPlaceholder):
            variables.add(entry.placeholder)
            continue
        try:
            parsed = _ENVIRONMENT.parse(entry.content)
            # Compiled too: an unknown filter or test passes parsing and fails
            # only here, which would otherwise be on the first request.
            _template(entry.content)
        except TemplateError as exc:
            raise ValueError(_refusal(place, entry, "cannot be read", exc)) from exc
        variables |= meta.find_undeclared_variables(parsed)
    return frozenset(variables)


def render_prompt(
    prompt: Prompt, variables: Mapping[str, object]
) -> list[dict[str, str]]:
    """Render a prompt into OpenAI-style messages.

    Args:
        prompt: The prompt's messages and placeholders.
        variables: The values its templates read and its placeholders splice,
            by name.

    Returns:
        One message per message of the prompt, and the messages of each
        placeholder's field in its place, each with ``role`` and ``content``.

    Raises:
        ValueError: A template cannot be read, reads an undefined variable or
            does what the sandbox refuses; the message names the message.

    Example:
        >>> render_prompt([PromptPlaceholder(placeholder="history")],
        ...               {"history": [{"role": "user", "content": "hi"}]})
        [{'role': 'user', 'content': 'hi'}]
    """
    messages: list[dict[str, str]] = []
    for place, entry in enumerate(prompt, start=1):
        if isinstance(entry, PromptPlaceholder):
            spliced = cast("Iterable[Any]", variables[entry.placeholder])
            messages.extend(_role_and_content(entry, m) for m in spliced)
            continue
        messages.append(
            {"role": entry.role, "content": _rendered(place, entry, variables)}
        )
    return messages


def _rendered(
    place: int, message: PromptMessage, variables: Mapping[str, object]
) -> str:
    try:
        return _template(message.content).render(variables)
    except TemplateError as exc:
        raise ValueError(_refusal(place, message, "cannot be rendered", exc)) from exc


@cache
def _template(content: str) -> Template:
    # Compiled once per text: a prompt is rendered on every request.
    return _ENVIRONMENT.from_string(content)


def _role_and_content(entry: PromptPlaceholder, message: Any) -> dict[str, str]:
    # A message of a field: a mapping as the request sent it, or an object.
    try:
        if isinstance(message, Mapping):
            return {"role": str(message["role"]), "content": str(message["content"])}
        return {"role": str(message.role), "content": str(message.content)}
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"the placeholder {entry.placeholder!r} splices {message!r} (expected a "
            "message with a role and a content)"
        ) from exc


def _refusal(place: int, message: PromptMessage, what: str, exc: TemplateError) -> str:
    return (
        f"message {place} ({message.role}) of the prompt {what}: {exc} "
        f"(its content: {message.content!r}; expected a Jinja2 template whose "
        "variables are fields the stage reads)"
    )
