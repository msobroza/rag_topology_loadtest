"""``BaseComponent``: the one abstract class every component type subclasses.

A component type is an abstract class that names its request and its answer, two
domain classes, and registers under its type name on its class line. Every
component is then called the same way: ``ainvoke(request)``, which by default
runs the blocking ``invoke(request)`` off the event loop. The names and that
default are LangChain's; nothing here subclasses LangChain's ``Runnable``.

Example:
    >>> class BaseAnonymizer(BaseComponent[AnonymizeRequest, AnonymizedText],
    ...                      component_type="anonymizer"):
    ...     '''Masks personal data in a text.'''
    >>> COMPONENT_TYPES.get("anonymizer")
    <class 'BaseAnonymizer'>
"""

from __future__ import annotations

import asyncio
from abc import ABC
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

from .registry import Registry

RequestT = TypeVar("RequestT")
AnswerT = TypeVar("AnswerT")


class BaseComponent(ABC, Generic[RequestT, AnswerT]):
    """A component: a request in, an answer out.

    A class defined with ``component_type="name"`` is a component type: it
    registers under that name, in ``registry`` when one is given and in
    ``COMPONENT_TYPES`` otherwise, and reads its request and answer classes from
    its ``BaseComponent[Request, Answer]`` base.

    Example:
        >>> class BaseEcho(BaseComponent[EchoRequest, Echo], component_type="echo"):
        ...     '''Answers a text with what it found in it.'''
        >>> BaseEcho.request_type, BaseEcho.answer_type
        (<class 'EchoRequest'>, <class 'Echo'>)
    """

    request_type: ClassVar[type[Any]]
    answer_type: ClassVar[type[Any]]
    component_type_name: ClassVar[str]

    def __init_subclass__(
        cls,
        component_type: str | None = None,
        registry: Registry[type[BaseComponent[Any, Any]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if component_type is None:
            return
        cls.request_type, cls.answer_type = _request_and_answer(cls)
        cls.component_type_name = component_type
        (COMPONENT_TYPES if registry is None else registry).register(component_type)(
            cls
        )

    def invoke(self, request: RequestT) -> AnswerT:
        """Answer a request, blocking: what an in-process implementation writes.

        Args:
            request: The component type's request.

        Returns:
            The component type's answer.

        Raises:
            NotImplementedError: The implementation wrote neither ``invoke`` nor
                ``ainvoke``; the message says what to write.

        Example:
            >>> Upper().invoke(EchoRequest(text="a"))
            Echo(text='A', length=1)
        """
        del request  # the signature an implementation overrides; unread here
        answer = getattr(type(self), "answer_type", Any)
        # The four ports answer through ainvoke alone, so the message says what
        # is missing without claiming the class is incomplete.
        raise NotImplementedError(
            f"{type(self).__name__} writes no invoke: call ainvoke, or write "
            f"invoke(self, request) -> {getattr(answer, '__name__', answer)} "
            "for a blocking implementation"
        )

    async def ainvoke(self, request: RequestT) -> AnswerT:
        """Answer a request: the one method the toolkit calls.

        By default it runs ``invoke`` in a worker thread, so a blocking
        implementation never blocks the event loop. An implementation that
        awaits, such as one that calls a service, writes this method instead.

        Args:
            request: The component type's request.

        Returns:
            The component type's answer.

        Example:
            >>> await Upper().ainvoke(EchoRequest(text="a"))
            Echo(text='A', length=1)
        """
        return await asyncio.to_thread(self.invoke, request)


def _request_and_answer(cls: type[Any]) -> tuple[type[Any], type[Any]]:
    # The two type arguments of the class's BaseComponent[Request, Answer] base.
    for base in cls.__dict__.get("__orig_bases__", ()):
        origin = get_origin(base)
        if isinstance(origin, type) and issubclass(origin, BaseComponent):
            request, answer = get_args(base)
            return request, answer
    raise TypeError(
        f"component type {cls.__name__} names no request and answer "
        f"(expected class {cls.__name__}(BaseComponent[Request, Answer], ...))"
    )


COMPONENT_TYPES: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")
"""Every component type, by its name: what the ports register on their class line."""
