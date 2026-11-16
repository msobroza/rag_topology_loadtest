"""``BaseComponent``: the one abstract class every component type subclasses.

A component type is an abstract class that names its request and its answer, two
domain classes, and registers under its type name on its class line. An
implementation is one subclass of it that names its backend, and its settings
class when it has settings, on its class line too; it builds itself with
``from_settings`` from the block of the workflow file named after its backend.
Every component is then called the same way: ``ainvoke(request)``, which by
default runs the blocking ``invoke(request)`` off the event loop. The names and
that default are LangChain's; nothing here subclasses LangChain's ``Runnable``.

Example:
    >>> class BaseAnonymizer(BaseComponent[AnonymizeRequest, AnonymizedText],
    ...                      component_type="anonymizer"):
    ...     '''Masks personal data in a text.'''
    >>> class RegexAnonymizer(BaseAnonymizer, backend="local",
    ...                       settings=RegexAnonymizerSettings): ...
    >>> COMPONENT_TYPES.get("anonymizer").backends
    {'local': <class 'RegexAnonymizer'>}
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC
from typing import Any, ClassVar, Generic, Self, TypeVar, get_args, get_origin

from pydantic import BaseModel

from .component_resources import ComponentResources
from .registry import Registry

RequestT = TypeVar("RequestT")
AnswerT = TypeVar("AnswerT")


class BaseComponent(ABC, Generic[RequestT, AnswerT]):
    """A component: a request in, an answer out.

    A class defined with ``component_type="name"`` is a component type: it
    registers under that name, in ``registry`` when one is given and in
    ``COMPONENT_TYPES`` otherwise, reads its request and answer classes from its
    ``BaseComponent[Request, Answer]`` base, and takes ``settings`` as the class
    of its own keys. A subclass of a component type defined with
    ``backend="name"`` is an implementation: it registers in its type's
    ``backends`` and takes ``settings`` as the class of its backend's block. A
    subclass that names neither is a plain subclass, registered nowhere.

    Example:
        >>> class BaseEcho(BaseComponent[EchoRequest, Echo], component_type="echo"):
        ...     '''Answers a text with what it found in it.'''
        >>> BaseEcho.request_type, BaseEcho.answer_type
        (<class 'EchoRequest'>, <class 'Echo'>)
    """

    request_type: ClassVar[type[Any]]
    answer_type: ClassVar[type[Any]]
    component_type_name: ClassVar[str]
    # A component type's own keys, read with this class; None when it has none.
    type_settings: ClassVar[type[BaseModel] | None] = None
    # A component type's implementations, by the backend each names.
    backends: ClassVar[dict[str, type[BaseComponent[Any, Any]]]]
    # An implementation's block, read with this class; None when it has none.
    settings_type: ClassVar[type[BaseModel] | None] = None

    def __init_subclass__(
        cls,
        component_type: str | None = None,
        registry: Registry[type[BaseComponent[Any, Any]]] | None = None,
        backend: str | None = None,
        settings: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_mixed_keywords(cls, component_type, backend, settings)
        if component_type is not None:
            _define_type(cls, component_type, settings)
            (COMPONENT_TYPES if registry is None else registry).register(
                component_type
            )(cls)
        elif backend is not None:
            _define_implementation(cls, backend, settings)

    @classmethod
    def backend_class(cls, backend: str) -> type[BaseComponent[Any, Any]]:
        """Return the implementation of this component type that names ``backend``.

        Args:
            backend: A backend's name, as a component's settings state it.

        Returns:
            The class registered under it.

        Raises:
            ValueError: If no implementation names ``backend``; the message
                lists those the component type has.

        Example:
            >>> BaseAnonymizer.backend_class("local")
            <class 'RegexAnonymizer'>
        """
        implementation = cls.backends.get(backend)
        if implementation is None:
            known = ", ".join(repr(name) for name in sorted(cls.backends)) or "none"
            raise ValueError(
                f"unknown {cls.component_type_name} backend {backend!r} "
                f"(expected one of: {known})"
            )
        return implementation

    @classmethod
    def from_settings(
        cls, settings: BaseModel | None, resources: ComponentResources
    ) -> Self:
        """Build the implementation from its backend's block.

        By default it takes no argument: an implementation that names a settings
        class writes this method.

        Args:
            settings: The block named after the backend, read with the class the
                implementation names; None when it names none.
            resources: What else it may use, such as another component's settings.

        Returns:
            The implementation, ready to answer.

        Example:
            >>> FakeAnonymizer.from_settings(None, resources)
            <FakeAnonymizer ...>
        """
        del settings, resources  # the signature an implementation overrides
        return cls()

    @classmethod
    def answer_labels(
        cls, type_settings: BaseModel | None
    ) -> dict[str, tuple[str, ...]]:
        """Return the labels an answer attribute takes, by the attribute's name.

        A plain call gives them to the field it writes that attribute to, and a
        label edge that reads the field has one row of ``path_map`` for each
        (ADR 0007). By default no attribute has labels.

        Args:
            type_settings: The keys of the component's type, as its settings
                read them; None for a type that has none.

        Returns:
            The labels of each answer attribute that takes one of a few.

        Example:
            >>> BaseRouter.answer_labels(settings.components["router"])
            {'name': ('self_contained', 'leans_on_history')}
        """
        del type_settings  # the signature a component type overrides
        return {}

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
        # The four ports answer through ainvoke alone, so the message says what
        # is missing without claiming the class is incomplete.
        raise NotImplementedError(
            f"{type(self).__name__} writes no invoke: call ainvoke, or write "
            f"invoke(self, request) -> {_answer_name(type(self))} "
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


def _refuse_mixed_keywords(
    cls: type[Any],
    component_type: str | None,
    backend: str | None,
    settings: type[BaseModel] | None,
) -> None:
    # A class is a component type or an implementation, never both; settings
    # alone would be read by nothing.
    if component_type is not None and backend is not None:
        raise TypeError(
            f"{cls.__name__} names both component_type {component_type!r} and "
            f"backend {backend!r} (expected a component type, then a subclass of "
            "it that names the backend)"
        )
    if component_type is None and backend is None and settings is not None:
        raise TypeError(
            f"{cls.__name__} names settings but no backend (expected "
            f"class {cls.__name__}(BaseX, backend='local', settings=...))"
        )


def _define_type(
    cls: type[BaseComponent[Any, Any]],
    component_type: str,
    settings: type[BaseModel] | None,
) -> None:
    cls.request_type, cls.answer_type = _request_and_answer(cls)
    _refuse_no_pydantic_model(cls, "takes", cls.request_type)
    _refuse_no_pydantic_model(cls, "answers", cls.answer_type)
    cls.component_type_name = component_type
    cls.type_settings = settings
    cls.backends = {}  # its own: the implementations of this type alone


def _refuse_no_pydantic_model(cls: type[Any], verb: str, model: Any) -> None:
    # Any, a TypeVar or a generic alias such as list[X] cannot be read as a class
    # here: the ComponentContract checks those, on an implementation.
    if model is Any or not isinstance(model, type) or issubclass(model, BaseModel):
        return
    raise TypeError(
        f"component type {cls.__name__} {verb} a {model.__name__}, which is no "
        "pydantic model: the toolkit reads the workflow state's fields from a "
        "request's and an answer's fields, and carries them as JSON to a model "
        f"server (expected class {model.__name__}(BaseModel), with its fields as "
        "annotations)"
    )


def _define_implementation(
    cls: type[BaseComponent[Any, Any]], backend: str, settings: type[BaseModel] | None
) -> None:
    # Refused when the class is defined, so that a mistake stops the load of
    # the package that holds it, naming what to write.
    if not hasattr(cls, "component_type_name"):
        raise TypeError(
            f"{cls.__name__} names backend {backend!r} but subclasses no component "
            f"type (expected class {cls.__name__}(BaseX, backend={backend!r}), "
            "BaseX a component type)"
        )
    if cls.invoke is BaseComponent.invoke and cls.ainvoke is BaseComponent.ainvoke:
        answer = _answer_name(cls)
        raise TypeError(
            f"{cls.__name__} still has to write: invoke(self, request) -> {answer}"
            f", or async ainvoke(self, request) -> {answer}"
        )
    _refuse_wrong_from_settings(cls, settings)
    existing = cls.backends.get(backend)
    if existing is not None:
        raise TypeError(
            f"{cls.component_type_name} backend {backend!r} is already "
            f"{existing.__name__}, so {cls.__name__} cannot name it too (expected "
            "one implementation for each backend)"
        )
    cls.settings_type = settings
    cls.backends[backend] = cls


def _refuse_wrong_from_settings(
    cls: type[Any], settings: type[BaseModel] | None
) -> None:
    # The composition calls cls.from_settings(block, resources): the default
    # takes no argument, so a class that names a block and keeps it would
    # answer with its constructor's defaults, the block read and then ignored.
    written = inspect.getattr_static(cls, "from_settings")
    expected = _from_settings_to_write(cls, settings)
    if written is vars(BaseComponent)["from_settings"]:
        if settings is not None:
            raise TypeError(
                f"{cls.__name__} names settings={settings.__name__} but writes no "
                "from_settings, so its block of the workflow file would be read "
                f"and never used {expected}"
            )
        return
    if not _callable_as_the_toolkit_calls(written):
        function = getattr(written, "__func__", written)
        raise TypeError(
            f"{cls.__name__}.from_settings{inspect.signature(function)} is not "
            "what the toolkit calls with the block of the workflow file and the "
            f"component's resources {expected}"
        )


def _callable_as_the_toolkit_calls(written: object) -> bool:
    if not isinstance(written, classmethod):
        return False
    try:  # cls, the backend's block and the resources, by position
        inspect.signature(written.__func__).bind(None, None, None)
    except TypeError:
        return False
    return True


def _from_settings_to_write(cls: type[Any], settings: type[BaseModel] | None) -> str:
    # What a refusal shows, in parentheses: the method as the user writes it.
    block = "BaseModel | None" if settings is None else settings.__name__
    name = cls.__name__
    return (
        f"(expected, in {name}:\n    @classmethod\n    def from_settings(cls, "
        f"settings: {block}, resources: ComponentResources) -> {name}: ...)"
    )


def _answer_name(cls: type[Any]) -> str:
    answer = getattr(cls, "answer_type", Any)
    return str(getattr(answer, "__name__", answer))


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
