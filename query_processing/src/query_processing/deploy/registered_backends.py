"""The backends each component type has, as the RAG system registers them.

A component type of the toolkit's four has its backends' openers in
``composition/openers/``, one module each, named ``<type>_on_<backend>.py``
(the README's rule 5). The deploy tool may not import the composition root
(rule 1), and importing the openers would load the adapters, so it reads those
backends from the modules' names, without importing them. A test checks that
the names give exactly the backends the openers register.

A registered component type, one that registers in the domain's
``COMPONENT_TYPES`` with no opener (an extension's, which the workflow file
loads before its components are read), has the backends its implementations
name on their class line, and ``served``, which the toolkit builds when the
type names none (ADR 0008).

Example:
    >>> registered_backends("llm")
    ('api', 'fake', 'served')
    >>> registered_backends("shouter")
    ('fake', 'local', 'served')
"""

from __future__ import annotations

from pathlib import Path

from ..domain.base_component import COMPONENT_TYPES

# The package of the openers, beside this one in query_processing.
OPENERS = Path(__file__).resolve().parents[1] / "composition" / "openers"
_OPENER_INFIX = "_on_"  # <type>_on_<backend>.py
# The backend every registered component type has: the class that calls a
# model server of ours, built by composition/registered_components.py.
SERVED_BACKEND = "served"


def registered_backends(
    component_type: str, openers: Path = OPENERS
) -> tuple[str, ...]:
    """Return the backends a component type has.

    Args:
        component_type: The type, such as ``vector_store``.
        openers: The folder of the opener modules.

    Returns:
        The backends, sorted: for a registered component type, those its
        implementations name and ``served``; else one per opener module of
        the type in ``openers``, none for a type with no opener.

    Example:
        >>> registered_backends("vector_store")
        ('chroma', 'elasticsearch', 'fake')
    """
    # Registered or not is judged against the real openers, never ``openers``:
    # the toolkit's four types register in COMPONENT_TYPES once their ports
    # are imported, and a folder without their openers must not make them so.
    if is_registered_type(component_type):
        port = COMPONENT_TYPES.get(component_type)
        return tuple(sorted({*port.backends, SERVED_BACKEND}))
    return tuple(sorted(_opener_backends(component_type, openers)))


def is_registered_type(component_type: str) -> bool:
    """Say whether a component type registers in ``COMPONENT_TYPES`` with no opener.

    The toolkit's own four types register there too once their ports are
    imported, so a type with an opener module in ``OPENERS`` is not one.

    Args:
        component_type: The type, such as ``shouter``.

    Returns:
        Whether it is a registered component type: one the FastAPI model
        server serves on its call route.

    Example:
        >>> is_registered_type("shouter"), is_registered_type("reranker")
        (True, False)
    """
    if _opener_backends(component_type, OPENERS):
        return False
    return component_type in COMPONENT_TYPES.names()


def _opener_backends(component_type: str, openers: Path) -> set[str]:
    prefix = f"{component_type}{_OPENER_INFIX}"
    return {path.stem.removeprefix(prefix) for path in openers.glob(f"{prefix}*.py")}
