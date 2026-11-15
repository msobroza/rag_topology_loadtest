"""The backends each component type has, as the RAG system registers them.

A component's backend picks the opener registered for its type and backend in
``composition/openers/``, one module each, named ``<type>_on_<backend>.py``
(the README's rule 5). The deploy tool may not import the composition root
(rule 1), and importing the openers would load the adapters, so it reads the
backends from those modules' names, without importing them. A test checks
that the names give exactly the backends the openers register.

Example:
    >>> registered_backends("llm")
    ('api', 'fake', 'served')
"""

from __future__ import annotations

from pathlib import Path

# The package of the openers, beside this one in query_processing.
OPENERS = Path(__file__).resolve().parents[1] / "composition" / "openers"
_OPENER_INFIX = "_on_"  # <type>_on_<backend>.py


def registered_backends(
    component_type: str, openers: Path = OPENERS
) -> tuple[str, ...]:
    """Return the backends a component type has: one per opener module of its type.

    Args:
        component_type: The type, such as ``vector_store``.
        openers: The folder of the opener modules.

    Returns:
        The backends, sorted, as the registry of the type's openers names them;
        none for a type with no opener.

    Example:
        >>> registered_backends("vector_store")
        ('chroma', 'elasticsearch', 'fake')
    """
    prefix = f"{component_type}{_OPENER_INFIX}"
    return tuple(
        sorted(path.stem.removeprefix(prefix) for path in openers.glob(f"{prefix}*.py"))
    )
