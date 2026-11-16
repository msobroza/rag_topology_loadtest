"""The toolkit's own component types, and the packages under ``extensions``.

A use case holds the implementations that are its own (ADR 0006): a component
type, an implementation of one, a stage type or an edge condition, each
registering itself when its module is imported. The workflow file names the
use case's packages, and every command that reads the file imports every
module of each before it reads the components, so that their classes are
registered when a component names them. The toolkit's own component types
register the same way, on their ports' class lines, and are loaded first, the
same way, whether the file names an extension or not.

Example:
    >>> load_extensions(["my_use_case"])
"""

from __future__ import annotations

from ..domain.registry import import_package_tree

# The toolkit's modules that register a component type, or an implementation
# of one, on their class line: the five ports, and the router's package (its
# fake backend and its API format). Named rather than imported, since config
# may not import adapters (LOADED_BY_NAME in tests/structure/test_layering.py).
TOOLKIT_COMPONENT_TYPES = (
    "query_processing.adapters.embedder.base_embedder",
    "query_processing.adapters.reranker.base_reranker",
    "query_processing.adapters.llm.base_llm",
    "query_processing.adapters.vector_store.base_vector_store",
    "query_processing.adapters.router",
)


def load_extensions(packages: object) -> None:
    """Import the toolkit's component types, then every module of each package named.

    The toolkit's own types are imported whatever ``packages`` is, so that a
    workflow file naming no extension can declare a router. A value that is
    no list of names is left for the settings to refuse.

    Args:
        packages: The ``extensions`` of a workflow file, as read.

    Raises:
        ValueError: If a package cannot be imported, its own error included,
            such as a backend that two of its classes name; the message names
            the package and that error.

    Example:
        >>> load_extensions(["no_such_package"])
        ValueError: extension 'no_such_package' cannot be imported: ...
    """
    for module in TOOLKIT_COMPONENT_TYPES:
        import_package_tree(module)
    if not isinstance(packages, list):
        return
    for package in packages:
        _load_extension(str(package))


def _load_extension(package: str) -> None:
    try:
        import_package_tree(package)
    except Exception as exc:  # any error of the package's code stops the load
        raise ValueError(
            f"extension {package!r} cannot be imported: "
            f"{type(exc).__name__}: {exc} (expected an installed package whose "
            "modules all import, named under extensions in the workflow file)"
        ) from exc
