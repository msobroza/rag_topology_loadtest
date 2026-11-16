"""The packages a workflow file names under ``extensions``, imported first.

A use case holds the implementations that are its own (ADR 0006): a component
type, an implementation of one, a stage type or an edge condition, each
registering itself when its module is imported. The workflow file names the
use case's packages, and every command that reads the file imports every
module of each before it reads the components, so that their classes are
registered when a component names them.

Example:
    >>> load_extensions(["my_use_case"])
"""

from __future__ import annotations

from ..domain.registry import import_package_tree


def load_extensions(packages: object) -> None:
    """Import every module of each package named, so that its classes register.

    A value that is no list of names is left for the settings to refuse.

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
    if not isinstance(packages, list):
        return
    for package in packages:
        try:
            import_package_tree(str(package))
        except Exception as exc:  # any error of the package's code stops the load
            raise ValueError(
                f"extension {package!r} cannot be imported: "
                f"{type(exc).__name__}: {exc} (expected an installed package whose "
                "modules all import, named under extensions in the workflow file)"
            ) from exc
