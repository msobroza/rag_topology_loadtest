"""The Run label: the name the workflow app reports for the measured configuration.

With none given, it lists each component with its backend, in the workflow
file's order, the parts joined by ``+`` as in the label the deploy tool sets
(``split-all+embedder=fastapi+reranker=ovms``). It is sent as a header, so it
must be header-safe.

Example:
    >>> default_run_label({"embedder": "local", "reranker": "none"})
    'embedder=local+reranker=none'
"""

from __future__ import annotations

import re
from collections.abc import Mapping

# Sent as the X-RAG-Run-Label header, so only header-safe characters.
RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9._+=:-]{1,128}")
RUN_LABEL_SHAPE = "letters, digits and ._+=:- only, 1 to 128 characters"


def default_run_label(backends: Mapping[str, str]) -> str:
    """Name a configuration after each component and the backend it runs on.

    Args:
        backends: Each component's backend, by name, in the workflow file's order.

    Returns:
        ``<component>=<backend>`` for each component, joined by ``+``.

    Raises:
        ValueError: If that label is not header-safe (a component name outside
            letters, digits and ``._+=:-``, or more than 128 characters); the
            message names the label and says to set ``QP_RUN_LABEL``.

    Example:
        >>> default_run_label({"embedder": "fake", "llm": "api"})
        'embedder=fake+llm=api'
    """
    label = "+".join(f"{name}={backend}" for name, backend in backends.items())
    if RUN_LABEL_PATTERN.fullmatch(label):
        return label
    raise ValueError(
        f"the default run label {label!r}, each component with its backend, is "
        f"not a run label ({RUN_LABEL_SHAPE}): set QP_RUN_LABEL to name this "
        "configuration"
    )


def validate_run_label(label: str) -> str:
    """Check that a run label is safe to send as a header.

    Args:
        label: The run label, sent as the ``X-RAG-Run-Label`` header.

    Returns:
        ``label`` unchanged.

    Raises:
        ValueError: If ``label`` is not header-safe (letters, digits and
            ``._+=:-`` only, 1 to 128 characters).

    Example:
        >>> validate_run_label("split-reranker-fastapi")
        'split-reranker-fastapi'
    """
    if RUN_LABEL_PATTERN.fullmatch(label):
        return label
    raise ValueError(
        f"QP_RUN_LABEL={label!r} is not a run label: use {RUN_LABEL_SHAPE}"
    )
