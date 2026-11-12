"""The Run label: the name the workflow app reports for the measured configuration.

It defaults to the topology plus every choice that differs from its default, and
is sent as a header, so it must be header-safe.

Example:
    >>> default_run_label("monolith", embedding="local", rerank="none",
    ...                   store="elasticsearch", generation="api")
    'monolith+rerank=none'
"""

from __future__ import annotations

import re

from .backends import DEFAULT_GENERATION_BACKEND, DEFAULT_VECTOR_STORE_BACKEND
from .topology import TOPOLOGY_BACKENDS, Topology
from .workflow_parameters import DEFAULT_RETRIEVAL_MODE

# Sent as the X-RAG-Run-Label header, so only header-safe characters.
RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9._+=:-]{1,128}")
RUN_LABEL_SHAPE = "letters, digits and ._+=:- only, 1 to 128 characters"


def default_run_label(
    topology: Topology,
    *,
    embedding: str,
    rerank: str,
    store: str,
    generation: str,
    retrieval: str = DEFAULT_RETRIEVAL_MODE,
) -> str:
    """Name a configuration after its topology and the choices off their default.

    Choices are listed in a fixed order (embedding, rerank, store, generation,
    retrieval) so the same configuration always gets the same label.

    Args:
        topology: The topology; it sets the default embedding and rerank backends.
        embedding: The embedding backend in use.
        rerank: The rerank backend in use.
        store: The vector store backend in use.
        generation: The generation backend in use.
        retrieval: The retrieval mode in use.

    Returns:
        The topology plus ``+<choice>=<value>`` for each choice off its default.

    Example:
        >>> default_run_label("split-all", embedding="api", rerank="api",
        ...                   store="elasticsearch", generation="fake")
        'split-all+generation=fake'
    """
    implied = TOPOLOGY_BACKENDS[topology]
    choices = (
        ("embedding", embedding, implied.embedding),
        ("rerank", rerank, implied.rerank),
        ("store", store, DEFAULT_VECTOR_STORE_BACKEND),
        ("generation", generation, DEFAULT_GENERATION_BACKEND),
        ("retrieval", retrieval, DEFAULT_RETRIEVAL_MODE),
    )
    variants = [
        f"+{name}={actual}" for name, actual, default in choices if actual != default
    ]
    return topology + "".join(variants)


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
        f"RAG_RUN_LABEL={label!r} is not a run label: use {RUN_LABEL_SHAPE}"
    )
