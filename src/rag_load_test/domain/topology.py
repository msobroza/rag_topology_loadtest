"""The topology rule: which embedding and rerank backends each topology implies.

A topology is a closed set; it says where the in-process models run, so it
decides the embedding and rerank backends. An explicit backend may only repeat
that choice or pick a variant that loads no model of ours into this process:
``api`` (an endpoint called over HTTP), ``fake``, or ``none`` for rerank. The
run label names the measured configuration and defaults to the topology plus
every choice that differs from its default. Pure functions only: settings call
them at construction; the Domino rule that ``rag-serve --domino`` applies lives
in ``config/domino_rules.py``.

Example:
    >>> resolve_rerank_backend("split-reranker", None)
    'api'
    >>> default_run_label("monolith", embedding="local", rerank="none",
    ...                   store="elasticsearch", generation="api")
    'monolith+rerank=none'
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

from .workflow_parameters import DEFAULT_RETRIEVAL_MODE

Topology = Literal["monolith", "split-reranker", "split-all"]
EmbeddingBackend = Literal["local", "api", "fake"]
RerankBackend = Literal["local", "api", "fake", "none"]
GenerationBackend = Literal["api", "fake"]
VectorStoreBackend = Literal["elasticsearch", "chroma", "fake"]
# The stores that rank text themselves; open-source Chroma does not (Cloud only).
LEXICAL_STORE_BACKENDS: tuple[VectorStoreBackend, ...] = ("elasticsearch", "fake")
# What runs an in-process model: OpenVINO on its exported files, or PyTorch.
LocalModelRuntime = Literal["openvino", "torch"]

# Allowed in every topology: they load no model of ours into this process.
EMBEDDING_VARIANTS: tuple[EmbeddingBackend, ...] = ("api", "fake")
RERANK_VARIANTS: tuple[RerankBackend, ...] = ("api", "fake", "none")

DEFAULT_VECTOR_STORE_BACKEND: VectorStoreBackend = "elasticsearch"
DEFAULT_GENERATION_BACKEND: GenerationBackend = "api"
# Sent as the X-RAG-Run-Label header, so only header-safe characters.
RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9._+=:-]{1,128}")
RUN_LABEL_SHAPE = "letters, digits and ._+=:- only, 1 to 128 characters"


class TopologyBackends(NamedTuple):
    """The embedding and rerank backends one topology implies."""

    embedding: EmbeddingBackend
    rerank: RerankBackend


TOPOLOGY_BACKENDS: dict[Topology, TopologyBackends] = {
    "monolith": TopologyBackends(embedding="local", rerank="local"),
    "split-reranker": TopologyBackends(embedding="local", rerank="api"),
    "split-all": TopologyBackends(embedding="api", rerank="api"),
}


def resolve_embedding_backend(
    topology: Topology, explicit: EmbeddingBackend | None
) -> EmbeddingBackend:
    """Return the embedding backend for ``topology``.

    Args:
        topology: The topology that implies the backend.
        explicit: The backend asked for, or None for the implied one; it may only
            be the implied backend or one of ``EMBEDDING_VARIANTS``.

    Returns:
        ``explicit`` when set, else the backend the topology implies.

    Raises:
        ValueError: If ``explicit`` contradicts the topology (``local`` where the
            topology implies ``api``); the message names the topologies that
            would allow it.

    Example:
        >>> resolve_embedding_backend("split-all", None)
        'api'
        >>> resolve_embedding_backend("split-all", "local")
        ValueError: RAG_EMBEDDING_BACKEND='local' contradicts ...
    """
    implied = TOPOLOGY_BACKENDS[topology].embedding
    if explicit is None:
        return implied
    allowed = _distinct(implied, *EMBEDDING_VARIANTS)
    _refuse_contradiction(topology, "embedding", explicit, allowed)
    return explicit


def resolve_rerank_backend(
    topology: Topology, explicit: RerankBackend | None
) -> RerankBackend:
    """Return the rerank backend for ``topology``.

    Args:
        topology: The topology that implies the backend.
        explicit: The backend asked for, or None for the implied one; it may only
            be the implied backend or one of ``RERANK_VARIANTS`` (``none`` skips
            reranking).

    Returns:
        ``explicit`` when set, else the backend the topology implies.

    Raises:
        ValueError: If ``explicit`` contradicts the topology (``local`` where the
            topology implies ``api``); the message names the topologies that
            would allow it.

    Example:
        >>> resolve_rerank_backend("monolith", "none")
        'none'
    """
    implied = TOPOLOGY_BACKENDS[topology].rerank
    if explicit is None:
        return implied
    allowed = _distinct(implied, *RERANK_VARIANTS)
    _refuse_contradiction(topology, "rerank", explicit, allowed)
    return explicit


def _refuse_contradiction(
    topology: Topology,
    component: Literal["embedding", "rerank"],
    explicit: str,
    allowed: Sequence[str],
) -> None:
    if explicit in allowed:
        return
    variable = f"RAG_{component.upper()}_BACKEND"
    raise ValueError(
        f"{variable}={explicit!r} contradicts RAG_TOPOLOGY={topology!r}, which allows "
        f"only {_either(allowed)} for {component}; "
        f"{_contradiction_fix(component, explicit, variable)}"
    )


def _contradiction_fix(
    component: Literal["embedding", "rerank"], explicit: str, variable: str
) -> str:
    matching = [
        f"RAG_TOPOLOGY={name}"
        for name, backends in TOPOLOGY_BACKENDS.items()
        if getattr(backends, component) == explicit
    ]
    if not matching:
        return f"drop {variable}"
    return f"set {' or '.join(matching)}, or drop {variable}"


def _distinct(*values: str) -> tuple[str, ...]:
    # The implied backend may also be a variant (``api``): name it once.
    return tuple(dict.fromkeys(values))


def _either(values: Sequence[str]) -> str:
    quoted = [repr(value) for value in values]
    return ", ".join(quoted[:-1]) + f" or {quoted[-1]}"


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
