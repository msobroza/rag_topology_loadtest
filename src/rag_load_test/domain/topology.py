"""The topology rule: which embedder and reranker backends each topology implies.

A topology is a closed set; it says where the in-process models run, so it
decides the embedder and reranker backends. An explicit backend may only repeat
that choice or pick a variant that runs no model of ours: a hosted API (``openai``
for the embedder; ``cohere`` or ``openai`` for the reranker), ``fake``, or
``none`` for the reranker. The run label names the measured
configuration and defaults to the topology plus every backend that differs from
its default. Pure functions only: settings call them at construction; the Domino
rule that ``rag-serve --domino`` applies lives in ``config/domino_rules.py``.

Example:
    >>> resolve_reranker_backend("split-reranker", None)
    'ovms'
    >>> default_run_label("monolith", embedder="local", reranker="none",
    ...                   store="elasticsearch", llm="openai")
    'monolith+reranker=none'
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

from .workflow_parameters import DEFAULT_RETRIEVAL_MODE

Topology = Literal["monolith", "split-reranker", "split-all"]
EmbedderBackend = Literal["local", "ovms", "openai", "fake"]
RerankerBackend = Literal["local", "ovms", "cohere", "openai", "fake", "none"]
VectorStoreBackend = Literal["elasticsearch", "chroma", "fake"]
LlmBackend = Literal["openai", "fake"]
# What runs an in-process model: OpenVINO on its exported files, or PyTorch.
LocalModelRuntime = Literal["openvino", "torch"]

# Backends any topology allows: they run no model of ours, so where our models
# run does not constrain them.
EMBEDDER_VARIANTS: tuple[EmbedderBackend, ...] = ("openai", "fake")
RERANKER_VARIANTS: tuple[RerankerBackend, ...] = ("cohere", "openai", "fake", "none")

DEFAULT_VECTOR_STORE_BACKEND: VectorStoreBackend = "elasticsearch"
DEFAULT_LLM_BACKEND: LlmBackend = "openai"
# Sent as the X-RAG-Run-Label header, so only header-safe characters.
RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9._+=:-]{1,128}")
RUN_LABEL_SHAPE = "letters, digits and ._+=:- only, 1 to 128 characters"


class TopologyBackends(NamedTuple):
    """The embedder and reranker backends one topology implies."""

    embedder: EmbedderBackend
    reranker: RerankerBackend


TOPOLOGY_BACKENDS: dict[Topology, TopologyBackends] = {
    "monolith": TopologyBackends(embedder="local", reranker="local"),
    "split-reranker": TopologyBackends(embedder="local", reranker="ovms"),
    "split-all": TopologyBackends(embedder="ovms", reranker="ovms"),
}


def resolve_embedder_backend(
    topology: Topology, explicit: EmbedderBackend | None
) -> EmbedderBackend:
    """Return the embedder backend for ``topology``.

    Args:
        topology: The topology that implies the backend.
        explicit: The backend asked for, or None for the implied one; it may only
            be the implied backend or one of ``EMBEDDER_VARIANTS``.

    Returns:
        ``explicit`` when set, else the backend the topology implies.

    Raises:
        ValueError: If ``explicit`` contradicts the topology; the message names
            the topologies that would allow it.

    Example:
        >>> resolve_embedder_backend("split-all", None)
        'ovms'
        >>> resolve_embedder_backend("monolith", "ovms")
        ValueError: ... contradicts RAG_TOPOLOGY='monolith' ...
    """
    implied = TOPOLOGY_BACKENDS[topology].embedder
    if explicit is None:
        return implied
    _refuse_contradiction(topology, "embedder", explicit, (implied, *EMBEDDER_VARIANTS))
    return explicit


def resolve_reranker_backend(
    topology: Topology, explicit: RerankerBackend | None
) -> RerankerBackend:
    """Return the reranker backend for ``topology``.

    Args:
        topology: The topology that implies the backend.
        explicit: The backend asked for, or None for the implied one; it may only
            be the implied backend or one of ``RERANKER_VARIANTS`` (``none``
            skips reranking).

    Returns:
        ``explicit`` when set, else the backend the topology implies.

    Raises:
        ValueError: If ``explicit`` contradicts the topology; the message names
            the topologies that would allow it.

    Example:
        >>> resolve_reranker_backend("monolith", "none")
        'none'
    """
    implied = TOPOLOGY_BACKENDS[topology].reranker
    if explicit is None:
        return implied
    allowed = (implied, *RERANKER_VARIANTS)
    _refuse_contradiction(topology, "reranker", explicit, allowed)
    return explicit


def _refuse_contradiction(
    topology: Topology,
    port: Literal["embedder", "reranker"],
    explicit: str,
    allowed: Sequence[str],
) -> None:
    if explicit in allowed:
        return
    variable = f"RAG_{port.upper()}_BACKEND"
    raise ValueError(
        f"{variable}={explicit!r} contradicts RAG_TOPOLOGY={topology!r}, which allows "
        f"only {_either(allowed)} for the {port}; "
        f"{_contradiction_fix(port, explicit, variable)}"
    )


def _contradiction_fix(
    port: Literal["embedder", "reranker"], explicit: str, variable: str
) -> str:
    matching = [
        f"RAG_TOPOLOGY={name}"
        for name, backends in TOPOLOGY_BACKENDS.items()
        if getattr(backends, port) == explicit
    ]
    if not matching:
        return f"drop {variable}"
    return f"set {' or '.join(matching)}, or drop {variable}"


def _either(values: Sequence[str]) -> str:
    quoted = [repr(value) for value in values]
    return ", ".join(quoted[:-1]) + f" or {quoted[-1]}"


def default_run_label(
    topology: Topology,
    *,
    embedder: str,
    reranker: str,
    store: str,
    llm: str,
    retrieval: str = DEFAULT_RETRIEVAL_MODE,
) -> str:
    """Name a configuration after its topology and the choices off their default.

    Choices are listed in a fixed order (embedder, reranker, store, llm, retrieval)
    so the same configuration always gets the same label.

    Args:
        topology: The topology; it sets the default embedder and reranker.
        embedder: The embedder backend in use.
        reranker: The reranker backend in use.
        store: The vector store backend in use.
        llm: The LLM backend in use.
        retrieval: The retrieval mode in use.

    Returns:
        The topology plus ``+<port>=<backend>`` for each backend off its default,
        then ``+retrieval=<mode>`` for a retrieval mode that is not ``dense``.

    Example:
        >>> default_run_label("split-all", embedder="ovms", reranker="ovms",
        ...                   store="elasticsearch", llm="fake")
        'split-all+llm=fake'
    """
    implied = TOPOLOGY_BACKENDS[topology]
    ports = (
        ("embedder", embedder, implied.embedder),
        ("reranker", reranker, implied.reranker),
        ("store", store, DEFAULT_VECTOR_STORE_BACKEND),
        ("llm", llm, DEFAULT_LLM_BACKEND),
        ("retrieval", retrieval, DEFAULT_RETRIEVAL_MODE),
    )
    variants = [
        f"+{port}={actual}" for port, actual, default in ports if actual != default
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
