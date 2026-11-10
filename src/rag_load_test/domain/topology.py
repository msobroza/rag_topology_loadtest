"""The topology rule: which embedder and reranker backends each topology implies.

A topology is a closed set; it decides the embedder and reranker backends, and an
explicit backend may only repeat that choice or pick an experiment variant
(``fake`` anywhere, ``none`` for the reranker). The run label names the measured
configuration and defaults to the topology plus every backend that differs from
its default. Pure functions only: settings call them at construction; the Domino
rule that ``rag-serve --domino`` applies lives in ``config/domino_rules.py``.

Example::

    resolve_reranker_backend("split-reranker", None)   # "ovms"
    default_run_label("monolith", embedder="local", reranker="none",
                      store="elasticsearch", llm="openai")  # "monolith+reranker=none"
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

Topology = Literal["monolith", "split-reranker", "split-all"]
EmbedderBackend = Literal["local", "ovms", "fake"]
RerankerBackend = Literal["local", "ovms", "fake", "none"]
VectorStoreBackend = Literal["elasticsearch", "fake"]
LlmBackend = Literal["openai", "fake"]

DEFAULT_VECTOR_STORE_BACKEND: VectorStoreBackend = "elasticsearch"
DEFAULT_LLM_BACKEND: LlmBackend = "openai"
# Sent as the X-RAG-Run-Label header, so only header-safe characters.
RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9._+=:-]{1,64}")
RUN_LABEL_SHAPE = "letters, digits and ._+=:- only, 1 to 64 characters"


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
    """The embedder backend for ``topology``; ``explicit`` may only be its own or fake.

    Example::

        resolve_embedder_backend("split-all", None)    # "ovms"
        resolve_embedder_backend("monolith", "ovms")   # ValueError: contradiction
    """
    implied = TOPOLOGY_BACKENDS[topology].embedder
    if explicit is None:
        return implied
    _refuse_contradiction(topology, "embedder", explicit, (implied, "fake"))
    return explicit


def resolve_reranker_backend(
    topology: Topology, explicit: RerankerBackend | None
) -> RerankerBackend:
    """The reranker backend for ``topology``; ``explicit`` may be its own, fake or none.

    Example::

        resolve_reranker_backend("monolith", "none")   # "none" (skip reranking)
    """
    implied = TOPOLOGY_BACKENDS[topology].reranker
    if explicit is None:
        return implied
    _refuse_contradiction(topology, "reranker", explicit, (implied, "fake", "none"))
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
) -> str:
    """The topology plus ``+<port>=<backend>`` for each backend off its default.

    Ports are listed in a fixed order (embedder, reranker, store, llm) so the same
    configuration always gets the same label.

    Example::

        default_run_label("split-all", embedder="ovms", reranker="ovms",
                          store="elasticsearch", llm="fake")   # "split-all+llm=fake"
    """
    implied = TOPOLOGY_BACKENDS[topology]
    ports = (
        ("embedder", embedder, implied.embedder),
        ("reranker", reranker, implied.reranker),
        ("store", store, DEFAULT_VECTOR_STORE_BACKEND),
        ("llm", llm, DEFAULT_LLM_BACKEND),
    )
    variants = [
        f"+{port}={actual}" for port, actual, default in ports if actual != default
    ]
    return topology + "".join(variants)


def validate_run_label(label: str) -> str:
    """``label`` unchanged when it is header-safe, else ``ValueError``.

    Example::

        validate_run_label("split-reranker-fastapi")   # "split-reranker-fastapi"
    """
    if RUN_LABEL_PATTERN.fullmatch(label):
        return label
    raise ValueError(
        f"RAG_RUN_LABEL={label!r} is not a run label: use {RUN_LABEL_SHAPE}"
    )
