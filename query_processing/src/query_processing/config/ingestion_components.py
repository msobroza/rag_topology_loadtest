"""Which components ingestion fills the store with: one store, one embedder.

``qp-ingest`` embeds records with one embedder component of the workflow file
and fills its one vector store. With several embedders, one must be named.

Example:
    >>> ingestion_embedder(["embedder"], None)
    'embedder'
"""

from __future__ import annotations


def ingestion_store(stores: list[str]) -> str:
    """Return the vector store that ingestion fills: the only one declared.

    Args:
        stores: The vector store components of the workflow file, in its order.

    Returns:
        The component's name.

    Raises:
        ValueError: If the workflow file declares no vector store or several.

    Example:
        >>> ingestion_store(["vector_store"])
        'vector_store'
    """
    if len(stores) != 1:
        known = ", ".join(map(repr, stores)) or "none"
        raise ValueError(
            "ingestion fills one vector store component, and the workflow "
            f"file declares {len(stores)} (its vector stores: {known})"
        )
    return stores[0]


def ingestion_embedder(embedders: list[str], name: str | None) -> str:
    """Return the embedder that fills the store: the one named, else the only one.

    Args:
        embedders: The embedder components of the workflow file, in its order.
        name: The embedder asked for; None for the only one declared.

    Returns:
        The component's name.

    Raises:
        ValueError: If ``name`` is not an embedder component, or none is
            given and the file declares no embedder or several.

    Example:
        >>> ingestion_embedder(["embedder", "router_embedder"], "embedder")
        'embedder'
    """
    known = ", ".join(map(repr, embedders)) or "none"
    if name is not None and name not in embedders:
        raise ValueError(
            f"{name!r} is not an embedder component of the workflow file "
            f"(its embedders: {known})"
        )
    if name is None and len(embedders) != 1:
        raise ValueError(
            "ingestion embeds with one embedder component, and the workflow "
            f"file declares {len(embedders)} (its embedders: {known}): name "
            "one, with qp-ingest --embedder"
        )
    return name or embedders[0]
