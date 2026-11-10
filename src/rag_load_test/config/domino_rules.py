"""The Domino rule: settings made fit for a Domino App (``rag-serve --domino``)."""

from __future__ import annotations

from .settings import RagSettings


def apply_domino_rules(settings: RagSettings) -> RagSettings:
    """``settings`` made fit for a Domino app, or ``ValueError`` naming what is missing.

    Every port on a model server needs its URL set explicitly (the localhost
    defaults cannot be right on Domino), and app-to-app calls need the run's token,
    so an unset ``ovms_auth`` becomes ``domino``. The input is not mutated.

    Example::

        settings = apply_domino_rules(RagSettings())   # rag-serve --domino
    """
    missing = _missing_model_server_urls(settings)
    if missing:
        raise ValueError(
            f"RAG_TOPOLOGY={settings.topology!r} calls a model server on Domino: set "
            f"{' and '.join(missing)} to the model server's App URL (the localhost "
            "default cannot be right there)"
        )
    if "ovms_auth" in settings.model_fields_set:
        return settings
    return settings.model_copy(update={"ovms_auth": "domino"})


def _missing_model_server_urls(settings: RagSettings) -> list[str]:
    needed = []
    if settings.embedder_backend == "ovms":
        needed.append("ovms_embeddings_url")
    if settings.reranker_backend == "ovms":
        needed.append("ovms_rerank_url")
    return [
        f"RAG_{field.upper()}"
        for field in needed
        if field not in settings.model_fields_set
    ]
