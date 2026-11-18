"""The ``ovms`` model-server kind: OpenVINO Model Server.

It serves an embedder or a reranker exported to OpenVINO IR, and an LLM, on
its own /v3 routes: the LLM on its chat completions route. Its launch script
is ``domino/model_servers/ovms.sh``, which runs the OVMS binary of the bundle
the deploy settings name (``ovms_bundle``): the deploy tool exports that path
as ``QP_OVMS_BUNDLE`` and refuses, before anything is published, a bundle that
does not exist.

Example:
    >>> OvmsModelServerKind().route_of("embedder")
    '/v3/embeddings'
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...domain.endpoints.api_endpoint import (
    MODEL_SERVER_CHAT_PATH,
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_MODEL_ROUTE,
    MODEL_SERVER_RERANK_PATH,
)
from ..apps_client import DominoDeployError
from .base_model_server_kind import MODEL_SERVER_KINDS, BaseModelServerKind

if TYPE_CHECKING:  # deploy_settings checks a kind's name against the registry
    from ..deploy_settings import DeploySettings


@MODEL_SERVER_KINDS.register("ovms")
class OvmsModelServerKind(BaseModelServerKind):
    """OpenVINO Model Server: an embedder, a reranker or an LLM, from the OVMS bundle.

    Example:
        >>> OvmsModelServerKind().launch_exports(settings)
        {'QP_OVMS_BUNDLE': '/mnt/data/ovms.tar.gz'}
    """

    @property
    def routes(self) -> Mapping[str, str]:
        """OVMS's /v3 route of each type it serves."""
        return {
            "embedder": MODEL_SERVER_EMBEDDINGS_PATH,
            "reranker": MODEL_SERVER_RERANK_PATH,
            "llm": MODEL_SERVER_CHAT_PATH,
        }

    @property
    def probe_route(self) -> str:
        """OVMS's model status route."""
        return MODEL_SERVER_MODEL_ROUTE

    def launch_exports(self, settings: DeploySettings) -> dict[str, str]:
        """Give the bundle the script's bootstrap runs OVMS from.

        Args:
            settings: The deploy settings, for ``ovms_bundle``.

        Returns:
            ``QP_OVMS_BUNDLE``.
        """
        return {"QP_OVMS_BUNDLE": settings.ovms_bundle}

    def refuse_unusable_settings(self, settings: DeploySettings) -> None:
        """Refuse a bundle that is not set or does not exist.

        Args:
            settings: The deploy settings, for ``ovms_bundle``.

        Raises:
            DominoDeployError: If ``ovms_bundle`` names no file or folder.
        """
        if settings.ovms_bundle and os.path.exists(settings.ovms_bundle):
            return
        raise DominoDeployError(
            f"ovms_bundle={settings.ovms_bundle} does not exist: set it to "
            "the OVMS bundle folder (bin/ovms, lib/) or its .tar.gz, at a path the "
            "Apps can read"
        )
