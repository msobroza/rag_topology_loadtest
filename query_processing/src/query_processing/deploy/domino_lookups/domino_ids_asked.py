"""The ids behind an App's tier name and revision number, asked of Domino.

The tiers are listed once, when the first App names a tier; a revision number
is looked for in the revision sources in order, the documented call first,
each asking Domino once an environment. An App that gives ids asks nothing.

Example:
    >>> DominoIdsAsked(client.http).find(settings.for_app("reranker")).found
    ('rag-reranker: hardware tier cpu-16 (id large-k8s): 16 cores, 64 GiB',)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import httpx

from ..apps_client import DominoDeployError
from ..deploy_settings import AppSettings
from .base_domino_ids import AppWithIds, BaseDominoIds, where_set
from .hardware_tiers import (
    TIERS_NEEDED_TO_DEPLOY,
    HardwareTier,
    list_hardware_tiers,
    tier_named,
)
from .internal_revisions import InternalRevisions
from .revision_sources import BaseRevisionSource, DocumentedRevisions, revision_numbered


class DominoIdsAsked(BaseDominoIds):
    """Finds the ids through Domino's calls, each made at most once.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).
        sources: Where a revision number is looked for, in order; None is the
            documented call, then the internal route.

    Example:
        >>> ids = DominoIdsAsked(client.http)
    """

    def __init__(
        self, http: httpx.Client, sources: Sequence[BaseRevisionSource] | None = None
    ) -> None:
        self._http = http
        self._sources = (
            [DocumentedRevisions(http), InternalRevisions(http)]
            if sources is None
            else list(sources)
        )
        self._tiers: list[HardwareTier] | None = None

    def find(self, app: AppSettings) -> AppWithIds:
        """Fill the ids of the tier and the revision the App names, asking Domino.

        Args:
            app: The App's merged settings.

        Returns:
            The App with the ids filled, and a line for each id found.

        Raises:
            DominoDeployError: As ``tier_named``, ``list_hardware_tiers`` and
                ``revision_numbered`` do, or if Domino gave the tier or the
                revision found without its id.

        Example:
            >>> DominoIdsAsked(client.http).find(app).app.environment_revision_id
            'rev-7'
        """
        tier_found = self._with_tier_id(app)
        revision_found = self._with_revision_id(tier_found.app)
        found = tier_found.found + revision_found.found
        return AppWithIds(revision_found.app, found)

    def _with_tier_id(self, app: AppSettings) -> AppWithIds:
        if not app.hardware_tier:
            return AppWithIds(app)
        if self._tiers is None:
            self._tiers = list_hardware_tiers(self._http, TIERS_NEEDED_TO_DEPLOY)
        tier = tier_named(self._tiers, app.hardware_tier, where_set(app))
        if not tier.id:
            raise DominoDeployError(
                f"hardware_tier {app.hardware_tier!r} of {where_set(app)} names a "
                "tier of Domino, but Domino's answer held no id for that tier: "
                "write the tier's id itself as hardware_tier_id"
            )
        line = f"{app.name}: hardware tier {tier.name} (id {tier.id}): {tier.size()}"
        return AppWithIds(replace(app, hardware_tier_id=tier.id), (line,))

    def _with_revision_id(self, app: AppSettings) -> AppWithIds:
        number = app.environment_revision
        if number is None:
            return AppWithIds(app)
        revision = revision_numbered(
            self._sources, app.environment_id, number, where_set(app)
        )
        if not revision.id:
            raise DominoDeployError(
                f"environment_revision {number} of {where_set(app)} is a revision "
                "Domino gives, but Domino's answer held no id for that revision of "
                f"environment {app.environment_id!r}: write the revision's id "
                "itself as environment_revision_id"
            )
        line = f"{app.name}: environment revision {number} (id {revision.id})"
        return AppWithIds(replace(app, environment_revision_id=revision.id), (line,))
