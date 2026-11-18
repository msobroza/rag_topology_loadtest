"""The ids of a dry run: a tier's name and a revision's number shown as written.

A dry run asks Domino nothing, so it cannot know an id. In its place the
version carries a text saying what is asked at deploy, which the dry run's
report shows; nothing is printed before publishing.

Example:
    >>> DominoIdsNotAsked().find(settings.for_app("reranker")).app.hardware_tier_id
    '(tier cpu-16: its id is asked of Domino at deploy)'
"""

from __future__ import annotations

from dataclasses import replace

from ..deploy_settings import AppSettings
from .base_domino_ids import AppWithIds, BaseDominoIds


class DominoIdsNotAsked(BaseDominoIds):
    """Puts in each id's place the name or the number written, asking nothing.

    Example:
        >>> ids = DominoIdsNotAsked()
    """

    def find(self, app: AppSettings) -> AppWithIds:
        """Put the tier's name and the revision's number where their ids go.

        Args:
            app: The App's merged settings.

        Returns:
            The App with each named id replaced by a text, and nothing found.

        Example:
            >>> DominoIdsNotAsked().find(app).app.environment_revision_id
            '(revision 7: its id is asked of Domino at deploy)'
        """
        if app.hardware_tier:
            tier = _asked_at_deploy(f"tier {app.hardware_tier}")
            app = replace(app, hardware_tier_id=tier)
        if app.environment_revision is not None:
            revision = _asked_at_deploy(f"revision {app.environment_revision}")
            app = replace(app, environment_revision_id=revision)
        return AppWithIds(app)


def _asked_at_deploy(written: str) -> str:
    return f"({written}: its id is asked of Domino at deploy)"
