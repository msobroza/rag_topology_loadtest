"""The ids behind an App's tier name and revision number: what finds them, its answer.

A deploy file may name an App's hardware tier (``hardware_tier``) and its
environment revision by number (``environment_revision``); an App's version
carries only ids. Before the first App is published, ``find_every_id`` finds
the ids of every App deployed and prints what it found, so a wrong value stops
the deploy with nothing published. A deploy asks Domino
(``domino_ids_asked.py``); a dry run asks nothing (``domino_ids_not_asked.py``).

Example:
    >>> find_every_id(DominoIdsAsked(client.http), [settings.for_app("reranker")])
    rag-reranker: hardware tier cpu-16 (id large-k8s): 16 cores, 64 GiB
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

from ..deploy_settings import AppSettings, deploy_file


@dataclass(frozen=True)
class AppWithIds:
    """An App's settings with its tier and revision ids filled, and what was found.

    ``found`` holds one line per id found, to print before publishing.
    """

    app: AppSettings
    found: tuple[str, ...] = ()


class BaseDominoIds(ABC):
    """What turns an App's tier name and revision number into its version's ids."""

    @abstractmethod
    def find(self, app: AppSettings) -> AppWithIds:
        """Fill the App's ``hardware_tier_id`` and ``environment_revision_id``.

        An App that names neither, giving ids or nothing, is returned as it is.
        Asking twice for the same App makes no new call to Domino.

        Args:
            app: The App's merged settings.

        Returns:
            The App with the ids filled, and a line for each id found.

        Raises:
            DominoDeployError: If a name or a number names nothing of Domino, or
                Domino cannot be asked (see ``tier_named`` and ``revision_numbered``).

        Example:
            >>> ids.find(settings.for_app("reranker")).app.hardware_tier_id
            'large-k8s'
        """


def where_set(app: AppSettings) -> str:
    """Say which App a value belongs to and the blocks of the deploy file that set it.

    Args:
        app: The App's merged settings.

    Returns:
        The App's name and its two blocks, for a refusal's message.

    Example:
        >>> where_set(settings.for_app("reranker"))
        'App rag-reranker (apps.reranker or defaults in deploy.yaml)'
    """
    return f"App {app.name} (apps.{app.part} or defaults in {deploy_file()})"


def find_every_id(ids: BaseDominoIds, apps: Iterable[AppSettings]) -> None:
    """Find the ids of every App, then print what was found, one line per id.

    Every App is looked up before anything is printed or published, so a wrong
    value of the last App stops the deploy as one of the first does.

    Args:
        ids: What finds the ids.
        apps: The merged settings of every App deployed, in the order published.

    Raises:
        DominoDeployError: As ``BaseDominoIds.find`` does.

    Example:
        >>> find_every_id(DominoIdsAsked(client.http), map(settings.for_app, parts))
        rag-embedder: hardware tier cpu-8 (id small-k8s): 8 cores, 32 GiB
    """
    found = [ids.find(app).found for app in apps]
    for line in chain.from_iterable(found):
        print(line)
