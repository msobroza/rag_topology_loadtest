"""Where an environment revision's number is turned into its id: sources asked in order.

The documented call (``DocumentedRevisions``) comes first: it gives the latest,
active and restricted revisions of an environment. A later source is asked only
for a number those three lack, as the CPU counters of ``cpu_count.py`` are asked.

Example:
    >>> revision_numbered([DocumentedRevisions(http)], "env-1", 7, "App x").id
    'rev-7'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from http import HTTPStatus
from urllib.parse import quote

import httpx

from ..apps_client import (
    DOMINO_API,
    DominoDeployError,
    domino_json,
    json_mapping,
    send_to_domino,
)
from .environments import (
    ENVIRONMENTS_PATH,
    DominoEnvironment,
    EnvironmentRevision,
    environment_from,
)

_NEEDS = (
    "reading an environment needs the permission ViewEnvironment: ask for it, or "
    "write the revision's id as environment_revision_id, which needs no such call"
)


class BaseRevisionSource(ABC):
    """A place that knows an environment's revisions, asked by ``revision_numbered``."""

    @abstractmethod
    def revisions(self, environment_id: str) -> list[EnvironmentRevision]:
        """Give every revision of the environment this source knows.

        A source whose failure is final, as the documented call's is, raises a
        ``DominoDeployError`` instead of answering an empty list; so may its
        ``describe``.

        Args:
            environment_id: The environment's id.

        Returns:
            The revisions; an empty list when the source has none or could not be
            asked.

        Example:
            >>> [revision.number for revision in source.revisions("env-1")]
            [7, 6, 3]
        """

    def left_out(self, environment_id: str) -> int:
        """Count the entries of the environment this source read but left out.

        Args:
            environment_id: The environment's id.

        Returns:
            How many of its entries this source could not use; 0 unless it says.

        Example:
            >>> source.left_out("env-1")
            0
        """
        del environment_id  # a source that leaves nothing out needs no id
        return 0

    @abstractmethod
    def describe(self, environment_id: str) -> str:
        """Say what this source holds of the environment, or what it answered.

        Args:
            environment_id: The environment's id.

        Returns:
            One clause of a refusal, e.g. ``"the documented API gives ..."``.

        Example:
            >>> source.describe("env-1")[:40]
            'the documented API gives its latest (7),'
        """


class DocumentedRevisions(BaseRevisionSource):
    """The documented call for one environment: its three revisions, asked once.

    Nothing is asked before a revision is.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).

    Example:
        >>> source = DocumentedRevisions(http)
    """

    def __init__(self, http: httpx.Client) -> None:
        self._http = http
        self._read: dict[str, DominoEnvironment] = {}

    def environment(self, environment_id: str) -> DominoEnvironment:
        """Give the environment, asking Domino the first time only.

        Args:
            environment_id: The environment's id.

        Returns:
            The environment with its three revisions.

        Raises:
            DominoDeployError: If Domino has no such environment (404), refuses it
                (401 or 403, with the permission ``ViewEnvironment``), answers
                without an environment, or the call fails otherwise. The failure
                is final: no later source is asked.

        Example:
            >>> DocumentedRevisions(http).environment("env-1").name
            'qp-rag'
        """
        if environment_id not in self._read:
            self._read[environment_id] = _read_environment(self._http, environment_id)
        return self._read[environment_id]

    def revisions(self, environment_id: str) -> list[EnvironmentRevision]:
        """Give the environment's latest, active and restricted revisions.

        Args:
            environment_id: The environment's id.

        Returns:
            Those of the three that exist, each once.

        Raises:
            DominoDeployError: As ``environment`` does.

        Example:
            >>> [revision.number for revision in source.revisions("env-1")]
            [7, 6, 3]
        """
        return self.environment(environment_id).three()

    def describe(self, environment_id: str) -> str:
        """Say the numbers of the three revisions, ``none`` for an absent one.

        Args:
            environment_id: The environment's id.

        Returns:
            E.g. ``"the documented API gives its latest (7), active (6) and
            restricted (none) revisions"``.

        Raises:
            DominoDeployError: As ``environment`` does.

        Example:
            >>> source.describe("env-1")[:40]
            'the documented API gives its latest (7),'
        """
        environment = self.environment(environment_id)
        latest, active, restricted = (
            "none" if revision is None else str(revision.number)
            for revision in (
                environment.latest,
                environment.active,
                environment.restricted,
            )
        )
        return (
            f"the documented API gives its latest ({latest}), active ({active}) "
            f"and restricted ({restricted}) revisions"
        )


def revision_numbered(
    sources: Sequence[BaseRevisionSource], environment_id: str, number: int, where: str
) -> EnvironmentRevision:
    """Find the revision numbered ``number``: a source is asked if those before lack it.

    Args:
        sources: The sources, in the order they are asked (the documented one first).
        environment_id: The environment's id.
        number: The value of ``environment_revision``.
        where: The App and the blocks that set it, for the message.

    Returns:
        The revision of the first source that has the number.

    Raises:
        DominoDeployError: If no source has it (the message says what each holds),
            or a source's call fails for good.

    Example:
        >>> revision_numbered([DocumentedRevisions(http)], "env-1", 7, "App x").id
        'rev-7'
    """
    for source in sources:
        found = [r for r in source.revisions(environment_id) if r.number == number]
        if found:
            return found[0]
    held = "; ".join(source.describe(environment_id) for source in sources)
    raise DominoDeployError(
        f"environment_revision {number} of {where} is no revision Domino gives for "
        f"environment {environment_id!r}: {held}. Write the revision's id as "
        "environment_revision_id"
    )


def _read_environment(http: httpx.Client, environment_id: str) -> DominoEnvironment:
    path = f"{ENVIRONMENTS_PATH}/{quote(environment_id, safe='')}"
    response = send_to_domino(http, "GET", path, api=DOMINO_API)
    if response.status_code == HTTPStatus.NOT_FOUND:
        raise DominoDeployError(
            f"Domino has no environment {environment_id!r} (GET {path} returned "
            "HTTP 404): check environment_id in deploy.yaml"
        )
    environment = json_mapping(
        domino_json(DOMINO_API, response, _NEEDS).get("environment")
    )
    if not environment:  # an answer of another shape: final, as a failure is
        raise DominoDeployError(
            f"{DOMINO_API} GET {path} answered HTTP 200 without an environment "
            f"(an EnvironmentEnvelopeV1 was expected): {response.text[:200]}"
        )
    return environment_from(environment)
