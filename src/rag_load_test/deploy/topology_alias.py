"""The topology alias deployed, resolved against the components of the workflow file.

The deploy settings choose a topology by its alias, and define each alias as a
row of ``topology_alias``: ``local`` or ``served`` for the components it names
(ADR 0003). Resolving the chosen one says which Apps to publish (one model
server App per served component, then the workflow App), what to write into the
workflow App's launch script, and refuses, before anything is published, an
alias the table lacks, a row naming a component the workflow file does not
declare, a served component no model server can serve, and an alias that makes
no valid run label.

Example:
    >>> topology = resolve_topology(DeploySettings())
    >>> topology.app_parts()
    ['embedder', 'reranker', 'workflow']
    >>> topology.run_label("fastapi")
    'split-all+embedder=fastapi+reranker=fastapi'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..domain.roles import MODEL_SERVER_ROUTES, WORKFLOW_APP, ModelServerKind
from ..domain.run_label import RUN_LABEL_PATTERN, RUN_LABEL_SHAPE
from ..domain.topology import SERVED, Topology, topology_that_moves_nothing
from .apps_client import DominoDeployError
from .deploy_settings import DeploySettings, Placement, deploy_file
from .workflow_components import (
    WorkflowComponent,
    read_workflow_components,
    workflow_file_for,
)


@dataclass(frozen=True)
class ResolvedTopology:
    """A topology alias and the components of the workflow file it places.

    Attributes:
        alias: The topology alias.
        placements: ``local`` or ``served`` for each component the alias names,
            in the workflow file's order.
        components: Every component of the workflow file, by name.
    """

    alias: str
    placements: dict[str, Placement]
    components: dict[str, WorkflowComponent]

    @property
    def served(self) -> list[str]:
        """The components a model server App serves, in the workflow file's order."""
        return [name for name, cell in self.placements.items() if cell == SERVED]

    def app_parts(self) -> list[str]:
        """List the Apps to publish, in order: each served component, then the workflow.

        Returns:
            The App parts, the workflow App last: it is handed the others' URLs.

        Example:
            >>> topology.app_parts()
            ['reranker', 'workflow']
        """
        return [*self.served, WORKFLOW_APP]

    def backends(self) -> dict[str, str]:
        """Give the backend each component of the workflow file runs on.

        Returns:
            The alias's cell for a component it names, else the backend the
            workflow file states.

        Example:
            >>> topology.backends()
            {'embedder': 'local', 'reranker': 'served', 'llm': 'api'}
        """
        return {
            name: self.placements.get(name, component.backend)
            for name, component in self.components.items()
        }

    def run_label(self, kind: ModelServerKind) -> str:
        """Name the run: the alias, then each served component and its server kind.

        Args:
            kind: The model-server kind of every model server App.

        Returns:
            ``<alias>+<component>=<kind>`` for each served component.

        Example:
            >>> topology.run_label("ovms")
            'split-reranker+reranker=ovms'
        """
        return self.alias + "".join(f"+{name}={kind}" for name in self.served)

    def topology_variable(self) -> Topology:
        """Name the topology the workflow App is told, as ``RAG_TOPOLOGY``.

        Returns:
            The alias when the workflow app knows it and its rule moves no
            backend this topology gives, else ``monolith`` (whose rule moves
            none): the deploy tool writes every backend itself.

        Example:
            >>> topology.topology_variable()
            'split-reranker'
        """
        return topology_that_moves_nothing(self.alias, self.backends())


def resolve_topology(settings: DeploySettings) -> ResolvedTopology:
    """Resolve the chosen alias against the workflow file the workflow App reads.

    Args:
        settings: The deploy settings: ``topology``, ``topology_alias``, the
            model-server kind and the workflow App's settings.

    Returns:
        The chosen alias and the components it places.

    Raises:
        DominoDeployError: If the workflow file cannot be read, ``apps`` has a
            block that is neither ``workflow`` nor a component of it, the alias
            is not in ``topology_alias``, its row names a component the file
            does not declare, it serves a component no model server can serve,
            or it would make no valid run label.

    Example:
        >>> resolve_topology(DeploySettings(topology="monolith")).app_parts()
        ['workflow']
    """
    path = workflow_file_for(settings.for_app(WORKFLOW_APP).settings)
    components = read_workflow_components(path)
    _refuse_an_app_block_of_no_app(settings, path, components)
    row = _row_of(settings)
    _refuse_an_unknown_component(settings.topology, row, path, components)
    placements = {name: row[name] for name in components if name in row}
    topology = ResolvedTopology(settings.topology, placements, components)
    for name in topology.served:
        _refuse_what_no_model_server_serves(topology, name, settings.model_server)
    _refuse_an_invalid_run_label(topology, settings.model_server)
    return topology


def _row_of(settings: DeploySettings) -> dict[str, Placement]:
    row = settings.topology_alias.get(settings.topology)
    if row is not None:
        return row
    raise DominoDeployError(
        f"topology {settings.topology!r} is not a topology alias of "
        f"{deploy_file()}: set it to a key of topology_alias (its aliases: "
        f"{_names(list(settings.topology_alias))}), or add a row for it, such as "
        "split-all: {embedder: served, reranker: served}"
    )


def _refuse_an_unknown_component(
    alias: str,
    row: dict[str, Placement],
    path: Path,
    components: dict[str, WorkflowComponent],
) -> None:
    unknown = [name for name in row if name not in components]
    if not unknown:
        return
    raise DominoDeployError(
        f"topology alias {alias!r} in {deploy_file()} names {_names(unknown)}, "
        f"which the workflow file {str(path)!r} does not declare: a row names "
        f"components of that file ({_names(list(components))})"
    )


def _refuse_what_no_model_server_serves(
    topology: ResolvedTopology, name: str, kind: ModelServerKind
) -> None:
    servable = MODEL_SERVER_ROUTES[kind]
    type_ = topology.components[name].type
    if type_ not in servable:
        raise DominoDeployError(
            f"topology alias {topology.alias!r} serves component {name!r}, of type "
            f"{type_!r}, which no model server can serve: a {kind} model server "
            f"serves a component of type {' or '.join(servable)}; leave {name!r} "
            "out of the row to keep the backend the workflow file gives it"
        )
    if name != type_:
        raise DominoDeployError(
            f"topology alias {topology.alias!r} serves component {name!r}, of type "
            f"{type_!r}: a model server App serves only the component named after "
            f"its type ({type_!r}), the one domino/app.sh starts"
        )


def _refuse_an_invalid_run_label(
    topology: ResolvedTopology, kind: ModelServerKind
) -> None:
    label = topology.run_label(kind)
    if RUN_LABEL_PATTERN.fullmatch(label):
        return
    raise DominoDeployError(
        f"topology alias {topology.alias!r} makes the run label {label!r}, which "
        f"the workflow app refuses: a run label is {RUN_LABEL_SHAPE}, so rename "
        "the alias"
    )


def _refuse_an_app_block_of_no_app(
    settings: DeploySettings, path: Path, components: dict[str, WorkflowComponent]
) -> None:
    # The blocks are keyed by an App part, which only the workflow file can
    # check: a misspelt one would otherwise be silently unused.
    parts = [WORKFLOW_APP, *components]
    unknown = [part for part in settings.apps if part not in parts]
    if not unknown:
        return
    raise DominoDeployError(
        f"apps.{unknown[0]} in {deploy_file()} is no App: an apps block is keyed "
        f"by workflow or by a component of the workflow file {str(path)!r} "
        f"({_names(list(components))})"
    )


def _names(names: list[str]) -> str:
    return ", ".join(map(repr, names)) or "none"
