"""The topology alias deployed, resolved against the components of the workflow file.

The deploy settings choose a topology by its alias, and define each alias as a
row of ``topology_alias``: ``local`` or ``served`` for the components it names
(ADR 0003). Resolving the chosen one says which Apps to publish (one model
server App per served component, then the workflow App), which model-server
kind serves each served component, what to write into the workflow App's
launch script (each backend the row gives, each served URL, the alias as a
label and the run label: the workflow app holds no topology rule of its own),
and refuses, before anything is published, an alias the table
lacks, a row naming a component the workflow file does not declare, a served
component its model-server kind cannot serve, and an alias that makes no valid
run label.

Example:
    >>> topology = resolve_topology(DeploySettings())
    >>> topology.app_parts()
    ['embedder', 'reranker', 'workflow']
    >>> topology.run_label()
    'split-all+embedder=fastapi+reranker=fastapi'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..domain.roles import WORKFLOW_APP
from ..domain.run_label import RUN_LABEL_PATTERN, RUN_LABEL_SHAPE
from .apps_client import DominoDeployError
from .deploy_settings import SERVED, DeploySettings, Placement, deploy_file
from .model_server_kinds.base_model_server_kind import (
    MODEL_SERVER_KINDS,
    model_server_kind,
)
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
        kinds: The model-server kind of each served component, by name.
    """

    alias: str
    placements: dict[str, Placement]
    components: dict[str, WorkflowComponent]
    kinds: dict[str, str]

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

    def run_label(self) -> str:
        """Name the run: the alias, then each served component and its server kind.

        Returns:
            ``<alias>+<component>=<kind>`` for each served component, each with
            the model-server kind of its own App.

        Example:
            >>> topology.run_label()
            'split-all+embedder=fastapi+reranker=ovms'
        """
        return self.alias + "".join(
            f"+{name}={self.kinds[name]}" for name in self.served
        )

    def served_route(self, name: str) -> str:
        """Give the route a served component is called on: its kind's for its type.

        Args:
            name: A served component.

        Returns:
            The route its model-server kind declares for its type, such as
            ``/v3/rerank``.

        Example:
            >>> topology.served_route("reranker")
            '/v3/rerank'
        """
        kind = model_server_kind(self.kinds[name])
        return kind.route_of(self.components[name].type)


def resolve_topology(settings: DeploySettings) -> ResolvedTopology:
    """Resolve the chosen alias against the workflow file the workflow App reads.

    Args:
        settings: The deploy settings: ``topology``, ``topology_alias``, the
            model-server kind of each App and the workflow App's settings.

    Returns:
        The chosen alias, the components it places and the model-server kind
        of each served one.

    Raises:
        DominoDeployError: If the workflow file cannot be read, ``apps`` has a
            block that is neither ``workflow`` nor a component of it, the alias
            is not in ``topology_alias``, its row names a component the file
            does not declare, it serves a component its model-server kind
            cannot serve, or it would make no valid run label.

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
    served = [name for name, cell in placements.items() if cell == SERVED]
    kinds = {name: settings.for_app(name).server for name in served}
    topology = ResolvedTopology(settings.topology, placements, components, kinds)
    for name in topology.served:
        _refuse_what_its_kind_cannot_serve(topology, name)
    _refuse_an_invalid_run_label(topology)
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


def _refuse_what_its_kind_cannot_serve(topology: ResolvedTopology, name: str) -> None:
    type_, kind = topology.components[name].type, topology.kinds[name]
    if model_server_kind(kind).serves(type_):
        return
    able = _kinds_that_serve(type_)
    if not able:
        raise DominoDeployError(
            f"topology alias {topology.alias!r} serves component {name!r}, of type "
            f"{type_!r}, which no model-server kind can serve: {_what_each_serves()}; "
            f"leave {name!r} out of the row to keep the backend the workflow file "
            "gives it"
        )
    raise DominoDeployError(
        f"topology alias {topology.alias!r} serves component {name!r}, of type "
        f"{type_!r}, with the model-server kind {kind!r} (apps.{name}.server, else "
        f"defaults.server, in {deploy_file()}), which cannot serve it: the kinds "
        f"that can are {_names(able)}"
    )


def _kinds_that_serve(component_type: str) -> list[str]:
    return [
        name
        for name in MODEL_SERVER_KINDS.names()
        if model_server_kind(name).serves(component_type)
    ]


def _what_each_serves() -> str:
    return ", ".join(
        f"{name} serves {' or '.join(model_server_kind(name).routes)}"
        for name in MODEL_SERVER_KINDS.names()
    )


def _refuse_an_invalid_run_label(topology: ResolvedTopology) -> None:
    label = topology.run_label()
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
