"""A registered component in a topology alias: served in an App of its own.

Through ``deploy`` over the fake Domino, against a workflow file that loads
the tests' extension (``shout_extension``) and declares a ``shouter`` and a
``murmurer`` beside the fixture's components: the alias puts the shouter on
``served``, the deploy publishes its App and hands the workflow App its
backend and the URL of its call route, and every refusal comes before
anything is sent to Domino. The fixtures are in domino_deploy_fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domino_deploy_helpers import (
    AGENT_EXPORTS,
    BOOTSTRAP_AND_EXEC,
    INSTANCES_URL,
    LAUNCH,
    TOPOLOGY_ALIASES,
    make_client,
    make_settings,
    model_server_exports,
    no_sleep,
)
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.model_server_kinds.fastapi_kind import (
    FastApiModelServerKind,
)
from query_processing.deploy.publish import deploy
from shout_components import SHOUT_EXTENSION, write_shout_workflow

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http
# The shouter's App is the first published, so Domino numbers it 1.
SHOUTER_CALL_URL = f"{INSTANCES_URL}/app-1-v1-i/call/shouter"


@pytest.fixture
def shout_workflow(tmp_path: Path) -> Path:
    """The fixture's components, a shouter on local and a murmurer on fake."""
    components = {
        "shouter": {"backend": "local"},
        "murmurer": {"backend": "fake"},
    }
    return write_shout_workflow(
        tmp_path / "workflow.yaml", [SHOUT_EXTENSION], components
    )


def shout_settings(
    workflow: Path,
    row: dict[str, str],
    apps: dict[str, object] | None = None,
    **given: object,
) -> DeploySettings:
    """The test settings deploying alias ``shout``, ``row``, on ``workflow``."""
    own = {"workflow": {"settings": {"QP_WORKFLOW_FILE": str(workflow)}}}
    return make_settings(
        topology="shout",
        topology_alias={**TOPOLOGY_ALIASES, "shout": row},
        apps={**own, **(apps or {})},
        **given,
    )


def run_deploy(apps: FakeDominoApps, settings: DeploySettings) -> None:
    deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)


def refusal(settings: DeploySettings) -> str:
    apps = FakeDominoApps()
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings)
    assert apps.received == []  # not even a lookup reached Domino
    return str(caught.value)


def workflow_exports(apps: FakeDominoApps) -> list[str]:
    """The workflow App's computed exports: after its own settings, before the Agent's."""
    script = str(apps.creates[-1].version.entryScript)
    script = script.removesuffix(f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}")
    return script[script.index("export QP_ROLE=") :].splitlines()


def test_an_alias_can_serve_a_registered_component_in_its_own_app(
    shout_workflow: Path,
) -> None:
    settings = shout_settings(
        shout_workflow, {"shouter": "served"}, {"shouter": {"kind": "app"}}
    )
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, settings)
    assert [body.name for body in apps.creates] == ["rag-shouter", "rag-workflow"]
    model_server = str(apps.creates[0].version.entryScript)
    assert model_server.startswith(model_server_exports("shouter"))
    assert workflow_exports(apps) == [
        "export QP_ROLE=workflow",
        "export QP_TOPOLOGY_ALIAS=shout",
        "export QP_COMPONENTS__SHOUTER__BACKEND=served",
        f"export QP_COMPONENTS__SHOUTER__SERVED__URL={SHOUTER_CALL_URL}",
        "export QP_RUN_LABEL=shout+shouter=fastapi",
    ]


def test_an_alias_can_keep_a_registered_component_in_the_workflow_app(
    shout_workflow: Path,
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, shout_settings(shout_workflow, {"shouter": "local"}))
    assert [body.name for body in apps.creates] == ["rag-workflow"]
    assert "export QP_COMPONENTS__SHOUTER__BACKEND=local" in workflow_exports(apps)


def test_the_ovms_kind_refuses_a_registered_component(
    shout_workflow: Path, tmp_path: Path
) -> None:
    settings = shout_settings(
        shout_workflow,
        {"shouter": "served"},
        {"shouter": {"kind": "app", "server": "ovms"}},
        ovms_bundle=str(tmp_path),
    )
    assert refusal(settings) == (
        "topology alias 'shout' serves component 'shouter', of type 'shouter', "
        "with the model-server kind 'ovms' (apps.shouter.server, else "
        "defaults.server, in deploy.yaml), which cannot serve it: the kinds that "
        "can are 'fastapi'"
    )


def test_a_type_no_kind_serves_is_refused_naming_each_kind_and_what_it_serves(
    shout_workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every type with a served backend has a kind today, since only ovms serves
    # an LLM and fastapi every registered type: a fastapi that served no
    # registered type leaves the shouter with none.
    monkeypatch.setattr(
        FastApiModelServerKind, "serves_registered_types", property(lambda _: False)
    )
    assert refusal(shout_settings(shout_workflow, {"shouter": "served"})) == (
        "topology alias 'shout' serves component 'shouter', of type 'shouter', "
        "which no model-server kind can serve: fastapi serves embedder or "
        "reranker, ovms serves embedder or reranker or llm; leave 'shouter' out "
        "of the row to keep the backend the workflow file gives it"
    )


def test_a_backend_the_registered_type_lacks_is_refused(shout_workflow: Path) -> None:
    # A murmurer has a fake and the built served class, no local implementation.
    assert refusal(shout_settings(shout_workflow, {"murmurer": "local"})) == (
        "topology alias 'shout' in deploy.yaml puts component 'murmurer', of type "
        "'murmurer', on 'local', a backend its type does not have (its backends: "
        "'fake', 'served'): leave 'murmurer' out of the row to keep the backend "
        "the workflow file gives it"
    )
