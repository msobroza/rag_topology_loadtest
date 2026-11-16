"""The model-server kinds: each one a registered class and a launch script.

What a kind declares is checked against what serves it: every registered kind
has its launch script and every script a registered kind, each script defines
the two functions app.sh calls, and the FastAPI kind's routes and probe are
where ``qp-model-server`` answers. Then the deploy settings: the kind of each
model server App, its own or the default.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.model_server_kinds.base_model_server_kind import (
    MODEL_SERVER_KINDS,
    model_server_kind,
)
from query_processing.entrypoints.qp_model_server import create_model_server
from repo_paths import TOOLKIT_ROOT

LAUNCH_SCRIPTS = TOOLKIT_ROOT / "domino" / "model_servers"
# What the FastAPI kind's test requests carry, by component type.
BODIES: dict[str, dict[str, object]] = {
    "embedder": {"input": ["a b"]},
    "reranker": {"query": "q", "documents": ["a", "b"]},
}


def test_each_registered_kind_has_a_launch_script_and_each_script_a_kind() -> None:
    scripts = sorted(path.stem for path in LAUNCH_SCRIPTS.glob("*.sh"))
    assert scripts == list(MODEL_SERVER_KINDS.names()) == ["fastapi", "ovms"]


@pytest.mark.parametrize("kind", MODEL_SERVER_KINDS.names())
def test_each_launch_script_defines_the_bootstrap_and_the_command(kind: str) -> None:
    script = LAUNCH_SCRIPTS / f"{kind}.sh"
    check = f'. "{script}"; declare -F bootstrap_model_server serve_model_server'
    result = subprocess.run(
        ["bash", "-c", check], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["bootstrap_model_server", "serve_model_server"]


@pytest.mark.parametrize("kind", MODEL_SERVER_KINDS.names())
def test_each_kind_serves_an_embedder_and_a_reranker_and_no_llm(kind: str) -> None:
    served = model_server_kind(kind)
    assert [served.serves(t) for t in ("embedder", "reranker", "llm")] == [
        True,
        True,
        False,
    ]
    assert served.route_of("embedder") == "/v3/embeddings"
    assert served.route_of("reranker") == "/v3/rerank"


def test_a_type_a_kind_does_not_serve_has_no_route() -> None:
    with pytest.raises(ValueError, match="no component of type") as caught:
        model_server_kind("ovms").route_of("llm")
    assert str(caught.value) == (
        "this model-server kind serves no component of type 'llm' "
        "(expected one of: 'embedder', 'reranker')"
    )


def test_only_the_fastapi_kind_serves_a_registered_type_on_its_call_route() -> None:
    import_module("shout_extension.shouting")  # registers shouter, as a file would
    fastapi, ovms = model_server_kind("fastapi"), model_server_kind("ovms")
    assert (fastapi.serves("shouter"), ovms.serves("shouter")) == (True, False)
    assert fastapi.route_of("shouter") == "/call/shouter"
    # The LLM registers in COMPONENT_TYPES too, but it has openers: no call route.
    assert not fastapi.serves("llm")
    with pytest.raises(ValueError, match="'reranker' or a registered type"):
        fastapi.route_of("llm")


def test_an_unknown_kind_names_every_registered_one() -> None:
    with pytest.raises(ValueError, match="unknown model-server kind") as caught:
        model_server_kind("vllm")
    assert str(caught.value) == (
        "unknown model-server kind 'vllm' (expected one of: 'fastapi', 'ovms')"
    )


@pytest.mark.parametrize("component_type", ["embedder", "reranker"])
def test_qp_model_server_answers_on_the_routes_and_probe_of_the_fastapi_kind(
    settings_factory: Callable[..., WorkflowSettings], component_type: str
) -> None:
    settings = settings_factory()
    alias = settings.components[component_type].alias
    composition = CompositionRoot(
        settings, embedder=FakeEmbedder(dim=4), reranker=FakeReranker()
    )
    kind = model_server_kind("fastapi")

    with TestClient(create_model_server(component_type, composition)) as client:
        body = {"model": alias, **BODIES[component_type]}
        answer = client.post(kind.route_of(component_type), json=body)
        probe = client.get(kind.probe_route.format(name=alias))

    assert (answer.status_code, probe.status_code) == (200, 200)


# --- the model-server kind of each App in the deploy settings ----------------------


@pytest.fixture
def deploy_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The path QP_DEPLOY_FILE names; no deploy.yaml of the shell is read.

    The prefix every deploy file must name is given by its variable.
    """
    path = tmp_path / "deploy.yaml"
    monkeypatch.setenv("QP_DEPLOY_FILE", str(path))
    monkeypatch.setenv("QP_DEPLOY_APP_NAME_PREFIX", "rag")
    return path


@pytest.mark.parametrize(
    ("text", "servers"),
    [
        ("defaults: {environment_id: env-1}\n", ("fastapi", "fastapi")),
        ("defaults: {server: ovms}\n", ("ovms", "ovms")),
        ("apps: {reranker: {server: ovms}}\n", ("fastapi", "ovms")),
        (
            "defaults: {server: ovms}\napps: {embedder: {server: fastapi}}\n",
            ("fastapi", "ovms"),
        ),
    ],
)
def test_each_model_server_app_takes_its_own_kind_else_the_default(
    deploy_file: Path, text: str, servers: tuple[str, str]
) -> None:
    deploy_file.write_text(text)
    settings = DeploySettings()
    assert (
        settings.for_app("embedder").server,
        settings.for_app("reranker").server,
    ) == servers


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("defaults: {server: vllm}\n", "defaults.server"),
        ("apps: {reranker: {server: OVMS}}\n", "apps.reranker.server"),
    ],
)
def test_a_kind_no_model_server_is_registered_under_is_refused_naming_the_kinds(
    deploy_file: Path, text: str, where: str
) -> None:
    deploy_file.write_text(text)
    with pytest.raises(ValidationError) as caught:
        DeploySettings()
    message = str(caught.value)
    assert where in message
    assert "unknown model-server kind" in message
    assert "(expected one of: 'fastapi', 'ovms')" in message


def test_a_kind_for_the_workflow_app_is_refused(deploy_file: Path) -> None:
    deploy_file.write_text("apps: {workflow: {server: ovms}}\n")
    with pytest.raises(ValidationError) as caught:
        DeploySettings()
    assert (
        "apps.workflow.server is 'ovms', but the workflow App is no model server: "
        "set server in the block of a served component, or in defaults for every "
        "model server App"
    ) in str(caught.value)
