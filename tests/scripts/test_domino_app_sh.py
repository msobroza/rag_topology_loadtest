"""domino/app.sh drives real binaries; here fakes (``ovms``, ``nginx``, ``rag-*``,
``uv``, ``sudo``, ``apt-get``; tests/helpers/launch_script_stand_ins.py) record their
argv instead, so nothing is installed.
Two roles: the workflow app, and a model server of one component, started by the
launch script of its model-server kind (domino/model_servers/<kind>.sh).
Every OVMS model server here finds its config; a missing one makes it run the real
ovms/export_models.sh, so that case lives in test_domino_app_sh_provisioning.py,
against a copy of the scripts beside a fake exporter."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path
from typing import get_args

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    model_server_env,
    models_dir,
    ovms_bundle,
    stand_in_env,
    started_env,
    write_fake,
)
from rag_load_test.domain.roles import Role


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """Fake binaries that append their argv (and nginx its config) to <name>.log."""
    return make_stand_ins(tmp_path)


def run_app(
    tmp_path: Path, bin_dir: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(APP_SH)],
        env=stand_in_env(tmp_path, bin_dir, **env),
        capture_output=True,
        text=True,
        timeout=60,
    )


def rag_serve_env(tmp_path: Path) -> list[str]:
    """The RAG_* variables rag-serve was started with, one ``NAME=value`` each."""
    return started_env(tmp_path, "rag-serve")


# --- the workflow app --------------------------------------------------------------


def test_the_workflow_role_is_the_default_and_sets_nothing_of_its_own(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "rag-serve").strip() == "--host 0.0.0.0 --port 8888 --domino"
    # Python reads backends, URLs and auth from its own variables; app.sh adds none.
    assert rag_serve_env(tmp_path) == []


@pytest.mark.parametrize("alias", ["monolith", "split-all", "split-embedder"])
def test_the_workflow_role_passes_any_topology_alias_through_untouched(
    tmp_path: Path, bin_dir: Path, alias: str
) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE="workflow", RAG_TOPOLOGY_ALIAS=alias)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "rag-serve").strip() == "--host 0.0.0.0 --port 8888 --domino"
    expected = ["RAG_ROLE=workflow", f"RAG_TOPOLOGY_ALIAS={alias}"]
    assert rag_serve_env(tmp_path) == expected


def test_workflow_role_passes_backends_and_auth_through_untouched(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(
        tmp_path,
        bin_dir,
        RAG_ROLE="workflow",
        RAG_COMPONENTS__RERANKER__BACKEND="none",
        RAG_COMPONENTS__RERANKER__SERVED__URL="http://a/v3/rerank",
    )

    assert result.returncode == 0, result.stderr
    env = rag_serve_env(tmp_path)
    assert "RAG_COMPONENTS__RERANKER__BACKEND=none" in env
    assert "RAG_COMPONENTS__RERANKER__SERVED__URL=http://a/v3/rerank" in env
    assert not any(
        line.startswith("RAG_COMPONENTS__EMBEDDER__BACKEND=") for line in env
    )
    assert not any(
        line.startswith("RAG_COMPONENTS__RERANKER__API__API_KEY=") for line in env
    )


# --- the roles, and a model server's component and kind ------------------------------


@pytest.mark.parametrize("role", get_args(Role))
def test_app_sh_starts_every_role_of_the_domain(
    tmp_path: Path, bin_dir: Path, role: str
) -> None:
    env = {**model_server_env("fastapi", "embedder"), "RAG_ROLE": role}

    result = run_app(tmp_path, bin_dir, **env)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "rag-serve") + log(tmp_path, "rag-model-server") != ""


@pytest.mark.parametrize("role", ["bogus", "monolith", "ovms-reranker"])
def test_an_unknown_role_is_rejected(tmp_path: Path, bin_dir: Path, role: str) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE=role)
    assert result.returncode == 1
    assert f"unknown RAG_ROLE '{role}'; expected workflow or model-server" in (
        result.stderr
    )


def test_a_model_server_without_its_component_names_the_variable(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(
        tmp_path, bin_dir, RAG_ROLE="model-server", RAG_MODEL_SERVER_KIND="fastapi"
    )
    assert result.returncode == 1
    assert "RAG_MODEL_SERVER_COMPONENT is not set" in result.stderr
    assert log(tmp_path, "rag-model-server") == ""


@pytest.mark.parametrize("kind", ["", "vllm", "../app", "Fast API"])
def test_a_kind_without_a_launch_script_names_the_kinds_that_have_one(
    tmp_path: Path, bin_dir: Path, kind: str
) -> None:
    result = run_app(tmp_path, bin_dir, **model_server_env(kind, "reranker"))
    assert result.returncode == 1
    assert f"app.sh[model-server {kind} reranker]: RAG_MODEL_SERVER_KIND='{kind}'" in (
        result.stderr
    )
    assert "expected one of: fastapi ovms" in result.stderr
    assert (log(tmp_path, "rag-model-server"), log(tmp_path, "ovms")) == ("", "")


@pytest.mark.parametrize("component", ["reranker", "coarse"])
def test_a_fastapi_model_server_serves_the_component_it_is_named(
    tmp_path: Path, bin_dir: Path, component: str
) -> None:
    result = run_app(tmp_path, bin_dir, **model_server_env("fastapi", component))
    assert result.returncode == 0, result.stderr
    assert (
        log(tmp_path, "rag-model-server").strip()
        == f"--serve {component} --host 0.0.0.0 --port 8888"
    )


# --- the ovms kind ------------------------------------------------------------------


def test_an_ovms_model_server_serves_its_components_config_directly_by_default(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        RAG_OVMS_MODELS_DIR=str(models),
    )
    assert result.returncode == 0, result.stderr
    assert (
        log(tmp_path, "ovms").strip()
        == f"--rest_port 8888 --config_path {models}/config_reranker.json"
    )
    assert log(tmp_path, "nginx") == ""


def test_an_ovms_model_server_of_any_name_serves_its_own_config(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    (models / "config_coarse.json").write_text("{}")

    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "coarse"),
        RAG_OVMS_MODELS_DIR=str(models),
    )

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_port 8888 --config_path {models}/config_coarse.json"
    )


def test_prefix_proxy_puts_nginx_in_front_of_ovms(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "embedder"),
        RAG_OVMS_MODELS_DIR=str(models),
        RAG_OVMS_PREFIX_PROXY="1",
        DOMINO_RUN_HOST_PATH="/apps/abc123/",
    )
    # The fakes exit at once, so app.sh reports the death and exits non-zero.
    assert result.returncode == 1
    assert "exited" in result.stderr
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_bind_address 127.0.0.1 --rest_port 9000 --config_path {models}/config_embedder.json"
    )
    assert log(tmp_path, "nginx").strip() == "-c /tmp/ovms-proxy.conf -g daemon off;"
    conf = (tmp_path / "nginx.conf").read_text()
    assert "listen 0.0.0.0:8888;" in conf
    assert "location /apps/abc123/ { proxy_pass http://ovms/; }" in conf
    assert "location / { proxy_pass http://ovms; }" in conf
    assert "server 127.0.0.1:9000; keepalive 32;" in conf
    assert "pid /tmp/ovms-proxy.pid;" in conf


def test_prefix_proxy_without_a_prefix_only_forwards_root(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        RAG_OVMS_MODELS_DIR=str(models),
        RAG_OVMS_PREFIX_PROXY="1",
        DOMINO_RUN_HOST_PATH="",
    )
    conf = (tmp_path / "nginx.conf").read_text()
    assert conf.count("location ") == 1
    assert "location / {" in conf


# --- the bootstrap ------------------------------------------------------------------


def test_without_bootstrap_nothing_is_installed(tmp_path: Path, bin_dir: Path) -> None:
    models = models_dir(tmp_path)
    run_app(tmp_path, bin_dir, RAG_ROLE="workflow")
    run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        RAG_OVMS_MODELS_DIR=str(models),
    )
    assert (log(tmp_path, "uv"), log(tmp_path, "apt-get")) == ("", "")
    assert log(tmp_path, "rag-serve") != ""
    assert log(tmp_path, "ovms") != ""


@pytest.mark.parametrize(
    ("env", "server", "argv"),
    [
        ({"RAG_ROLE": "workflow"}, "rag-serve", "--host 0.0.0.0 --port 8888 --domino"),
        (
            model_server_env("fastapi", "embedder"),
            "rag-model-server",
            "--serve embedder --host 0.0.0.0 --port 8888",
        ),
    ],
)
def test_bootstrap_installs_the_project_in_a_venv_before_serving(
    tmp_path: Path, bin_dir: Path, env: dict[str, str], server: str, argv: str
) -> None:
    result = run_app(tmp_path, bin_dir, **env, RAG_BOOTSTRAP="1")
    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "rag-venv"
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",  # the project's Python, from .python-version
        f"pip install --python {venv}/bin/python .",
    ]
    assert log(tmp_path, f"venv-{server}").strip() == argv
    assert log(tmp_path, server) == ""  # not the one already on PATH
    assert log(tmp_path, "apt-get") == ""


@pytest.mark.parametrize(
    "env", [{"RAG_ROLE": "workflow"}, model_server_env("fastapi", "reranker")]
)
def test_bootstrap_installs_the_extras_an_agent_asks_for(
    tmp_path: Path, bin_dir: Path, env: dict[str, str]
) -> None:
    result = run_app(
        tmp_path, bin_dir, **env, RAG_BOOTSTRAP="1", RAG_BOOTSTRAP_EXTRAS="domino"
    )
    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "rag-venv"
    assert log(tmp_path, "uv").splitlines()[1] == (
        f"pip install --python {venv}/bin/python .[domino]"
    )


def test_bootstrap_runs_ovms_from_an_unpacked_bundle(
    tmp_path: Path, bin_dir: Path
) -> None:
    models, bundle = models_dir(tmp_path), ovms_bundle(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        RAG_BOOTSTRAP="1",
        RAG_OVMS_BUNDLE=str(bundle),
        RAG_OVMS_MODELS_DIR=str(models),
    )
    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "sudo").splitlines() == [
        "apt-get update",
        "apt-get install -y --no-install-recommends libxml2 curl",
    ]
    assert log(tmp_path, "apt-get").splitlines() == [  # sudo ran it
        "update",
        "install -y --no-install-recommends libxml2 curl",
    ]
    assert log(tmp_path, "bundle-ovms").strip() == (
        f"--rest_port 8888 --config_path {models}/config_reranker.json"
    )
    assert log(tmp_path, "ovms") == ""
    ld_path = (tmp_path / "bundle-ovms.ld").read_text().strip()
    assert ld_path.split(":")[0] == f"{bundle}/lib"
    assert log(tmp_path, "uv") == ""


def test_bootstrap_unpacks_a_bundle_tarball(tmp_path: Path, bin_dir: Path) -> None:
    models, bundle = models_dir(tmp_path), ovms_bundle(tmp_path)
    tarball = tmp_path / "ovms.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(bundle, arcname="ovms")
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "embedder"),
        RAG_BOOTSTRAP="1",
        RAG_OVMS_BUNDLE=str(tarball),
        RAG_OVMS_MODELS_DIR=str(models),
    )
    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "bundle-ovms").strip() == (
        f"--rest_port 8888 --config_path {models}/config_embedder.json"
    )
    ld_path = (tmp_path / "bundle-ovms.ld").read_text().strip()
    assert ld_path.split(":")[0].endswith("/ovms/lib")
    assert not ld_path.startswith(str(bundle))  # the unpacked copy, not the source


def test_bootstrap_runs_apt_get_directly_as_root(tmp_path: Path, bin_dir: Path) -> None:
    write_fake(bin_dir / "id", tmp_path / "id.log", "echo 0\n")
    models, bundle = models_dir(tmp_path), ovms_bundle(tmp_path)
    run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        RAG_BOOTSTRAP="1",
        RAG_OVMS_BUNDLE=str(bundle),
        RAG_OVMS_MODELS_DIR=str(models),
    )
    assert log(tmp_path, "sudo") == ""
    assert len(log(tmp_path, "apt-get").splitlines()) == 2


def test_bootstrap_installs_nginx_only_with_the_prefix_proxy(
    tmp_path: Path, bin_dir: Path
) -> None:
    models, bundle = models_dir(tmp_path), ovms_bundle(tmp_path)
    run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        RAG_BOOTSTRAP="1",
        RAG_OVMS_BUNDLE=str(bundle),
        RAG_OVMS_MODELS_DIR=str(models),
        RAG_OVMS_PREFIX_PROXY="1",
    )
    assert log(tmp_path, "apt-get").splitlines()[1] == (
        "install -y --no-install-recommends libxml2 curl nginx"
    )


@pytest.mark.parametrize("bundle", [None, "/no/such/ovms.tar.gz"])
def test_bootstrap_without_a_bundle_names_the_variable(
    tmp_path: Path, bin_dir: Path, bundle: str | None
) -> None:
    env = {"RAG_OVMS_BUNDLE": bundle} if bundle else {}
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        RAG_BOOTSTRAP="1",
        **env,
    )
    assert result.returncode == 1
    assert "RAG_OVMS_BUNDLE" in result.stderr
    assert log(tmp_path, "apt-get") == ""
    assert log(tmp_path, "ovms") == ""
