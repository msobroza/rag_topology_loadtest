"""domino/app.sh drives real binaries; here fakes (``ovms``, ``nginx``, ``rag-*``,
``uv``, ``sudo``, ``apt-get``; tests/helpers/launch_script_stand_ins.py) record their
argv instead, so nothing is installed.
Every OVMS role here finds its config; a missing one makes app.sh run the real
ovms/export_models.sh, so that case lives in test_domino_app_sh_provisioning.py,
against a copy of app.sh beside a fake exporter."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    models_dir,
    ovms_bundle,
    stand_in_env,
    started_env,
    write_fake,
)


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


def test_ovms_role_serves_directly_by_default(tmp_path: Path, bin_dir: Path) -> None:
    models = models_dir(tmp_path)
    result = run_app(
        tmp_path, bin_dir, RAG_ROLE="ovms-reranker", RAG_OVMS_MODELS_DIR=str(models)
    )
    assert result.returncode == 0, result.stderr
    assert (
        log(tmp_path, "ovms").strip()
        == f"--rest_port 8888 --config_path {models}/config_reranker.json"
    )
    assert log(tmp_path, "nginx") == ""


def test_prefix_proxy_puts_nginx_in_front_of_ovms(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        RAG_ROLE="ovms-embedder",
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
        RAG_ROLE="ovms-reranker",
        RAG_OVMS_MODELS_DIR=str(models),
        RAG_OVMS_PREFIX_PROXY="1",
        DOMINO_RUN_HOST_PATH="",
    )
    conf = (tmp_path / "nginx.conf").read_text()
    assert conf.count("location ") == 1
    assert "location / {" in conf


def test_monolith_role_is_the_default_and_serves_the_monolith(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "rag-serve").strip() == "--host 0.0.0.0 --port 8888 --domino"
    # Python resolves backends and auth from the topology; app.sh sets nothing else.
    assert rag_serve_env(tmp_path) == ["RAG_TOPOLOGY=monolith"]


@pytest.mark.parametrize(
    ("role", "topology", "served"),
    [
        ("monolith", "monolith", "monolith"),
        ("workflow", None, "split-reranker"),
        ("workflow", "split-reranker", "split-reranker"),
        ("workflow", "split-all", "split-all"),
    ],
)
def test_workflow_app_roles_pass_their_topology_to_rag_serve(
    tmp_path: Path, bin_dir: Path, role: str, topology: str | None, served: str
) -> None:
    env = {"RAG_ROLE": role} | ({"RAG_TOPOLOGY": topology} if topology else {})

    result = run_app(tmp_path, bin_dir, **env)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "rag-serve").strip() == "--host 0.0.0.0 --port 8888 --domino"
    assert f"RAG_TOPOLOGY={served}" in rag_serve_env(tmp_path)


@pytest.mark.parametrize(
    ("role", "topology", "allowed"),
    [
        ("monolith", "split-all", "monolith"),
        ("workflow", "monolith", "split-reranker split-all"),
        ("workflow", "split-reranker-fastapi", "split-reranker split-all"),
    ],
)
def test_role_refuses_a_topology_it_does_not_serve(
    tmp_path: Path, bin_dir: Path, role: str, topology: str, allowed: str
) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE=role, RAG_TOPOLOGY=topology)

    assert result.returncode == 1
    assert f"RAG_TOPOLOGY='{topology}'" in result.stderr
    assert f"allowed: {allowed}" in result.stderr
    assert log(tmp_path, "rag-serve") == ""


def test_workflow_role_passes_backends_and_auth_through_untouched(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(
        tmp_path,
        bin_dir,
        RAG_ROLE="workflow",
        RAG_RERANK_BACKEND="none",
        RAG_RERANK_API_URL="http://a/v3/rerank",
    )

    assert result.returncode == 0, result.stderr
    env = rag_serve_env(tmp_path)
    assert "RAG_RERANK_BACKEND=none" in env
    assert "RAG_RERANK_API_URL=http://a/v3/rerank" in env
    assert not any(line.startswith("RAG_EMBEDDING_BACKEND=") for line in env)
    assert not any(line.startswith("RAG_RERANK_API_KEY=") for line in env)


def test_fastapi_roles_run_the_model_server(tmp_path: Path, bin_dir: Path) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE="fastapi-reranker")
    assert result.returncode == 0, result.stderr
    assert (
        log(tmp_path, "rag-model-server").strip()
        == "--serve reranker --host 0.0.0.0 --port 8888"
    )


def test_unknown_role_is_rejected(tmp_path: Path, bin_dir: Path) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE="bogus")
    assert result.returncode == 1
    assert "unknown RAG_ROLE" in result.stderr


def test_without_bootstrap_nothing_is_installed(tmp_path: Path, bin_dir: Path) -> None:
    models = models_dir(tmp_path)
    run_app(tmp_path, bin_dir, RAG_ROLE="workflow")
    run_app(
        tmp_path, bin_dir, RAG_ROLE="ovms-reranker", RAG_OVMS_MODELS_DIR=str(models)
    )
    assert (log(tmp_path, "uv"), log(tmp_path, "apt-get")) == ("", "")
    assert log(tmp_path, "rag-serve") != ""
    assert log(tmp_path, "ovms") != ""


@pytest.mark.parametrize(
    ("role", "server", "argv"),
    [
        ("workflow", "rag-serve", "--host 0.0.0.0 --port 8888 --domino"),
        (
            "fastapi-embedder",
            "rag-model-server",
            "--serve embedder --host 0.0.0.0 --port 8888",
        ),
    ],
)
def test_bootstrap_installs_the_project_in_a_venv_before_serving(
    tmp_path: Path, bin_dir: Path, role: str, server: str, argv: str
) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE=role, RAG_BOOTSTRAP="1")
    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "rag-venv"
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",  # the project's Python, from .python-version
        f"pip install --python {venv}/bin/python .",
    ]
    assert log(tmp_path, f"venv-{server}").strip() == argv
    assert log(tmp_path, server) == ""  # not the one already on PATH
    assert log(tmp_path, "apt-get") == ""


def test_bootstrap_installs_the_extras_an_agent_asks_for(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(
        tmp_path,
        bin_dir,
        RAG_ROLE="workflow",
        RAG_BOOTSTRAP="1",
        RAG_BOOTSTRAP_EXTRAS="domino",
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
        RAG_ROLE="ovms-reranker",
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
        RAG_ROLE="ovms-embedder",
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
        RAG_ROLE="ovms-reranker",
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
        RAG_ROLE="ovms-reranker",
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
        tmp_path, bin_dir, RAG_ROLE="ovms-reranker", RAG_BOOTSTRAP="1", **env
    )
    assert result.returncode == 1
    assert "RAG_OVMS_BUNDLE" in result.stderr
    assert log(tmp_path, "apt-get") == ""
    assert log(tmp_path, "ovms") == ""
