"""domino/app.sh drives real binaries; here fakes (``ovms``, ``nginx``, ``qp-*``,
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
    served_config,
    stand_in_env,
    started_env,
    started_in,
    wheels_env,
    write_fake,
)
from query_processing.domain.roles import Role
from repo_paths import CORE_ROOT, TOOLKIT_ROOT


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


def qp_serve_env(tmp_path: Path) -> list[str]:
    """The QP_* variables qp-serve was started with, one ``NAME=value`` each."""
    return started_env(tmp_path, "qp-serve")


# --- the workflow app --------------------------------------------------------------


def test_the_workflow_role_is_the_default_and_sets_nothing_of_its_own(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "qp-serve").strip() == "--host 0.0.0.0 --port 8888 --domino"
    # Python reads backends, URLs and auth from its own variables; app.sh adds none.
    assert qp_serve_env(tmp_path) == []


@pytest.mark.parametrize("alias", ["monolith", "split-all", "split-embedder"])
def test_the_workflow_role_passes_any_topology_alias_through_untouched(
    tmp_path: Path, bin_dir: Path, alias: str
) -> None:
    result = run_app(tmp_path, bin_dir, QP_ROLE="workflow", QP_TOPOLOGY_ALIAS=alias)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "qp-serve").strip() == "--host 0.0.0.0 --port 8888 --domino"
    expected = ["QP_ROLE=workflow", f"QP_TOPOLOGY_ALIAS={alias}"]
    assert qp_serve_env(tmp_path) == expected


def test_workflow_role_passes_backends_and_auth_through_untouched(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(
        tmp_path,
        bin_dir,
        QP_ROLE="workflow",
        QP_COMPONENTS__RERANKER__BACKEND="none",
        QP_COMPONENTS__RERANKER__SERVED__URL="http://a/v3/rerank",
    )

    assert result.returncode == 0, result.stderr
    env = qp_serve_env(tmp_path)
    assert "QP_COMPONENTS__RERANKER__BACKEND=none" in env
    assert "QP_COMPONENTS__RERANKER__SERVED__URL=http://a/v3/rerank" in env
    assert not any(line.startswith("QP_COMPONENTS__EMBEDDER__BACKEND=") for line in env)
    assert not any(
        line.startswith("QP_COMPONENTS__RERANKER__API__KEY__") for line in env
    )


# --- the roles, and a model server's component and kind ------------------------------


@pytest.mark.parametrize("role", get_args(Role))
def test_app_sh_starts_every_role_of_the_domain(
    tmp_path: Path, bin_dir: Path, role: str
) -> None:
    env = {**model_server_env("fastapi", "embedder"), "QP_ROLE": role}

    result = run_app(tmp_path, bin_dir, **env)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "qp-serve") + log(tmp_path, "qp-model-server") != ""


@pytest.mark.parametrize("role", ["bogus", "monolith", "ovms-reranker"])
def test_an_unknown_role_is_rejected(tmp_path: Path, bin_dir: Path, role: str) -> None:
    result = run_app(tmp_path, bin_dir, QP_ROLE=role)
    assert result.returncode == 1
    assert f"unknown QP_ROLE '{role}'; expected workflow or model-server" in (
        result.stderr
    )


def test_a_model_server_without_its_component_names_the_variable(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(
        tmp_path, bin_dir, QP_ROLE="model-server", QP_MODEL_SERVER_KIND="fastapi"
    )
    assert result.returncode == 1
    assert "QP_MODEL_SERVER_COMPONENT is not set" in result.stderr
    assert log(tmp_path, "qp-model-server") == ""


@pytest.mark.parametrize("kind", ["", "triton", "../app", "Fast API"])
def test_a_kind_without_a_launch_script_names_the_kinds_that_have_one(
    tmp_path: Path, bin_dir: Path, kind: str
) -> None:
    result = run_app(tmp_path, bin_dir, **model_server_env(kind, "reranker"))
    assert result.returncode == 1
    assert f"app.sh[model-server {kind} reranker]: QP_MODEL_SERVER_KIND='{kind}'" in (
        result.stderr
    )
    assert "expected one of: fastapi ovms vllm" in result.stderr
    assert (log(tmp_path, "qp-model-server"), log(tmp_path, "ovms")) == ("", "")


@pytest.mark.parametrize("component", ["reranker", "coarse"])
def test_a_fastapi_model_server_serves_the_component_it_is_named(
    tmp_path: Path, bin_dir: Path, component: str
) -> None:
    result = run_app(tmp_path, bin_dir, **model_server_env("fastapi", component))
    assert result.returncode == 0, result.stderr
    assert (
        log(tmp_path, "qp-model-server").strip()
        == f"--serve {component} --host 0.0.0.0 --port 8888"
    )


# --- the ovms kind ------------------------------------------------------------------


# A component of any name is served from its own config, not only reranker.
@pytest.mark.parametrize("component", ["reranker", "coarse"])
def test_an_ovms_model_server_serves_its_components_config_directly_by_default(
    tmp_path: Path, bin_dir: Path, component: str
) -> None:
    models = models_dir(tmp_path)
    (models / "config_coarse.json").write_text("{}")
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", component),
        QP_OVMS_MODELS_DIR=str(models),
    )
    assert result.returncode == 0, result.stderr
    config = served_config(tmp_path)
    assert log(tmp_path, "ovms").strip() == f"--rest_port 8888 --config_path {config}"
    assert log(tmp_path, "nginx") == ""
    flags_asked = log(tmp_path, "qp-ovms-config-server-flags").strip()
    assert flags_asked == f"--server-flags {component}"
    serving_dir = config.removesuffix(f"/config_{component}.json")
    assert log(tmp_path, "qp-ovms-config").strip() == (
        f"--serve {component} --models-dir {models} --serving-dir {serving_dir}"
    )
    assert serving_dir != str(models)  # a fresh folder of the App's own
    assert Path(serving_dir).is_dir()


def test_prefix_proxy_puts_nginx_in_front_of_ovms(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "embedder"),
        QP_OVMS_MODELS_DIR=str(models),
        QP_OVMS_PREFIX_PROXY="1",
        DOMINO_RUN_HOST_PATH="/apps/abc123/",
    )
    # The fakes exit at once, so app.sh reports the death and exits non-zero.
    assert result.returncode == 1
    assert "exited" in result.stderr
    config = served_config(tmp_path)
    assert config.endswith("/config_embedder.json")
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_bind_address 127.0.0.1 --rest_port 9000 --config_path {config}"
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
        QP_OVMS_MODELS_DIR=str(models),
        QP_OVMS_PREFIX_PROXY="1",
        DOMINO_RUN_HOST_PATH="",
    )
    conf = (tmp_path / "nginx.conf").read_text()
    assert conf.count("location ") == 1
    assert "location / {" in conf


# --- the bootstrap ------------------------------------------------------------------


def test_without_bootstrap_nothing_is_installed(tmp_path: Path, bin_dir: Path) -> None:
    models = models_dir(tmp_path)
    run_app(tmp_path, bin_dir, QP_ROLE="workflow")
    run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        QP_OVMS_MODELS_DIR=str(models),
    )
    assert (log(tmp_path, "uv"), log(tmp_path, "apt-get")) == ("", "")
    assert log(tmp_path, "qp-serve") != ""
    assert log(tmp_path, "ovms") != ""


@pytest.mark.parametrize(
    ("env", "server", "argv"),
    [
        ({"QP_ROLE": "workflow"}, "qp-serve", "--host 0.0.0.0 --port 8888 --domino"),
        (
            model_server_env("fastapi", "embedder"),
            "qp-model-server",
            "--serve embedder --host 0.0.0.0 --port 8888",
        ),
    ],
)
def test_bootstrap_installs_the_toolkit_in_a_venv_before_serving(
    tmp_path: Path, bin_dir: Path, env: dict[str, str], server: str, argv: str
) -> None:
    wheels = wheels_env(tmp_path)
    result = run_app(tmp_path, bin_dir, **env, QP_BOOTSTRAP="1", **wheels)
    assert result.returncode == 0, result.stderr
    venv, found = tmp_path / "home" / "qp-venv", wheels["QP_BOOTSTRAP_WHEELS"]
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",  # the project's Python, from .python-version
        f"pip install --python {venv}/bin/python --find-links {found} "
        f"{CORE_ROOT} {TOOLKIT_ROOT}",
    ]
    assert log(tmp_path, f"venv-{server}").strip() == argv
    assert log(tmp_path, server) == ""  # not the one already on PATH
    assert log(tmp_path, "apt-get") == ""


@pytest.mark.parametrize(
    "env", [{"QP_ROLE": "workflow"}, model_server_env("fastapi", "reranker")]
)
def test_bootstrap_installs_the_toolkit_then_the_use_case_in_one_command(
    tmp_path: Path, bin_dir: Path, env: dict[str, str]
) -> None:
    # No package index has the toolkit: the use case's dependency on it is
    # met only when both are named in the same install.
    use_case = tmp_path / "use case"
    use_case.mkdir()
    wheels = wheels_env(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        **env,
        QP_BOOTSTRAP="1",
        QP_BOOTSTRAP_EXTRAS="domino",
        QP_USE_CASE_DIR=str(use_case),
        **wheels,
    )
    assert result.returncode == 0, result.stderr
    venv, found = tmp_path / "home" / "qp-venv", wheels["QP_BOOTSTRAP_WHEELS"]
    assert log(tmp_path, "uv").splitlines()[1] == (
        f"pip install --python {venv}/bin/python --find-links {found} "
        f"{CORE_ROOT} {TOOLKIT_ROOT}[domino] {use_case}"
    )


# --- the use case's directory ---------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "server"),
    [
        ({"QP_ROLE": "workflow"}, "qp-serve"),
        (model_server_env("fastapi", "embedder"), "qp-model-server"),
        (model_server_env("ovms", "reranker"), "ovms"),
    ],
)
def test_the_roles_command_runs_in_the_use_cases_directory(
    tmp_path: Path, bin_dir: Path, env: dict[str, str], server: str
) -> None:
    # There it finds the use case's workflow file; the kinds' scripts are found
    # from app.sh's own place.
    use_case = tmp_path / "use case"
    use_case.mkdir()
    models = models_dir(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        **env,
        QP_USE_CASE_DIR=str(use_case),
        QP_OVMS_MODELS_DIR=str(models),
    )
    assert result.returncode == 0, result.stderr
    assert started_in(tmp_path, server) == use_case


def test_the_prefix_proxy_reads_the_toolkits_config_from_the_use_case(
    tmp_path: Path, bin_dir: Path
) -> None:
    use_case = tmp_path / "use case"
    use_case.mkdir()
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        QP_USE_CASE_DIR=str(use_case),
        QP_OVMS_MODELS_DIR=str(models_dir(tmp_path)),
        QP_OVMS_PREFIX_PROXY="1",
        DOMINO_RUN_HOST_PATH="/apps/abc123/",
    )
    # The fakes exit at once, so app.sh reports the death and exits non-zero.
    assert "exited" in result.stderr
    conf = (tmp_path / "nginx.conf").read_text()
    assert "location /apps/abc123/ { proxy_pass http://ovms/; }" in conf
    assert started_in(tmp_path, "nginx") == use_case


def test_without_a_use_case_the_command_runs_where_app_sh_started(
    tmp_path: Path, bin_dir: Path
) -> None:
    started = tmp_path / "started here"
    started.mkdir()
    result = subprocess.run(
        ["bash", str(APP_SH)],
        env=stand_in_env(tmp_path, bin_dir),
        cwd=started,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert started_in(tmp_path, "qp-serve") == started


def test_a_use_case_directory_that_is_missing_is_named(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(tmp_path, bin_dir, QP_USE_CASE_DIR=str(tmp_path / "gone"))
    assert result.returncode == 1
    assert (
        f"QP_USE_CASE_DIR='{tmp_path / 'gone'}' is no directory; expected the use "
        "case's directory, as qp-deploy sets it"
    ) in result.stderr
    assert log(tmp_path, "qp-serve") == ""


def test_bootstrap_runs_ovms_from_an_unpacked_bundle(
    tmp_path: Path, bin_dir: Path
) -> None:
    models, bundle = models_dir(tmp_path), ovms_bundle(tmp_path)
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        QP_BOOTSTRAP="1",
        QP_OVMS_BUNDLE=str(bundle),
        QP_OVMS_MODELS_DIR=str(models),
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
        f"--rest_port 8888 --config_path {served_config(tmp_path)}"
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
        QP_BOOTSTRAP="1",
        QP_OVMS_BUNDLE=str(tarball),
        QP_OVMS_MODELS_DIR=str(models),
    )
    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "bundle-ovms").strip() == (
        f"--rest_port 8888 --config_path {served_config(tmp_path)}"
    )
    assert served_config(tmp_path).endswith("/config_embedder.json")
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
        QP_BOOTSTRAP="1",
        QP_OVMS_BUNDLE=str(bundle),
        QP_OVMS_MODELS_DIR=str(models),
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
        QP_BOOTSTRAP="1",
        QP_OVMS_BUNDLE=str(bundle),
        QP_OVMS_MODELS_DIR=str(models),
        QP_OVMS_PREFIX_PROXY="1",
    )
    assert log(tmp_path, "apt-get").splitlines()[1] == (
        "install -y --no-install-recommends libxml2 curl nginx"
    )


@pytest.mark.parametrize("bundle", [None, "/no/such/ovms.tar.gz"])
def test_bootstrap_without_a_bundle_names_the_variable(
    tmp_path: Path, bin_dir: Path, bundle: str | None
) -> None:
    env = {"QP_OVMS_BUNDLE": bundle} if bundle else {}
    result = run_app(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", "reranker"),
        QP_BOOTSTRAP="1",
        **env,
    )
    assert result.returncode == 1
    assert "QP_OVMS_BUNDLE" in result.stderr
    assert log(tmp_path, "apt-get") == ""
    assert log(tmp_path, "ovms") == ""
