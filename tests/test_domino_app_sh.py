"""domino/app.sh drives real binaries; here fake ``ovms``/``nginx``/``rag-*`` record argv."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

APP_SH = Path(__file__).resolve().parents[1] / "domino" / "app.sh"
FAKES = ("ovms", "nginx", "rag-serve", "rag-model-server")


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """Fake binaries that append their argv (and nginx its config) to <name>.log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in FAKES:
        script = bin_dir / name
        script.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$@" >> "{tmp_path}/{name}.log"\n'
            f'env | grep "^RAG_" | sort >> "{tmp_path}/{name}.env" || true\n'
            + (
                f'[ "$1" = -c ] && cp "$2" "{tmp_path}/nginx.conf"\n'
                if name == "nginx"
                else ""
            )
            + "exit 0\n"
        )
        script.chmod(0o755)
    return bin_dir


def run_app(
    tmp_path: Path, bin_dir: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", **env}
    return subprocess.run(
        ["bash", str(APP_SH)], env=full_env, capture_output=True, text=True, timeout=60
    )


def log(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.log"
    return path.read_text() if path.exists() else ""


def models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "config_reranker.json").write_text("{}")
    (models / "config_embedder.json").write_text("{}")
    return models


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
    assert result.returncode == 1 and "exited" in result.stderr
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
    assert conf.count("location ") == 1 and "location / {" in conf


def test_missing_ovms_config_fails_with_a_clear_message(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(
        tmp_path, bin_dir, RAG_ROLE="ovms-reranker", RAG_OVMS_MODELS_DIR=str(tmp_path)
    )
    assert result.returncode == 1 and "OVMS config not found" in result.stderr


def test_workflow_role_requires_the_reranker_url_and_derives_the_topology(
    tmp_path: Path, bin_dir: Path
) -> None:
    missing = run_app(tmp_path, bin_dir, RAG_ROLE="workflow")
    assert missing.returncode == 1 and "RAG_OVMS_RERANK_URL" in missing.stderr
    ok = run_app(
        tmp_path,
        bin_dir,
        RAG_ROLE="workflow",
        RAG_OVMS_RERANK_URL="http://a",
        RAG_EMBEDDER_BACKEND="ovms",
        RAG_OVMS_EMBEDDINGS_URL="http://c",
    )
    assert ok.returncode == 0, ok.stderr
    env = (tmp_path / "rag-serve.env").read_text()
    assert "RAG_TOPOLOGY=split-all" in env and "RAG_OVMS_AUTH=domino" in env
    assert log(tmp_path, "rag-serve").strip() == "--host 0.0.0.0 --port 8888"


def test_fastapi_roles_run_the_model_server(tmp_path: Path, bin_dir: Path) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE="fastapi-reranker")
    assert result.returncode == 0, result.stderr
    assert (
        log(tmp_path, "rag-model-server").strip()
        == "--serve reranker --host 0.0.0.0 --port 8888"
    )


def test_unknown_role_is_rejected(tmp_path: Path, bin_dir: Path) -> None:
    result = run_app(tmp_path, bin_dir, RAG_ROLE="bogus")
    assert result.returncode == 1 and "unknown RAG_ROLE" in result.stderr
