"""The join: every entry script rag-deploy writes, run through the real domino/app.sh.

The deploy tool and the launch script agree on variable names (``RAG_ROLE``,
``RAG_TOPOLOGY``, ``RAG_OVMS_BUNDLE``, ``RAG_BOOTSTRAP``, the model servers'
URLs) without sharing code. For each combination the deploy tool can produce
(a topology, one of its Apps, a model-server kind) the script it writes is run by
bash with the launch script's stand-in binaries, and the test asserts what
app.sh starts. The generator App is in every topology's combinations; only its
own script asks for it, so the other scripts are those of a deploy without one. What is expected is worked out from the combination alone, never
from a Role string, so a Role that app.sh does not handle, refuses for the
topology or maps to another server fails here rather than at App start on Domino.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import get_args

import pytest

from domino_deploy_helpers import make_settings
from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    models_dir,
    ovms_bundle,
    served_config,
    stand_in_env,
    started_env,
    write_fake,
)
from rag_load_test.deploy.publish import entry_script
from rag_load_test.domain.roles import (
    AppPart,
    GeneratorServer,
    ModelServerKind,
    apps_for_topology,
    model_servers_for_topology,
)
from rag_load_test.domain.topology import Topology

APP_URLS = {
    "embedder": "https://domino.test/apps/e1",
    "reranker": "https://domino.test/apps/r1",
    "generator": "https://domino.test/apps/g1",
}
COMBINATIONS = [
    (topology, part, kind)
    for topology in get_args(Topology)
    for part in apps_for_topology(topology, generator=True)
    for kind in get_args(ModelServerKind)
]
# What the rag-served-models stand-in prints: the generator and its OVMS options.
SERVED_GENERATOR = (
    "GENERATOR_SOURCE_MODEL=OpenVINO/Qwen3.5-2B-int4-ov",
    "GENERATOR_MODEL_NAME=qwen3.5-2b",
    "GENERATOR_DYNAMIC_SPLIT_FUSE=true",
    "GENERATOR_MAX_BATCHED_TOKENS=''",
)


def expected_start(
    part: AppPart, kind: ModelServerKind, tmp_path: Path
) -> tuple[str, str]:
    """``(stand-in, argv)`` app.sh must start for an App part and model-server kind.

    Every script bootstraps (``RAG_BOOTSTRAP=1``), so the project's commands run
    from the venv ``uv`` made and OVMS from the bundle, on the config the
    ``rag-ovms-config`` stand-in printed.
    """
    if part == "workflow":
        return "venv-rag-serve", "--host 0.0.0.0 --port 8888 --domino"
    if part == "generator":  # OVMS whatever the model servers' kind, no config
        return "bundle-ovms", (
            "--rest_port 8888 --source_model OpenVINO/Qwen3.5-2B-int4-ov "
            f"--model_name qwen3.5-2b --model_repository_path {tmp_path / 'models'} "
            "--task text_generation --target_device CPU --dynamic_split_fuse true"
        )
    if kind == "fastapi":
        return "venv-rag-model-server", f"--serve {part} --host 0.0.0.0 --port 8888"
    config = served_config(tmp_path)
    assert config.endswith(f"/config_{part}.json")
    return "bundle-ovms", f"--rest_port 8888 --config_path {config}"


def run_entry_script(
    tmp_path: Path,
    topology: Topology,
    part: AppPart,
    kind: ModelServerKind,
    generator: GeneratorServer = "none",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Write ``rag-<part>``'s entry script as rag-deploy would, then run it."""
    bin_dir, models = make_stand_ins(tmp_path), models_dir(tmp_path)
    served = "".join(f"echo {shlex.quote(line)}\n" for line in SERVED_GENERATOR)
    write_fake(bin_dir / "rag-served-models", tmp_path / "served.log", served)
    bundle = ovms_bundle(tmp_path)
    settings = make_settings(
        topology=topology,
        model_server=kind,
        ovms_bundle=str(bundle),
        generator_server=generator,
    )
    servers = apps_for_topology(topology, generator=generator != "none")[:-1]
    urls = {server: APP_URLS[server] for server in servers if part == "workflow"}
    script = entry_script(settings.for_app(part), settings, urls, str(APP_SH))
    env = stand_in_env(tmp_path, bin_dir, RAG_OVMS_MODELS_DIR=str(models))
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    return result, models


def test_every_role_the_deploy_tool_writes_is_exercised() -> None:
    assert {(t, p) for t, p, _ in COMBINATIONS} == {
        ("monolith", "workflow"),
        ("split-reranker", "reranker"),
        ("split-reranker", "workflow"),
        ("split-all", "embedder"),
        ("split-all", "reranker"),
        ("split-all", "workflow"),
        ("monolith", "generator"),
        ("split-reranker", "generator"),
        ("split-all", "generator"),
    }


@pytest.mark.parametrize(
    ("topology", "part", "kind"),
    COMBINATIONS,
    ids=[f"{t}-{p}-{k}" for t, p, k in COMBINATIONS],
)
def test_each_entry_script_starts_its_app_through_app_sh(
    tmp_path: Path, topology: Topology, part: AppPart, kind: ModelServerKind
) -> None:
    generator: GeneratorServer = "ovms" if part == "generator" else "none"
    result, _ = run_entry_script(tmp_path, topology, part, kind, generator)

    assert result.returncode == 0, result.stderr
    server, argv = expected_start(part, kind, tmp_path)
    assert log(tmp_path, server).strip() == argv
    if part == "workflow":
        assert f"RAG_TOPOLOGY={topology}" in started_env(tmp_path, server)


@pytest.mark.parametrize("topology", ["split-reranker", "split-all"])
def test_a_split_workflow_starts_with_the_url_of_each_model_server(
    tmp_path: Path, topology: Topology
) -> None:
    result, _ = run_entry_script(tmp_path, topology, "workflow", "fastapi")

    assert result.returncode == 0, result.stderr
    started = started_env(tmp_path, "venv-rag-serve")
    expected = {
        "embedder": f"RAG_EMBEDDING_API_URL={APP_URLS['embedder']}/v3/embeddings",
        "reranker": f"RAG_RERANK_API_URL={APP_URLS['reranker']}/v3/rerank",
    }
    servers = model_servers_for_topology(topology)
    assert [expected[server] in started for server in servers] == [True] * len(servers)


@pytest.mark.parametrize("topology", get_args(Topology))
def test_a_workflow_with_a_generator_starts_with_its_chat_route(
    tmp_path: Path, topology: Topology
) -> None:
    result, _ = run_entry_script(tmp_path, topology, "workflow", "fastapi", "ovms")

    assert result.returncode == 0, result.stderr
    started = started_env(tmp_path, "venv-rag-serve")
    route = f"{APP_URLS['generator']}/v3/chat/completions"
    assert {
        "RAG_GENERATION_BACKEND=api",
        f"RAG_GENERATION_API_URL={route}",
        "RAG_GENERATION_API_KEY=domino",
    } <= set(started)
