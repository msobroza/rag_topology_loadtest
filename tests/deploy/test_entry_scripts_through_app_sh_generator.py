"""The join with a generator: the scripts rag-deploy writes when one is asked for,
run through the real domino/app.sh.

``test_entry_scripts_through_app_sh.py`` proves a deploy with no generator starts
its Apps as before; this file covers a deploy with ``generator_server: ovms``, in
every topology: the generator's own script, which must reach app.sh's
``ovms-generator`` arm, and the workflow's script, which must start with the
generator's chat route. A ``rag-served-models`` stand-in prints the generator's
model and options, as the real command would.
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
    stand_in_env,
    started_env,
    write_fake,
)
from rag_load_test.deploy.publish import entry_script
from rag_load_test.domain.roles import AppPart, Role, apps_for_topology
from rag_load_test.domain.topology import Topology

APP_URLS = {
    "embedder": "https://domino.test/apps/e1",
    "reranker": "https://domino.test/apps/r1",
    "generator": "https://domino.test/apps/g1",
}
# The parts whose script a generator changes: its own, and the workflow's.
GENERATOR_COMBINATIONS: list[tuple[Topology, AppPart]] = [
    (topology, part)
    for topology in get_args(Topology)
    for part in apps_for_topology(topology, generator=True)
    if part in ("generator", "workflow")
]
# What the rag-served-models stand-in prints: the generator and its OVMS options.
SERVED_GENERATOR = (
    "GENERATOR_SOURCE_MODEL=OpenVINO/Qwen3.5-2B-int4-ov",
    "GENERATOR_MODEL_NAME=qwen3.5-2b",
    "GENERATOR_DYNAMIC_SPLIT_FUSE=true",
    "GENERATOR_MAX_BATCHED_TOKENS=''",
)


def write_served_models_stand_in(bin_dir: Path, tmp_path: Path) -> None:
    """A ``rag-served-models`` on PATH that prints ``SERVED_GENERATOR``."""
    printed = "".join(f"echo {shlex.quote(line)}\n" for line in SERVED_GENERATOR)
    write_fake(bin_dir / "rag-served-models", tmp_path / "served.log", printed)


def run_generator_deploy_script(
    tmp_path: Path, topology: Topology, part: AppPart
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Write ``rag-<part>``'s script for a deploy with a generator, then run it."""
    bin_dir, models = make_stand_ins(tmp_path), models_dir(tmp_path)
    write_served_models_stand_in(bin_dir, tmp_path)
    settings = make_settings(
        topology=topology,
        generator_server="ovms",
        ovms_bundle=str(ovms_bundle(tmp_path)),
    )
    servers = apps_for_topology(topology, generator=True)[:-1]
    urls = {server: APP_URLS[server] for server in servers if part == "workflow"}
    script = entry_script(settings.for_app(part), settings, urls, str(APP_SH))
    env = stand_in_env(tmp_path, bin_dir, RAG_OVMS_MODELS_DIR=str(models))
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    return result, models


def exported_role(topology: Topology, part: AppPart) -> str:
    """The ``RAG_ROLE`` that ``rag-<part>``'s script exports with a generator."""
    settings = make_settings(topology=topology, generator_server="ovms")
    urls = {server: APP_URLS[server] for server in APP_URLS}
    script = entry_script(settings.for_app(part), settings, urls, str(APP_SH))
    return next(
        line.removeprefix("export RAG_ROLE=")
        for line in script.splitlines()
        if line.startswith("export RAG_ROLE=")
    )


def test_every_topology_with_a_generator_is_exercised() -> None:
    assert set(GENERATOR_COMBINATIONS) == {
        (topology, part)
        for topology in get_args(Topology)
        for part in ("generator", "workflow")
    }
    roles = {exported_role(t, p) for t, p in GENERATOR_COMBINATIONS}
    generator: Role = "ovms-generator"
    assert generator in roles


@pytest.mark.parametrize(
    ("topology", "part"),
    GENERATOR_COMBINATIONS,
    ids=[f"{t}-{p}" for t, p in GENERATOR_COMBINATIONS],
)
def test_each_script_of_a_deploy_with_a_generator_starts_through_app_sh(
    tmp_path: Path, topology: Topology, part: AppPart
) -> None:
    result, models = run_generator_deploy_script(tmp_path, topology, part)

    assert result.returncode == 0, result.stderr
    if part == "generator":  # the bundle's OVMS pulls the model, no config
        assert log(tmp_path, "bundle-ovms").strip() == (
            "--rest_port 8888 --source_model OpenVINO/Qwen3.5-2B-int4-ov "
            f"--model_name qwen3.5-2b --model_repository_path {models} "
            "--task text_generation --target_device CPU --dynamic_split_fuse true"
        )
        return
    assert log(tmp_path, "venv-rag-serve").strip() == (
        "--host 0.0.0.0 --port 8888 --domino"
    )
    route = f"{APP_URLS['generator']}/v3/chat/completions"
    assert {
        f"RAG_TOPOLOGY={topology}",
        "RAG_GENERATION_BACKEND=api",
        f"RAG_GENERATION_API_URL={route}",
        "RAG_GENERATION_API_KEY=domino",
    } <= set(started_env(tmp_path, "venv-rag-serve"))
