"""The join: every entry script rag-deploy writes, run through the real domino/app.sh.

The deploy tool and the launch script agree on variable names (``RAG_ROLE``,
``RAG_TOPOLOGY``, ``RAG_OVMS_BUNDLE``, ``RAG_BOOTSTRAP``, each component's
backend and served URL) without sharing code. For each combination the deploy
tool can produce (a topology alias, one of its Apps, a model-server kind) the
script it writes is run by bash with the launch script's stand-in binaries, and
the test asserts what app.sh starts. What is expected is worked out from the
combination alone, never from a Role string, so a Role that app.sh does not
handle, refuses for the topology or maps to another server fails here rather
than at App start on Domino. The aliases are the three of deploy.example.yaml
and a fourth that the workflow app's own topology rule does not know.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import get_args

import pytest

from domino_deploy_helpers import TOPOLOGY_ALIASES, make_settings
from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    models_dir,
    ovms_bundle,
    stand_in_env,
    started_env,
)
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.workflow_file import WORKFLOW_FILE_VARIABLE
from rag_load_test.deploy.deploy_settings import DeploySettings
from rag_load_test.deploy.publish import entry_script
from rag_load_test.deploy.topology_alias import resolve_topology
from rag_load_test.domain.roles import ModelServerKind

APP_URLS = {
    "embedder": "https://domino.test/apps/e1",
    "reranker": "https://domino.test/apps/r1",
}
ROUTES = {"embedder": "/v3/embeddings", "reranker": "/v3/rerank"}
# The example's three, and one more added to the table with no change in code.
ALIASES = {
    **TOPOLOGY_ALIASES,
    "split-embedder": {"embedder": "served", "reranker": "local"},
}
# What the workflow app is told as RAG_TOPOLOGY: the alias when its own rule
# knows the name and agrees, else monolith, whose rule moves nothing.
TOLD = {
    "monolith": "monolith",
    "split-reranker": "split-reranker",
    "split-all": "split-all",
    "split-embedder": "monolith",
}
# The run label: the alias, then each served component and its server kind.
LABELS = {
    "monolith": "monolith",
    "split-reranker": "split-reranker+reranker=fastapi",
    "split-all": "split-all+embedder=fastapi+reranker=fastapi",
    "split-embedder": "split-embedder+embedder=fastapi",
}


def settings_for(alias: str, kind: ModelServerKind, bundle: str = "") -> DeploySettings:
    return make_settings(
        topology=alias, topology_alias=ALIASES, model_server=kind, ovms_bundle=bundle
    )


COMBINATIONS = [
    (alias, part, kind)
    for alias in ALIASES
    for part in resolve_topology(settings_for(alias, "fastapi")).app_parts()
    for kind in get_args(ModelServerKind)
]


def expected_start(part: str, kind: ModelServerKind, models: Path) -> tuple[str, str]:
    """``(stand-in, argv)`` app.sh must start for an App part and model-server kind.

    Every script bootstraps (``RAG_BOOTSTRAP=1``), so the project's commands run
    from the venv ``uv`` made and OVMS from the bundle.
    """
    if part == "workflow":
        return "venv-rag-serve", "--host 0.0.0.0 --port 8888 --domino"
    if kind == "fastapi":
        return "venv-rag-model-server", f"--serve {part} --host 0.0.0.0 --port 8888"
    return "bundle-ovms", f"--rest_port 8888 --config_path {models}/config_{part}.json"


def run_entry_script(
    tmp_path: Path, alias: str, part: str, kind: ModelServerKind
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Write ``rag-<part>``'s entry script as rag-deploy would, then run it."""
    bin_dir, models = make_stand_ins(tmp_path), models_dir(tmp_path)
    settings = settings_for(alias, kind, str(ovms_bundle(tmp_path)))
    topology = resolve_topology(settings)
    servers = topology.served if part == "workflow" else []
    urls = {server: APP_URLS[server] for server in servers}
    script = entry_script(settings.for_app(part), settings, topology, urls, str(APP_SH))
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
        ("split-embedder", "embedder"),
        ("split-embedder", "workflow"),
    }


@pytest.mark.parametrize(
    ("alias", "part", "kind"),
    COMBINATIONS,
    ids=[f"{t}-{p}-{k}" for t, p, k in COMBINATIONS],
)
def test_each_entry_script_starts_its_app_through_app_sh(
    tmp_path: Path, alias: str, part: str, kind: ModelServerKind
) -> None:
    result, models = run_entry_script(tmp_path, alias, part, kind)

    assert result.returncode == 0, result.stderr
    server, argv = expected_start(part, kind, models)
    assert log(tmp_path, server).strip() == argv
    if part == "workflow":
        assert f"RAG_TOPOLOGY={TOLD[alias]}" in started_env(tmp_path, server)


@pytest.mark.parametrize("alias", list(ALIASES))
def test_the_workflow_app_reads_each_component_where_the_alias_puts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    result, _ = run_entry_script(tmp_path, alias, "workflow", "fastapi")
    assert result.returncode == 0, result.stderr
    # The settings rag-serve reads from what app.sh started it with, and nothing
    # else: no RAG_* of the shell but the workflow file the tests read.
    for name in [n for n in os.environ if n.startswith("RAG_")]:
        if name != WORKFLOW_FILE_VARIABLE:
            monkeypatch.delenv(name)
    for line in started_env(tmp_path, "venv-rag-serve"):
        monkeypatch.setenv(*line.split("=", 1))

    settings = RagSettings(_env_file=None)

    components = settings.components
    placed = {
        name: (components[name].backend, components[name].served.url)
        for name in ("embedder", "reranker")
    }
    expected = {
        name: (
            ("served", APP_URLS[name] + ROUTES[name])
            if cell == "served"
            else ("local", "")
        )
        for name, cell in ALIASES[alias].items()
    }
    assert placed == expected
    assert (settings.topology, settings.run_label) == (TOLD[alias], LABELS[alias])
