"""The join: every entry script rag-deploy writes, run through the real domino/app.sh.

The deploy tool and the launch script agree on variable names (``RAG_ROLE``,
``RAG_MODEL_SERVER_COMPONENT``, ``RAG_MODEL_SERVER_KIND``, ``RAG_TOPOLOGY_ALIAS``,
``RAG_OVMS_BUNDLE``, ``RAG_BOOTSTRAP``, each component's backend and served URL)
without sharing code. For each combination the deploy tool can produce (a
topology alias, one of its Apps, the model-server kinds of the Apps) the script
it writes is run by bash with the launch script's stand-in binaries, and the test
asserts what app.sh starts. What is expected is worked out from the combination
alone, never from a Role string, so a Role or a kind that app.sh does not handle
or maps to another server fails here rather than at App start on Domino. The
aliases are the three of deploy.example.yaml and a fourth added to the table;
the kinds are every registered one for every model server App, and FastAPI
beside OVMS.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
from rag_load_test.deploy.model_server_kinds.base_model_server_kind import (
    MODEL_SERVER_KINDS,
)
from rag_load_test.deploy.publish import entry_script
from rag_load_test.deploy.topology_alias import resolve_topology

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
# The model-server kind of each model server App: one kind for all, each
# registered kind in turn, and the embedder by FastAPI beside the reranker by OVMS.
KINDS = {
    **{
        kind: {"embedder": kind, "reranker": kind}
        for kind in MODEL_SERVER_KINDS.names()
    },
    "mixed": {"embedder": "fastapi", "reranker": "ovms"},
}


def settings_for(alias: str, kinds: str, bundle: str = "") -> DeploySettings:
    """The test settings, each model server App's kind set in its own block."""
    apps = {part: {"server": kind} for part, kind in KINDS[kinds].items()}
    return make_settings(
        topology=alias, topology_alias=ALIASES, apps=apps, ovms_bundle=bundle
    )


COMBINATIONS = [
    (alias, part, kinds)
    for alias in ALIASES
    for part in resolve_topology(settings_for(alias, "fastapi")).app_parts()
    for kinds in KINDS
]


def expected_start(part: str, kinds: str, models: Path) -> tuple[str, str]:
    """``(stand-in, argv)`` app.sh must start for an App part and the Apps' kinds.

    Every script bootstraps (``RAG_BOOTSTRAP=1``), so the project's commands run
    from the venv ``uv`` made and OVMS from the bundle.
    """
    if part == "workflow":
        return "venv-rag-serve", "--host 0.0.0.0 --port 8888 --domino"
    if KINDS[kinds][part] == "fastapi":
        return "venv-rag-model-server", f"--serve {part} --host 0.0.0.0 --port 8888"
    return "bundle-ovms", f"--rest_port 8888 --config_path {models}/config_{part}.json"


def expected_label(alias: str, kinds: str) -> str:
    """The run label: the alias, then each served component and its App's kind."""
    served = [name for name, cell in ALIASES[alias].items() if cell == "served"]
    return alias + "".join(f"+{name}={KINDS[kinds][name]}" for name in served)


def run_entry_script(
    tmp_path: Path, alias: str, part: str, kinds: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Write ``rag-<part>``'s entry script as rag-deploy would, then run it."""
    bin_dir, models = make_stand_ins(tmp_path), models_dir(tmp_path)
    settings = settings_for(alias, kinds, str(ovms_bundle(tmp_path)))
    topology = resolve_topology(settings)
    servers = topology.served if part == "workflow" else []
    urls = {server: APP_URLS[server] for server in servers}
    script = entry_script(settings.for_app(part), settings, topology, urls, str(APP_SH))
    env = stand_in_env(tmp_path, bin_dir, RAG_OVMS_MODELS_DIR=str(models))
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    return result, models


def test_every_app_the_deploy_tool_writes_is_exercised_under_each_kind() -> None:
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
    assert {k for _, _, k in COMBINATIONS} == {"fastapi", "ovms", "mixed"}


@pytest.mark.parametrize(
    ("alias", "part", "kinds"),
    COMBINATIONS,
    ids=[f"{t}-{p}-{k}" for t, p, k in COMBINATIONS],
)
def test_each_entry_script_starts_its_app_through_app_sh(
    tmp_path: Path, alias: str, part: str, kinds: str
) -> None:
    result, models = run_entry_script(tmp_path, alias, part, kinds)

    assert result.returncode == 0, result.stderr
    server, argv = expected_start(part, kinds, models)
    assert log(tmp_path, server).strip() == argv
    if part == "workflow":
        assert f"RAG_TOPOLOGY_ALIAS={alias}" in started_env(tmp_path, server)


@pytest.mark.parametrize(
    ("alias", "kinds"), [(alias, kinds) for alias in ALIASES for kinds in KINDS]
)
def test_the_workflow_app_reads_each_component_where_the_alias_puts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str, kinds: str
) -> None:
    result, _ = run_entry_script(tmp_path, alias, "workflow", kinds)
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
    # Any alias, a fourth one too, is reported under its own name.
    assert (settings.topology_alias, settings.run_label) == (
        alias,
        expected_label(alias, kinds),
    )
