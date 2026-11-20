"""The join: every entry script qp-deploy writes, run through the real domino/app.sh.

The deploy tool and the launch script agree on variable names (``QP_ROLE``,
``QP_MODEL_SERVER_COMPONENT``, ``QP_MODEL_SERVER_KIND``, ``QP_TOPOLOGY_ALIAS``,
``QP_OVMS_BUNDLE``, ``QP_BOOTSTRAP``, ``QP_BOOTSTRAP_WHEELS``, ``QP_USE_CASE_DIR``,
each component's backend and served URL) without sharing code. For each combination the deploy
tool can produce (a topology alias, one of its Apps, the model-server kinds of
the Apps) the script it writes is run by bash with the launch script's stand-in
binaries, and the test asserts what app.sh starts, and where: in the use case's
directory. What is expected is worked out from the combination alone, never from
a Role string, so a Role or a kind that app.sh does not handle or maps to
another server fails here rather than at App start on Domino. The aliases are
the three a use case compares and a fourth added to the table; the kinds are
every registered one for every model server App, and FastAPI beside OVMS. And the overrides, written in the workflow file's shape, reach
the workflow App and a model server App as variables that their settings read
back to the same values, over the workflow file.

An alias row that serves the LLM, by OVMS, the one kind that serves it, is run
with the toolkit's real ``qp-served-models`` and ``qp-ovms-config`` in place of
their stand-ins, so what the deploy tool exports is read by the commands the
script calls: the LLM's App starts OVMS on its model, and the workflow App
calls it there.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from domino_deploy_helpers import TOPOLOGY_ALIASES, make_settings
from launch_script_stand_ins import (
    APP_SH,
    WHEELS_VARIABLE,
    log,
    make_stand_ins,
    models_dir,
    ovms_bundle,
    put_the_real_toolkit_command_on_path,
    served_config,
    stand_in_env,
    started_env,
    started_in,
    wheels_env,
)
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.shared_settings import DefaultHttpSettings
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.model_server_kinds.base_model_server_kind import (
    MODEL_SERVER_KINDS,
    model_server_kind,
)
from query_processing.deploy.publish import LaunchPaths, entry_script
from query_processing.deploy.topology_alias import resolve_topology
from query_processing.domain.components.component_requests import RerankRequest

APP_URLS = {
    "embedder": "https://domino.test/apps/e1",
    "reranker": "https://domino.test/apps/r1",
    "llm": "https://domino.test/apps/l1",
}
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


def settings_for(
    alias: str,
    kinds: str,
    bundle: str = "",
    overrides: dict[str, object] | None = None,
) -> DeploySettings:
    """The test settings, each model server App's kind set in its own block."""
    apps = {part: {"server": kind} for part, kind in KINDS[kinds].items()}
    return make_settings(
        topology=alias,
        topology_alias=ALIASES,
        apps=apps,
        ovms_bundle=bundle,
        overrides=overrides or {},
    )


COMBINATIONS = [
    (alias, part, kinds)
    for alias in ALIASES
    for part in resolve_topology(settings_for(alias, "fastapi")).app_parts()
    for kinds in KINDS
]


def expected_start(part: str, kinds: str, tmp_path: Path) -> tuple[str, str]:
    """``(stand-in, argv)`` app.sh must start for an App part and the Apps' kinds.

    Every script bootstraps (``QP_BOOTSTRAP=1``), so the project's commands run
    from the venv ``uv`` made, vLLM from its own venv on the words the
    ``qp-vllm-config`` stand-in printed (one argument a line), and OVMS from the
    bundle, on the config the ``qp-ovms-config`` stand-in printed.
    """
    if part == "workflow":
        return "venv-qp-serve", "--host 0.0.0.0 --port 8888 --domino"
    if KINDS[kinds][part] == "fastapi":
        return "venv-qp-model-server", f"--serve {part} --host 0.0.0.0 --port 8888"
    if KINDS[kinds][part] == "vllm":
        source_id = f"fixture-org/fixture-{part}"
        served = [source_id, "--served-model-name", f"fixture-{part}"]
        cache = ["--kv-cache-memory-bytes", str(8 * 2**30)]
        listens = ["--host", "0.0.0.0", "--port", "8888"]
        return "venv-vllm", "\n".join(["serve", *served, *cache, *listens])
    config = served_config(tmp_path)
    assert config.endswith(f"/config_{part}.json")
    return "bundle-ovms", f"--rest_port 8888 --config_path {config}"


def expected_label(alias: str, kinds: str) -> str:
    """The run label: the alias, then each served component and its App's kind."""
    served = [name for name, cell in ALIASES[alias].items() if cell == "served"]
    return alias + "".join(f"+{name}={KINDS[kinds][name]}" for name in served)


def run_entry_script(
    tmp_path: Path,
    alias: str,
    part: str,
    kinds: str,
    overrides: dict[str, object] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Write ``rag-<part>``'s entry script as qp-deploy would, then run it."""
    bin_dir, models = make_stand_ins(tmp_path), models_dir(tmp_path)
    settings = settings_for(alias, kinds, str(ovms_bundle(tmp_path)), overrides)
    return run_script_of(tmp_path, settings, part, bin_dir, models)


def run_script_of(
    tmp_path: Path,
    settings: DeploySettings,
    part: str,
    bin_dir: Path,
    models: Path,
    **extra_env: str,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the entry script qp-deploy writes for ``part`` under ``settings``.

    ``run_entry_script`` builds the settings from a combination of the table;
    this runs any settings, with ``extra_env`` beside the stand-ins' own. Their
    wheels' directory is one made here, holding doclang's core as the Dataset's.
    """
    wheels = wheels_env(tmp_path)[WHEELS_VARIABLE]
    settings = settings.model_copy(update={"bootstrap_wheels": wheels})
    topology = resolve_topology(settings)
    servers = topology.served if part == "workflow" else []
    urls = {server: APP_URLS[server] for server in servers}
    use_case = tmp_path / "use_case"
    use_case.mkdir()
    launch = LaunchPaths(str(APP_SH), str(use_case))
    script = entry_script(settings.for_app(part), settings, topology, urls, launch)
    env = stand_in_env(tmp_path, bin_dir, QP_OVMS_MODELS_DIR=str(models), **extra_env)
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
    assert {k for _, _, k in COMBINATIONS} == {"fastapi", "ovms", "vllm", "mixed"}


@pytest.mark.parametrize(
    ("alias", "part", "kinds"),
    COMBINATIONS,
    ids=[f"{t}-{p}-{k}" for t, p, k in COMBINATIONS],
)
def test_each_entry_script_starts_its_app_through_app_sh(
    tmp_path: Path, alias: str, part: str, kinds: str
) -> None:
    result, _ = run_entry_script(tmp_path, alias, part, kinds)

    assert result.returncode == 0, result.stderr
    server, argv = expected_start(part, kinds, tmp_path)
    assert log(tmp_path, server).strip() == argv
    # In the use case's directory, where it finds the use case's workflow file.
    assert started_in(tmp_path, server) == tmp_path / "use_case"
    if part == "workflow":
        assert f"QP_TOPOLOGY_ALIAS={alias}" in started_env(tmp_path, server)


@pytest.mark.parametrize(
    ("alias", "kinds"), [(alias, kinds) for alias in ALIASES for kinds in KINDS]
)
def test_the_workflow_app_reads_each_component_where_the_alias_puts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str, kinds: str
) -> None:
    result, _ = run_entry_script(tmp_path, alias, "workflow", kinds)
    assert result.returncode == 0, result.stderr

    settings = read_back(tmp_path, monkeypatch, "venv-qp-serve")

    components = settings.components
    placed = {
        name: (components[name].backend, components[name].served.url)
        for name in ("embedder", "reranker")
    }
    expected = {
        name: (
            ("served", APP_URLS[name] + route_of(kinds, name))
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


def route_of(kinds: str, name: str) -> str:
    """The route a component is served on, by the kind of its App in ``kinds``."""
    return model_server_kind(KINDS[kinds][name]).route_of(name)


def read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_in: str
) -> WorkflowSettings:
    """The settings an App reads from what app.sh started ``stand_in`` with.

    Nothing else: no QP_* of the shell but the workflow file the tests read.
    """
    for name in [n for n in os.environ if n.startswith("QP_")]:
        if name != WORKFLOW_FILE_VARIABLE:
            monkeypatch.delenv(name)
    for line in started_env(tmp_path, stand_in):
        monkeypatch.setenv(*line.split("=", 1))
    return WorkflowSettings(_env_file=None)


# A list of conditions, each value of another type: the site's prefilter.
PREFILTER = [
    {"attribute": "ingestion_pipeline_version", "comparator": "eq", "value": "1"},
    {"attribute": "active", "comparator": "eq", "value": True},
    {"attribute": "sequence", "comparator": "in", "value": [0, 1]},
]
# A site's values in the workflow file's shape, one of each kind: whole and
# decimal numbers, a boolean, a text with spaces and shell symbols, a list, a
# value of a component, one of a stage and a label.
SITE_OVERRIDES: dict[str, object] = {
    "http": {"timeout_s": 45, "retries": 0, "backoff_s": 0.5},
    "components": {
        "llm": {
            "http": {"timeout_s": 90.5},  # the file's is 120
            "api": {"structured_output": "format_instructions"},
        },
        "vector_store": {
            "elasticsearch": {"url": "https://es.site:9200/?a=b&c d"},
            "prefilter": PREFILTER,  # its fields need not be declared
        },
        "embedder": {"query_prefix": "passage: "},
    },
    "domino": {"tracing": False},
    "local_models": {"cpus": 1.5},
    "graph": {
        "stages": {"rerank": {"top_k": 5}},
        "edges": [["START", "embed", "retrieve", "rerank"], ["generate", "END"]],
    },
    "run_label": "site-a",
}
STARTED = {"workflow": "venv-qp-serve", "reranker": "venv-qp-model-server"}


@pytest.mark.parametrize("part", list(STARTED))
def test_every_app_reads_each_override_back_to_its_value_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part: str
) -> None:
    result, _ = run_entry_script(
        tmp_path, "split-reranker", part, "fastapi", SITE_OVERRIDES
    )
    assert result.returncode == 0, result.stderr

    settings = read_back(tmp_path, monkeypatch, STARTED[part])

    assert settings.http == DefaultHttpSettings(timeout_s=45, retries=0, backoff_s=0.5)
    components = settings.components
    assert components["llm"].http.timeout_s == 90.5
    assert components["llm"].api.structured_output == "format_instructions"
    url = components["vector_store"].elasticsearch.url
    assert url == "https://es.site:9200/?a=b&c d"
    assert components["embedder"].query_prefix == "passage: "
    prefilter = [c.model_dump() for c in components["vector_store"].prefilter]
    assert prefilter == PREFILTER
    assert (settings.domino.tracing, settings.local_models.cpus) == (False, 1.5)
    assert settings.graph.edges == SITE_OVERRIDES["graph"]["edges"]  # type: ignore[index]
    # A plain call's keys fill its component's request, which reads the text back.
    rerank = settings.graph.stages["rerank"].stage_settings
    request = RerankRequest.model_validate({"query": "q", "records": [], **rerank})
    assert request.top_k == 5
    assert settings.run_label == "site-a"


# --- an LLM served by OVMS: the join with the real toolkit commands ---------------

# The alias row that serves the LLM, by OVMS: the only kind that serves one.
LLM_ALIAS = "split-all-llm"
LLM_ROW = {"embedder": "served", "reranker": "served", "llm": "served"}
LLM_SOURCE_ID, LLM_SERVED_NAME = "OpenVINO/Fixture-Chat-int4-ov", "fixture-chat"
# The LLM App's own block: OVMS, split fuse off as by default, and batched tokens
# away from the default 8192, to show the block reaches the App.
LLM_APP = {
    "server": "ovms",
    "settings": {
        "QP_OVMS_DYNAMIC_SPLIT_FUSE": "false",
        "QP_OVMS_MAX_BATCHED_TOKENS": "4096",
    },
}


def run_llm_deploy_script(
    tmp_path: Path, part: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run ``rag-<part>``'s script of a deploy serving the LLM, real qp-* commands.

    ``qp-served-models`` and ``qp-ovms-config`` are the toolkit's own, reading
    the tests' workflow file and what the script exports; ``ovms`` is the
    bundle's stand-in, and ``uv`` a stand-in that is never needed.
    """
    bin_dir, models = make_stand_ins(tmp_path), models_dir(tmp_path)
    for command in ("qp-served-models", "qp-ovms-config"):
        put_the_real_toolkit_command_on_path(bin_dir, command)
    kinds = {"embedder": {"server": "ovms"}, "reranker": {"server": "ovms"}}
    settings = make_settings(
        topology=LLM_ALIAS,
        topology_alias={**ALIASES, LLM_ALIAS: LLM_ROW},
        apps={**kinds, "llm": LLM_APP},
        ovms_bundle=str(ovms_bundle(tmp_path)),
        overrides={
            "components": {
                "llm": {"source_id": LLM_SOURCE_ID, "alias": LLM_SERVED_NAME}
            }
        },
    )
    workflow = os.environ[WORKFLOW_FILE_VARIABLE]  # the tests' workflow file
    return run_script_of(
        tmp_path, settings, part, bin_dir, models, QP_WORKFLOW_FILE=workflow
    )


def test_the_llms_entry_script_starts_ovms_on_its_model_through_app_sh(
    tmp_path: Path,
) -> None:
    result, models = run_llm_deploy_script(tmp_path, "llm")

    assert result.returncode == 0, result.stderr
    # The bundle's OVMS pulls the model the overrides name, under their alias,
    # with the two batching flags of the LLM App's block: no export, no config.
    assert log(tmp_path, "bundle-ovms").strip() == (
        f"--rest_port 8888 --source_model {LLM_SOURCE_ID} "
        f"--model_name {LLM_SERVED_NAME} --model_repository_path {models} "
        "--task text_generation --target_device CPU "
        "--dynamic_split_fuse false --max_num_batched_tokens 4096"
    )
    assert log(tmp_path, "uv") == ""
    assert not (models / "config_llm.json").exists()


def test_the_workflow_app_calls_the_llm_at_its_apps_chat_route_by_its_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = run_llm_deploy_script(tmp_path, "workflow")
    assert result.returncode == 0, result.stderr

    settings = read_back(tmp_path, monkeypatch, "venv-qp-serve")

    llm = settings.model_components()["llm"]
    assert (llm.backend, llm.served.url, llm.alias) == (
        "served",
        APP_URLS["llm"] + model_server_kind("ovms").route_of("llm"),
        LLM_SERVED_NAME,
    )
    assert settings.run_label == f"{LLM_ALIAS}+embedder=ovms+reranker=ovms+llm=ovms"
