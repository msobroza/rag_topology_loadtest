"""The RAG's deploy settings example: a dry run of each topology alias it defines.

``qp-deploy --dry-run`` runs from this directory, as ``make deploy-dry-run``
does, against the in-process fake Domino: it reads ``deploy.example.yaml``
(through ``QP_DEPLOY_FILE``, each example placeholder replaced by a real-looking
value) and this directory's workflow file, and prints what it would publish.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest
import yaml
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_settings import (
    DeploySettings,
    refuse_placeholders,
    refuse_secrets,
)
from query_processing.deploy.qp_deploy import main

# The Apps each topology alias publishes, by App part, in the order published:
# one model server per served component, then the workflow App.
TOPOLOGY_APPS = {
    "monolith": ["workflow"],
    "split-reranker": ["reranker", "workflow"],
    "split-all": ["embedder", "reranker", "workflow"],
}
# The run label the deploy tool gives the workflow App: the alias, then each
# served component's model-server kind.
RUN_LABELS = {
    "monolith": "monolith",
    "split-reranker": "split-reranker+reranker=fastapi",
    "split-all": "split-all+embedder=fastapi+reranker=fastapi",
}
# The two site values the example shows commented, under its overrides: how an
# LLM endpoint that imposes no schema is asked for one, and the passages this
# site's workflow can read at all.
STRUCTURED_OUTPUT = "    # llm: {api: {structured_output: format_instructions}}\n"
PREFILTER = (
    "      # prefilter:\n"
    '      #   - {attribute: ingestion_pipeline_version, comparator: eq, value: "1"}\n'
    "      #   - {attribute: active, comparator: eq, value: true}\n"
)
PREFILTER_READ = [
    {"attribute": "ingestion_pipeline_version", "comparator": "eq", "value": "1"},
    {"attribute": "active", "comparator": "eq", "value": True},
]


def example(use_case_dir: Path) -> str:
    return (use_case_dir / "deploy.example.yaml").read_text(encoding="utf-8")


@pytest.fixture
def filled_example(
    use_case_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """The example with every <placeholder> made a real value, nothing else changed."""
    filled = tmp_path / "deploy.yaml"
    filled.write_text(re.sub(r"<([^<>]+)>", "real", example(use_case_dir)))
    monkeypatch.setenv("QP_DEPLOY_FILE", str(filled))
    return filled


def dry_run(capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """``qp-deploy --dry-run deploy all``; its exit code, stdout and stderr."""
    code = 0
    try:
        main(["--dry-run", "deploy", "all"])
    except SystemExit as exc:
        code = int(exc.code or 0)
    out, err = capsys.readouterr()
    return code, out, err


def published(report: str) -> list[str]:
    """The names of the Apps a dry-run report lists, in order."""
    report = report[report.index("Dry run: ") :]
    return re.findall(r"^([a-z][\w-]*)$", report, re.M)


def test_the_example_defines_the_three_topology_aliases_compared(
    use_case_dir: Path,
) -> None:
    aliases = yaml.safe_load(example(use_case_dir))["topology_alias"]
    assert aliases == {
        "monolith": {"embedder": "local", "reranker": "local"},
        "split-reranker": {"embedder": "local", "reranker": "served"},
        "split-all": {"embedder": "served", "reranker": "served"},
    }


@pytest.mark.parametrize("alias", list(TOPOLOGY_APPS))
def test_a_dry_run_of_each_topology_alias_publishes_its_apps(
    filled_example: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    alias: str,
) -> None:
    monkeypatch.setenv("QP_DEPLOY_TOPOLOGY", alias)

    code, out, err = dry_run(capsys)

    assert (code, err) == (0, "")
    # The RAG's Apps: qp-rag-workflow, qp-rag-embedder, qp-rag-reranker.
    assert published(out) == [f"qp-rag-{part}" for part in TOPOLOGY_APPS[alias]]
    assert f"    export QP_RUN_LABEL={RUN_LABELS[alias]}\n" in out
    assert f"    export QP_TOPOLOGY_ALIAS={alias}\n" in out
    # No alias names the router or its embedder: both keep the backend the
    # workflow file gives them, local, in the workflow App of every layout.
    assert "QP_COMPONENTS__ROUTER__BACKEND" not in out
    assert "QP_COMPONENTS__ROUTER_EMBEDDER__BACKEND" not in out


def test_the_routers_embedder_is_loaded_from_this_sites_model_path(
    filled_example: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = dry_run(capsys)

    assert code == 0
    # Solon's folder, whose last part is the alias the pipeline file records.
    solon = "/domino/datasets/local/real/models/Solon-embeddings-large-0.1"
    export = f"    export QP_COMPONENTS__ROUTER_EMBEDDER__SOURCE_ID={solon}\n"
    assert out.count(export) == len(TOPOLOGY_APPS["split-all"])  # every App


def uncommented(lines: str) -> str:
    """Lines of the example, the comment sign of each taken away."""
    return "".join(line.replace("# ", "", 1) + "\n" for line in lines.splitlines())


def test_the_two_commented_site_values_pass_the_dry_run_once_uncommented(
    filled_example: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = filled_example.read_text()
    assert text.count(STRUCTURED_OUTPUT) == text.count(PREFILTER) == 1
    for lines in (STRUCTURED_OUTPUT, PREFILTER):
        text = text.replace(lines, uncommented(lines))
    filled_example.write_text(text)

    code, out, err = dry_run(capsys)

    assert (code, err) == (0, "")
    method = "QP_COMPONENTS__LLM__API__STRUCTURED_OUTPUT=format_instructions"
    assert out.count(f"    export {method}\n") == len(TOPOLOGY_APPS["split-all"])
    (prefilter,) = set(
        re.findall(r"export QP_COMPONENTS__VECTOR_STORE__PREFILTER=(.+)", out)
    )
    assert json.loads(shlex.split(prefilter)[0]) == PREFILTER_READ


# The three commented pieces that serve the LLM by OVMS: the alias row, the
# LLM's App block, and its model and served name under the overrides.
LLM_SOURCE_ID, LLM_ALIAS = "OpenVINO/Qwen3.5-2B-int4-ov", "qwen3.5-2b"
SERVED_LLM = (
    "  # split-all-llm: {embedder: served, reranker: served, llm: served}\n",
    "  # llm:\n  #   server: ovms\n  #   hardware_tier_id: real\n",
    f"    # llm: {{source_id: {LLM_SOURCE_ID}, alias: {LLM_ALIAS}}}\n",
)


def test_the_commented_served_llm_deploys_it_by_ovms_once_uncommented(
    filled_example: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = filled_example.read_text()
    for lines in SERVED_LLM:
        assert text.count(lines) == 1
        text = text.replace(lines, uncommented(lines))
    filled_example.write_text(text)
    monkeypatch.setenv("QP_DEPLOY_TOPOLOGY", "split-all-llm")
    monkeypatch.setenv("QP_DEPLOY_OVMS_BUNDLE", str(tmp_path))

    code, out, err = dry_run(capsys)

    assert (code, err) == (0, "")
    parts = ["embedder", "reranker", "llm", "workflow"]
    assert published(out) == [f"qp-rag-{part}" for part in parts]
    assert "    export QP_MODEL_SERVER_KIND=ovms\n" in out
    for export in (f"SOURCE_ID={LLM_SOURCE_ID}", f"ALIAS={LLM_ALIAS}"):
        assert out.count(f"    export QP_COMPONENTS__LLM__{export}\n") == 4
    url = re.findall(r"export QP_COMPONENTS__LLM__SERVED__URL=(\S+)", out)
    assert [u.endswith("/v3/chat/completions") for u in url] == [True]
    label = "split-all-llm+embedder=fastapi+reranker=fastapi+llm=ovms"
    assert f"    export QP_RUN_LABEL={label}\n" in out


def test_every_app_runs_the_toolkits_launch_script_in_this_directory(
    filled_example: Path, use_case_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = dry_run(capsys)

    assert code == 0
    launch_script = use_case_dir.parent / "query_processing" / "domino" / "app.sh"
    tail = (
        f"    export QP_USE_CASE_DIR={shlex.quote(str(use_case_dir))}\n"
        "    export QP_BOOTSTRAP=1\n"
        f"    exec bash {shlex.quote(str(launch_script))}\n"
    )
    assert out.count(tail) == len(TOPOLOGY_APPS["split-all"])  # the example's
    # This site's values, written once in the example, reach every App.
    es_url = "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL"
    assert out.count(f"    export {es_url}=https://real:9200\n") == 3
    assert "  kind: agent (configurationType AISYSTEM)\n" in out
    # The example holds no revision: Domino keeps choosing it.
    assert out.count("  environmentRevisionId: (chosen by Domino)\n") == 3


def test_the_example_copied_as_is_stops_on_a_placeholder(
    use_case_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    copied = tmp_path / "deploy.yaml"
    copied.write_text(example(use_case_dir))
    monkeypatch.setenv("QP_DEPLOY_FILE", str(copied))

    code, out, err = dry_run(capsys)

    assert (code, out) == (2, "")
    assert err.endswith(
        f"qp-deploy: error: environment_id in defaults of {copied} still holds "
        "the example placeholder '<environment id>': replace it with a real value\n"
    )
    with pytest.raises(DominoDeployError, match="<environment id>"):
        refuse_placeholders(DeploySettings(), ["embedder", "reranker", "workflow"])


def test_the_example_keeps_secrets_out_and_its_site_values_in_the_overrides(
    filled_example: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Its `project_id: ""` means "not set": the Workspace's project is used.
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-of-the-workspace")
    settings = DeploySettings()

    assert settings.project_id == "proj-of-the-workspace"
    refuse_secrets(settings)  # the example holds no secret
    # The store's URL and the model paths: once, in the workflow file's shape.
    components = settings.overrides["components"]
    assert set(components) == {
        "vector_store",
        "embedder",
        "router_embedder",
        "reranker",
    }
    for model in ("embedder", "router_embedder", "reranker"):
        assert "source_id" in components[model]
    for part in TOPOLOGY_APPS["split-all"]:
        assert settings.for_app(part).settings == {}
