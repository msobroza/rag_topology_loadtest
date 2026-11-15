"""The RAG's deploy settings example: a dry run of each topology alias it defines.

``qp-deploy --dry-run`` runs from this directory, as ``make deploy-dry-run``
does, against the in-process fake Domino: it reads ``deploy.example.yaml``
(through ``QP_DEPLOY_FILE``, each example placeholder replaced by a real-looking
value) and this directory's workflow file, and prints what it would publish.
"""

from __future__ import annotations

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
    assert set(components) == {"vector_store", "embedder", "reranker"}
    assert "source_id" in components["embedder"]
    assert "source_id" in components["reranker"]
    for part in TOPOLOGY_APPS["split-all"]:
        assert settings.for_app(part).settings == {}
