"""``rag-deploy --dry-run``: the same code against an in-process fake Domino.

The fixtures (domino_env, no_real_http, run_main) are in domino_deploy_fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from domino_deploy_helpers import (
    AGENT_EXPORTS,
    CACHE_FILE,
    ENV_1,
    INSTANCES_URL,
    NO_ENVIRONMENT,
    RunMain,
    deployed_lines,
    launched,
)
from repo_paths import REPO_ROOT

pytest_plugins = ["domino_deploy_fixtures"]
EXAMPLE_YAML = REPO_ROOT / "deploy.example.yaml"


def dry_run_report(tier: str = "(the project default)") -> str:
    tail = launched()
    model_servers = "".join(
        f"\nrag-{part}\n  kind: app (configurationType STANDARD)\n"
        f"  environmentId: env-1\n  hardwareTierId: {tier}\n  autoscaling: off\n"
        f"  entryScript:\n    export RAG_ROLE=fastapi-{part}\n"
        f"    export RAG_BOOTSTRAP=1\n    {tail}\n"
        for part in ("embedder", "reranker")
    )
    agent_exports = "".join(f"    {line}\n" for line in AGENT_EXPORTS.splitlines())
    return (
        "\nDry run: nothing was sent to Domino. rag-deploy would publish:\n"
        f"{model_servers}"
        "\nrag-workflow\n  kind: agent (configurationType AISYSTEM)\n"
        f"  environmentId: env-1\n  hardwareTierId: {tier}\n"
        "  autoscaling: minReplicas 1, maxReplicas 4, targetCpuAvgUtilizationPct 70\n"
        "  entryScript:\n"
        "    export RAG_ROLE=workflow\n    export RAG_TOPOLOGY=split-all\n"
        f"    export RAG_OVMS_RERANK_URL={INSTANCES_URL}/app-2-v1-i\n"
        f"    export RAG_OVMS_EMBEDDINGS_URL={INSTANCES_URL}/app-1-v1-i\n"
        "    export RAG_RUN_LABEL=split-all-fastapi\n"
        f"{agent_exports}    export RAG_BOOTSTRAP=1\n    {tail}\n"
    )


WORKFLOW_SCALES = (
    "apps:\n  workflow:\n" "    autoscaling: {max_replicas: 4, target_cpu_pct: 70}\n"
)


def test_dry_run_deploy_needs_no_workspace_and_prints_each_entry_script(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY")
    monkeypatch.delenv("DOMINO_PROJECT_ID")
    Path("deploy.yaml").write_text(ENV_1 + WORKFLOW_SCALES)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, err) == (0, "")
    assert out == deployed_lines(3) + dry_run_report()
    assert not Path(CACHE_FILE).exists()  # a dry run remembers nothing


def test_dry_run_shows_the_hardware_tier_when_set(run_main: RunMain) -> None:
    tier = "  hardware_tier_id: tier-8\n"
    Path("deploy.yaml").write_text(ENV_1 + tier + WORKFLOW_SCALES)
    code, out, _ = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 0
    assert out.endswith(dry_run_report("tier-8"))


def test_dry_run_still_requires_the_environment_id(run_main: RunMain) -> None:
    Path("deploy.yaml").unlink()
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert err.endswith(NO_ENVIRONMENT)


def test_dry_run_still_refuses_secrets_in_deploy_yaml(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(ENV_1 + "  settings: {RAG_ES_API_KEY: secret}\n")
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert "RAG_ES_API_KEY is set in deploy.yaml under defaults.settings" in err


def test_dry_run_still_requires_the_repository_root(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    code, _, err = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 2
    assert "has no domino/app.sh" in err


def test_dry_run_still_checks_the_ovms_bundle(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_MODEL_SERVER", "ovms")
    monkeypatch.setenv("RAG_DEPLOY_OVMS_BUNDLE", "/no/such/bundle")
    code, _, err = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 2
    assert "ovms_bundle=/no/such/bundle does not exist" in err


def test_dry_run_status_lists_nothing_deployed(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY")
    code, out, _ = run_main(["--dry-run", "status"], None)
    assert (code, out) == (
        0,
        "rag-embedder   not deployed\n"
        "rag-reranker   not deployed\n"
        "rag-workflow   not deployed\n",
    )


def test_dry_run_of_the_example_copied_as_is_stops_on_a_placeholder(
    run_main: RunMain,
) -> None:
    Path("deploy.yaml").write_text(EXAMPLE_YAML.read_text())
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert err.endswith(
        "rag-deploy: error: environment_id in defaults of deploy.yaml still holds "
        "the example placeholder '<environment id>': replace it with a real value\n"
    )


def test_dry_run_of_the_example_with_real_values_publishes_every_app(
    run_main: RunMain,
) -> None:
    # Every <placeholder> replaced by a real-looking value, nothing else changed.
    filled = re.sub(r"<([^<>]+)>", "real", EXAMPLE_YAML.read_text())
    Path("deploy.yaml").write_text(filled)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, err) == (0, "")
    assert out.startswith(deployed_lines(3))
    assert "    export RAG_ES_URL=https://real:9200\n" in out
    assert "  kind: agent (configurationType AISYSTEM)\n" in out
