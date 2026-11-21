"""``qp-deploy --dry-run``: the same code against an in-process fake Domino.

The fixtures (domino_env, no_real_http, run_main) are in domino_deploy_fixtures.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from domino_deploy_helpers import (
    AGENT_EXPORTS,
    CACHE_FILE,
    ENV_1,
    INSTANCES_URL,
    LAUNCH,
    NO_ENVIRONMENT,
    PREFIX_YAML,
    TOPOLOGY_ALIAS_YAML,
    WHEELS,
    WHEELS_YAML,
    RunMain,
    deployed_lines,
    launched,
    make_client,
    make_settings,
    model_server_exports,
    no_sleep,
)
from fake_domino.server import create_fake_domino
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.publish import deploy
from repo_paths import REPO_ROOT

pytest_plugins = ["domino_deploy_fixtures"]


def dry_run_report(
    tier: str = "(the project default)", revision: str = "(chosen by Domino)"
) -> str:
    tail = "".join(f"    {line}\n" for line in launched().splitlines())
    environment = f"  environmentId: env-1\n  environmentRevisionId: {revision}\n"
    served = f"{INSTANCES_URL}/app-{{}}-v1-i/v3/{{}}"
    model_servers = "".join(
        f"\nrag-{part}\n  kind: app (configurationType STANDARD)\n"
        f"{environment}  hardwareTierId: {tier}\n  autoscaling: off\n"
        "  entryScript:\n"
        + "".join(f"    {line}\n" for line in model_server_exports(part).splitlines())
        + tail
        for part in ("embedder", "reranker")
    )
    agent_exports = "".join(f"    {line}\n" for line in AGENT_EXPORTS.splitlines())
    return (
        "\nDry run: nothing was sent to Domino. qp-deploy would publish:\n"
        f"{model_servers}"
        "\nrag-workflow\n  kind: agent (configurationType AISYSTEM)\n"
        f"{environment}  hardwareTierId: {tier}\n"
        "  autoscaling: minReplicas 1, maxReplicas 4, targetCpuAvgUtilizationPct 70\n"
        "  entryScript:\n"
        "    export QP_ROLE=workflow\n    export QP_TOPOLOGY_ALIAS=split-all\n"
        "    export QP_COMPONENTS__EMBEDDER__BACKEND=served\n"
        f"    export QP_COMPONENTS__EMBEDDER__SERVED__URL={served.format(1, 'embeddings')}\n"
        "    export QP_COMPONENTS__RERANKER__BACKEND=served\n"
        f"    export QP_COMPONENTS__RERANKER__SERVED__URL={served.format(2, 'rerank')}\n"
        "    export QP_RUN_LABEL=split-all+embedder=fastapi+reranker=fastapi\n"
        f"{agent_exports}{tail}"
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


def test_dry_run_shows_a_name_and_a_number_as_written_and_asks_domino_nothing(
    run_main: RunMain,
) -> None:
    named = "  environment_revision: 7\n  hardware_tier: cpu-16\n"
    Path("deploy.yaml").write_text(ENV_1 + named + WORKFLOW_SCALES)
    code, out, _ = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 0
    assert out.endswith(
        dry_run_report(
            tier="(tier cpu-16: its id is asked of Domino at deploy)",
            revision="(revision 7: its id is asked of Domino at deploy)",
        )
    )


def test_dry_run_shows_the_hardware_tier_when_set(run_main: RunMain) -> None:
    tier = "  hardware_tier_id: tier-8\n"
    Path("deploy.yaml").write_text(ENV_1 + tier + WORKFLOW_SCALES)
    code, out, _ = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 0
    assert out.endswith(dry_run_report("tier-8"))


def test_dry_run_shows_the_environment_revision_when_set(run_main: RunMain) -> None:
    revision = "  environment_revision_id: rev-7\n"
    Path("deploy.yaml").write_text(ENV_1 + revision + WORKFLOW_SCALES)
    code, out, _ = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 0
    assert out.endswith(dry_run_report(revision="rev-7"))


def test_dry_run_refuses_a_default_revision_without_the_default_environment(
    run_main: RunMain,
) -> None:
    no_environment = "defaults: {environment_revision_id: rev-7}\n"
    Path("deploy.yaml").write_text(PREFIX_YAML + TOPOLOGY_ALIAS_YAML + no_environment)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert "defaults.environment_revision_id 'rev-7' in deploy.yaml has no " in err


def test_dry_run_still_requires_the_environment_id(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(PREFIX_YAML + TOPOLOGY_ALIAS_YAML)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert err.endswith(NO_ENVIRONMENT)


def test_dry_run_still_refuses_secrets_in_deploy_yaml(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(ENV_1 + "  settings: {QP_ES_API_KEY: secret}\n")
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert "QP_ES_API_KEY is set in deploy.yaml under defaults.settings" in err


def test_dry_run_still_requires_the_toolkits_launch_script_above(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A use case's directory outside the repository: no toolkit beside it.
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    Path("deploy.yaml").write_text(ENV_1)
    code, _, err = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 2
    assert f"no query_processing/domino/app.sh in {tmp_path / 'elsewhere'} " in err


def test_dry_run_without_deploy_settings_says_where_it_looked(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    Path("deploy.yaml").unlink()
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert err.endswith(
        "qp-deploy: error: no deploy settings: QP_DEPLOY_FILE is not set, and "
        f"the working directory {Path.cwd()} has no deploy.yaml; run qp-deploy "
        "from a use case's directory (cp deploy.example.yaml deploy.yaml there), "
        "or set QP_DEPLOY_FILE to the file\n"
    )
    monkeypatch.setenv("QP_DEPLOY_FILE", "nowhere.yaml")
    code, _, err = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 2
    assert "QP_DEPLOY_FILE names 'nowhere.yaml', which is no file" in err


def test_dry_run_without_a_workflow_file_says_where_it_looked(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QP_WORKFLOW_FILE")
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert (
        "qp-deploy reads the workflow file the workflow App reads: no workflow "
        f"file: QP_WORKFLOW_FILE is not set, and the working directory {Path.cwd()} "
        "has no workflow.yaml"
    ) in err


def test_dry_run_still_checks_the_ovms_bundle(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    Path("deploy.yaml").write_text(ENV_1 + "  server: ovms\n")
    monkeypatch.setenv("QP_DEPLOY_OVMS_BUNDLE", "/no/such/bundle")
    code, _, err = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 2
    assert "ovms_bundle=/no/such/bundle does not exist" in err


def test_a_dry_run_deploys_one_component_served_by_fastapi_and_one_by_ovms(
    run_main: RunMain, tmp_path: Path
) -> None:
    bundle = tmp_path / "ovms.tar.gz"
    bundle.write_text("")
    kinds = "apps: {embedder: {server: fastapi}, reranker: {server: ovms}}\n"
    Path("deploy.yaml").write_text(ENV_1 + kinds + f"ovms_bundle: {bundle}\n")

    code, out, err = run_main(["--dry-run", "deploy", "all"], None)

    assert (code, err) == (0, "")
    report = out[out.index("Dry run: ") :]
    scripts = re.findall(r"entryScript:\n((?:    .*\n)+)", report)
    embedder, reranker, workflow = (script.replace("    ", "") for script in scripts)
    assert embedder == f"{model_server_exports('embedder', 'fastapi')}" + (
        f"{launched()}\n"
    )
    assert reranker == f"{model_server_exports('reranker', 'ovms')}" + (
        f"export QP_OVMS_BUNDLE={bundle}\n{launched()}\n"
    )
    served = f"{INSTANCES_URL}/app-{{}}-v1-i/v3/{{}}"
    assert (
        f"export QP_COMPONENTS__EMBEDDER__SERVED__URL={served.format(1, 'embeddings')}\n"
        f"export QP_COMPONENTS__RERANKER__BACKEND=served\n"
        f"export QP_COMPONENTS__RERANKER__SERVED__URL={served.format(2, 'rerank')}\n"
        "export QP_RUN_LABEL=split-all+embedder=fastapi+reranker=ovms\n"
    ) in workflow


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


def test_dry_run_hands_every_app_the_wheels_directory_beside_the_bootstrap(
    run_main: RunMain,
) -> None:
    # Where an App that installs finds doclang's core: a directory on the Dataset.
    Path("deploy.yaml").write_text(ENV_1)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, err) == (0, "")
    wheels = shlex.quote(WHEELS)
    beside = f"    export QP_BOOTSTRAP=1\n    export QP_BOOTSTRAP_WHEELS={wheels}\n"
    assert out.count(beside) == 3  # the two model servers and the workflow App


def test_a_deploy_file_without_the_wheels_directory_is_refused_in_one_line(
    run_main: RunMain,
) -> None:
    Path("deploy.yaml").write_text(ENV_1.replace(WHEELS_YAML, ""))
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    line = err.splitlines()[-1]  # after argparse's usage
    assert line.startswith(
        "qp-deploy: error: bootstrap_wheels is not set in deploy.yaml"
    )


def test_a_deploy_without_the_wheels_directory_publishes_nothing() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(bootstrap_wheels="")
    with pytest.raises(DominoDeployError, match="bootstrap_wheels is not set"):
        deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    assert apps.received == []  # not even a lookup reached Domino


def test_the_readmes_copy_of_the_example_says_to_set_the_wheels_directory() -> None:
    # A deploy refuses it unset, so the line that copies the example names it.
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    [copy] = [line for line in readme.splitlines() if line.startswith("cp deploy.")]

    assert "bootstrap_wheels" in copy.split("#", 1)[1]


def a_wheels_directory(where: Path, *versions: str) -> Path:
    """A directory at ``where`` holding a wheel of doclang's core per version."""
    where.mkdir(parents=True)
    for version in versions:
        (where / f"doclang_domain_core-{version}-py3-none-any.whl").touch()
    return where


@pytest.mark.parametrize(
    ("given", "refusal"),
    [
        ("missing", "does not exist"),
        ("empty", "holds no wheel of doclang's core (doclang_domain_core-*.whl)"),
        ("three versions", "holds 3 wheels of doclang's core"),
    ],
)
def test_a_deploy_refuses_a_wheels_directory_the_apps_cannot_install_from(
    run_main: RunMain, tmp_path: Path, given: str, refusal: str
) -> None:
    # As an OVMS bundle that does not exist is: before anything is published,
    # not at each App's start. A deploy, not a dry run: it runs in a Workspace,
    # which mounts the Apps' Dataset, so it can look.
    paths = {
        "missing": str(tmp_path / "no such directory"),
        "empty": str(a_wheels_directory(tmp_path / "empty")),
        "three versions": str(
            a_wheels_directory(tmp_path / "three", "0.1.0", "0.1.1", "0.2.0")
        ),
    }
    Path("deploy.yaml").write_text(
        ENV_1.replace(WHEELS_YAML, f"bootstrap_wheels: '{paths[given]}'\n")
    )
    apps = FakeDominoApps(["Running"])

    code, out, err = run_main(["deploy", "all"], create_fake_domino(apps))

    assert (code, out, apps.creates) == (2, "", [])
    line = err.splitlines()[-1]  # after argparse's usage
    assert line.startswith(f"qp-deploy: error: bootstrap_wheels={paths[given]} ")
    assert refusal in line


@pytest.mark.parametrize(
    ("written", "refusal"),
    [("wheels", "bootstrap_wheels=wheels is a relative path"), ("   ", "is not set")],
)
def test_a_dry_run_refuses_a_wheels_directory_no_app_could_read(
    run_main: RunMain, written: str, refusal: str
) -> None:
    Path("deploy.yaml").write_text(
        ENV_1.replace(WHEELS_YAML, f"bootstrap_wheels: '{written}'\n")
    )

    code, out, err = run_main(["--dry-run", "deploy", "all"], None)

    assert (code, out) == (2, "")
    assert refusal in err.splitlines()[-1]  # after argparse's usage


def test_a_dry_run_does_not_look_for_the_wheels_directory_on_its_machine(
    run_main: RunMain, tmp_path: Path
) -> None:
    # A dry run also runs off Domino, on a laptop that mounts no Dataset: the
    # README's first run copies the example, whose directory is not there.
    nowhere = tmp_path / "datasets" / "wheels"
    Path("deploy.yaml").write_text(
        ENV_1.replace(WHEELS_YAML, f"bootstrap_wheels: '{nowhere}'\n")
    )

    code, out, err = run_main(["--dry-run", "deploy", "all"], None)

    assert (code, err) == (0, "")
    assert f"export QP_BOOTSTRAP_WHEELS={nowhere}" in out
