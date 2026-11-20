"""Tests of qp-deploy's settings: ``deploy.yaml``, its per-App blocks, the
``QP_DEPLOY_*`` overrides, the refused secrets and the project id fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from domino_deploy_helpers import DEPLOY_YAML
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_refusals import refuse_placeholders, refuse_secrets
from query_processing.deploy.deploy_settings import (
    AppSettings,
    Autoscaling,
    DeploySettings,
    require_deploy_file,
)


@pytest.fixture
def deploy_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The path QP_DEPLOY_FILE names; nothing from the shell leaks in.

    The prefix every deploy file must name is given by its variable, so each
    file below holds only what its test is about.
    """
    for name in ("QP_DEPLOY_TOPOLOGY", "QP_DEPLOY_PROJECT_ID", "DOMINO_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QP_DEPLOY_APP_NAME_PREFIX", "rag")
    path = tmp_path / "deploy.yaml"
    monkeypatch.setenv("QP_DEPLOY_FILE", str(path))
    return path


def test_each_app_takes_its_own_value_else_the_default(deploy_file: Path) -> None:
    deploy_file.write_text(DEPLOY_YAML)
    settings = DeploySettings()
    assert settings.topology == "split-reranker"
    assert settings.for_app("workflow") == AppSettings(
        part="workflow",
        name="rag-workflow",
        kind="agent",
        environment_id="env-123",
        environment_revision_id="",
        hardware_tier_id="small",
        autoscaling=Autoscaling(min_replicas=1, max_replicas=4),
        server="fastapi",
        settings={"QP_ES_URL": "https://es.test:9200"},
    )
    embedder = settings.for_app("embedder")
    assert (embedder.kind, embedder.hardware_tier_id) == ("app", "cpu-8")
    assert embedder.autoscaling is None
    assert list(embedder.settings) == [  # merged, sorted by name
        "QP_BOOTSTRAP_EXTRAS",
        "QP_ES_URL",
    ]
    # The overrides are the deploy's, for every App, in the file's own shape.
    assert settings.overrides == {
        "components": {
            "reranker": {"source_id": "/mnt/data/models/bge reranker"},
            "embedder": {"source_id": "/mnt/data/models/bge-small"},
        }
    }


def test_an_apps_settings_win_over_the_defaults(deploy_file: Path) -> None:
    deploy_file.write_text(
        "defaults: {settings: {QP_ES_URL: https://shared.test, QP_A: a}}\n"
        "apps: {reranker: {settings: {QP_ES_URL: https://own.test}}}\n"
    )
    assert DeploySettings().for_app("reranker").settings == {
        "QP_A": "a",
        "QP_ES_URL": "https://own.test",
    }


def environment_of(settings: DeploySettings, part: str) -> tuple[str, str]:
    app = settings.for_app(part)  # type: ignore[arg-type]
    return app.environment_id, app.environment_revision_id


def test_a_revision_set_for_an_app_or_in_the_defaults_is_taken(
    deploy_file: Path,
) -> None:
    deploy_file.write_text(
        "defaults: {environment_id: env-1, environment_revision_id: rev-1}\n"
        "apps: {embedder: {environment_revision_id: rev-3}}\n"
    )
    settings = DeploySettings()
    assert environment_of(settings, "workflow") == ("env-1", "rev-1")
    assert environment_of(settings, "embedder") == ("env-1", "rev-3")


def test_an_app_with_its_own_environment_does_not_take_the_default_revision(
    deploy_file: Path,
) -> None:
    deploy_file.write_text(
        "defaults: {environment_id: env-1, environment_revision_id: rev-1}\n"
        "apps:\n"
        "  reranker: {environment_id: env-2}\n"
        "  embedder: {environment_id: env-2, environment_revision_id: rev-2}\n"
    )
    settings = DeploySettings()
    assert environment_of(settings, "reranker") == ("env-2", "")
    assert environment_of(settings, "embedder") == ("env-2", "rev-2")


def test_with_no_revision_set_none_is_taken(deploy_file: Path) -> None:
    deploy_file.write_text("defaults: {environment_id: env-1}\n")
    assert environment_of(DeploySettings(), "workflow") == ("env-1", "")


def test_kind_is_agent_for_the_workflow_and_app_for_model_servers_by_default(
    deploy_file: Path,
) -> None:
    deploy_file.write_text("defaults: {environment_id: env-1}\n")
    settings = DeploySettings()
    kinds = [settings.for_app(part).kind for part in ("workflow", "embedder")]
    assert kinds == ["agent", "app"]


def test_a_kind_in_the_defaults_applies_to_every_app(deploy_file: Path) -> None:
    deploy_file.write_text("defaults: {kind: app}\n")
    assert DeploySettings().for_app("workflow").kind == "app"


def test_yaml_scalars_are_exported_as_strings(deploy_file: Path) -> None:
    deploy_file.write_text(
        "defaults:\n  settings:\n    QP_MODEL_WORKERS: 2\n    QP_FLAG: true\n"
        "    QP_RATIO: 0.5\n    QP_EMPTY: ''\n"
    )
    assert DeploySettings().for_app("embedder").settings == {
        "QP_EMPTY": "",
        "QP_FLAG": "true",
        "QP_MODEL_WORKERS": "2",
        "QP_RATIO": "0.5",
    }


def test_a_setting_that_is_not_a_scalar_is_refused_with_its_name(
    deploy_file: Path,
) -> None:
    deploy_file.write_text("defaults: {settings: {QP_LIST: [1, 2]}}\n")
    with pytest.raises(ValidationError) as caught:
        DeploySettings()
    assert "QP_LIST must be a string, a number or a boolean, not [1, 2]" in str(
        caught.value
    )


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("topolgy: monolith\n", "topolgy"),
        ("apps: {embedder: {hardware_teir: cpu-8}}\n", "apps.embedder.hardware_teir"),
        ("defaults: {autoscaling: {max: 4}}\n", "defaults.autoscaling.max"),
    ],
)
def test_an_unknown_key_is_refused_with_its_name(
    deploy_file: Path, text: str, where: str
) -> None:
    deploy_file.write_text(text)
    with pytest.raises(ValidationError) as caught:
        DeploySettings()
    assert where in str(caught.value)


def test_qp_deploy_variables_override_the_file(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_file.write_text("topology: split-all\nwait_timeout_s: 30\n")
    monkeypatch.setenv("QP_DEPLOY_TOPOLOGY", "monolith")
    settings = DeploySettings()
    assert (settings.topology, settings.wait_timeout_s) == ("monolith", 30.0)


def test_without_a_file_every_value_but_the_prefix_is_its_default(
    deploy_file: Path,
) -> None:
    settings = DeploySettings()
    # The prefix has no default: deploy_file gives it.
    assert settings.model_dump(exclude={"defaults", "apps", "app_name_prefix"}) == {
        "topology": "split-all",
        "topology_alias": {},  # no alias in code: deploy.yaml defines them
        "project_id": "",
        "ovms_bundle": "",
        "domino_host": "",
        "wait_timeout_s": 600.0,
        "poll_interval_s": 5.0,
        "no_instance_grace_s": 60.0,
        "http_timeout_s": 30.0,
        "http_retries": 2,
        "run_token_url": "http://localhost:8899/access-token",
        "overrides": {},  # the workflow file as it is
    }
    assert settings.for_app("reranker").environment_id == ""
    assert settings.for_app("reranker").server == "fastapi"


def test_deploy_yaml_in_the_working_directory_is_read_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("QP_DEPLOY_FILE", raising=False)
    monkeypatch.delenv("QP_DEPLOY_TOPOLOGY", raising=False)
    monkeypatch.chdir(tmp_path)
    Path("deploy.yaml").write_text("app_name_prefix: rag\ntopology: monolith\n")
    assert DeploySettings().topology == "monolith"


def test_app_settings_come_from_the_file_only(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_ES_URL", "https://from-the-shell.test")
    assert DeploySettings().for_app("workflow").settings == {}


@pytest.mark.parametrize(
    ("autoscaling", "message"),
    [
        (
            "{min_replicas: 3, max_replicas: 2}",
            "max_replicas 2 is below min_replicas 3",
        ),
        (
            "{max_replicas: 2, target_cpu_pct: 0}",
            "target_cpu_pct 0 is outside 1..100",
        ),
        (
            "{max_replicas: 2, target_cpu_pct: 101}",
            "target_cpu_pct 101 is outside 1..100",
        ),
    ],
)
def test_impossible_autoscaling_is_refused_with_its_values(
    deploy_file: Path, autoscaling: str, message: str
) -> None:
    deploy_file.write_text(f"apps: {{workflow: {{autoscaling: {autoscaling}}}}}\n")
    with pytest.raises(ValidationError) as caught:
        DeploySettings()
    assert message in str(caught.value)


def test_autoscaling_at_its_bounds_is_accepted(deploy_file: Path) -> None:
    deploy_file.write_text(
        "defaults: {autoscaling: {min_replicas: 2, max_replicas: 2, "
        "target_cpu_pct: 100}}\n"
    )
    scaling = DeploySettings().for_app("embedder").autoscaling
    assert scaling == Autoscaling(min_replicas=2, max_replicas=2, target_cpu_pct=100)


# --- placeholders left from the example ---


def placeholder_message(key: str, block: str, value: str, path: Path) -> str:
    return (
        f"{key} in {block} of {path} still holds the example placeholder "
        f"{value!r}: replace it with a real value"
    )


@pytest.mark.parametrize(
    ("text", "key", "block", "value"),
    [
        (
            "defaults: {settings: {QP_ES_URL: 'https://<es host>:9200'}}",
            "QP_ES_URL",
            "defaults.settings",
            "https://<es host>:9200",
        ),
        (
            "apps: {embedder: {settings: {QP_COMPONENTS__EMBEDDER__SOURCE_ID: /d/<dataset>/m}}}",
            "QP_COMPONENTS__EMBEDDER__SOURCE_ID",
            "apps.embedder.settings",
            "/d/<dataset>/m",
        ),
        (
            "defaults: {environment_id: '<environment id>'}",
            "environment_id",
            "defaults",
            "<environment id>",
        ),
        (
            "defaults: {environment_revision_id: '<revision id>'}",
            "environment_revision_id",
            "defaults",
            "<revision id>",
        ),
        (
            "apps: {workflow: {environment_revision_id: '<revision id>'}}",
            "environment_revision_id",
            "apps.workflow",
            "<revision id>",
        ),
        (
            "apps: {reranker: {hardware_tier_id: '<tier>'}}",
            "hardware_tier_id",
            "apps.reranker",
            "<tier>",
        ),
        (
            "ovms_bundle: /domino/datasets/local/<dataset>/ovms.tar.gz",
            "ovms_bundle",
            "the top level",
            "/domino/datasets/local/<dataset>/ovms.tar.gz",
        ),
        (
            "overrides: {components: {embedder: {source_id: /d/<dataset>/m}}}",
            "components.embedder.source_id",
            "overrides",
            "/d/<dataset>/m",
        ),
    ],
)
def test_a_placeholder_left_in_a_value_is_refused(
    deploy_file: Path, text: str, key: str, block: str, value: str
) -> None:
    deploy_file.write_text(text + "\n")
    with pytest.raises(DominoDeployError) as caught:
        refuse_placeholders(DeploySettings(), ["embedder", "reranker", "workflow"])
    assert str(caught.value) == placeholder_message(key, block, value, deploy_file)


def test_placeholders_of_apps_not_deployed_are_left_alone(deploy_file: Path) -> None:
    deploy_file.write_text("apps: {embedder: {settings: {QP_X: /d/<dataset>}}}\n")
    refuse_placeholders(DeploySettings(), ["reranker", "workflow"])


def test_a_placeholder_in_the_overrides_is_refused_whichever_apps_deploy(
    deploy_file: Path,
) -> None:
    # Every App gets every override, so the workflow App alone uses each one.
    deploy_file.write_text("overrides: {local_models: {openvino_dir: /d/<dataset>}}\n")
    with pytest.raises(DominoDeployError) as caught:
        refuse_placeholders(DeploySettings(), ["workflow"])
    assert str(caught.value) == placeholder_message(
        "local_models.openvino_dir", "overrides", "/d/<dataset>", deploy_file
    )


def test_a_comparison_is_not_mistaken_for_a_placeholder(deploy_file: Path) -> None:
    deploy_file.write_text("defaults: {settings: {QP_RULE: 'a < b > c'}}\n")
    refuse_placeholders(DeploySettings(), ["workflow"])


# --- secrets ---


def secret_message(name: str, path: Path, block: str) -> str:
    return (
        f"{name} is set in {path} under {block}, whose values are written into the "
        "Apps' launch scripts: set it as a Domino project or user environment "
        "variable instead"
    )


@pytest.mark.parametrize(
    ("text", "name", "block"),
    [
        ("defaults: {settings: {QP_ES_API_KEY: s}}", "QP_ES_API_KEY", "defaults"),
        (
            "apps: {workflow: {settings: {QP_COMPONENTS__LLM__API__API_KEY: sk-1}}}",
            "QP_COMPONENTS__LLM__API__API_KEY",
            "apps.workflow",
        ),
        (
            "apps: {workflow: {settings: {QP_COMPONENTS__RERANKER__API__API_KEY: co-1}}}",
            "QP_COMPONENTS__RERANKER__API__API_KEY",
            "apps.workflow",
        ),
        (
            "apps: {embedder: {settings: {QP_COMPONENTS__EMBEDDER__API__API_KEY: dom_pat_abc123}}}",
            "QP_COMPONENTS__EMBEDDER__API__API_KEY",
            "apps.embedder",
        ),
        (
            "apps: {workflow: {settings: {OPENAI_API_KEY: sk-1}}}",
            "OPENAI_API_KEY",
            "apps.workflow",
        ),
        (
            "apps: {workflow: {settings: {COHERE_API_KEY: co-1}}}",
            "COHERE_API_KEY",
            "apps.workflow",
        ),
    ],
)
def test_a_secret_in_any_settings_block_is_refused(
    deploy_file: Path, text: str, name: str, block: str
) -> None:
    deploy_file.write_text(text + "\n")
    with pytest.raises(DominoDeployError) as caught:
        refuse_secrets(DeploySettings())
    assert str(caught.value) == secret_message(name, deploy_file, f"{block}.settings")


def test_only_a_name_ending_with_api_key_is_a_secret(deploy_file: Path) -> None:
    values = (
        "{QP_API_KEY_FILE: /run/key, QP_COMPONENTS__EMBEDDER__API__URL: 'http://e/v3'}"
    )
    deploy_file.write_text(f"defaults: {{settings: {values}}}\n")
    refuse_secrets(DeploySettings())  # neither name ends with _API_KEY


# --- the project id ---


def test_an_empty_project_id_takes_the_project_from_the_workspace(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A use case's example has `project_id: ""`: empty means "not set".
    deploy_file.write_text('project_id: ""\n')
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-of-the-workspace")
    assert DeploySettings().project_id == "proj-of-the-workspace"


# --- where the deploy settings are found ---


def test_the_deploy_file_a_variable_names_must_exist(deploy_file: Path) -> None:
    with pytest.raises(DominoDeployError) as caught:
        require_deploy_file()
    assert str(caught.value) == (
        f"QP_DEPLOY_FILE names {str(deploy_file)!r}, which is no file: point it "
        "at the deploy settings, or unset it to read deploy.yaml here"
    )
    deploy_file.write_text("topology: monolith\n")
    require_deploy_file()


def test_without_a_variable_the_deploy_file_is_the_working_directorys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("QP_DEPLOY_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DominoDeployError, match="no deploy settings: ") as caught:
        require_deploy_file()
    assert f"the working directory {tmp_path} has no deploy.yaml" in str(caught.value)
    (tmp_path / "deploy.yaml").write_text("topology: monolith\n")
    require_deploy_file()


def test_an_explicit_project_id_wins_over_the_workspace(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-of-the-workspace")
    monkeypatch.setenv("QP_DEPLOY_PROJECT_ID", "proj-named-here")
    assert DeploySettings().project_id == "proj-named-here"
