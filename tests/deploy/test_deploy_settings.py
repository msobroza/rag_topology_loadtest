"""Tests of rag-deploy's settings: ``deploy.yaml``, its per-App blocks, the
``RAG_DEPLOY_*`` overrides, the refused secrets and the project id fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from domino_deploy_helpers import DEPLOY_YAML
from rag_load_test.deploy.apps_client import DominoDeployError
from rag_load_test.deploy.deploy_settings import (
    AppSettings,
    Autoscaling,
    DeploySettings,
    refuse_placeholders,
    refuse_secrets,
)
from repo_paths import REPO_ROOT


@pytest.fixture
def deploy_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The path RAG_DEPLOY_FILE names; nothing from the shell leaks in."""
    for name in ("RAG_DEPLOY_TOPOLOGY", "RAG_DEPLOY_PROJECT_ID", "DOMINO_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "deploy.yaml"
    monkeypatch.setenv("RAG_DEPLOY_FILE", str(path))
    return path


def test_each_app_takes_its_own_value_else_the_default(deploy_file: Path) -> None:
    deploy_file.write_text(DEPLOY_YAML)
    settings = DeploySettings()
    assert settings.topology == "split-reranker"
    assert settings.for_app("workflow") == AppSettings(
        part="workflow",
        kind="agent",
        environment_id="env-123",
        hardware_tier_id="small",
        autoscaling=Autoscaling(min_replicas=1, max_replicas=4),
        settings={
            "RAG_ES_URL": "https://es.test:9200",
            "RAG_RERANK_MODEL": "/mnt/data/models/bge reranker",
        },
    )
    embedder = settings.for_app("embedder")
    assert (embedder.kind, embedder.hardware_tier_id) == ("app", "cpu-8")
    assert embedder.autoscaling is None
    assert list(embedder.settings) == [  # merged, sorted by name
        "RAG_EMBEDDING_MODEL",
        "RAG_ES_URL",
        "RAG_RERANK_MODEL",
    ]


def test_an_apps_settings_win_over_the_defaults(deploy_file: Path) -> None:
    deploy_file.write_text(
        "defaults: {settings: {RAG_ES_URL: https://shared.test, RAG_A: a}}\n"
        "apps: {reranker: {settings: {RAG_ES_URL: https://own.test}}}\n"
    )
    assert DeploySettings().for_app("reranker").settings == {
        "RAG_A": "a",
        "RAG_ES_URL": "https://own.test",
    }


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
        "defaults:\n  settings:\n    RAG_MODEL_WORKERS: 2\n    RAG_FLAG: true\n"
        "    RAG_RATIO: 0.5\n    RAG_EMPTY: ''\n"
    )
    assert DeploySettings().for_app("embedder").settings == {
        "RAG_EMPTY": "",
        "RAG_FLAG": "true",
        "RAG_MODEL_WORKERS": "2",
        "RAG_RATIO": "0.5",
    }


def test_a_setting_that_is_not_a_scalar_is_refused_with_its_name(
    deploy_file: Path,
) -> None:
    deploy_file.write_text("defaults: {settings: {RAG_LIST: [1, 2]}}\n")
    with pytest.raises(ValidationError) as caught:
        DeploySettings()
    assert "RAG_LIST must be a string, a number or a boolean, not [1, 2]" in str(
        caught.value
    )


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("topolgy: monolith\n", "topolgy"),
        ("apps: {embedder: {hardware_tier: cpu-8}}\n", "apps.embedder.hardware_tier"),
        ("defaults: {autoscaling: {max: 4}}\n", "defaults.autoscaling.max"),
        ("apps: {orchestrator: {kind: agent}}\n", "apps.orchestrator"),
    ],
)
def test_an_unknown_key_is_refused_with_its_name(
    deploy_file: Path, text: str, where: str
) -> None:
    deploy_file.write_text(text)
    with pytest.raises(ValidationError) as caught:
        DeploySettings()
    assert where in str(caught.value)


def test_rag_deploy_variables_override_the_file(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_file.write_text("topology: split-all\nwait_timeout_s: 30\n")
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "monolith")
    settings = DeploySettings()
    assert (settings.topology, settings.wait_timeout_s) == ("monolith", 30.0)


def test_without_a_file_every_value_is_its_default(deploy_file: Path) -> None:
    settings = DeploySettings()
    assert settings.model_dump(exclude={"defaults", "apps"}) == {
        "topology": "split-all",
        "model_server": "fastapi",
        "project_id": "",
        "ovms_bundle": "",
        "domino_host": "",
        "wait_timeout_s": 600.0,
        "poll_interval_s": 5.0,
        "no_instance_grace_s": 60.0,
        "http_timeout_s": 30.0,
        "http_retries": 2,
        "run_token_url": "http://localhost:8899/access-token",
    }
    assert settings.for_app("reranker").environment_id == ""


def test_deploy_yaml_in_the_working_directory_is_read_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RAG_DEPLOY_FILE", raising=False)
    monkeypatch.delenv("RAG_DEPLOY_TOPOLOGY", raising=False)
    monkeypatch.chdir(tmp_path)
    Path("deploy.yaml").write_text("topology: monolith\n")
    assert DeploySettings().topology == "monolith"


def test_app_settings_come_from_the_file_only(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_ES_URL", "https://from-the-shell.test")
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
            "defaults: {settings: {RAG_ES_URL: 'https://<es host>:9200'}}",
            "RAG_ES_URL",
            "defaults.settings",
            "https://<es host>:9200",
        ),
        (
            "apps: {embedder: {settings: {RAG_EMBEDDING_MODEL: /d/<dataset>/m}}}",
            "RAG_EMBEDDING_MODEL",
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
    deploy_file.write_text("apps: {embedder: {settings: {RAG_X: /d/<dataset>}}}\n")
    refuse_placeholders(DeploySettings(), ["reranker", "workflow"])


def test_a_comparison_is_not_mistaken_for_a_placeholder(deploy_file: Path) -> None:
    deploy_file.write_text("defaults: {settings: {RAG_RULE: 'a < b > c'}}\n")
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
        ("defaults: {settings: {RAG_ES_API_KEY: s}}", "RAG_ES_API_KEY", "defaults"),
        (
            "apps: {workflow: {settings: {RAG_GENERATION_API_KEY: sk-1}}}",
            "RAG_GENERATION_API_KEY",
            "apps.workflow",
        ),
        (
            "apps: {workflow: {settings: {RAG_RERANK_API_KEY: co-1}}}",
            "RAG_RERANK_API_KEY",
            "apps.workflow",
        ),
        (
            "apps: {embedder: {settings: {RAG_EMBEDDING_API_KEY: dom_pat_abc123}}}",
            "RAG_EMBEDDING_API_KEY",
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


@pytest.mark.parametrize(
    "name",
    [
        "RAG_EMBEDDING_API_KEY",
        "RAG_RERANK_API_KEY",
        "RAG_GENERATION_API_KEY",
        "RAG_ES_API_KEY",
    ],
)
@pytest.mark.parametrize("value", ["domino", ""])
def test_an_empty_or_domino_api_key_passes_through(
    deploy_file: Path, name: str, value: str
) -> None:
    deploy_file.write_text(f"defaults: {{settings: {{{name}: '{value}'}}}}\n")
    settings = DeploySettings()
    refuse_secrets(settings)
    assert settings.for_app("workflow").settings == {name: value}


def test_only_a_name_ending_with_api_key_is_a_secret(deploy_file: Path) -> None:
    values = "{RAG_API_KEY_FILE: /run/key, RAG_EMBEDDING_API_URL: 'http://e/v3'}"
    deploy_file.write_text(f"defaults: {{settings: {values}}}\n")
    refuse_secrets(DeploySettings())  # neither name ends with _API_KEY


# --- the project id ---

EXAMPLE_YAML = REPO_ROOT / "deploy.example.yaml"


def test_the_example_file_copied_as_is_takes_the_project_from_the_workspace(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The example has `project_id: ""`: empty means "not set".
    deploy_file.write_text(EXAMPLE_YAML.read_text())
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-of-the-workspace")
    settings = DeploySettings()
    assert settings.project_id == "proj-of-the-workspace"
    refuse_secrets(settings)  # the example holds no secret


def test_the_example_file_copied_as_is_stops_on_its_first_placeholder(
    deploy_file: Path,
) -> None:
    deploy_file.write_text(EXAMPLE_YAML.read_text())
    with pytest.raises(DominoDeployError) as caught:
        refuse_placeholders(DeploySettings(), ["embedder", "reranker", "workflow"])
    message = placeholder_message(
        "environment_id", "defaults", "<environment id>", deploy_file
    )
    assert str(caught.value) == message


def test_the_example_file_lists_the_model_paths_once_for_every_app(
    deploy_file: Path,
) -> None:
    deploy_file.write_text(EXAMPLE_YAML.read_text())
    settings = DeploySettings()
    for part in ("embedder", "reranker", "workflow"):
        assert {"RAG_ES_URL", "RAG_EMBEDDING_MODEL", "RAG_RERANK_MODEL"} <= set(
            settings.for_app(part).settings
        )
    assert settings.for_app("workflow").kind == "agent"


def test_an_explicit_project_id_wins_over_the_workspace(
    deploy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-of-the-workspace")
    monkeypatch.setenv("RAG_DEPLOY_PROJECT_ID", "proj-named-here")
    assert DeploySettings().project_id == "proj-named-here"
