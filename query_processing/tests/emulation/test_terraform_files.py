"""The emulation's Terraform, read with ``hcl2`` (no ``terraform`` binary, no cloud).

Checks what would otherwise only show after an apply, or never: no machine has a
public address, every machine deletes itself and carries the emulation label, only
IAP's range is let in from outside, the placement file matches the fake Domino's
contract, every template exists and gets exactly the variables it uses, and every
startup script parses with ``bash -n`` once its placeholders are rendered.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from emulation_terraform import (
    FILE_CALL,
    IAP_RANGE,
    LABELLED_TYPES,
    PLACEHOLDER,
    PLACEMENT_SAMPLE,
    SCRIPTS,
    SETUP_SH,
    TF_DIR,
    USE_CASE_ROOT,
    heredoc,
    locals_,
    render,
    resources,
    template_calls,
    terraform,
    tf_sources,
    variables,
)

from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.overrides import (
    check_overrides,
    refuse_workflow_settings_in_free_lists,
)
from repo_paths import REPO_ROOT, TOOLKIT_ROOT


def test_every_tf_file_parses_and_names_the_backend_and_providers() -> None:
    settings = terraform()["terraform"][0]
    assert settings["backend"] == [{"gcs": {}}]
    assert set(settings["required_providers"][0]) == {"google", "tls"}


def test_no_instance_has_a_public_address() -> None:
    for name, instance in resources("google_compute_instance").items():
        for interface in instance["network_interface"]:
            assert "access_config" not in interface, name
            assert "ipv6_access_config" not in interface, name


def test_every_instance_deletes_itself_after_the_max_run_duration() -> None:
    assert locals_()["max_run_seconds"] == "${var.max_run_hours * 3600}"
    assert variables()["max_run_hours"]["default"] == 8
    for name, instance in resources("google_compute_instance").items():
        (scheduling,) = instance["scheduling"]
        assert scheduling["instance_termination_action"] == "DELETE", name
        assert scheduling["max_run_duration"] == [
            {"seconds": "${local.max_run_seconds}"}
        ]


def test_every_labelled_resource_carries_the_emulation_label() -> None:
    assert locals_()["labels"] == '${merge(var.labels, {"rag-emulation" = var.name})}'
    for kind in LABELLED_TYPES:
        for name, body in resources(kind).items():
            assert "local.labels" in body.get("labels", ""), f"{kind}.{name}"


def test_every_instance_disk_carries_the_emulation_label() -> None:
    for name, instance in resources("google_compute_instance").items():
        (disk,) = instance["boot_disk"]
        assert disk["initialize_params"][0]["labels"] == "${local.labels}", name


def test_only_iaps_range_gets_in_from_outside() -> None:
    assert locals_()["iap_range"] == IAP_RANGE
    for name, rule in resources("google_compute_firewall").items():
        assert rule["direction"] == "INGRESS", name
        if "source_ranges" in rule:
            assert rule["source_ranges"] == ["${local.iap_range}"], name
        else:
            assert rule["source_tags"], f"{name} must name its sources"


def test_ssh_is_open_only_to_iap_and_to_the_workspace() -> None:
    for name, rule in resources("google_compute_firewall").items():
        ports = [port for allow in rule["allow"] for port in allow.get("ports", [])]
        if "22" not in ports:
            continue
        sources = rule.get("source_ranges", rule.get("source_tags"))
        assert sources in (["${local.iap_range}"], ["${local.tags.workspace}"]), name


def test_the_placement_file_matches_the_fake_dominos_contract() -> None:
    values = locals_()
    placement = values["placement"]
    assert set(placement) == {"ssh", "app_port", "tiers"}
    assert placement["ssh"] == {"user": "${local.ssh_user}", "key": "${local.ssh_key}"}
    assert placement["app_port"] == "${local.app_port}"
    assert (values["ssh_user"], values["ssh_key"]) == (
        "ubuntu",
        "/home/ubuntu/.ssh/id_emulation",
    )
    assert values["app_port"] == 8888
    tier_keys = set(re.findall(r"[{,]\s*(\w+) = ", placement["tiers"]))
    assert tier_keys == {"id", "cores", "memory_gib", "default", "machines"}


def test_the_fake_domino_runs_with_the_contracts_command_line() -> None:
    script = (SCRIPTS / "workspace.sh.tftpl").read_text(encoding="utf-8")
    command = (
        "qp-domino-emulation --host 0.0.0.0 --port $DOMINO_PORT"
        " --placement $CONFIG_DIR/placement.json --project-env $CONFIG_DIR/project.env"
    )
    assert command in script
    assert locals_()["config_dir"] == "/etc/rag-emulation"
    assert locals_()["domino_port"] == 8899


def test_the_machines_install_both_and_deploy_from_the_use_cases_directory() -> None:
    toolkit, use_case = TOOLKIT_ROOT.name, USE_CASE_ROOT.name
    workspace = (SCRIPTS / "workspace.sh.tftpl").read_text(encoding="utf-8")
    pool = (SCRIPTS / "pool.sh.tftpl").read_text(encoding="utf-8")
    # The repository's one virtualenv, at the clone's root, and in one command the
    # toolkit, then the use case that depends on it.
    assert (
        'uv venv --python "$(cat .python-version)" .venv\' _ "$REPO_PATH"' in workspace
    )
    installed = f'-e "./{toolkit}[loadtest,emulation]" -e ./{use_case}\' _ "$REPO_PATH"'
    assert installed in workspace
    # qp-deploy runs in the use case's directory and reads its deploy.yaml there.
    assert f'tee "$REPO_PATH/{use_case}/deploy.yaml"' in workspace
    # The OVMS pin is the toolkit's: common.sh reads <toolkit>/ovms/versions.env.
    assert f'fetch_ovms_bundle "$REPO_PATH/{toolkit}" "$OVMS_BUNDLE"' in pool
    assert (USE_CASE_ROOT / "workflow.yaml").is_file()
    assert (REPO_ROOT / ".python-version").is_file()
    assert (TOOLKIT_ROOT / "ovms" / "versions.env").is_file()


def test_the_forward_timeout_is_a_variable_of_at_least_one_second() -> None:
    # A generated answer can outlast the emulated Domino's 30 s forward timeout.
    variable = variables()["forward_timeout_seconds"]
    assert (variable["type"], variable["default"]) == ("number", 30)
    (rule,) = variable["validation"]
    assert rule["condition"] == "${var.forward_timeout_seconds >= 1}"
    workspace = template_calls()["scripts/workspace.sh.tftpl"]
    assert "forward_timeout_seconds" in workspace


def test_the_generated_deploy_yaml_uses_only_documented_keys() -> None:
    deploy = yaml.safe_load(heredoc(render("scripts/workspace.sh.tftpl"), "DEPLOY"))
    example = (USE_CASE_ROOT / "deploy.example.yaml").read_text(encoding="utf-8")
    documented = set(re.findall(r"^[ #]*(\w+):", example, re.M))
    assert set(deploy) <= documented
    assert set(deploy["defaults"]) <= documented
    # The free list holds only what the launch script reads; the values of the
    # workflow file are overrides, which qp-deploy checks against it.
    assert set(deploy["defaults"]["settings"]) == {"QP_OVMS_MODELS_DIR"}
    assert deploy["overrides"] == {
        "components": {"vector_store": {"elasticsearch": {"url": "dummy-es_url"}}},
        "local_models": {"openvino_dir": "dummy-dataset_path/openvino"},
    }


def test_the_generated_deploy_yaml_passes_qp_deploys_checks_of_its_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A value of the workflow file in a free list, or a misspelt override, would
    # stop every deploy of the emulation before anything is published.
    path = tmp_path / "deploy.yaml"
    path.write_text(heredoc(render("scripts/workspace.sh.tftpl"), "DEPLOY"))
    monkeypatch.setenv("QP_DEPLOY_FILE", str(path))
    # Checked against the workflow file of the use case the emulation deploys,
    # and the routes file beside it, which qp-deploy finds in that directory:
    # the file declares a router, whose routes the settings look up there.
    monkeypatch.setenv("QP_WORKFLOW_FILE", str(USE_CASE_ROOT / "workflow.yaml"))
    monkeypatch.setenv("QP_ROUTES_FILE", str(USE_CASE_ROOT / "routes.yaml"))
    settings = DeploySettings()
    refuse_workflow_settings_in_free_lists(settings)
    check_overrides(settings)


def test_the_generated_deploy_yaml_defines_the_examples_topology_aliases() -> None:
    # qp-deploy defines no alias in code: without the table it deploys nothing.
    deploy = yaml.safe_load(heredoc(render("scripts/workspace.sh.tftpl"), "DEPLOY"))
    example = yaml.safe_load((USE_CASE_ROOT / "deploy.example.yaml").read_text())
    assert deploy["topology_alias"] == example["topology_alias"]
    assert set(example["topology_alias"]) == {"monolith", "split-reranker", "split-all"}


def test_the_workspace_shell_sets_only_variables_the_project_reads() -> None:
    shell = heredoc(render("scripts/workspace.sh.tftpl"), "SHELLENV")
    exported = set(re.findall(r"^export (\w+)=", shell, re.M))
    assert exported == {
        "DOMINO_API_PROXY",
        "DOMINO_PROJECT_ID",
        "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL",
        "QP_LOCAL_MODELS__OPENVINO_DIR",
        "PATH",
    }


@pytest.mark.parametrize(
    "template", ["scripts/pool.sh.tftpl", "scripts/workspace.sh.tftpl"]
)
def test_pool_and_workspace_machines_fetch_the_ovms_bundle(template: str) -> None:
    # qp-deploy checks deploy.yaml's ovms_bundle on the Workspace machine: in the
    # sitting of 15 November 2026 only the pool machines held it.
    script = (TF_DIR / template).read_text(encoding="utf-8")
    assert "OVMS_BUNDLE='${ovms_bundle_path}'" in script
    # The pin is the toolkit's: query_processing/ovms/versions.env in the clone.
    call = 'fetch_ovms_bundle "$REPO_PATH/query_processing" "$OVMS_BUNDLE"'
    assert call in script.splitlines()
    assert "fetch_ovms_bundle()" not in script, "defined once, in common.sh"
    assert "ovms_bundle_path" in template_calls()[template]


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck absent")
@pytest.mark.parametrize("template", sorted(template_calls()))
def test_each_rendered_startup_script_passes_shellcheck(template: str) -> None:
    result = subprocess.run(
        ["shellcheck", "-s", "bash", "-S", "warning", "-"],
        input=render(template),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout


def test_the_default_hardware_tier_is_one_pool_of_three_n2_standard_8() -> None:
    tiers = variables()["hardware_tiers"]
    assert tiers["default"] == {
        "cpu-8": {"machine_type": "n2-standard-8", "count": 3, "default": True}
    }
    conditions = [rule["condition"] for rule in tiers["validation"]]
    assert any("tier.default" in rule and "== 1" in rule for rule in conditions)


def test_only_the_project_and_the_git_ref_are_required() -> None:
    required = {name for name, body in variables().items() if "default" not in body}
    assert required == {"project_id", "git_ref"}
    undescribed = [
        name for name, body in variables().items() if not body.get("description")
    ]
    assert not undescribed


def test_every_referenced_file_exists_and_every_script_is_used() -> None:
    referenced = set(FILE_CALL.findall(tf_sources()))
    missing = [path for path in referenced if not (TF_DIR / path).is_file()]
    assert not missing
    on_disk = {str(path.relative_to(TF_DIR)) for path in SCRIPTS.iterdir()}
    assert on_disk == referenced


@pytest.mark.parametrize("template", sorted(template_calls()))
def test_each_template_gets_exactly_the_variables_it_uses(template: str) -> None:
    used = set(PLACEHOLDER.findall((TF_DIR / template).read_text(encoding="utf-8")))
    assert used == template_calls()[template]


@pytest.mark.parametrize("template", sorted(template_calls()))
def test_each_rendered_startup_script_parses_with_bash(template: str) -> None:
    # Only the template is rendered by Terraform: common.sh is read with file(), so a
    # "%{" there is curl's -w format, not a directive.
    source = (TF_DIR / template).read_text(encoding="utf-8")
    assert not re.search(r"(?<!%)%\{", source), "directives are not rendered here"
    script = render(template)
    result = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("template", sorted(template_calls()))
def test_each_startup_script_ends_by_reporting_done(template: str) -> None:
    lines = render(template).rstrip().splitlines()
    assert 'guest_attribute startup "done"' in lines[-2:]


def test_the_placement_json_reaches_its_file_unchanged() -> None:
    body = heredoc(render("scripts/workspace.sh.tftpl"), "PLACEMENT")
    assert json.loads(body) == PLACEMENT_SAMPLE


def test_the_private_key_is_never_in_instance_metadata() -> None:
    for name, instance in resources("google_compute_instance").items():
        metadata = json.dumps(instance["metadata"])
        assert not re.search(r"private_key_\w+|secret_data", metadata), name
        assert instance["metadata"]["enable-guest-attributes"] == "TRUE", name
    pool = resources("google_compute_instance")["pool"]
    assert "public_key_openssh" in pool["metadata"]["ssh-keys"]


def test_the_shared_script_and_the_cloud_shell_setup_parse_with_bash() -> None:
    for path in (SCRIPTS / "common.sh", SETUP_SH):
        result = subprocess.run(
            ["bash", "-n", str(path)], text=True, capture_output=True
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_the_outputs_give_the_iap_commands() -> None:
    outputs = {
        name: body["value"]
        for entry in terraform()["output"]
        for name, body in entry.items()
    }
    assert "--tunnel-through-iap" in outputs["ssh_workspace"]
    assert "start-iap-tunnel" in outputs["tunnel_fake_domino"]
    assert "local.domino_port" in outputs["tunnel_fake_domino"]
    assert "local.locust_port" in outputs["tunnel_locust"]
    assert {"workspace_name", "workspace_zone", "elasticsearch_url"} <= set(outputs)


def test_no_project_id_is_hard_coded() -> None:
    assert not re.search(r'project\s*=\s*"[a-z]', tf_sources())
    assert Path(TF_DIR / "terraform.tfvars.example").is_file()
