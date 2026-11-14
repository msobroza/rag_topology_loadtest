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
from functools import cache
from pathlib import Path
from typing import Any

import hcl2
import pytest
import yaml
from hcl2.utils import SerializationOptions

from repo_paths import REPO_ROOT

TF_DIR = REPO_ROOT / "emulation" / "terraform"
SCRIPTS = TF_DIR / "scripts"
SETUP_SH = REPO_ROOT / "emulation" / "cloudshell" / "setup.sh"
OPTIONS = SerializationOptions(
    with_comments=False, explicit_blocks=False, strip_string_quotes=True
)
IAP_RANGE = "35.235.240.0/20"
# Every resource type used here that takes labels.
LABELLED_TYPES = (
    "google_compute_instance",
    "google_compute_address",
    "google_secret_manager_secret",
)
FILE_CALL = re.compile(r'(?:templatefile|file)\("\$\{path\.module\}/([^"]+)"')
TEMPLATE_CALL = re.compile(
    r'templatefile\("\$\{path\.module\}/(?P<path>[^"]+)", \{\n(?P<body>.*?)\n\s*\}\)',
    re.S,
)
PLACEHOLDER = re.compile(r"(?<!\$)\$\{([^}]*)\}")
# The contract's example, rendered into the Workspace script in place of the real one.
PLACEMENT_SAMPLE = {
    "ssh": {"user": "ubuntu", "key": "/home/ubuntu/.ssh/id_emulation"},
    "app_port": 8888,
    "tiers": [
        {
            "id": "cpu-8",
            "cores": 8,
            "memory_gib": 32,
            "default": True,
            "machines": ["10.10.0.21", "10.10.0.22", "10.10.0.23"],
        }
    ],
}
DUMMIES = {
    "placement_json": json.dumps(PLACEMENT_SAMPLE),
    "pool_addresses": "10.10.0.21 10.10.0.22 10.10.0.23",
    "domino_port": "8899",
}


@cache
def terraform() -> dict[str, list[Any]]:
    """Every ``.tf`` file parsed, each top-level key's entries concatenated."""
    merged: dict[str, list[Any]] = {}
    for path in sorted(TF_DIR.glob("*.tf")):
        parsed = hcl2.loads(
            path.read_text(encoding="utf-8"), serialization_options=OPTIONS
        )
        for key, entries in parsed.items():
            merged.setdefault(key, []).extend(entries)
    return merged


def resources(kind: str) -> dict[str, dict[str, Any]]:
    """``{name: body}`` of every resource of ``kind``."""
    found: dict[str, dict[str, Any]] = {}
    for entry in terraform().get("resource", []):
        found.update(entry.get(kind, {}))
    return found


def locals_() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for block in terraform()["locals"]:
        merged.update(block)
    return merged


def variables() -> dict[str, dict[str, Any]]:
    return {
        name: body for entry in terraform()["variable"] for name, body in entry.items()
    }


def tf_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(TF_DIR.glob("*.tf"))
    )


def template_calls() -> dict[str, set[str]]:
    """``{template file: names of the variables passed to it}``."""
    return {
        match["path"]: set(re.findall(r"^\s*(\w+)\s*=", match["body"], re.M))
        for match in TEMPLATE_CALL.finditer(tf_sources())
    }


def render(template: str) -> str:
    """``scripts/common.sh`` then the template, placeholders given dummy values."""
    text = (TF_DIR / template).read_text(encoding="utf-8")
    rendered = PLACEHOLDER.sub(lambda m: DUMMIES.get(m[1], f"dummy-{m[1]}"), text)
    rendered = rendered.replace("$${", "${").replace("%%{", "%{")
    return (SCRIPTS / "common.sh").read_text(encoding="utf-8") + "\n" + rendered


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
        "rag-domino-emulation --host 0.0.0.0 --port $DOMINO_PORT"
        " --placement $CONFIG_DIR/placement.json --project-env $CONFIG_DIR/project.env"
    )
    assert command in script
    assert '-e ".[loadtest,emulation]"' in script
    assert locals_()["config_dir"] == "/etc/rag-emulation"
    assert locals_()["domino_port"] == 8899


def heredoc(script: str, marker: str) -> str:
    return script.split(f"<<'{marker}'\n", 1)[1].split(f"\n{marker}\n", 1)[0]


def test_the_generated_deploy_yaml_uses_only_documented_keys() -> None:
    deploy = yaml.safe_load(heredoc(render("scripts/workspace.sh.tftpl"), "DEPLOY"))
    example = (REPO_ROOT / "deploy.example.yaml").read_text(encoding="utf-8")
    documented = set(re.findall(r"^[ #]*(\w+):", example, re.M))
    assert set(deploy) <= documented
    assert set(deploy["defaults"]) <= documented
    assert set(deploy["defaults"]["settings"]) == {"RAG_ES_URL"}


def test_the_workspace_shell_sets_only_variables_the_project_reads() -> None:
    shell = heredoc(render("scripts/workspace.sh.tftpl"), "SHELLENV")
    exported = set(re.findall(r"^export (\w+)=", shell, re.M))
    assert exported == {"DOMINO_API_PROXY", "DOMINO_PROJECT_ID", "RAG_ES_URL", "PATH"}


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
    script = render(template)
    assert "%{" not in script, "template directives are not rendered by this test"
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
