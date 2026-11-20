"""Reading the emulation's Terraform for the tests of ``tests/emulation/``.

Parsed with ``hcl2`` (no ``terraform`` binary, no cloud). Lives here, not in
``tests/helpers``, because it goes away with the emulation.

Example::

    resources("google_compute_instance")["pool"]["scheduling"]
    heredoc(render("scripts/workspace.sh.tftpl"), "DEPLOY")  # the deploy.yaml text
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

import hcl2
from hcl2.utils import SerializationOptions

from repo_paths import REPO_ROOT

TF_DIR = REPO_ROOT / "emulation" / "terraform"
SCRIPTS = TF_DIR / "scripts"
# The use case the emulation deploys. The toolkit's own tests read none of its
# files; the emulation's do, since it installs and deploys that use case.
USE_CASE_ROOT = REPO_ROOT / "query_processing_rag_usecase"
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
    "google_storage_bucket",
)
FILE_CALL = re.compile(r'(?:templatefile|file)\("\$\{path\.module\}/([^"]+)"')
TEMPLATE_CALL = re.compile(
    r'templatefile\("\$\{path\.module\}/(?P<path>[^"]+)", \{\n(?P<body>.*?)\n\s*\}\)',
    re.S,
)
PLACEHOLDER = re.compile(r"(?<!\$)\$\{([^}]*)\}")
# What render() puts in place of ${dataset_path}.
DATASET_PATH = "dummy-dataset_path"
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
    """Every ``locals`` block merged."""
    merged: dict[str, Any] = {}
    for block in terraform()["locals"]:
        merged.update(block)
    return merged


def variables() -> dict[str, dict[str, Any]]:
    """``{name: body}`` of every variable."""
    return {
        name: body for entry in terraform()["variable"] for name, body in entry.items()
    }


def tf_sources() -> str:
    """Every ``.tf`` file's text, in name order."""
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(TF_DIR.glob("*.tf"))
    )


def template_calls() -> dict[str, set[str]]:
    """``{template file: names of the variables passed to it}``."""
    return {
        match["path"]: set(re.findall(r"^\s*(\w+)\s*=", match["body"], re.M))
        for match in TEMPLATE_CALL.finditer(tf_sources())
    }


def render(template: str, values: dict[str, str] | None = None) -> str:
    """``scripts/common.sh`` then the template, placeholders given dummy values.

    ``values`` replaces the dummy of the placeholders it names, as a variable set in
    ``terraform.tfvars`` would.
    """
    given = DUMMIES | (values or {})
    text = (TF_DIR / template).read_text(encoding="utf-8")
    rendered = PLACEHOLDER.sub(lambda m: given.get(m[1], f"dummy-{m[1]}"), text)
    rendered = rendered.replace("$${", "${").replace("%%{", "%{")
    return (SCRIPTS / "common.sh").read_text(encoding="utf-8") + "\n" + rendered


def common_constant(name: str) -> str:
    """A ``NAME=value`` constant of ``scripts/common.sh``."""
    common = (SCRIPTS / "common.sh").read_text(encoding="utf-8")
    return re.search(rf"^{name}=(\S+)$", common, re.M)[1]


def heredoc(script: str, marker: str) -> str:
    """The body of the quoted here-document ``<<'marker'`` in ``script``."""
    return script.split(f"<<'{marker}'\n", 1)[1].split(f"\n{marker}\n", 1)[0]
