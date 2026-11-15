"""The cache bucket, Spot machines and the dataset folder, in the emulation's files.

Read with ``hcl2`` and from the startup templates (no ``terraform``, no cloud): the
bucket is private and labelled, the three service accounts hold one role on it and
nothing project-wide, the pool and Elasticsearch machines follow ``var.spot`` while
the Workspace stays on-demand, every App and the Workspace's shell keep their models
under the dataset path, and the pool machines warm exactly what ``domino/app.sh``
installs at every App start.
"""

from __future__ import annotations

import re

import yaml
from emulation_terraform import (
    SCRIPTS,
    TF_DIR,
    heredoc,
    locals_,
    render,
    resources,
    terraform,
    variables,
)

from repo_paths import REPO_ROOT

# What render() puts in place of ${dataset_path}.
DATASET_PATH = "dummy-dataset_path"
SPOT_MACHINES = ("pool", "elasticsearch")
SERVICE_ACCOUNTS = ("workspace", "pool", "elasticsearch")


def common_constant(name: str) -> str:
    """A ``NAME=value`` constant of ``scripts/common.sh``."""
    common = (SCRIPTS / "common.sh").read_text(encoding="utf-8")
    return re.search(rf"^{name}=(\S+)$", common, re.M)[1]


def install_project_commands() -> list[str]:
    """The ``uv`` lines of ``install_project`` in ``domino/app.sh``."""
    app_sh = (REPO_ROOT / "domino" / "app.sh").read_text(encoding="utf-8")
    body = app_sh.split("install_project() {", 1)[1].split("\n}", 1)[0]
    return [
        line.strip() for line in body.splitlines() if line.strip().startswith("uv ")
    ]


def test_the_cache_bucket_is_private_labelled_and_goes_with_destroy() -> None:
    bucket = resources("google_storage_bucket")["cache"]

    assert bucket["name"] == "${local.cache_bucket}"
    assert bucket["uniform_bucket_level_access"] is True
    assert bucket["public_access_prevention"] == "enforced"
    assert bucket["force_destroy"] is True
    assert bucket["labels"] == "${local.labels}"
    assert "storage.googleapis.com" in locals_()["apis"]


def test_the_cache_bucket_name_defaults_to_the_project_and_the_emulation() -> None:
    assert variables()["cache_bucket"]["default"] == ""
    expected = '"${var.project_id}-${var.name}-cache"'
    assert expected in locals_()["cache_bucket"]


def test_each_service_account_uses_the_cache_bucket_and_nothing_project_wide() -> None:
    grants = resources("google_storage_bucket_iam_member")
    assert set(grants) == {"cache_users", "elasticsearch_reads_cache_bucket"}
    grant = grants["cache_users"]

    assert grant["bucket"] == "${google_storage_bucket.cache.name}"
    assert grant["role"] == "roles/storage.objectUser"
    assert grant["member"] == "serviceAccount:${each.value}"
    assert grant["for_each"] == {
        account: f"${{google_service_account.{account}.email}}"
        for account in SERVICE_ACCOUNTS
    }
    kinds = {kind for entry in terraform()["resource"] for kind in entry}
    assert not [kind for kind in kinds if kind.startswith("google_project_iam")]


def test_elasticsearch_also_reads_the_cache_buckets_metadata() -> None:
    grant = resources("google_storage_bucket_iam_member")[
        "elasticsearch_reads_cache_bucket"
    ]

    assert grant["bucket"] == "${google_storage_bucket.cache.name}"
    assert grant["role"] == "roles/storage.legacyBucketReader"
    member = "serviceAccount:${google_service_account.elasticsearch.email}"
    assert grant["member"] == member
    elasticsearch = resources("google_compute_instance")["elasticsearch"]
    waited_for = "${google_storage_bucket_iam_member.elasticsearch_reads_cache_bucket}"
    assert waited_for in elasticsearch["depends_on"]


def test_every_machine_boots_after_its_grant_on_the_cache_bucket() -> None:
    for name, instance in resources("google_compute_instance").items():
        grant = "${google_storage_bucket_iam_member.cache_users}"
        assert grant in instance["depends_on"], name


def test_the_pool_and_elasticsearch_machines_follow_var_spot() -> None:
    assert variables()["spot"]["default"] is True
    for name in SPOT_MACHINES:
        (scheduling,) = resources("google_compute_instance")[name]["scheduling"]
        assert scheduling["provisioning_model"] == '${var.spot ? "SPOT" : "STANDARD"}'
        assert scheduling["preemptible"] == "${var.spot}"
        assert scheduling["automatic_restart"] == "${!var.spot}"


def test_the_workspace_machine_is_never_spot() -> None:
    (scheduling,) = resources("google_compute_instance")["workspace"]["scheduling"]

    assert scheduling["provisioning_model"] == "STANDARD"
    assert "preemptible" not in scheduling


def test_every_app_keeps_its_models_under_the_dataset_path() -> None:
    deploy = yaml.safe_load(heredoc(render("scripts/workspace.sh.tftpl"), "DEPLOY"))
    settings = deploy["defaults"]["settings"]
    local_models = deploy["overrides"]["local_models"]  # a value of the workflow file

    openvino, ovms = common_constant("DATASET_OPENVINO"), common_constant(
        "DATASET_OVMS"
    )
    assert local_models["openvino_dir"] == f"{DATASET_PATH}/{openvino}"
    assert settings["RAG_OVMS_MODELS_DIR"] == f"{DATASET_PATH}/{ovms}"


def test_the_workspace_shell_ingests_with_the_dataset_folder() -> None:
    shell = heredoc(render("scripts/workspace.sh.tftpl"), "SHELLENV")
    openvino = common_constant("DATASET_OPENVINO")

    assert f"export RAG_LOCAL_MODELS__OPENVINO_DIR='{DATASET_PATH}/{openvino}'" in shell


def test_the_dataset_path_default_is_a_domino_dataset_folder() -> None:
    assert variables()["dataset_path"]["default"] == "/domino/datasets/local/emulation"
    example = (TF_DIR / "terraform.tfvars.example").read_text(encoding="utf-8")
    for name in ("cache_bucket", "dataset_path", "spot"):
        assert re.search(rf"^# {name}\s+=", example, re.M), name


def test_the_pool_warms_what_app_sh_installs_at_every_app_start() -> None:
    commands = install_project_commands()
    pool = (TF_DIR / "scripts" / "pool.sh.tftpl").read_text(encoding="utf-8")

    assert len(commands) == 2, commands
    for command in commands:
        warmed = command.replace('"$1', '"$HOME/rag-venv').replace(
            '"$2"', '".[domino]"'
        )
        assert warmed in pool, f"{warmed!r} not in pool.sh.tftpl"


def test_each_machine_restores_its_models_after_writing_the_cache_config() -> None:
    expected = {
        "pool.sh.tftpl": "$CACHE_MODELS_PREFIX",
        "workspace.sh.tftpl": "$CACHE_OPENVINO_PREFIX",
    }
    for template, prefix in expected.items():
        lines = (SCRIPTS / template).read_text(encoding="utf-8").splitlines()
        sync = lines.index('sync_repo "$REPO_URL" "$GIT_REF" "$REPO_PATH"')
        write = lines.index('write_cache_config "$CACHE_BUCKET" "$DATASET_PATH"')
        restore = lines.index(
            f'restore_cached_models "$CACHE_BUCKET" "{prefix}" "$DATASET_PATH"'
        )
        assert sync < write < restore, template
