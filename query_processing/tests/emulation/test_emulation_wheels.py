"""A machine of the emulation takes doclang's core from the cache bucket.

It has no clone of ``document_doclang``: on the laptop, where the clone is and
Terraform is not, ``make -C emulation wheel-save CACHE_BUCKET=<name>`` copies
the wheel that ``make doclang-wheel`` built into the bucket, and at boot each
machine copies every wheel there to its dataset folder, names it with
``QP_BOOTSTRAP_WHEELS`` for its own install, and the Workspace's ``deploy.yaml``
names it for the Apps. ``curl`` plays the bucket
(``tests/emulation/emulation_machine_fakes.py``); ``wheel-save`` runs on
stand-ins of ``gcloud`` and Terraform; nothing reaches the network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from emulation_machine_fakes import (
    BUCKET,
    COMMON_SH,
    FakeMachine,
    machine_env,
    make_machine,
)
from emulation_terraform import (
    DATASET_PATH,
    SCRIPTS,
    common_constant,
    heredoc,
    render,
)

from repo_paths import REPO_ROOT

DOCLANG_WHEEL = "doclang_domain_core-0.1.0-py3-none-any.whl"
EMULATION = REPO_ROOT / "emulation"
# Each startup script that installs the project, and the line that installs it.
INSTALLS = {
    "pool.sh.tftpl": "warm_app_dependencies",
    "workspace.sh.tftpl": "install_project",
}


@pytest.fixture
def machine(tmp_path: Path) -> FakeMachine:
    return make_machine(tmp_path)


def in_the_bucket(machine: FakeMachine, *names: str) -> None:
    """Objects under the bucket's wheels/ prefix, as wheel-save leaves them."""
    for name in names:
        path = machine.bucket / "wheels" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"the wheel {name}")


def restore_wheels(machine: FakeMachine) -> subprocess.CompletedProcess[str]:
    """``restore_wheels`` of ``common.sh`` into the dataset, then what it exported."""
    command = (
        '. "$1" && restore_wheels "$2" "$3" '
        '&& echo "QP_BOOTSTRAP_WHEELS=$QP_BOOTSTRAP_WHEELS"'
    )
    return subprocess.run(
        ["bash", "-c", command, "_", str(COMMON_SH), BUCKET, str(machine.dataset)],
        env=machine_env(machine, QP_EMULATION_SOURCED="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_machine_copies_every_wheel_of_the_bucket_and_names_the_folder(
    machine: FakeMachine,
) -> None:
    in_the_bucket(
        machine, DOCLANG_WHEEL, "doclang_domain_ingestion-0.1.0-py3-none-any.whl"
    )
    folder = machine.dataset / "wheels"

    result = restore_wheels(machine)

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in folder.iterdir()) == [
        DOCLANG_WHEEL,
        "doclang_domain_ingestion-0.1.0-py3-none-any.whl",
    ]
    assert (folder / DOCLANG_WHEEL).read_text() == f"the wheel {DOCLANG_WHEEL}"
    assert result.stdout.splitlines() == [f"QP_BOOTSTRAP_WHEELS={folder}"]


def test_a_wheel_no_longer_in_the_bucket_leaves_the_machine(
    machine: FakeMachine,
) -> None:
    # A reboot after another build: the folder holds what the bucket holds.
    folder = machine.dataset / "wheels"
    folder.mkdir()
    (folder / "doclang_domain_core-0.0.9-py3-none-any.whl").write_text("older")
    in_the_bucket(machine, DOCLANG_WHEEL)

    result = restore_wheels(machine)

    assert result.returncode == 0, result.stderr
    assert [path.name for path in folder.iterdir()] == [DOCLANG_WHEEL]


@pytest.mark.parametrize("held", [(), ("other-1.0-py3-none-any.whl",)])
def test_with_no_wheel_of_doclangs_core_in_the_bucket_the_boot_stops_in_one_line(
    machine: FakeMachine, held: tuple[str, ...]
) -> None:
    in_the_bucket(machine, *held)

    result = restore_wheels(machine)

    assert result.returncode != 0
    [line] = result.stderr.splitlines()
    assert f"gs://{BUCKET}/wheels/" in line
    assert "doclang_domain_core-*.whl" in line
    assert "make -C emulation wheel-save" in line
    assert result.stdout == ""


@pytest.mark.parametrize(("template", "install"), INSTALLS.items())
def test_each_machine_installs_with_the_wheels_it_copied_at_boot(
    template: str, install: str
) -> None:
    lines = (SCRIPTS / template).read_text(encoding="utf-8").splitlines()
    write = lines.index('write_cache_config "$CACHE_BUCKET" "$DATASET_PATH"')
    restore = lines.index('restore_wheels "$CACHE_BUCKET" "$DATASET_PATH"')
    installed = lines.index(install)

    assert write < restore < installed, template
    uv_installs = [line for line in lines if "uv pip install" in line]
    assert uv_installs, f"no uv pip install in {template}"
    for line in uv_installs:
        assert '--find-links "$2"' in line, line
        assert line.endswith(' "$QP_BOOTSTRAP_WHEELS"'), line


def test_every_app_is_handed_the_wheels_folder_of_its_machine() -> None:
    # The same path on every machine, as on Domino the same Dataset.
    deploy = yaml.safe_load(heredoc(render("scripts/workspace.sh.tftpl"), "DEPLOY"))

    wheels = common_constant("DATASET_WHEELS")
    assert deploy["bootstrap_wheels"] == f"{DATASET_PATH}/{wheels}"


TERRAFORMS_BUCKET = "bucket-from-terraform"


def a_machine_for_wheel_save(where: Path, *, terraform: bool) -> Path:
    """Stand-ins of ``gcloud`` and, if ``terraform``, of Terraform, under ``where/bin``.

    ``gcloud`` logs its argv to ``where/gcloud.log``; ``terraform`` answers the
    cache bucket's name with ``TERRAFORMS_BUCKET``, as after the first apply.

    Returns:
        The folder of the stand-ins, to put first on the PATH.
    """
    bin_dir = where / "bin"
    bin_dir.mkdir(parents=True)
    stand_ins = {"gcloud": f'echo "$@" >> "{where}/gcloud.log"'}
    if terraform:
        question = "-chdir=terraform output -raw cache_bucket"
        stand_ins["terraform"] = (
            f'[ "$*" = "{question}" ] || exit 1\nprintf {TERRAFORMS_BUCKET}'
        )
    for name, body in stand_ins.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    return bin_dir


def wheel_save(
    where: Path, *args: str, terraform: bool
) -> subprocess.CompletedProcess[str]:
    """``make wheel-save`` in the emulation's folder, on stand-ins, a wheel built.

    The wheel is in ``where/wheels``; no Terraform, gcloud or network is reached.
    """
    wheels = where / "wheels"
    wheels.mkdir()
    (wheels / DOCLANG_WHEEL).write_text("the wheel")
    bin_dir = a_machine_for_wheel_save(where, terraform=terraform)
    return subprocess.run(
        ["make", "wheel-save", f"WHEELS={wheels}", *args],
        cwd=EMULATION,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(where)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def gcloud_was_asked(where: Path) -> list[str]:
    log = where / "gcloud.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize(
    ("terraform", "given", "bucket"),
    [
        (True, (), TERRAFORMS_BUCKET),
        (False, ("CACHE_BUCKET=named-bucket",), "named-bucket"),
        (True, ("CACHE_BUCKET=named-bucket",), "named-bucket"),
    ],
    ids=["terraform-names-it", "named-without-terraform", "named-over-terraform"],
)
def test_wheel_save_copies_the_built_wheel_under_the_prefix_the_machines_read(
    tmp_path: Path, terraform: bool, given: tuple[str, ...], bucket: str
) -> None:
    # The laptop builds the wheel and has no Terraform: it names the bucket.
    run = wheel_save(tmp_path, *given, terraform=terraform)

    assert run.returncode == 0, run.stderr
    prefix = common_constant("CACHE_WHEELS_PREFIX")
    wheel = tmp_path / "wheels" / DOCLANG_WHEEL
    assert gcloud_was_asked(tmp_path) == [f"storage cp {wheel} gs://{bucket}/{prefix}"]


def test_wheel_save_without_terraform_nor_a_name_stops_in_one_line(
    tmp_path: Path,
) -> None:
    # It went on with an empty name: gs:///wheels/.
    run = wheel_save(tmp_path, terraform=False)

    assert run.returncode != 0
    [line, *make_says] = run.stderr.splitlines()
    assert "CACHE_BUCKET=<cache bucket>" in line
    assert "terraform" in line
    assert all(said.startswith("make: ***") for said in make_says)
    assert gcloud_was_asked(tmp_path) == []


def test_the_readme_says_where_the_wheel_is_built_and_saved() -> None:
    # The wheel is built where the clone is, the laptop, which has no Terraform:
    # it names the bucket, whose name Cloud Shell's Terraform prints.
    readme = (EMULATION / "README.md").read_text(encoding="utf-8")

    assert "make -C emulation apply wheel-save" not in readme
    assert "make doclang-wheel" in readme
    assert "make -C emulation wheel-save CACHE_BUCKET=<cache bucket>" in readme
    assert "terraform -chdir=emulation/terraform output -raw cache_bucket" in readme
