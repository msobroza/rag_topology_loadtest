"""The workflow image holds every file this use case's settings read.

No docker here: the test reads the ``COPY`` lines of the toolkit's Dockerfile
for the ``rag`` stage, with ``USE_CASE`` set to this directory's name, copies
those files and nothing else into a folder laid out as the image's ``/opt/qp``,
and reads the settings there, the first thing the image's export step does.
The three packages, the core's, the toolkit's and this one's, are found on
``PYTHONPATH`` in their copied folders, as the image installs them from there.
A file the use case gains and the Dockerfile forgets fails here, not in a
build.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

USE_CASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = USE_CASE_DIR.parent
DOCKERFILE = REPO_ROOT / "query_processing" / "domino" / "Dockerfile"
IMAGE_ROOT = "/opt/qp/"  # where the rag stage lays out the repository's files
# Read the settings, and say where the use case's package was imported from.
READ_SETTINGS = (
    "import query_processing_rag_usecase as package\n"
    "from query_processing.extension_api import WorkflowSettings\n"
    "settings = WorkflowSettings(_env_file=None)\n"
    "print(package.__file__)\n"
    "print(settings.model_dump_json())\n"
)


def rag_stage_copies(dockerfile: str, use_case: str) -> list[tuple[str, str]]:
    """The ``(sources, destination)`` of each ``COPY`` of the rag stage.

    Example::

        rag_stage_copies(text, "query_processing_rag_usecase")[0]
        # (".python-version", "/opt/qp/")
    """
    joined = dockerfile.replace("\\\n", " ")
    stage = re.search(r"^FROM \S+ AS rag$(.*?)(?=^FROM |\Z)", joined, re.M | re.S)
    assert stage, f"no stage 'rag' in {DOCKERFILE}"
    lines = re.findall(r"^COPY (.+)$", stage.group(1), re.M)
    copies = []
    for line in lines:
        *sources, destination = line.replace("${USE_CASE}", use_case).split()
        copies += [(source, destination) for source in sources]
    return copies


def copy_as_the_image(copies: list[tuple[str, str]], image: Path) -> None:
    """Copy each source from the repository as ``COPY`` does, ``/opt/qp/`` at ``image``."""
    for source, destination in copies:
        assert destination.startswith(IMAGE_ROOT), f"{destination} is not under /opt/qp"
        target = image / destination.removeprefix(IMAGE_ROOT)
        origin = REPO_ROOT / source
        if origin.is_dir():  # a folder's contents land in the destination
            ignore = shutil.ignore_patterns("__pycache__", "*.egg-info")
            shutil.copytree(origin, target, ignore=ignore, dirs_exist_ok=True)
            continue
        if destination.endswith("/"):
            target = target / origin.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)


def settings_read_in(directory: Path, *python_path: Path) -> list[str]:
    """Read the settings in ``directory``, in a process of its own; its two lines."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QP_", "DOMINO_"))}
    env["PYTHONPATH"] = os.pathsep.join(map(str, python_path))
    run = subprocess.run(
        [sys.executable, "-c", READ_SETTINGS],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    return run.stdout.splitlines()[-2:]


def toolkit_sources(root: Path) -> tuple[Path, Path]:
    """The source folders of the core and the toolkit, laid out under ``root``."""
    return root / "query_processing_core" / "src", root / "query_processing" / "src"


def test_the_files_the_image_copies_are_enough_to_read_the_settings(
    tmp_path: Path,
) -> None:
    use_case = USE_CASE_DIR.name
    copies = rag_stage_copies(DOCKERFILE.read_text("utf-8"), use_case)
    copy_as_the_image(copies, tmp_path)
    copied = tmp_path / use_case

    package_file, in_image = settings_read_in(
        copied, *toolkit_sources(tmp_path), copied / "src"
    )
    _, here = settings_read_in(
        USE_CASE_DIR, *toolkit_sources(REPO_ROOT), USE_CASE_DIR / "src"
    )

    assert Path(package_file).is_relative_to(copied / "src")
    assert in_image == here  # the same prompts, routes and components
