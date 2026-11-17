"""``emulation/machine/``, the restore at boot and the emulated Domino's unit.

``curl`` plays the metadata server, the cache bucket and Elasticsearch;
``chown`` and ``qp-served-models`` are stand-ins too
(``tests/emulation/emulation_machine_fakes.py``). The scripts run with the
system's bash, so they stay within bash 3.2 and BSD tools, as on macOS.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from emulation_machine_fakes import (
    BUCKET,
    CACHE_BUCKET_SH,
    EMBEDDER,
    ES_URL,
    INDEX_SNAPSHOT_SH,
    PASSAGES,
    RERANKER,
    SERVED_MODEL_LINES,
    TOKEN,
    FakeMachine,
    make_machine,
    restore_from_common_sh,
    run_common_sh_function,
    run_machine_script,
)
from emulation_terraform import render, variables

from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_served_models import served_model_lines

MACHINE_SCRIPTS = (CACHE_BUCKET_SH, INDEX_SNAPSHOT_SH)
SNAPSHOT = "quebec-insurance-french-snowflake-snowflake-arctic-embed-l-v2.0"
REPOSITORY = "/_snapshot/rag-emulation-cache"


@pytest.fixture
def machine(tmp_path: Path) -> FakeMachine:
    return make_machine(tmp_path)


def exported_model(machine: FakeMachine, folder: str, weights: str) -> Path:
    """A model folder in the dataset holding one file."""
    path = machine.dataset / folder
    path.mkdir(parents=True)
    (path / "model.bin").write_text(weights)
    return path


def ovms_export(machine: FakeMachine, *configs: str) -> Path:
    path = exported_model(machine, "ovms-models", "ir")
    for config in configs:
        (path / config).write_text("{}")
    return path


def models_save(machine: FakeMachine, **env: str) -> str:
    result = run_machine_script(machine, CACHE_BUCKET_SH, "models-save", **env)
    assert result.returncode == 0, result.stderr
    return result.stdout


def snapshot(machine: FakeMachine, *args: str) -> subprocess.CompletedProcess[str]:
    env = {"QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL": ES_URL}
    return run_machine_script(machine, INDEX_SNAPSHOT_SH, *args, **env)


def test_the_stand_in_prints_what_qp_served_models_prints(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # Regression: the stand-in kept an older output (EMBEDDER_SOURCE_MODEL=...)
    # after the command changed, so index_snapshot.sh passed here and could name
    # no snapshot on a real machine.
    settings = settings_factory(
        components={
            "embedder": {"source_id": EMBEDDER},
            "reranker": {"source_id": RERANKER},
        }
    )

    assert sorted(served_model_lines(settings)) == sorted(SERVED_MODEL_LINES)


@pytest.mark.parametrize("script", MACHINE_SCRIPTS, ids=lambda p: p.name)
def test_each_machine_script_parses_with_bash(script: Path) -> None:
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck absent")
@pytest.mark.parametrize("script", MACHINE_SCRIPTS, ids=lambda p: p.name)
def test_each_machine_script_passes_shellcheck(script: Path) -> None:
    result = subprocess.run(
        ["shellcheck", "-s", "bash", "-S", "warning", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


def test_models_save_uploads_one_tar_per_model_folder(machine: FakeMachine) -> None:
    exported_model(machine, "openvino/embedder", "e")
    exported_model(machine, "openvino/reranker", "r")
    ovms_export(machine, "config_reranker.json", "config_embedder.json")

    out = models_save(machine)

    assert machine.bucket_objects() == [
        "models/openvino/embedder.tar",
        "models/openvino/reranker.tar",
        "models/ovms-models.tar",
    ]
    assert out.count("saved: ") == 3
    assert TOKEN not in machine.curl_argv(), "the token reached a command line"


def test_models_save_keeps_an_object_the_bucket_has(machine: FakeMachine) -> None:
    kept = machine.bucket / "models" / "openvino" / "embedder.tar"
    kept.parent.mkdir(parents=True)
    kept.write_text("earlier")
    exported_model(machine, "openvino/embedder", "e")

    out = models_save(machine)

    assert kept.read_text() == "earlier"
    assert f"kept: gs://{BUCKET}/models/openvino/embedder.tar" in out


def test_force_replaces_an_object_the_bucket_has(machine: FakeMachine) -> None:
    kept = machine.bucket / "models" / "openvino" / "embedder.tar"
    kept.parent.mkdir(parents=True)
    kept.write_text("earlier")
    exported_model(machine, "openvino/embedder", "e")

    out = models_save(machine, FORCE="1")

    assert kept.read_bytes() != b"earlier"
    assert f"saved: gs://{BUCKET}/models/openvino/embedder.tar" in out


def test_models_save_skips_an_unfinished_ovms_export(machine: FakeMachine) -> None:
    ovms_export(machine, "config_reranker.json")

    out = models_save(machine)

    assert machine.bucket_objects() == []
    assert "not saved: " in out


def test_the_boot_restore_puts_a_saved_model_back(machine: FakeMachine) -> None:
    folder = exported_model(machine, "openvino/embedder", "weights")
    models_save(machine)
    shutil.rmtree(folder)

    result = restore_from_common_sh(machine, "models/")

    assert result.returncode == 0, result.stderr
    assert (folder / "model.bin").read_text() == "weights"
    assert "model restored: " in result.stderr
    assert (
        f"-R ubuntu:ubuntu {machine.dataset}/openvino/"
        in (machine.root / "chown.log").read_text()
    )


def test_the_boot_restore_leaves_an_existing_folder_alone(
    machine: FakeMachine,
) -> None:
    folder = exported_model(machine, "openvino/embedder", "saved")
    models_save(machine)
    (folder / "model.bin").write_text("newer")

    result = restore_from_common_sh(machine, "models/")

    assert result.returncode == 0, result.stderr
    assert (folder / "model.bin").read_text() == "newer"
    assert "model kept: " in result.stderr


def test_the_boot_restore_of_an_empty_bucket_is_no_error(machine: FakeMachine) -> None:
    result = restore_from_common_sh(machine, "models/openvino/")

    assert result.returncode == 0, result.stderr
    assert list(machine.dataset.iterdir()) == []


def test_the_boot_restore_refuses_an_object_of_another_shape(
    machine: FakeMachine,
) -> None:
    stray = machine.bucket / "models" / "openvino" / "nested" / "model.tar"
    stray.parent.mkdir(parents=True)
    stray.write_text("x")

    result = restore_from_common_sh(machine, "models/")

    assert result.returncode != 0
    assert "models/openvino/nested/model.tar" in result.stderr


def pinned_ovms_bundle(machine: FakeMachine, name: str, content: str) -> Path:
    """A clone whose ovms/versions.env pins a tarball the download host serves."""
    served = machine.root / "downloads" / name
    served.parent.mkdir(exist_ok=True)
    served.write_text(content)
    repo = machine.root / "repo"
    (repo / "ovms").mkdir(parents=True, exist_ok=True)
    pin = f"OVMS_BINARY_URL=https://downloads.test/{name}\n"
    (repo / "ovms" / "versions.env").write_text(pin)
    return repo


def fetch_ovms_bundle(
    machine: FakeMachine, repo: Path, bundle: Path
) -> subprocess.CompletedProcess[str]:
    return run_common_sh_function(machine, "fetch_ovms_bundle", str(repo), str(bundle))


def test_fetch_ovms_bundle_downloads_the_pinned_tarball(machine: FakeMachine) -> None:
    repo = pinned_ovms_bundle(machine, "ovms-1.tar.gz", "binary one")
    bundle = machine.root / "opt" / "ovms.tar.gz"

    result = fetch_ovms_bundle(machine, repo, bundle)

    assert result.returncode == 0, result.stderr
    assert bundle.read_text() == "binary one"
    url = "https://downloads.test/ovms-1.tar.gz"
    assert Path(f"{bundle}.url").read_text() == f"{url}\n"


def test_fetch_ovms_bundle_fetches_again_only_when_the_pin_changes(
    machine: FakeMachine,
) -> None:
    repo = pinned_ovms_bundle(machine, "ovms-1.tar.gz", "binary one")
    bundle = machine.root / "ovms.tar.gz"
    assert fetch_ovms_bundle(machine, repo, bundle).returncode == 0
    calls = machine.curl_argv().count("downloads.test")

    assert fetch_ovms_bundle(machine, repo, bundle).returncode == 0
    assert machine.curl_argv().count("downloads.test") == calls
    pinned_ovms_bundle(machine, "ovms-2.tar.gz", "binary two")
    assert fetch_ovms_bundle(machine, repo, bundle).returncode == 0
    assert bundle.read_text() == "binary two"


def test_fetch_ovms_bundle_without_a_pin_names_the_file(machine: FakeMachine) -> None:
    repo = machine.root / "repo"
    (repo / "ovms").mkdir(parents=True)
    (repo / "ovms" / "versions.env").write_text("OVMS_VERSION=1\n")

    result = fetch_ovms_bundle(machine, repo, machine.root / "ovms.tar.gz")

    assert result.returncode != 0
    assert f"no OVMS_BINARY_URL in {repo}/ovms/versions.env" in result.stderr


def test_snapshot_save_replaces_the_snapshot_named_after_the_embedder(
    machine: FakeMachine,
) -> None:
    result = snapshot(machine, "save", "quebec-insurance-french")

    assert result.returncode == 0, result.stderr
    calls = machine.es_calls()
    assert calls[0].startswith(f"PUT {REPOSITORY} ")
    assert f'"bucket": "{BUCKET}"' in calls[0]
    assert '"base_path": "elasticsearch"' in calls[0]
    assert calls[1:3] == [
        "GET /rag_passages/_count",
        f"DELETE {REPOSITORY}/{SNAPSHOT}",
    ]
    assert calls[3].startswith(f"PUT {REPOSITORY}/{SNAPSHOT}?wait_for_completion=true ")
    assert f'"passages": {PASSAGES}' in calls[3]
    assert '"include_global_state": false' in calls[3]


def test_snapshot_restore_replaces_the_index(machine: FakeMachine) -> None:
    assert snapshot(machine, "save", "quebec-insurance-french").returncode == 0
    saved = len(machine.es_calls())

    result = snapshot(machine, "restore", "quebec-insurance-french")

    assert result.returncode == 0, result.stderr
    restore_path = f"{REPOSITORY}/{SNAPSHOT}/_restore?wait_for_completion=true"
    assert [call.split(" {")[0] for call in machine.es_calls()[saved:]] == [
        f"PUT {REPOSITORY}",
        f"GET {REPOSITORY}/{SNAPSHOT}",
        "DELETE /rag_passages",
        f"POST {restore_path}",
        "GET /rag_passages/_count",
    ]
    assert f"holds {PASSAGES} passages" in result.stdout


def test_snapshot_restore_of_a_missing_snapshot_names_it(machine: FakeMachine) -> None:
    result = snapshot(machine, "restore", "quebec-insurance-french")

    assert result.returncode != 0
    assert f"no snapshot {SNAPSHOT}" in result.stderr
    assert "make -C emulation corpus corpus-save" in result.stderr


def test_index_snapshot_without_the_es_url_names_it(machine: FakeMachine) -> None:
    result = run_machine_script(machine, INDEX_SNAPSHOT_SH, "list")

    assert result.returncode != 0
    assert "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL" in result.stderr
    assert machine.es_calls() == []


def test_results_save_uploads_one_archive_that_status_lists(
    machine: FakeMachine,
) -> None:
    results = machine.root / "results"
    results.mkdir()
    (results / "run_stats.csv").write_text("Name,Requests\n")
    snapshot_file = machine.bucket / "elasticsearch" / "index-0"
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.write_text("one of Elasticsearch's own files")

    saved = run_machine_script(machine, CACHE_BUCKET_SH, "results-save", str(results))
    status = run_machine_script(machine, CACHE_BUCKET_SH, "status")

    assert saved.returncode == 0, saved.stderr
    (archive,) = [o for o in machine.bucket_objects() if o.startswith("results/")]
    assert archive.endswith(".tar.gz")
    assert saved.stdout.strip() == f"gs://{BUCKET}/{archive}"
    assert status.returncode == 0, status.stderr
    assert archive in status.stdout
    assert "elasticsearch/" not in status.stdout


def emulated_domino_exec_start(values: dict[str, str]) -> str:
    """The ``ExecStart`` line of the emulated Domino's unit in the Workspace script."""
    script = render("scripts/workspace.sh.tftpl", values)
    (line,) = [
        line
        for line in script.splitlines()
        if line.startswith("ExecStart=") and "qp-domino-emulation" in line
    ]
    return line


def test_the_emulated_domino_forwards_for_30_seconds_by_default() -> None:
    default = str(variables()["forward_timeout_seconds"]["default"])

    line = emulated_domino_exec_start({"forward_timeout_seconds": default})

    assert line.endswith(" --forward-timeout 30")


def test_the_emulated_domino_forwards_for_as_long_as_the_variable_says() -> None:
    line = emulated_domino_exec_start({"forward_timeout_seconds": "300"})

    assert line.endswith(" --forward-timeout 300")
