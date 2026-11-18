"""``write_ovms_serving_folder``: the export linked, a graph of our own, nothing written into the export."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ovms_stand_ins import MODEL_FILES, write_ovms_export
from query_processing.infrastructure.ovms.ovms_serving_folder import (
    OvmsExport,
    recorded_weight_format,
    write_ovms_serving_folder,
)

CONFIG = {"NUM_STREAMS": "2", "INFERENCE_NUM_THREADS": "2"}


@pytest.fixture
def export(tmp_path: Path) -> OvmsExport:
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", "rr")
    return OvmsExport(models, "config_reranker.json", "rr")


def snapshot(folder: Path) -> dict[str, tuple[float, bytes]]:
    return {
        str(p.relative_to(folder)): (p.stat().st_mtime, p.read_bytes())
        for p in folder.rglob("*")
        if p.is_file()
    }


def test_the_config_is_a_copy_and_the_model_files_are_links(
    export: OvmsExport, tmp_path: Path
) -> None:
    serving = tmp_path / "serving"
    config = write_ovms_serving_folder(export, serving, CONFIG)

    assert config == serving / "config_reranker.json"
    assert (
        config.read_text() == (export.models_dir / "config_reranker.json").read_text()
    )
    assert not config.is_symlink()
    for name in MODEL_FILES:
        link = serving / "rr" / name
        assert link.is_symlink()
        assert link.resolve() == (export.models_dir / "rr" / name).resolve()


def test_the_graph_is_ours_and_holds_the_plugin_config(
    export: OvmsExport, tmp_path: Path
) -> None:
    write_ovms_serving_folder(export, tmp_path / "serving", CONFIG)

    graph = tmp_path / "serving" / "rr" / "graph.pbtxt"
    assert not graph.is_symlink()
    assert f"plugin_config: '{json.dumps(CONFIG)}'," in graph.read_text()


def test_the_export_is_never_written(export: OvmsExport, tmp_path: Path) -> None:
    before = snapshot(export.models_dir)
    folders = (export.models_dir, export.models_dir / "rr")
    for folder in folders:  # read-only, as a shared Dataset may be
        os.chmod(folder, 0o555)
    try:
        write_ovms_serving_folder(export, tmp_path / "serving", CONFIG)
    finally:
        for folder in folders:
            os.chmod(folder, 0o755)

    assert snapshot(export.models_dir) == before


def test_building_again_replaces_the_graph_and_the_links(
    export: OvmsExport, tmp_path: Path
) -> None:
    serving = tmp_path / "serving"
    write_ovms_serving_folder(export, serving, CONFIG)
    (export.models_dir / "rr" / "tokenizer.json").unlink()  # the export changed

    write_ovms_serving_folder(export, serving, {"NUM_STREAMS": "1"})

    names = sorted(p.name for p in (serving / "rr").iterdir())
    assert names == ["graph.pbtxt", "openvino_model.bin", "openvino_model.xml"]
    assert '{"NUM_STREAMS": "1"}' in (serving / "rr" / "graph.pbtxt").read_text()


def test_a_missing_export_names_the_path_and_the_exporter(tmp_path: Path) -> None:
    export = OvmsExport(tmp_path / "empty", "config_embedder.json", "emb")

    with pytest.raises(FileNotFoundError) as raised:
        write_ovms_serving_folder(export, tmp_path / "serving", CONFIG)

    message = str(raised.value)
    assert str(tmp_path / "empty" / "emb" / "graph.pbtxt") in message
    assert "ovms/export_models.sh" in message


# --- the weight format an export says it holds -------------------------------------


def test_an_export_holds_the_weight_format_its_record_says(export: OvmsExport) -> None:
    (export.models_dir / "weight-format").write_text("fp32\n")

    assert recorded_weight_format(export.models_dir) == "fp32"


def test_an_export_without_a_record_holds_int8(export: OvmsExport) -> None:
    # ovms/export_models.sh exported int8 only, before it recorded the format.
    (export.models_dir / "weight-format").unlink()

    assert recorded_weight_format(export.models_dir) == "int8"
