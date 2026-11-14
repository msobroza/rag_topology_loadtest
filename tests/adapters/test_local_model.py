"""Which files an in-process model loads from, and the arguments that pick its runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_load_test.adapters.local_models.local_model import (
    LocalModel,
    exported_model_dir,
    local_model,
)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Snowflake/snowflake-arctic-embed-l-v2.0", "snowflake-arctic-embed-l-v2.0"),
        ("plain-name", "plain-name"),
        ("org/name/", "name"),
        ("/abs/path/to/model", "model"),
    ],
)
def test_exported_model_dir_is_the_last_segment_of_the_id(
    model_id: str, expected: str
) -> None:
    assert exported_model_dir("models/openvino", model_id) == Path(
        "models/openvino", expected
    )


def test_openvino_loads_the_exported_folder_when_it_exists(tmp_path: Path) -> None:
    (tmp_path / "name").mkdir()

    model = local_model("org/name", runtime="openvino", models_dir=str(tmp_path))

    assert model == LocalModel("org/name", str(tmp_path / "name"), "openvino")


def test_openvino_falls_back_to_the_id_without_an_export(tmp_path: Path) -> None:
    model = local_model("org/name", runtime="openvino", models_dir=str(tmp_path))

    assert model == LocalModel("org/name", "org/name", "openvino")


def test_torch_ignores_an_existing_export(tmp_path: Path) -> None:
    (tmp_path / "name").mkdir()

    model = local_model("org/name", runtime="torch", models_dir=str(tmp_path))

    assert model == LocalModel("org/name", "org/name", "torch")


def test_a_file_named_like_the_export_is_not_an_export(tmp_path: Path) -> None:
    (tmp_path / "name").write_text("not a folder")

    model = local_model("org/name", runtime="openvino", models_dir=str(tmp_path))

    assert model.source == "org/name"


def test_torch_load_arguments_are_empty() -> None:
    assert LocalModel("m", "m", "torch").load_arguments(3) == {}


def test_openvino_load_arguments_pick_the_backend_and_cap_its_threads() -> None:
    arguments = LocalModel("m", "m", "openvino").load_arguments(3)

    assert arguments == {
        "backend": "openvino",
        "model_kwargs": {"ov_config": {"INFERENCE_NUM_THREADS": "3"}},
    }


def test_openvino_load_arguments_are_a_fresh_dict_each_call() -> None:
    # sentence-transformers pops keys out of model_kwargs while loading.
    model = LocalModel("m", "m", "openvino")
    first = model.load_arguments(2)
    first["model_kwargs"]["ov_config"].clear()

    assert model.load_arguments(2)["model_kwargs"]["ov_config"] != {}
