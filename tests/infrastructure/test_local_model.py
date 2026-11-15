"""Which files an in-process model loads from, and the arguments that pick its runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_load_test.domain.backends import OpenVinoWeightFormat
from rag_load_test.infrastructure.local_models.local_model import (
    LocalModel,
    OpenVinoExports,
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
    assert exported_model_dir(
        "models/openvino", model_id, weight_format="fp32"
    ) == Path("models/openvino", expected)


def test_an_int8_export_has_a_folder_of_its_own_beside_the_fp32_one() -> None:
    # fp32 keeps the bare name: every export made before the setting is fp32.
    assert exported_model_dir("m", "org/name", weight_format="int8") == Path(
        "m", "name-int8"
    )


@pytest.mark.parametrize(
    ("weight_format", "folder"), [("fp32", "name"), ("int8", "name-int8")]
)
def test_openvino_loads_the_export_of_the_weight_format_asked(
    tmp_path: Path, weight_format: OpenVinoWeightFormat, folder: str
) -> None:
    (tmp_path / "name").mkdir()
    (tmp_path / "name-int8").mkdir()

    model = local_model(
        "org/name",
        runtime="openvino",
        models_dir=str(tmp_path),
        weight_format=weight_format,
    )

    assert model.source == str(tmp_path / folder)


def test_an_export_of_another_weight_format_is_never_loaded(tmp_path: Path) -> None:
    (tmp_path / "name").mkdir()  # fp32 only

    model = local_model(
        "org/name", runtime="openvino", models_dir=str(tmp_path), weight_format="int8"
    )

    assert model.source == "org/name"


def test_openvino_loads_the_exported_folder_when_it_exists(tmp_path: Path) -> None:
    (tmp_path / "name").mkdir()

    model = local_model(
        "org/name", runtime="openvino", models_dir=str(tmp_path), weight_format="fp32"
    )

    assert model == LocalModel("org/name", str(tmp_path / "name"), "openvino")


def test_openvino_falls_back_to_the_id_without_an_export(tmp_path: Path) -> None:
    model = local_model(
        "org/name", runtime="openvino", models_dir=str(tmp_path), weight_format="fp32"
    )

    assert model == LocalModel("org/name", "org/name", "openvino")


def test_torch_ignores_an_existing_export(tmp_path: Path) -> None:
    (tmp_path / "name").mkdir()

    model = local_model(
        "org/name", runtime="torch", models_dir=str(tmp_path), weight_format="fp32"
    )

    assert model == LocalModel("org/name", "org/name", "torch")


def test_a_file_named_like_the_export_is_not_an_export(tmp_path: Path) -> None:
    (tmp_path / "name").write_text("not a folder")

    model = local_model(
        "org/name", runtime="openvino", models_dir=str(tmp_path), weight_format="fp32"
    )

    assert model.source == "org/name"


def test_torch_load_arguments_are_empty() -> None:
    assert LocalModel("m", "m", "torch").load_arguments(3) == {}


def test_openvino_load_arguments_pick_the_backend_and_its_threads() -> None:
    arguments = LocalModel("m", "m", "openvino").load_arguments(3)

    assert arguments == {
        "backend": "openvino",
        "model_kwargs": {
            "ov_config": {"INFERENCE_NUM_THREADS": "3", "ENABLE_HYPER_THREADING": "YES"}
        },
    }


def test_openvino_is_allowed_the_hyper_threads_it_is_given() -> None:
    # Without it, asked for 2 threads on one core with two hyper-threads, OpenVINO
    # ran 1 (GCP sitting of 2026-11-15).
    ov_config = LocalModel("m", "m", "openvino").load_arguments(2)["model_kwargs"][
        "ov_config"
    ]

    assert ov_config["ENABLE_HYPER_THREADING"] == "YES"


def test_openvino_load_arguments_are_a_fresh_dict_each_call() -> None:
    # sentence-transformers pops keys out of model_kwargs while loading.
    model = LocalModel("m", "m", "openvino")
    first = model.load_arguments(2)
    first["model_kwargs"]["ov_config"].clear()

    assert model.load_arguments(2)["model_kwargs"]["ov_config"] != {}


def test_the_exports_name_the_folder_of_their_weight_format() -> None:
    exports = OpenVinoExports("models/openvino", "int8")

    assert exports.folder("org/name") == Path("models/openvino/name-int8")
