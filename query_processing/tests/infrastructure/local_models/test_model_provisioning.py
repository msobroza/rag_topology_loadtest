"""``provision_local_model``: the export on a first start, the package check always.

The exporter and the OpenVINO package check are named fakes: no weights, no
OpenVINO package imported.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openvino_package_fakes import FakeOpenVinoPackages
from query_processing.domain.backends import LocalModelRuntime, OpenVinoWeightFormat
from query_processing.infrastructure.local_models.local_model import (
    LocalModel,
    OpenVinoExports,
    exported_model_dir,
)
from query_processing.infrastructure.local_models.model_provisioning import (
    provision_local_model,
)
from query_processing.infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
)

MODEL_ID = "org/reranker"


class RecordingExporter:
    """An exporter that writes the folder and records each model it exported."""

    def __init__(self) -> None:
        self.exported: list[str] = []
        self.weight_formats: list[OpenVinoWeightFormat] = []

    def __call__(
        self, model_id: str, target: Path, weight_format: OpenVinoWeightFormat
    ) -> Path:
        self.exported.append(model_id)
        self.weight_formats.append(weight_format)
        target.mkdir(parents=True)
        return target


@pytest.fixture
def pool() -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(1) as executor:
        yield executor


async def provision(
    tmp_path: Path,
    pool: ThreadPoolExecutor,
    packages: FakeOpenVinoPackages,
    exporter: RecordingExporter,
    runtime: LocalModelRuntime = "openvino",
    weight_format: OpenVinoWeightFormat = "fp32",
) -> LocalModel:
    return await provision_local_model(
        MODEL_ID,
        runtime=runtime,
        exports=OpenVinoExports(str(tmp_path), weight_format),
        export=exporter,
        require_packages=packages,
        executor=pool,
    )


@pytest.mark.parametrize("exported", [True, False], ids=["folder", "no-folder"])
async def test_missing_openvino_packages_fail_before_any_load_or_export(
    tmp_path: Path, pool: ThreadPoolExecutor, exported: bool
) -> None:
    # Regression, sitting of 15 November 2026: with the folder present nothing
    # checked, and the library's bare Exception reached qp-calibrate as a traceback.
    if exported:
        exported_model_dir(str(tmp_path), MODEL_ID, weight_format="fp32").mkdir()
    exporter = RecordingExporter()

    with pytest.raises(OpenVinoUnavailableError, match="'openvino'"):
        await provision(tmp_path, pool, FakeOpenVinoPackages("openvino"), exporter)

    assert exporter.exported == []


async def test_an_exported_folder_loads_once_the_packages_are_there(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    folder = exported_model_dir(str(tmp_path), MODEL_ID, weight_format="fp32")
    folder.mkdir()
    packages, exporter = FakeOpenVinoPackages(), RecordingExporter()

    model = await provision(tmp_path, pool, packages, exporter)

    assert (model.source, model.runtime) == (str(folder), "openvino")
    assert (packages.checks, exporter.exported) == (1, [])


async def test_a_first_start_checks_the_packages_then_exports(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    packages, exporter = FakeOpenVinoPackages(), RecordingExporter()

    await provision(tmp_path, pool, packages, exporter)

    assert (packages.checks, exporter.exported) == (1, [MODEL_ID])


async def test_the_torch_runtime_never_checks_the_openvino_packages(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    packages, exporter = FakeOpenVinoPackages("openvino"), RecordingExporter()

    model = await provision(tmp_path, pool, packages, exporter, runtime="torch")

    assert (model.source, model.runtime) == (MODEL_ID, "torch")
    assert (packages.checks, exporter.exported) == (0, [])


async def test_an_fp32_export_alone_does_not_serve_int8_and_int8_is_exported(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    # Changing the setting re-exports: the fp32 folder stays, int8 gets its own.
    fp32 = exported_model_dir(str(tmp_path), MODEL_ID, weight_format="fp32")
    fp32.mkdir()
    exporter = RecordingExporter()

    model = await provision(
        tmp_path, pool, FakeOpenVinoPackages(), exporter, weight_format="int8"
    )

    int8 = exported_model_dir(str(tmp_path), MODEL_ID, weight_format="int8")
    assert (exporter.exported, exporter.weight_formats) == ([MODEL_ID], ["int8"])
    assert model.source == str(int8)
    assert fp32.is_dir()


async def test_the_export_logs_its_weight_format(
    tmp_path: Path, pool: ThreadPoolExecutor, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger="query_processing"):
        await provision(
            tmp_path,
            pool,
            FakeOpenVinoPackages(),
            RecordingExporter(),
            weight_format="int8",
        )

    [line] = [json.loads(r.message) for r in caplog.records]
    assert (line["event"], line["weight_format"]) == ("local_model_provisioned", "int8")
