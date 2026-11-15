"""``provision_local_model``: the export on a first start, the package check always.

The exporter and the OpenVINO package check are named fakes: no weights, no
OpenVINO package imported.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openvino_package_fakes import FakeOpenVinoPackages
from rag_load_test.domain.backends import LocalModelRuntime
from rag_load_test.infrastructure.local_models.local_model import (
    LocalModel,
    exported_model_dir,
)
from rag_load_test.infrastructure.local_models.model_provisioning import (
    provision_local_model,
)
from rag_load_test.infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
)

MODEL_ID = "org/reranker"


class RecordingExporter:
    """An exporter that writes the folder and records each model it exported."""

    def __init__(self) -> None:
        self.exported: list[str] = []

    def __call__(self, model_id: str, target: Path) -> Path:
        self.exported.append(model_id)
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
) -> LocalModel:
    return await provision_local_model(
        MODEL_ID,
        runtime=runtime,
        models_dir=str(tmp_path),
        export=exporter,
        require_packages=packages,
        executor=pool,
    )


@pytest.mark.parametrize("exported", [True, False], ids=["folder", "no-folder"])
async def test_missing_openvino_packages_fail_before_any_load_or_export(
    tmp_path: Path, pool: ThreadPoolExecutor, exported: bool
) -> None:
    # Regression, sitting of 15 November 2026: with the folder present nothing
    # checked, and the library's bare Exception reached rag-calibrate as a traceback.
    if exported:
        exported_model_dir(str(tmp_path), MODEL_ID).mkdir()
    exporter = RecordingExporter()

    with pytest.raises(OpenVinoUnavailableError, match="'openvino'"):
        await provision(tmp_path, pool, FakeOpenVinoPackages("openvino"), exporter)

    assert exporter.exported == []


async def test_an_exported_folder_loads_once_the_packages_are_there(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    folder = exported_model_dir(str(tmp_path), MODEL_ID)
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
