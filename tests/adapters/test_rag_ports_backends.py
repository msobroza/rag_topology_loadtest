"""``RagPorts`` over Chroma and the in-process model runtimes. Settings in, the
right adapter out; the ``api`` backends are in ``test_rag_ports_api.py``."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_load_test.adapters.local_models.local_model import (
    exported_model_dir,
    local_model,
)
from rag_load_test.adapters.rag_ports import RagPorts
from rag_load_test.adapters.vector_store.chroma.chroma_store import ChromaVectorStore
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import Passage
from sentence_transformers_fakes import (
    FakeSentenceTransformersModule,
    fake_openvino_modules,
)

SettingsFactory = Callable[..., RagSettings]


# --- Chroma ----------------------------------------------------------------------


async def test_the_chroma_backend_opens_the_configured_embedded_collection(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    settings = settings_factory(
        embedding_backend="fake",
        vector_store_backend="chroma",
        chroma_path=str(tmp_path / "chroma"),
        chroma_collection="passages_under_test",
    )

    async with RagPorts(settings) as ports:
        embedder, store = await ports.ingest_ports()
        passage = Passage(id="p1", text="Employees accrue vacation days")
        await store.upsert([passage], await embedder.embed([passage.text]))

        assert isinstance(store, ChromaVectorStore)
        assert await store.ready() == (True, "1 passages in passages_under_test")
        [vector] = await embedder.embed([passage.text])
        [hit] = await store.query(vector, 5)
        assert hit.id == "p1"
    assert (tmp_path / "chroma").is_dir()  # embedded: written where the settings say


async def test_the_chroma_backend_gives_the_workflow_no_lexical_search(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    settings = settings_factory(
        embedding_backend="fake",
        rerank_backend="fake",
        generation_backend="fake",
        vector_store_backend="chroma",
        chroma_path=str(tmp_path / "chroma"),
    )

    async with RagPorts(settings) as ports:
        deps = await ports.workflow_dependencies()

        assert isinstance(deps.vector_store, ChromaVectorStore)
        assert deps.lexical_search is None  # open-source Chroma ranks no text


# --- in-process runtime ---------------------------------------------------------------

# What the export loads a model with: convert it to OpenVINO, in memory.
EXPORT_ARGUMENTS = {"backend": "openvino", "model_kwargs": {"export": True}}


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for name, fake in fake_openvino_modules().items():
        monkeypatch.setitem(sys.modules, name, fake)
    return module


def local_settings(
    settings_factory: SettingsFactory, **overrides: object
) -> RagSettings:
    """In-process models on a known CPU budget; everything else fake."""
    return settings_factory(
        vector_store_backend="fake",
        generation_backend="fake",
        model_cpus=4,
        **overrides,
    )


def folders(settings: RagSettings, models_dir: Path) -> tuple[Path, Path]:
    """Where the embedder and the reranker are exported under ``models_dir``."""
    return (
        exported_model_dir(str(models_dir), settings.embedding_model),
        exported_model_dir(str(models_dir), settings.rerank_model),
    )


def provisioned_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    lines = [json.loads(r.message) for r in caplog.records]
    return [line for line in lines if line["event"] == "local_model_provisioned"]


@pytest.mark.parametrize("runtime", ["openvino", "torch"])
async def test_local_models_load_with_the_arguments_of_the_configured_runtime(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
    runtime: str,
) -> None:
    settings = local_settings(
        settings_factory, local_model_runtime=runtime, openvino_models_dir=str(tmp_path)
    )
    for folder in folders(settings, tmp_path):
        folder.mkdir()  # exported already: nothing to provision

    async with RagPorts(settings) as ports:
        await ports.local_embedder()
        await ports.local_reranker()

    # 4 CPUs over 4 workers: one thread per call, for both models.
    described = local_model(settings.embedding_model, runtime=runtime, models_dir="")
    assert fake_weights.load_arguments == [described.load_arguments(1)] * 2
    assert fake_weights.saved == []


async def test_a_first_openvino_start_exports_each_model_then_loads_its_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = local_settings(settings_factory, openvino_models_dir=str(tmp_path))
    embedder_dir, reranker_dir = folders(settings, tmp_path)

    with caplog.at_level("INFO", logger="rag_load_test"):
        async with RagPorts(settings) as ports:
            embedder = await ports.local_embedder()
            await ports.local_reranker()
            assert await embedder.ready() == (True, settings.embedding_model)

    assert fake_weights.loaded == [
        f"embedder:{settings.embedding_model}",  # the export: download, convert
        f"embedder:{embedder_dir}",  # then the folder it wrote
        f"reranker:{settings.rerank_model}",
        f"reranker:{reranker_dir}",
    ]
    assert fake_weights.load_arguments[0] == EXPORT_ARGUMENTS
    assert len(fake_weights.saved) == 2  # one export per model
    assert (embedder_dir / "openvino_model.xml").read_text() == settings.embedding_model
    assert (reranker_dir / "openvino_model.xml").read_text() == settings.rerank_model
    # Off the event loop: on the model pool.
    assert all(name.startswith("rag-model") for name in fake_weights.save_threads)
    lines = provisioned_lines(caplog)
    assert [(line["model"], line["folder"]) for line in lines] == [
        (settings.embedding_model, str(embedder_dir)),
        (settings.rerank_model, str(reranker_dir)),
    ]
    assert all(isinstance(line["seconds"], float) for line in lines)


async def test_a_later_openvino_start_loads_the_folder_without_exporting(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    settings = local_settings(settings_factory, openvino_models_dir=str(tmp_path))
    embedder_dir, _ = folders(settings, tmp_path)

    for _start in range(2):
        async with RagPorts(settings) as ports:
            await ports.local_embedder()

    assert fake_weights.loaded == [
        f"embedder:{settings.embedding_model}",
        f"embedder:{embedder_dir}",
        f"embedder:{embedder_dir}",  # the second start: the folder, no export
    ]
    assert len(fake_weights.saved) == 1


async def test_the_torch_runtime_never_exports_and_ignores_an_exported_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    settings = local_settings(
        settings_factory, local_model_runtime="torch", openvino_models_dir=str(tmp_path)
    )
    embedder_dir, _ = folders(settings, tmp_path)
    embedder_dir.mkdir()

    async with RagPorts(settings) as ports:
        await ports.local_embedder()
        await ports.local_reranker()

    assert fake_weights.loaded == [
        f"embedder:{settings.embedding_model}",
        f"reranker:{settings.rerank_model}",
    ]
    assert fake_weights.saved == []
    assert list(tmp_path.iterdir()) == [embedder_dir]


async def test_a_failed_export_names_the_model_and_folder_and_leaves_no_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    fake_weights.save_error = OSError("No space left on device")
    models = tmp_path / "models"
    settings = local_settings(settings_factory, openvino_models_dir=str(models))
    _, reranker_dir = folders(settings, models)

    async with RagPorts(settings) as ports:
        with pytest.raises(RagDependencyError) as raised:
            await ports.local_reranker()

    message = str(raised.value)
    assert message.startswith("local_reranker unavailable")
    assert f"{settings.rerank_model!r}" in message
    assert str(reranker_dir) in message
    assert "No space left on device" in message
    assert list(models.iterdir()) == []  # neither the folder nor its partial copy
    assert fake_weights.loaded == [f"reranker:{settings.rerank_model}"]  # no load


async def test_an_export_without_the_openvino_packages_fails_before_any_download(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openvino", None)  # not installed
    settings = local_settings(settings_factory, openvino_models_dir=str(tmp_path))
    embedder_dir, _ = folders(settings, tmp_path)

    async with RagPorts(settings) as ports:
        with pytest.raises(RagDependencyError) as raised:
            await ports.local_embedder()

    assert str(embedder_dir) in str(raised.value)
    assert "sentence-transformers[openvino]" in str(raised.value)
    assert fake_weights.loaded == []
    assert list(tmp_path.iterdir()) == []
