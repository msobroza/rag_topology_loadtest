"""``CompositionRoot`` over Chroma and the in-process model runtimes. Settings in, the
right adapter out; the ``api`` backends are in ``test_composition_root_api.py``."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from query_processing_core.retrieval.vector_store_record import (
    ScoredVectorStoreRecord,
    VectorStoreRecord,
)

from cpu_fakes import FakeCpuCounter
from local_model_fakes import FakeOneRequestModule
from query_processing.adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from query_processing.adapters.vector_store.base_vector_store import BaseLexicalSearch
from query_processing.adapters.vector_store.chroma.chroma_store import ChromaVectorStore
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import RagDependencyError
from query_processing.infrastructure.local_models.local_model import (
    exported_model_dir,
    local_model,
)
from sentence_transformers_fakes import (
    FakeSentenceTransformersModule,
    fake_openvino_modules,
)

SettingsFactory = Callable[..., WorkflowSettings]


# --- Chroma ----------------------------------------------------------------------


async def test_the_chroma_backend_opens_the_configured_embedded_collection(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    settings = settings_factory(
        components__embedder__backend="fake",
        components__vector_store__backend="chroma",
        components__vector_store__chroma__path=str(tmp_path / "chroma"),
        components__vector_store__chroma__collection="passages_under_test",
    )

    async with CompositionRoot(settings) as composition:
        embedder, store = await composition.ingest_ports()
        passage = VectorStoreRecord(
            text="Employees accrue vacation days", metadata={"id": "p1"}
        )
        await store.upsert([passage], await embedder.embed_documents([passage.text]))

        assert isinstance(store, ChromaVectorStore)
        assert await store.ready() == (True, "1 passages in passages_under_test")
        [vector] = await embedder.embed_documents([passage.text])
        [hit] = await store.query(vector, 5)
        assert hit.id == "p1"
    assert (tmp_path / "chroma").is_dir()  # embedded: written where the settings say


async def test_the_chroma_backend_gives_the_workflow_a_store_that_ranks_no_text(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    settings = settings_factory(
        components__embedder__backend="fake",
        components__reranker__backend="fake",
        components__llm__backend="fake",
        components__vector_store__backend="chroma",
        components__vector_store__chroma__path=str(tmp_path / "chroma"),
    )

    async with CompositionRoot(settings) as composition:
        deps = await composition.workflow_dependencies()

        store = deps.by_name["vector_store"]
        assert isinstance(store, ChromaVectorStore)
        assert not isinstance(store, BaseLexicalSearch)  # Chroma ranks no text


# Opening a store reaches no network: Elasticsearch connects on the first call.
@pytest.mark.parametrize("backend", ["fake", "chroma", "elasticsearch"])
async def test_every_vector_store_opens_under_the_prefilter_of_its_settings(
    settings_factory: SettingsFactory, tmp_path: Path, backend: str
) -> None:
    prefilter = [{"attribute": "active", "comparator": "eq", "value": True}]
    settings = settings_factory(
        components__embedder__backend="fake",
        components__vector_store__backend=backend,
        components__vector_store__chroma__path=str(tmp_path / "chroma"),
        components__vector_store__prefilter=prefilter,
    )

    async with CompositionRoot(settings) as composition:
        store = (await composition.ingest_ports()).vector_store

        assert [c.model_dump() for c in store.prefilter] == prefilter


# --- in-process runtime ---------------------------------------------------------------

# What the export loads a model with: convert it to OpenVINO, in memory, at fp32
# (the default weight format).
EXPORT_ARGUMENTS = {
    "backend": "openvino",
    "model_kwargs": {"export": True, "load_in_8bit": False},
}


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for name, fake in fake_openvino_modules().items():
        monkeypatch.setitem(sys.modules, name, fake)
    return module


def local_settings(
    settings_factory: SettingsFactory, **overrides: object
) -> WorkflowSettings:
    """In-process models on a known CPU budget; everything else fake.

    The calibration budget is too small for a second trial, so each model loads
    once, for the thread budget's share.
    """
    return settings_factory(
        components__vector_store__backend="fake",
        components__llm__backend="fake",
        local_models__cpus=4,
        calibration__budget_s=1e-9,
        **overrides,
    )


def folders(settings: WorkflowSettings, models_dir: Path) -> tuple[Path, Path]:
    """Where the embedder and the reranker are exported under ``models_dir``."""
    return (
        exported_model_dir(
            str(models_dir),
            settings.components["embedder"].source_id,
            weight_format=settings.local_models.openvino_weight_format,
        ),
        exported_model_dir(
            str(models_dir),
            settings.components["reranker"].source_id,
            weight_format=settings.local_models.openvino_weight_format,
        ),
    )


def provisioned_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    lines = [json.loads(r.message) for r in caplog.records]
    return [line for line in lines if line["event"] == "local_model_provisioned"]


# 4 CPUs: openvino gives each model all 4 threads, torch 1 per call over 4 workers.
@pytest.mark.parametrize(("runtime", "threads"), [("openvino", 4), ("torch", 1)])
async def test_local_models_load_with_the_arguments_of_the_configured_runtime(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
    runtime: str,
    threads: int,
) -> None:
    settings = local_settings(
        settings_factory,
        local_models__runtime=runtime,
        local_models__openvino_dir=str(tmp_path),
    )
    for folder in folders(settings, tmp_path):
        folder.mkdir()  # exported already: nothing to provision

    async with CompositionRoot(settings) as composition:
        await composition.local_embedder("embedder")
        await composition.local_reranker("reranker")

    described = local_model(
        settings.components["embedder"].source_id,
        runtime=runtime,
        models_dir="",
        weight_format="fp32",
        openvino_properties={"ENABLE_HYPER_THREADING": "YES"},
    )
    assert fake_weights.load_arguments == [described.load_arguments(threads)] * 2
    assert fake_weights.saved == []


async def test_the_local_models_load_with_the_sites_openvino_properties(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    settings = local_settings(
        settings_factory,
        local_models__runtime="openvino",
        local_models__openvino_dir=str(tmp_path),
        local_models__openvino__performance_hint="LATENCY",
    )
    for folder in folders(settings, tmp_path):
        folder.mkdir()

    async with CompositionRoot(settings) as composition:
        await composition.local_embedder("embedder")
        await composition.local_reranker("reranker")

    ov_configs = [a["model_kwargs"]["ov_config"] for a in fake_weights.load_arguments]
    assert [c["PERFORMANCE_HINT"] for c in ov_configs] == ["LATENCY", "LATENCY"]


async def test_a_first_openvino_start_exports_each_model_then_loads_its_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = local_settings(
        settings_factory, local_models__openvino_dir=str(tmp_path)
    )
    embedder_dir, reranker_dir = folders(settings, tmp_path)

    with caplog.at_level("INFO", logger="query_processing"):
        async with CompositionRoot(settings) as composition:
            embedder = await composition.local_embedder("embedder")
            await composition.local_reranker("reranker")
            assert await embedder.ready() == (
                True,
                settings.components["embedder"].source_id,
            )

    assert fake_weights.loaded == [
        f"embedder:{settings.components['embedder'].source_id}",  # the export: download, convert
        f"embedder:{embedder_dir}",  # then the folder it wrote
        f"reranker:{settings.components['reranker'].source_id}",
        f"reranker:{reranker_dir}",
    ]
    assert fake_weights.load_arguments[0] == EXPORT_ARGUMENTS
    assert len(fake_weights.saved) == 2  # one export per model
    assert (embedder_dir / "openvino_model.xml").read_text() == settings.components[
        "embedder"
    ].source_id
    assert (reranker_dir / "openvino_model.xml").read_text() == settings.components[
        "reranker"
    ].source_id
    # Off the event loop: on the model pool.
    assert all(name.startswith("rag-model") for name in fake_weights.save_threads)
    lines = provisioned_lines(caplog)
    assert [(line["model"], line["folder"]) for line in lines] == [
        (settings.components["embedder"].source_id, str(embedder_dir)),
        (settings.components["reranker"].source_id, str(reranker_dir)),
    ]
    assert all(isinstance(line["seconds"], float) for line in lines)


async def test_a_later_openvino_start_loads_the_folder_without_exporting(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    settings = local_settings(
        settings_factory, local_models__openvino_dir=str(tmp_path)
    )
    embedder_dir, _ = folders(settings, tmp_path)

    for _start in range(2):
        async with CompositionRoot(settings) as composition:
            await composition.local_embedder("embedder")

    assert fake_weights.loaded == [
        f"embedder:{settings.components['embedder'].source_id}",
        f"embedder:{embedder_dir}",
        f"embedder:{embedder_dir}",  # the second start: the folder, no export
    ]
    assert len(fake_weights.saved) == 1


async def test_an_int8_start_exports_at_8_bits_into_its_own_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    settings = local_settings(
        settings_factory,
        local_models__openvino_dir=str(tmp_path),
        local_models__openvino_weight_format="int8",
    )
    embedder_dir, _ = folders(settings, tmp_path)
    source_id = settings.components["embedder"].source_id
    fp32_dir = exported_model_dir(str(tmp_path), source_id, weight_format="fp32")
    fp32_dir.mkdir()  # an fp32 export is never loaded for int8

    async with CompositionRoot(settings) as composition:
        await composition.local_embedder("embedder")

    assert embedder_dir.name.endswith("-int8")
    assert fake_weights.loaded == [
        f"embedder:{source_id}",
        f"embedder:{embedder_dir}",
    ]
    assert fake_weights.load_arguments[0]["model_kwargs"] == {
        "export": True,
        "load_in_8bit": True,
    }


async def test_the_torch_runtime_never_exports_and_ignores_an_exported_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    settings = local_settings(
        settings_factory,
        local_models__runtime="torch",
        local_models__openvino_dir=str(tmp_path),
    )
    embedder_dir, _ = folders(settings, tmp_path)
    embedder_dir.mkdir()

    async with CompositionRoot(settings) as composition:
        await composition.local_embedder("embedder")
        await composition.local_reranker("reranker")

    assert fake_weights.loaded == [
        f"embedder:{settings.components['embedder'].source_id}",
        f"reranker:{settings.components['reranker'].source_id}",
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
    settings = local_settings(settings_factory, local_models__openvino_dir=str(models))
    _, reranker_dir = folders(settings, models)

    async with CompositionRoot(settings) as composition:
        with pytest.raises(RagDependencyError) as raised:
            await composition.local_reranker("reranker")

    message = str(raised.value)
    assert message.startswith("local_reranker unavailable")
    assert f"{settings.components['reranker'].source_id!r}" in message
    assert str(reranker_dir) in message
    assert "No space left on device" in message
    assert list(models.iterdir()) == []  # neither the folder nor its partial copy
    assert fake_weights.loaded == [
        f"reranker:{settings.components['reranker'].source_id}"
    ]  # no load


async def test_an_export_without_the_openvino_packages_fails_before_any_download(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openvino", None)  # not installed
    settings = local_settings(
        settings_factory, local_models__openvino_dir=str(tmp_path)
    )

    async with CompositionRoot(settings) as composition:
        with pytest.raises(RagDependencyError) as raised:
            await composition.local_embedder("embedder")

    # Checked before the folder is looked at, so the error names the package.
    assert "needs module 'openvino'" in str(raised.value)
    assert "sentence-transformers[openvino]" in str(raised.value)
    assert fake_weights.loaded == []
    assert list(tmp_path.iterdir()) == []


# --- openvino: one model call at a time -------------------------------------------


def exported_settings(
    settings_factory: SettingsFactory, models_dir: Path
) -> WorkflowSettings:
    """Default settings (openvino, 4 workers), both models exported already.

    The calibration budget is too small for a second trial, so each model keeps
    the thread budget's share.
    """
    settings = settings_factory(
        components__vector_store__backend="fake",
        components__llm__backend="fake",
        local_models__openvino_dir=str(models_dir),
        calibration__budget_s=1e-9,
    )
    for folder in folders(settings, models_dir):
        folder.mkdir()
    return settings


@pytest.fixture
def one_request_weights(monkeypatch: pytest.MonkeyPatch) -> FakeOneRequestModule:
    module = FakeOneRequestModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    # The openvino runtime checks its packages even when the folders exist.
    for name, fake in fake_openvino_modules().items():
        monkeypatch.setitem(sys.modules, name, fake)
    return module


@pytest.mark.usefixtures("one_request_weights")
async def test_openvino_runs_one_call_on_every_cpu_and_logs_the_runtime(
    settings_factory: SettingsFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = exported_settings(settings_factory, tmp_path)
    container = FakeCpuCounter(8, "container")
    with caplog.at_level("INFO", logger="query_processing"):
        async with CompositionRoot(settings, container_cpus=container) as composition:
            embedder = await composition.local_embedder("embedder")

    [line] = [
        json.loads(r.message) for r in caplog.records if "thread_budget" in r.message
    ]
    assert line == {
        "event": "thread_budget",
        "cpus": 8,
        "source": "container",
        "runtime": "openvino",
        "workers": 1,
        "embedder_threads": 8,
        "reranker_threads": 8,
    }
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.scope.threads == 8


# Regression: under openvino, calls on one model overlapping on the shared pool
# raised "Infer Request is busy" (837 times, 98% of requests failed, GCP load
# test of 2026-11-14).
@pytest.mark.usefixtures("one_request_weights")
async def test_concurrent_openvino_embeds_never_overlap_on_one_model(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    settings = exported_settings(settings_factory, tmp_path)
    container = FakeCpuCounter(8, "container")
    async with CompositionRoot(settings, container_cpus=container) as composition:
        embedder = await composition.local_embedder("embedder")
        vectors = await asyncio.gather(
            *(embedder.embed_documents([f"q{i}"]) for i in range(4))
        )

    assert vectors == [[[1.0, 0.0]]] * 4


@pytest.mark.usefixtures("one_request_weights")
async def test_concurrent_openvino_reranks_never_overlap_on_one_model(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    settings = exported_settings(settings_factory, tmp_path)
    container = FakeCpuCounter(8, "container")
    passages = [
        ScoredVectorStoreRecord(
            text="vacation", metadata={"id": "p1"}, retrieval_score=0.5
        )
    ]
    async with CompositionRoot(settings, container_cpus=container) as composition:
        reranker = await composition.local_reranker("reranker")
        ranked = await asyncio.gather(
            *(reranker.rerank(f"q{i}", passages, top_n=1) for i in range(4))
        )

    assert [[p.id for p in result] for result in ranked] == [["p1"]] * 4
