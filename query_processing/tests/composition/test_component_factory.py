"""The generic factory: a component opened once, by the opener of its type and backend.

What each opener does with a component's settings is tested through
``CompositionRoot`` in ``test_composition_root*.py``; here, the rules every
component type shares, and a new backend added as one module, nothing else.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.composition.component_factory import ComponentFactory
from query_processing.composition.composition_root import CompositionRoot
from query_processing.composition.http_resources import HttpResources
from query_processing.composition.local_model_resources import LocalModelResources
from query_processing.composition.openers.component_openers import (
    EMBEDDER_OPENERS,
    LLM_OPENERS,
    RERANKER_OPENERS,
    VECTOR_STORE_OPENERS,
    ComponentOpeners,
    SharedResources,
)
from query_processing.composition.resource_scope import ResourceScope
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import RagDependencyError
from query_processing.infrastructure.local_models.cpu_count import ContainerCpus
from rag_components import FAKE_BACKENDS, without_shell_settings
from repo_paths import TOOLKIT_ROOT

# Each component type's openers, and the backends registered for it: today's.
OPENERS: dict[str, tuple[ComponentOpeners[Any, Any], tuple[str, ...]]] = {
    "embedder": (EMBEDDER_OPENERS, ("api", "fake", "local", "served")),
    "reranker": (RERANKER_OPENERS, ("api", "fake", "local", "none", "served")),
    "llm": (LLM_OPENERS, ("api", "fake", "served")),
    "vector_store": (VECTOR_STORE_OPENERS, ("chroma", "elasticsearch", "fake")),
}


async def no_embedder() -> Any:
    raise AssertionError("no test here seeds a store")


def factory_for(
    settings: WorkflowSettings, scope: ResourceScope, **supplied: object
) -> ComponentFactory:
    """The factory, given what ``CompositionRoot`` gives it."""
    http = HttpResources(settings, scope)
    local_models = LocalModelResources(settings, scope, http, ContainerCpus())
    shared = SharedResources(scope, http, local_models, no_embedder)
    return ComponentFactory(shared, supplied)


# --- one opener per backend ------------------------------------------------------------


@pytest.mark.parametrize("component_type", sorted(OPENERS))
def test_each_component_type_has_one_opener_per_backend_value(
    component_type: str,
) -> None:
    openers, backends = OPENERS[component_type]
    known = ", ".join(repr(backend) for backend in sorted(backends))

    assert openers.component_type == component_type
    assert openers.names() == tuple(sorted(backends))
    for backend in backends:
        assert callable(openers.get(backend))
    # The known names, exactly: one opener per value, and none more.
    message = f"unknown {component_type} backend 'gpu' (expected one of: {known})"
    with pytest.raises(ValueError, match=re.escape(message)):
        openers.get("gpu")


async def test_an_unknown_backend_fails_naming_the_known_ones(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # The settings take any backend name: the opener's registry refuses it.
    settings = settings_factory(
        **FAKE_BACKENDS | {"components__reranker__backend": "gpu"}
    )
    known = "'api', 'fake', 'local', 'none', 'served'"

    async with CompositionRoot(settings) as composition:
        with pytest.raises(RagDependencyError) as raised:
            await composition.workflow_dependencies()

    assert str(raised.value) == (
        "reranker unavailable: could not open backend 'gpu': ValueError: "
        f"unknown reranker backend 'gpu' (expected one of: {known})"
    )


# --- once per process, and what is handed in ---------------------------------------------


async def test_a_component_opens_once_per_scope(
    fake_settings: WorkflowSettings,
) -> None:
    reranker = fake_settings.components["reranker"]
    async with ResourceScope("test") as scope:
        factory = factory_for(fake_settings, scope)
        opened = await factory.open(RERANKER_OPENERS, "reranker", reranker)

        assert isinstance(opened, FakeReranker)
        assert await factory.open(RERANKER_OPENERS, "reranker", reranker) is opened


async def test_two_components_of_one_type_open_apart(
    fake_settings: WorkflowSettings,
) -> None:
    reranker = fake_settings.components["reranker"]
    async with ResourceScope("test") as scope:
        factory = factory_for(fake_settings, scope)
        coarse = await factory.open(RERANKER_OPENERS, "coarse", reranker)
        fine = await factory.open(RERANKER_OPENERS, "fine", reranker)

        assert coarse is not fine
        assert await factory.open(RERANKER_OPENERS, "fine", reranker) is fine


async def test_a_supplied_port_is_used_as_given_even_outside_a_scope(
    fake_settings: WorkflowSettings,
) -> None:
    supplied = FakeLLM()
    factory = factory_for(fake_settings, ResourceScope("test"), llm=supplied)
    llm = fake_settings.components["llm"]

    assert await factory.open(LLM_OPENERS, "llm", llm) is supplied  # scope unopened
    assert await factory.open_local(LLM_OPENERS, "llm", llm) is supplied


class ClosingRecorderStore(InMemoryVectorStore):
    """An in-memory store that records being closed, as Elasticsearch's is."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_a_supplied_port_is_never_closed(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # The elasticsearch opener closes the store it opens when the scope ends.
    settings = settings_factory(
        components__embedder__backend="fake",
        components__vector_store__backend="elasticsearch",
    )
    supplied = ClosingRecorderStore()
    async with CompositionRoot(settings, vector_store=supplied) as composition:
        assert (await composition.ingest_ports()).vector_store is supplied

    assert supplied.closed is False


# --- a new backend is one new module -----------------------------------------------------

# Dropped into composition/openers/ of a copy of the package: the only change.
REVERSED_RERANKER_OPENER = '''"""A reranker backend of the tests: the candidates in reverse order."""

from __future__ import annotations

from collections.abc import Sequence

from ...adapters.reranker.base_reranker import BaseReranker
from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord
from .component_openers import RERANKER_OPENERS


class ReversedReranker(BaseReranker):
    async def rerank(self, query, passages, top_n):
        kept = list(passages)[::-1][:top_n]
        return [
            passage.model_copy(update={"rerank_score": float(len(kept) - i)})
            for i, passage in enumerate(kept)
        ]

    async def ready(self):
        return True, "reversed"


@RERANKER_OPENERS.register("reversed")
async def open_reranker_on_reversed(component, shared):
    return ReversedReranker()
'''
# Serves the workflow app of that copy, its reranker on the new backend, and
# asks /retrieve.
DRIVER = """
import json

from fastapi.testclient import TestClient

import query_processing
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app

settings = WorkflowSettings(
    _env_file=None,
    components={
        "embedder": {"backend": "fake"},
        "reranker": {"backend": "reversed"},
        "llm": {"backend": "fake"},
        "vector_store": {"backend": "fake"},
    },
    graph={"stages": {"retrieve": {"top_k": 5}, "rerank": {"top_k": 3}}},
)
with TestClient(create_app(CompositionRoot(settings))) as client:
    answer = client.post("/retrieve", json={"question": "vacation days"}).json()
passages = answer["records"]
print(json.dumps({
    "package": query_processing.__file__,
    "retrieval": [passage["retrieval_score"] for passage in passages],
    "rerank": [passage["rerank_score"] for passage in passages],
}))
"""


def test_a_new_backend_is_one_new_module(tmp_path: Path) -> None:
    package = tmp_path / "src" / "query_processing"
    shutil.copytree(
        TOOLKIT_ROOT / "src" / "query_processing",
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    opener = package / "composition" / "openers" / "reranker_on_reversed.py"
    opener.write_text(REVERSED_RERANKER_OPENER, encoding="utf-8")
    (tmp_path / "driver.py").write_text(DRIVER, encoding="utf-8")
    # No QP_* variable of this shell, and the copy ahead of any installed package.
    env = without_shell_settings(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "src")

    run = subprocess.run(
        [sys.executable, str(tmp_path / "driver.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env=env,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    served = json.loads(run.stdout.splitlines()[-1])
    assert Path(served["package"]).parent == package  # the copy, not the repository
    assert served["rerank"] == [3.0, 2.0, 1.0]
    # The last of the 5 candidates first: retrieval order, reversed.
    assert served["retrieval"] == sorted(served["retrieval"])
