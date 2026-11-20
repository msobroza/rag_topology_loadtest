"""The workflow app and the API formats: an unknown one stops it at start-up, and a
new one is a module dropped into its component's ``api`` package, nothing else."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import RagDependencyError
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKENDS, without_shell_settings
from repo_paths import TOOLKIT_ROOT

SettingsFactory = Callable[..., WorkflowSettings]
URLS = {
    "embedder": "http://embedder.test/v3/embeddings",
    "reranker": "http://reranker.test/v3/rerank",
    "llm": "http://gateway.test/v1/chat/completions",
}


# --- an unknown API format -----------------------------------------------------------


@pytest.mark.parametrize(
    ("component", "sort", "known"),
    [
        ("embedder", "embedding", "'openai'"),
        ("reranker", "rerank", "'cohere'"),
        ("llm", "generation", "'openai'"),
    ],
)
def test_an_unknown_api_format_stops_the_app_at_start_up(
    settings_factory: SettingsFactory, component: str, sort: str, known: str
) -> None:
    on_an_unknown_format = {
        f"components__{component}__backend": "api",
        f"components__{component}__api__url": URLS[component],
        f"components__{component}__api__api_format": "jina",
    }
    settings = settings_factory(**(FAKE_BACKENDS | on_an_unknown_format))
    message = f"unknown {sort} API format 'jina' (expected one of: {known})"

    with (
        pytest.raises(RagDependencyError, match=re.escape(message)),
        TestClient(create_app(CompositionRoot(settings))),
    ):
        pass  # the lifespan opens every port before the first request


# --- a new API format is one new module ----------------------------------------------

# Dropped into adapters/reranker/api/ of a copy of the package: the only change.
FLAT_RERANK_FORMAT = '''"""A rerank format of the tests: texts in, one score per text out."""

from __future__ import annotations

from collections.abc import Sequence

from .base_rerank_api_format import RERANK_API_FORMATS, BaseRerankApiFormat


@RERANK_API_FORMATS.register("flat")
class FlatRerankFormat(BaseRerankApiFormat):
    def request_body(
        self, model: str, query: str, documents: Sequence[str], top_n: int
    ) -> dict[str, object]:
        return {"model": model, "query": query, "texts": list(documents)}

    def scores(self, body: object) -> dict[int, float]:
        return dict(enumerate(body["scores"]))
'''
# Serves the workflow app of that copy, its reranker on an endpoint in the new
# format whose scores rise with the position of the text, and asks /retrieve.
DRIVER = """
import json

import httpx
from fastapi.testclient import TestClient

import query_processing
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app


def flat_endpoint(request):
    texts = json.loads(request.content)["texts"]
    return httpx.Response(200, json={"scores": [float(i) for i in range(len(texts))]})


reranker = {
    "backend": "api",
    "api": {"url": "http://reranker.test/flat", "api_format": "flat"},
}
settings = WorkflowSettings(
    _env_file=None,
    components={
        "embedder": {"backend": "fake"},
        "reranker": reranker,
        "llm": {"backend": "fake"},
        "vector_store": {"backend": "fake"},
    },
    graph={"stages": {"retrieve": {"top_k": 5}, "rerank": {"top_k": 3}}},
)
composition = CompositionRoot(settings, transport=httpx.MockTransport(flat_endpoint))
with TestClient(create_app(composition)) as client:
    answer = client.post("/retrieve", json={"question": "vacation days"}).json()
scores = [passage["rerank_score"] for passage in answer["records"]]
print(json.dumps({"package": query_processing.__file__, "scores": scores}))
"""


def package_copy_with(root: Path, module: Path, source: str) -> Path:
    """Copy the package under ``root/src`` and add one module to the copy."""
    package = root / "src" / "query_processing"
    shutil.copytree(
        TOOLKIT_ROOT / "src" / "query_processing",
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (package / module).write_text(source, encoding="utf-8")
    return package


def test_a_new_api_format_is_one_new_module(tmp_path: Path) -> None:
    module = Path("adapters", "reranker", "api", "flat_rerank_format.py")
    package = package_copy_with(tmp_path, module, FLAT_RERANK_FORMAT)
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
    assert served["scores"] == [4.0, 3.0, 2.0]  # the last texts sent score highest
