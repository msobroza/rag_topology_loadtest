"""What importing the adapters costs: no heavy library until a port opens."""

from __future__ import annotations

import subprocess
import sys

HEAVY_MODULES = (
    "sentence_transformers",
    "torch",
    "elasticsearch",
    "openai",
    "joblib",
    "chromadb",
    "openvino",
    "optimum",
)
ADAPTER_MODULES = (
    "query_processing.adapters.embedder.local.sentence_transformer_embedder",
    "query_processing.adapters.reranker.local.cross_encoder_reranker",
    "query_processing.infrastructure.local_models.model_pool",
    "query_processing.adapters.vector_store.elasticsearch.elasticsearch_store",
    "query_processing.adapters.vector_store.chroma.chroma_store",
    "query_processing.adapters.embedder.api.api_embedder",
    "query_processing.adapters.reranker.api.api_reranker",
    "query_processing.adapters.llm.api.api_llm",
    "query_processing.adapters.reranker.none.no_reranker",
    "query_processing.infrastructure.local_models.local_model",
    "query_processing.infrastructure.local_models.openvino_export",
    "query_processing.infrastructure.local_models.native_threads",
    "query_processing.infrastructure.local_models.cpu_count",
    # Imports every opener, and so every adapter, whenever composition loads.
    "query_processing.composition.openers",
)


def test_importing_adapters_leaves_heavy_libraries_unimported() -> None:
    # Fresh interpreter: this process may already have torch loaded.
    code = (
        f"import sys; import {', '.join(ADAPTER_MODULES)}; "
        f"print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )
    command = [sys.executable, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
