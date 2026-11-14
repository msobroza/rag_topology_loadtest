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
    "rag_load_test.adapters.embedder.local.sentence_transformer_embedder",
    "rag_load_test.adapters.reranker.local.cross_encoder_reranker",
    "rag_load_test.adapters.local_models.model_pool",
    "rag_load_test.adapters.vector_store.elasticsearch.elasticsearch_store",
    "rag_load_test.adapters.vector_store.chroma.chroma_store",
    "rag_load_test.adapters.embedder.api.api_embedder",
    "rag_load_test.adapters.reranker.api.api_reranker",
    "rag_load_test.adapters.llm.api.api_llm",
    "rag_load_test.adapters.reranker.none.no_reranker",
    "rag_load_test.adapters.local_models.local_model",
    "rag_load_test.adapters.local_models.openvino_export",
    "rag_load_test.adapters.local_models.native_threads",
    "rag_load_test.adapters.local_models.cpu_count",
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
