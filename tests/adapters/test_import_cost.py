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
    "rag_load_test.adapters.local_models.sentence_transformer_models",
    "rag_load_test.adapters.vector_store.elasticsearch_store",
    "rag_load_test.adapters.vector_store.chroma_store",
    "rag_load_test.adapters.model_api.endpoint_adapters",
    "rag_load_test.adapters.model_api.model_server_contract",
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
