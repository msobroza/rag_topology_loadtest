"""``ApiEndpoint`` and the Domino run-token key: the endpoint settings' value types."""

from __future__ import annotations

from rag_load_test.domain.model_api import (
    DOMINO_RUN_TOKEN,
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_MODEL_ROUTE,
    MODEL_SERVER_RERANK_PATH,
    ApiEndpoint,
)


def test_api_endpoint_fields_in_order() -> None:
    endpoint = ApiEndpoint("http://h/v3/embeddings", "openai", "", "bge")
    assert endpoint._asdict() == {
        "url": "http://h/v3/embeddings",
        "api_format": "openai",
        "api_key": "",
        "model": "bge",
    }


def test_the_domino_run_token_key_is_the_settings_value() -> None:
    assert DOMINO_RUN_TOKEN == "domino"


def test_model_server_routes_are_the_ovms_ones() -> None:
    assert MODEL_SERVER_EMBEDDINGS_PATH == "/v3/embeddings"
    assert MODEL_SERVER_RERANK_PATH == "/v3/rerank"
    assert MODEL_SERVER_MODEL_ROUTE.format(name="bge") == "/v3/models/bge"
