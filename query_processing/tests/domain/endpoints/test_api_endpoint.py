"""``ApiEndpoint``, the Domino run-token key, and the routes model servers answer on."""

from __future__ import annotations

from query_processing.domain.endpoints.api_endpoint import (
    DOMINO_RUN_TOKEN,
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_MODEL_ROUTE,
    MODEL_SERVER_RERANK_PATH,
    VLLM_CHAT_PATH,
    VLLM_EMBEDDINGS_PATH,
    VLLM_MODELS_ROUTE,
    VLLM_RERANK_PATH,
    ApiEndpoint,
    call_path,
)


def test_api_endpoint_fields_in_order() -> None:
    endpoint = ApiEndpoint("http://h/v3/embeddings", "openai", "bge")
    assert endpoint._asdict() == {
        "url": "http://h/v3/embeddings",
        "api_format": "openai",
        "model": "bge",
    }


def test_the_domino_run_token_key_is_the_settings_value() -> None:
    assert DOMINO_RUN_TOKEN == "domino"


def test_model_server_routes_are_the_ovms_ones() -> None:
    assert MODEL_SERVER_EMBEDDINGS_PATH == "/v3/embeddings"
    assert MODEL_SERVER_RERANK_PATH == "/v3/rerank"
    assert MODEL_SERVER_MODEL_ROUTE.format(name="bge") == "/v3/models/bge"


def test_vllm_routes_are_its_openai_compatible_v1_ones() -> None:
    routes = (VLLM_EMBEDDINGS_PATH, VLLM_RERANK_PATH, VLLM_CHAT_PATH, VLLM_MODELS_ROUTE)
    assert routes == (
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/chat/completions",
        "/v1/models",
    )


def test_a_registered_component_is_called_on_the_call_route_of_its_type() -> None:
    assert call_path("shouter") == "/call/shouter"
