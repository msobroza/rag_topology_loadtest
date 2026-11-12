"""``RagSettings``: the three model components' endpoint settings and their check."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.api_endpoint import ApiEndpoint

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder:8002/v3/embeddings"
RERANK_URL = "http://reranker:8001/v3/rerank"


# --- defaults ------------------------------------------------------------------


def test_embedding_defaults(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert s.embedding_backend == "local"
    assert s.embedding_query_prefix == "query: "
    assert (s.embedding_api_url, s.embedding_api_format) == ("", "openai")
    assert (s.embedding_api_key, s.embedding_api_model) == (
        "",
        "snowflake-arctic-embed-l-v2.0",
    )
    assert (s.embedding_batch_size, s.embedding_threads) == (64, 0)


def test_rerank_defaults(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert s.rerank_backend == "local"
    assert (s.rerank_api_url, s.rerank_api_format) == ("", "cohere")
    assert (s.rerank_api_key, s.rerank_api_model) == (
        "",
        "crossencoder-camembert-base-mmarcoFR",
    )
    assert (s.rerank_batch_size, s.rerank_threads) == (32, 0)


def test_generation_defaults(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert s.generation_backend == "api"
    assert s.generation_api_url == "https://api.openai.com/v1/chat/completions"
    assert (s.generation_api_format, s.generation_api_key) == ("openai", "")
    assert s.generation_api_model == "gpt-4o-mini"
    assert (s.generation_temperature, s.generation_timeout_s) == (0.0, 120.0)


# --- bounds and closed sets ----------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_batch_size", 0),
        ("embedding_threads", -1),
        ("rerank_batch_size", 0),
        ("rerank_threads", -1),
        ("generation_timeout_s", 0),
        ("generation_temperature", 2.5),
        ("embedding_api_format", "cohere"),
        ("rerank_api_format", "openai"),
        ("generation_api_format", "anthropic"),
        ("generation_backend", "local"),
        ("rerank_backend", "ovms"),
        ("embedding_backend", "openai"),
    ],
)
def test_out_of_range_or_unknown_values_are_refused(
    settings_factory: SettingsFactory, field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match=field):
        settings_factory(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_batch_size", 1),
        ("embedding_threads", 0),
        ("rerank_batch_size", 1),
        ("rerank_threads", 0),
        ("generation_timeout_s", 0.5),
    ],
)
def test_values_at_the_bounds_are_accepted(
    settings_factory: SettingsFactory, field: str, value: object
) -> None:
    assert getattr(settings_factory(**{field: value}), field) == value


# --- the environment -------------------------------------------------------------


def test_endpoint_settings_read_rag_prefixed_variables(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_RERANK_BACKEND", "api")
    monkeypatch.setenv("RAG_RERANK_API_URL", RERANK_URL)
    monkeypatch.setenv("RAG_RERANK_API_KEY", "domino")
    monkeypatch.setenv("RAG_GENERATION_API_KEY", "sk-key")
    monkeypatch.setenv("RAG_GENERATION_TIMEOUT_S", "60")

    s = settings_factory()

    assert (s.rerank_backend, s.rerank_api_url) == ("api", RERANK_URL)
    assert (s.rerank_api_key, s.generation_api_key) == ("domino", "sk-key")
    assert s.generation_timeout_s == 60.0


@pytest.mark.parametrize(
    "name", ["OPENAI_API_KEY", "OPENAI_BASE_URL", "COHERE_API_KEY"]
)
def test_unprefixed_provider_variables_are_not_read(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "https://must-not-be-read")

    s = settings_factory()

    assert s.generation_api() == ApiEndpoint(
        "https://api.openai.com/v1/chat/completions", "openai", "", "gpt-4o-mini"
    )


# --- the three endpoints ---------------------------------------------------------


def test_embedding_api_is_built_from_its_four_fields(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(
        embedding_backend="api",
        embedding_api_url=EMBED_URL,
        embedding_api_key="domino",
        embedding_api_model="bge-small",
    )

    assert s.embedding_api() == ApiEndpoint(EMBED_URL, "openai", "domino", "bge-small")


def test_rerank_api_is_built_from_its_four_fields(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(
        rerank_api_url="https://api.cohere.com/v2/rerank",
        rerank_api_key="co-key",
        rerank_api_model="rerank-v3.5",
    )

    assert s.rerank_api() == ApiEndpoint(
        "https://api.cohere.com/v2/rerank", "cohere", "co-key", "rerank-v3.5"
    )


def test_generation_api_is_built_from_its_four_fields(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(
        generation_api_url="https://gateway/v1/chat/completions",
        generation_api_key="sk-key",
        generation_api_model="llama-3",
    )

    assert s.generation_api() == ApiEndpoint(
        "https://gateway/v1/chat/completions", "openai", "sk-key", "llama-3"
    )


# --- check_api_endpoints ---------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "variable", "example"),
    [
        (
            {"embedding_backend": "api"},
            "RAG_EMBEDDING",
            "http://localhost:8002/v3/embeddings",
        ),
        (
            {"topology": "split-reranker"},
            "RAG_RERANK",
            "http://localhost:8001/v3/rerank",
        ),
        (
            {"generation_api_url": ""},
            "RAG_GENERATION",
            "https://api.openai.com/v1/chat/completions",
        ),
    ],
    ids=["embedding", "rerank-from-the-topology", "generation"],
)
def test_an_api_backend_without_a_url_is_refused(
    settings_factory: SettingsFactory,
    overrides: dict[str, str],
    variable: str,
    example: str,
) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_factory(**overrides)

    assert (
        f"{variable}_BACKEND='api' needs {variable}_API_URL, the full route of the "
        f"endpoint (for example {example}); got ''"
    ) in str(caught.value)


@pytest.mark.parametrize("url", ["localhost:8001/v3/rerank", "ftp://h/rerank", " "])
def test_a_url_that_is_not_http_is_refused_with_its_value(
    settings_factory: SettingsFactory, url: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_factory(rerank_backend="api", rerank_api_url=url)

    assert "RAG_RERANK_API_URL, the full route" in str(caught.value)
    assert f"got {url!r}" in str(caught.value)


@pytest.mark.parametrize("url", [RERANK_URL, "https://api.cohere.com/v2/rerank"])
def test_an_http_or_https_url_is_accepted(
    settings_factory: SettingsFactory, url: str
) -> None:
    assert (
        settings_factory(rerank_backend="api", rerank_api_url=url).rerank_api_url == url
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # monolith: both local
        {"rerank_backend": "fake", "embedding_backend": "fake"},
        {
            "topology": "split-all",
            "embedding_backend": "fake",
            "rerank_backend": "none",
        },
        {"generation_backend": "fake", "generation_api_url": ""},
    ],
    ids=["local", "fake", "fake-and-none", "fake-generation"],
)
def test_the_url_is_not_checked_for_another_backend(
    settings_factory: SettingsFactory, overrides: dict[str, str]
) -> None:
    s = settings_factory(**overrides)

    assert (s.embedding_api_url, s.rerank_api_url) == ("", "")
