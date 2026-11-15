"""``RagSettings``: the model components' endpoint settings, their bounds and check."""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import Any

import pytest
from pydantic import ValidationError

from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.api_endpoint import ApiEndpoint

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder:8002/v3/embeddings"
RERANK_URL = "http://reranker:8001/v3/rerank"


def value_at(settings: RagSettings, path: str) -> Any:
    """The value a keyword named by its path sets, read back from the settings."""
    first, *levels = path.split("__")
    if first == "components":
        name, *levels = levels
        start: Any = settings.components[name]
    else:
        start = getattr(settings, first)
    return reduce(getattr, levels, start)


# --- bounds ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("components__embedder__local__batch_size", 0),
        ("components__embedder__local__threads", -1),
        ("components__embedder__api__batch_size", 0),
        ("components__reranker__local__batch_size", -1),
        ("components__reranker__local__threads", -1),
        ("components__reranker__local__calibration_passages", 0),
        ("calibration_budget_s", 0),
        ("calibration_rounds", 0),
        ("components__llm__http__timeout_s", 0),
        ("components__llm__api__temperature", 2.5),
    ],
)
def test_out_of_range_values_are_refused_naming_their_path(
    settings_factory: SettingsFactory, path: str, value: object
) -> None:
    with pytest.raises(ValidationError) as refused:
        settings_factory(**{path: value})

    assert path.split("__")[-1] in str(refused.value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("components__embedder__local__batch_size", 1),
        ("components__embedder__local__threads", 0),
        ("components__reranker__local__batch_size", 0),
        ("components__reranker__local__batch_size", 1),
        ("components__reranker__local__threads", 0),
        ("calibration_budget_s", 0.001),
        ("calibration_rounds", 1),
        ("components__llm__http__timeout_s", 0.5),
    ],
)
def test_values_at_the_bounds_are_accepted(
    settings_factory: SettingsFactory, path: str, value: object
) -> None:
    assert value_at(settings_factory(**{path: value}), path) == value


# --- the environment -------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["OPENAI_API_KEY", "OPENAI_BASE_URL", "COHERE_API_KEY"]
)
def test_unprefixed_provider_variables_are_not_read(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "https://must-not-be-read")

    llm = settings_factory().components["llm"]

    assert llm.api.endpoint(llm.alias) == ApiEndpoint(
        "https://api.openai.com/v1/chat/completions", "openai", "", "gpt-4o-mini"
    )


# --- the endpoints ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("component", "url", "api_format"),
    [
        ("embedder", EMBED_URL, "openai"),
        ("reranker", "https://api.cohere.com/v2/rerank", "cohere"),
        ("llm", "https://gateway/v1/chat/completions", "openai"),
    ],
)
def test_an_api_block_gives_its_endpoint(
    settings_factory: SettingsFactory, component: str, url: str, api_format: str
) -> None:
    s = settings_factory(
        **{
            f"components__{component}__api__url": url,
            f"components__{component}__api__api_key": "key",
            f"components__{component}__api__model": "served-name",
        }
    )

    model = s.components[component]
    assert model.api.endpoint(model.alias) == ApiEndpoint(
        url, api_format, "key", "served-name"
    )


def test_an_api_block_without_a_model_sends_the_alias(
    settings_factory: SettingsFactory,
) -> None:
    # Our model servers serve a model under its alias: a split topology sends it.
    s = settings_factory(
        topology="split-all",
        components__embedder__api__url=EMBED_URL,
        components__reranker__api__url=RERANK_URL,
    )

    embedder = s.components["embedder"]
    assert (
        embedder.api.endpoint(embedder.alias).model == "snowflake-arctic-embed-l-v2.0"
    )


@pytest.mark.parametrize("component", ["embedder", "reranker", "llm"])
def test_an_api_format_is_passed_on_for_its_registry_to_check(
    settings_factory: SettingsFactory, component: str
) -> None:
    # The registries are in the adapters, which the settings may not import: each
    # command refuses an unknown format before it opens anything.
    s = settings_factory(**{f"components__{component}__api__api_format": "jina"})

    assert s.components[component].api.api_format == "jina"


# --- check_api_endpoints ---------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "component", "example"),
    [
        (
            {"components__embedder__backend": "api"},
            "EMBEDDER",
            "http://localhost:8002/v3/embeddings",
        ),
        (
            {"topology": "split-reranker"},
            "RERANKER",
            "http://localhost:8001/v3/rerank",
        ),
        (
            {"components__llm__api__url": ""},
            "LLM",
            "https://api.openai.com/v1/chat/completions",
        ),
    ],
    ids=["embedder", "reranker-from-the-topology", "llm"],
)
def test_an_api_backend_without_a_url_is_refused_naming_the_paths(
    settings_factory: SettingsFactory,
    overrides: dict[str, str],
    component: str,
    example: str,
) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_factory(**overrides)

    variable = f"RAG_COMPONENTS__{component}"
    assert (
        f"{variable}__BACKEND='api' needs {variable}__API__URL, the full route of "
        f"the endpoint (for example {example}); got ''"
    ) in str(caught.value)


@pytest.mark.parametrize("url", ["localhost:8001/v3/rerank", "ftp://h/rerank", " "])
def test_a_url_that_is_not_http_is_refused_with_its_value(
    settings_factory: SettingsFactory, url: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_factory(
            components__reranker__backend="api", components__reranker__api__url=url
        )

    assert "RAG_COMPONENTS__RERANKER__API__URL, the full route" in str(caught.value)
    assert f"got {url!r}" in str(caught.value)


@pytest.mark.parametrize("url", [RERANK_URL, "https://api.cohere.com/v2/rerank"])
def test_an_http_or_https_url_is_accepted(
    settings_factory: SettingsFactory, url: str
) -> None:
    settings = settings_factory(
        components__reranker__backend="api", components__reranker__api__url=url
    )

    assert settings.components["reranker"].api.url == url


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # monolith: both local
        {
            "components__reranker__backend": "fake",
            "components__embedder__backend": "fake",
        },
        {
            "topology": "split-all",
            "components__embedder__backend": "fake",
            "components__reranker__backend": "none",
        },
        {"components__llm__backend": "fake", "components__llm__api__url": ""},
    ],
    ids=["local", "fake", "fake-and-none", "fake-generation"],
)
def test_the_url_is_not_checked_for_another_backend(
    settings_factory: SettingsFactory, overrides: dict[str, str]
) -> None:
    s = settings_factory(**overrides)

    urls = (s.components["embedder"].api.url, s.components["reranker"].api.url)
    assert urls == ("", "")
