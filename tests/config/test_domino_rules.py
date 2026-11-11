"""The Domino rule ``rag-serve --domino`` applies: model-server keys default to the run token."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_load_test.config.domino_rules import apply_domino_rules
from rag_load_test.config.settings import RagSettings

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder-app/v3/embeddings"
RERANK_URL = "http://reranker-app/v3/rerank"


def test_api_embedding_and_rerank_keys_default_to_the_run_token(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        topology="split-all", embedding_api_url=EMBED_URL, rerank_api_url=RERANK_URL
    )

    ruled = apply_domino_rules(settings)

    assert (ruled.embedding_api_key, ruled.rerank_api_key) == ("domino", "domino")
    assert (settings.embedding_api_key, settings.rerank_api_key) == ("", "")  # input


def test_only_the_api_components_get_the_run_token(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(topology="split-reranker", rerank_api_url=RERANK_URL)

    ruled = apply_domino_rules(settings)

    assert (ruled.embedding_api_key, ruled.rerank_api_key) == ("", "domino")


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # monolith: both local
        {"embedding_backend": "fake", "rerank_backend": "none"},
    ],
    ids=["local", "fake-and-none"],
)
def test_no_api_component_leaves_the_settings_as_they_are(
    settings_factory: SettingsFactory, overrides: dict[str, str]
) -> None:
    settings = settings_factory(**overrides)

    assert apply_domino_rules(settings) is settings


@pytest.mark.parametrize("explicit", ["", "static-token", "domino"])
def test_an_explicit_key_is_kept(
    settings_factory: SettingsFactory, explicit: str
) -> None:
    settings = settings_factory(
        topology="split-all",
        embedding_api_url=EMBED_URL,
        rerank_api_url=RERANK_URL,
        rerank_api_key=explicit,
    )

    ruled = apply_domino_rules(settings)

    assert (ruled.embedding_api_key, ruled.rerank_api_key) == ("domino", explicit)


def test_a_key_set_in_the_environment_is_explicit(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_RERANK_API_KEY", "")
    settings = settings_factory(topology="split-reranker", rerank_api_url=RERANK_URL)

    assert apply_domino_rules(settings).rerank_api_key == ""


@pytest.mark.parametrize("generation_key", ["", "sk-key"])
def test_the_generation_key_is_never_touched(
    settings_factory: SettingsFactory, generation_key: str
) -> None:
    settings = settings_factory(
        topology="split-reranker",
        rerank_api_url=RERANK_URL,
        generation_api_key=generation_key,
    )

    assert apply_domino_rules(settings).generation_api_key == generation_key
