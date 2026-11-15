"""The Domino rule ``rag-serve --domino`` applies: model-server keys default to the run token."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_load_test.config.domino_rules import apply_domino_rules
from rag_load_test.config.rag_settings import RagSettings

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder-app/v3/embeddings"
RERANK_URL = "http://reranker-app/v3/rerank"


def key_of(settings: RagSettings, component: str) -> str:
    return settings.components[component].api.api_key


def keys(settings: RagSettings) -> tuple[str, str]:
    return key_of(settings, "embedder"), key_of(settings, "reranker")


def test_api_embedding_and_rerank_keys_default_to_the_run_token(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        topology="split-all",
        components__embedder__api__url=EMBED_URL,
        components__reranker__api__url=RERANK_URL,
    )

    ruled = apply_domino_rules(settings)

    assert keys(ruled) == ("domino", "domino")
    assert keys(settings) == ("", "")  # the input is left as it was


def test_only_the_api_components_get_the_run_token(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        topology="split-reranker", components__reranker__api__url=RERANK_URL
    )

    assert keys(apply_domino_rules(settings)) == ("", "domino")


def test_every_reranker_on_api_gets_the_run_token_whatever_its_name(
    settings_factory: SettingsFactory,
) -> None:
    coarse = {
        "type": "reranker",
        "backend": "api",
        "source_id": "org/coarse",
        "api": {"url": RERANK_URL},
    }
    settings = settings_factory(components={"coarse": coarse})

    assert key_of(apply_domino_rules(settings), "coarse") == "domino"


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # monolith: both local
        {
            "components__embedder__backend": "fake",
            "components__reranker__backend": "none",
        },
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
        components__embedder__api__url=EMBED_URL,
        components__reranker__api__url=RERANK_URL,
        components__reranker__api__api_key=explicit,
    )

    assert keys(apply_domino_rules(settings)) == ("domino", explicit)


def test_a_key_set_in_the_environment_is_explicit(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__API__API_KEY", "")
    settings = settings_factory(
        topology="split-reranker", components__reranker__api__url=RERANK_URL
    )

    assert key_of(apply_domino_rules(settings), "reranker") == ""


@pytest.mark.parametrize("generation_key", ["", "sk-key"])
def test_the_generation_key_is_never_touched(
    settings_factory: SettingsFactory, generation_key: str
) -> None:
    settings = settings_factory(
        topology="split-reranker",
        components__reranker__api__url=RERANK_URL,
        components__llm__api__api_key=generation_key,
    )

    assert key_of(apply_domino_rules(settings), "llm") == generation_key
