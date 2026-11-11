"""The Domino rule ``rag-serve --domino`` applies: model server URLs and the run token."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_load_test.config.domino_rules import apply_domino_rules
from rag_load_test.config.settings import RagSettings

SettingsFactory = Callable[..., RagSettings]


# --- Domino rules ----------------------------------------------------------------


def test_domino_rules_require_each_model_server_url(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(topology="split-all")

    with pytest.raises(ValueError, match="RAG_OVMS_EMBEDDINGS_URL") as caught:
        apply_domino_rules(settings)

    message = str(caught.value)
    assert "RAG_OVMS_EMBEDDINGS_URL" in message
    assert "RAG_OVMS_RERANK_URL" in message
    assert "RAG_TOPOLOGY='split-all'" in message


def test_domino_rules_name_only_the_missing_url(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(topology="split-all", ovms_rerank_url="http://a")

    with pytest.raises(ValueError, match="RAG_OVMS_EMBEDDINGS_URL") as caught:
        apply_domino_rules(settings)

    assert "RAG_OVMS_EMBEDDINGS_URL" in str(caught.value)
    assert "RAG_OVMS_RERANK_URL" not in str(caught.value)


def test_domino_rules_ignore_urls_of_ports_not_on_a_model_server(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        topology="split-all", embedder_backend="fake", ovms_rerank_url="http://a"
    )

    assert apply_domino_rules(settings).ovms_auth == "domino"


def test_domino_rules_default_the_auth_to_the_run_token(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(topology="split-reranker", ovms_rerank_url="http://a")

    ruled = apply_domino_rules(settings)

    assert ruled.ovms_auth == "domino"
    assert settings.ovms_auth == ""  # the input is not mutated
    assert apply_domino_rules(settings_factory()).ovms_auth == "domino"


def test_domino_rules_keep_an_explicit_auth(settings_factory: SettingsFactory) -> None:
    for explicit in ("", "static-token"):
        settings = settings_factory(ovms_auth=explicit)
        assert apply_domino_rules(settings).ovms_auth == explicit
