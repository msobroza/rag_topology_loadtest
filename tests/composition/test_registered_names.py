"""``check_registered_names``: an unknown backend or API format, refused before anything opens.

What a command does with the refusal is tested in
``tests/entrypoints/test_unknown_names_at_start_up.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from rag_load_test.composition.registered_names import check_registered_names
from rag_load_test.config.rag_settings import RagSettings

SettingsFactory = Callable[..., RagSettings]


def test_the_default_names_are_all_registered(
    settings_factory: SettingsFactory, fake_settings: RagSettings
) -> None:
    check_registered_names(settings_factory())
    check_registered_names(fake_settings)


@pytest.mark.parametrize(
    ("component", "known"),
    [("embedding", "'openai'"), ("rerank", "'cohere'"), ("generation", "'openai'")],
)
def test_an_unknown_api_format_is_refused_whatever_the_backend(
    fake_settings: RagSettings, component: str, known: str
) -> None:
    # Every backend fake: the format is refused although nothing would call it,
    # as the settings refused it before the formats moved to a registry.
    settings = fake_settings.model_copy(update={f"{component}_api_format": "jina"})
    message = f"unknown {component} API format 'jina' (expected one of: {known})"

    with pytest.raises(ValueError, match=re.escape(message)):
        check_registered_names(settings)


@pytest.mark.parametrize(
    ("field", "component_type", "known"),
    [
        ("embedding_backend", "embedder", "'api', 'fake', 'local'"),
        ("rerank_backend", "reranker", "'api', 'fake', 'local', 'none'"),
        ("generation_backend", "llm", "'api', 'fake'"),
        ("vector_store_backend", "vector_store", "'chroma', 'elasticsearch', 'fake'"),
    ],
)
def test_a_backend_without_an_opener_is_refused(
    fake_settings: RagSettings, field: str, component_type: str, known: str
) -> None:
    # model_copy keeps a name the flat settings would refuse themselves.
    settings = fake_settings.model_copy(update={field: "gpu"})
    message = f"unknown {component_type} backend 'gpu' (expected one of: {known})"

    with pytest.raises(ValueError, match=re.escape(message)):
        check_registered_names(settings)
