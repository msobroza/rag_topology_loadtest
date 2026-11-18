"""``check_registered_names``: an unknown backend or API format, refused before anything opens.

What a command does with the refusal is tested in
``tests/entrypoints/start_up/test_unknown_names_at_start_up.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from query_processing.composition.registered_names import check_registered_names
from query_processing.config.workflow_settings import WorkflowSettings
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]


def test_the_default_names_are_all_registered(
    settings_factory: SettingsFactory, fake_settings: WorkflowSettings
) -> None:
    check_registered_names(settings_factory())
    check_registered_names(fake_settings)


@pytest.mark.parametrize(
    ("component", "sort", "known"),
    [
        ("embedder", "embedding", "'openai'"),
        ("reranker", "rerank", "'cohere'"),
        ("llm", "generation", "'openai'"),
    ],
)
def test_an_unknown_api_format_is_refused_whatever_the_backend(
    settings_factory: SettingsFactory, component: str, sort: str, known: str
) -> None:
    # Every backend fake: the format is refused although nothing would call it,
    # as the settings refused it before the formats moved to a registry.
    path = f"components__{component}__api__api_format"
    settings = settings_factory(**FAKE_BACKENDS | {path: "jina"})
    message = (
        f"QP_COMPONENTS__{component.upper()}__API__API_FORMAT: "
        f"unknown {sort} API format 'jina' (expected one of: {known})"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        check_registered_names(settings)


@pytest.mark.parametrize("block", ["api", "served"])
def test_an_unknown_structured_output_method_is_refused_whatever_the_backend(
    settings_factory: SettingsFactory, block: str
) -> None:
    path = f"components__llm__{block}__structured_output"
    settings = settings_factory(**FAKE_BACKENDS | {path: "grammar"})
    message = (
        f"QP_COMPONENTS__LLM__{block.upper()}__STRUCTURED_OUTPUT: unknown "
        "structured output method 'grammar' "
        "(expected one of: 'format_instructions', 'json_schema')"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        check_registered_names(settings)


@pytest.mark.parametrize(
    ("component", "known"),
    [
        ("embedder", "'api', 'fake', 'local', 'served'"),
        ("reranker", "'api', 'fake', 'local', 'none', 'served'"),
        ("llm", "'api', 'fake', 'served'"),
    ],
)
def test_a_backend_without_an_opener_is_refused_naming_its_path(
    settings_factory: SettingsFactory, component: str, known: str
) -> None:
    # The settings take any name: the registered openers are the backends.
    path = f"components__{component}__backend"
    settings = settings_factory(**FAKE_BACKENDS | {path: "gpu"})
    message = (
        f"QP_COMPONENTS__{component.upper()}__BACKEND: "
        f"unknown {component} backend 'gpu' (expected one of: {known})"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        check_registered_names(settings)


def test_a_component_of_any_name_is_checked_by_its_type(
    settings_factory: SettingsFactory,
) -> None:
    coarse = {"type": "reranker", "backend": "gpu", "source_id": "org/coarse"}
    settings = settings_factory(**FAKE_BACKENDS, components={"coarse": coarse})

    with pytest.raises(ValueError, match="QP_COMPONENTS__COARSE__BACKEND: unknown"):
        check_registered_names(settings)


def test_a_store_backend_without_an_opener_is_refused(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # The settings keep any name: the registry of openers is what refuses it.
    settings = settings_factory(components__vector_store__backend="gpu")
    known = "'chroma', 'elasticsearch', 'fake'"
    message = (
        "QP_COMPONENTS__VECTOR_STORE__BACKEND: unknown vector_store backend 'gpu' "
        f"(expected one of: {known})"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        check_registered_names(settings)
