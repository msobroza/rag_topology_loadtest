"""``LoadTestSettings``: every variable a run reads, its default, and what it refuses."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from loadtest_settings_helpers import settings_from
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.loadtest.loadtest_settings import LoadTestSettings

BEARER, API_KEY = "RAG_LOADTEST_BEARER_TOKEN", "DOMINO_API_KEY"


def test_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_from(monkeypatch, {})

    assert (settings.target, settings.query_weight, settings.retrieve_weight) == (
        "rag",
        1,
        3,
    )
    assert (settings.shape, settings.step_shape) == (None, False)
    assert (settings.step_users, settings.step_seconds) == ((5, 10, 20, 40), 60.0)
    assert (settings.wait_min_s, settings.wait_max_s) == (0.2, 1.0)
    assert (settings.question_count, settings.questions_file) == (50, None)
    assert settings.auth_headers == {}


def test_the_four_model_values_default_to_the_rag_systems(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., RagSettings]
) -> None:
    settings = settings_from(monkeypatch, {})
    rag = settings_factory()

    assert (settings.top_k_retrieve, settings.top_k_rerank) == (
        rag.top_k_retrieve,
        rag.top_k_rerank,
    )
    assert (settings.rerank_api_model, settings.embedding_api_model) == (
        rag.rerank_api_model,
        rag.embedding_api_model,
    )


def test_every_variable_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_from(
        monkeypatch,
        {
            "RAG_LOADTEST_TARGET": "rerank",
            "RAG_LOADTEST_QUERY_WEIGHT": "2",
            "RAG_LOADTEST_RETRIEVE_WEIGHT": "0",
            "RAG_LOADTEST_SHAPE": "step",
            "RAG_LOADTEST_STEP_USERS": "2, 4,8",
            "RAG_LOADTEST_STEP_SECONDS": "30",
            "RAG_LOADTEST_WAIT_MIN_S": "0",
            "RAG_LOADTEST_WAIT_MAX_S": "0.5",
            "RAG_LOADTEST_QUESTIONS": "12",
            "RAG_LOADTEST_QUESTIONS_FILE": "data/q.jsonl",
            "RAG_TOP_K_RETRIEVE": "7",
            "RAG_TOP_K_RERANK": "3",
            "RAG_RERANK_API_MODEL": "my-reranker",
            "RAG_EMBEDDING_API_MODEL": "my-embedder",
            API_KEY: "key",
        },
    )

    assert (settings.target, settings.query_weight, settings.retrieve_weight) == (
        "rerank",
        2,
        0,
    )
    assert settings.step_shape is True
    assert (settings.step_users, settings.step_seconds) == ((2, 4, 8), 30.0)
    assert (settings.wait_min_s, settings.wait_max_s) == (0.0, 0.5)
    assert settings.question_count == 12
    assert settings.questions_file == Path("data/q.jsonl")
    assert (settings.top_k_retrieve, settings.top_k_rerank) == (7, 3)
    assert settings.rerank_api_model == "my-reranker"
    assert settings.embedding_api_model == "my-embedder"
    assert settings.auth_headers == {"X-Domino-Api-Key": "key"}


def test_an_empty_shape_or_questions_file_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = {"RAG_LOADTEST_SHAPE": "", "RAG_LOADTEST_QUESTIONS_FILE": ""}
    settings = settings_from(monkeypatch, variables)

    assert (settings.shape, settings.questions_file) == (None, None)


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({BEARER: "tok", API_KEY: "key"}, {"Authorization": "Bearer tok"}),
        ({API_KEY: "key"}, {"X-Domino-Api-Key": "key"}),
        ({BEARER: "", API_KEY: ""}, {}),
        ({}, {}),
    ],
)
def test_auth_headers_a_bearer_token_wins(
    monkeypatch: pytest.MonkeyPatch,
    variables: dict[str, str],
    expected: dict[str, str],
) -> None:
    assert settings_from(monkeypatch, variables).auth_headers == expected


WHOLE = "expected a whole number of"


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        (
            {"RAG_LOADTEST_TARGET": "ovms"},
            r"RAG_LOADTEST_TARGET='ovms'; expected one of rag, rerank, embeddings",
        ),
        (
            {"RAG_LOADTEST_QUERY_WEIGHT": "many"},
            rf"RAG_LOADTEST_QUERY_WEIGHT='many'; {WHOLE} 0 or more",
        ),
        (
            {"RAG_LOADTEST_RETRIEVE_WEIGHT": "1.5"},
            rf"RAG_LOADTEST_RETRIEVE_WEIGHT='1\.5'; {WHOLE} 0 or more",
        ),
        (
            {"RAG_LOADTEST_QUESTIONS": "lots"},
            rf"RAG_LOADTEST_QUESTIONS='lots'; {WHOLE} 1 or more",
        ),
        (
            {"RAG_LOADTEST_QUESTIONS": "0"},
            rf"RAG_LOADTEST_QUESTIONS='0'; {WHOLE} 1 or more",
        ),
    ],
)
def test_bad_values_are_refused_naming_the_value(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        settings_from(monkeypatch, variables)


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        (
            {"RAG_LOADTEST_SHAPE": "ramp"},
            r"RAG_LOADTEST_SHAPE='ramp'; expected 'step' or unset",
        ),
        (
            {"RAG_LOADTEST_SHAPE": "Step"},
            r"RAG_LOADTEST_SHAPE='Step'; expected 'step' or unset",
        ),
        (
            {"RAG_LOADTEST_QUERY_WEIGHT": "-1"},
            rf"RAG_LOADTEST_QUERY_WEIGHT='-1'; {WHOLE} 0 or more",
        ),
        (
            {"RAG_LOADTEST_RETRIEVE_WEIGHT": "-3"},
            rf"RAG_LOADTEST_RETRIEVE_WEIGHT='-3'; {WHOLE} 0 or more",
        ),
    ],
)
def test_an_unknown_shape_or_a_negative_weight_is_refused(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        settings_from(monkeypatch, variables)


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        (
            {"RAG_LOADTEST_STEP_USERS": "5,ten"},
            r"RAG_LOADTEST_STEP_USERS='5,ten'; expected comma-separated whole numbers",
        ),
        (
            {"RAG_LOADTEST_STEP_USERS": "5,0"},
            r"RAG_LOADTEST_STEP_USERS='5,0'; expected comma-separated whole numbers",
        ),
        ({"RAG_LOADTEST_STEP_SECONDS": "0"}, r"step_seconds"),
        (
            {"RAG_LOADTEST_WAIT_MIN_S": "2", "RAG_LOADTEST_WAIT_MAX_S": "1"},
            r"RAG_LOADTEST_WAIT_MAX_S=1\.0 is below RAG_LOADTEST_WAIT_MIN_S=2\.0",
        ),
        (
            {"RAG_TOP_K_RETRIEVE": "3", "RAG_TOP_K_RERANK": "5"},
            r"RAG_TOP_K_RERANK=5 exceeds RAG_TOP_K_RETRIEVE=3",
        ),
    ],
)
def test_the_new_settings_refuse_what_cannot_run(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        settings_from(monkeypatch, variables)


def test_each_field_names_the_variable_it_is_read_from() -> None:
    assert LoadTestSettings.variable_of("target") == "RAG_LOADTEST_TARGET"
    assert LoadTestSettings.variable_of("question_count") == "RAG_LOADTEST_QUESTIONS"
    assert LoadTestSettings.variable_of("top_k_retrieve") == "RAG_TOP_K_RETRIEVE"
    assert LoadTestSettings.variable_of("domino_api_key") == "DOMINO_API_KEY"
