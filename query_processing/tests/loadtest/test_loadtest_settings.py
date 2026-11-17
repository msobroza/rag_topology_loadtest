"""``LoadTestSettings``: every variable a run reads, its default, and what it refuses."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from loadtest_settings_helpers import settings_from
from query_processing.config.shared_settings import DefaultHttpSettings
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.loadtest.loadtest_settings import LoadTestSettings

BEARER, API_KEY = "QP_LOADTEST_BEARER_TOKEN", "DOMINO_API_KEY"


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


def test_the_workflow_apps_target_reads_no_model_value_and_no_workflow_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("QP_WORKFLOW_FILE")
    monkeypatch.chdir(tmp_path)  # no workflow file here either

    settings = settings_from(monkeypatch, {})

    assert (settings.top_k_retrieve, settings.top_k_rerank) == (None, None)
    assert settings.rerank_sizes == (0, 0)
    assert (settings.reranker_alias, settings.embedder_alias) == ("", "")


@pytest.mark.parametrize(
    ("target", "field", "component"),
    [
        ("rerank", "reranker_alias", "reranker"),
        ("embeddings", "embedder_alias", "embedder"),
    ],
)
def test_a_model_servers_target_sends_the_workflow_files_alias(
    monkeypatch: pytest.MonkeyPatch,
    settings_factory: Callable[..., WorkflowSettings],
    target: str,
    field: str,
    component: str,
) -> None:
    sizes = {"QP_TOP_K_RETRIEVE": "20", "QP_TOP_K_RERANK": "10"}
    settings = settings_from(monkeypatch, {"QP_LOADTEST_TARGET": target, **sizes})

    # As a model server serves its model: the alias the workflow file gives it.
    assert getattr(settings, field) == settings_factory().components[component].alias


def test_a_model_servers_target_without_a_workflow_file_says_where_it_looked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("QP_WORKFLOW_FILE")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="no workflow file: QP_WORKFLOW_FILE is not"):
        settings_from(monkeypatch, {"QP_LOADTEST_TARGET": "embeddings"})


@pytest.mark.parametrize(
    "sizes",
    [{}, {"QP_TOP_K_RETRIEVE": "20"}, {"QP_TOP_K_RERANK": "10"}],
    ids=["neither", "no-top-n", "no-documents"],
)
def test_the_rerank_target_needs_both_sizes_which_have_no_default(
    monkeypatch: pytest.MonkeyPatch, sizes: dict[str, str]
) -> None:
    # They are the use case's sizes: its environment example states them.
    with pytest.raises(ValueError, match="QP_LOADTEST_TARGET=rerank") as caught:
        settings_from(monkeypatch, {"QP_LOADTEST_TARGET": "rerank", **sizes})

    assert (
        "QP_LOADTEST_TARGET=rerank needs QP_TOP_K_RETRIEVE and QP_TOP_K_RERANK, "
        "the documents and top_n of each rerank request, as the workflow file's "
        "retrieve and rerank stages set them (top_k)"
    ) in str(caught.value)


def test_every_variable_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_from(
        monkeypatch,
        {
            "QP_LOADTEST_TARGET": "rerank",
            "QP_LOADTEST_QUERY_WEIGHT": "2",
            "QP_LOADTEST_RETRIEVE_WEIGHT": "0",
            "QP_LOADTEST_SHAPE": "step",
            "QP_LOADTEST_STEP_USERS": "2, 4,8",
            "QP_LOADTEST_STEP_SECONDS": "30",
            "QP_LOADTEST_WAIT_MIN_S": "0",
            "QP_LOADTEST_WAIT_MAX_S": "0.5",
            "QP_LOADTEST_QUESTIONS": "12",
            "QP_LOADTEST_QUESTIONS_FILE": "data/q.jsonl",
            "QP_TOP_K_RETRIEVE": "7",
            "QP_TOP_K_RERANK": "3",
            "QP_COMPONENTS__RERANKER__ALIAS": "my-reranker",
            "QP_COMPONENTS__EMBEDDER__ALIAS": "my-embedder",
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
    assert settings.reranker_alias == "my-reranker"
    assert settings.embedder_alias == "my-embedder"
    assert settings.auth_headers == {"X-Domino-Api-Key": "key"}


def test_an_empty_shape_or_questions_file_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = {"QP_LOADTEST_SHAPE": "", "QP_LOADTEST_QUESTIONS_FILE": ""}
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
            {"QP_LOADTEST_TARGET": "ovms"},
            r"QP_LOADTEST_TARGET='ovms'; expected one of rag, rerank, embeddings",
        ),
        (
            {"QP_LOADTEST_QUERY_WEIGHT": "many"},
            rf"QP_LOADTEST_QUERY_WEIGHT='many'; {WHOLE} 0 or more",
        ),
        (
            {"QP_LOADTEST_RETRIEVE_WEIGHT": "1.5"},
            rf"QP_LOADTEST_RETRIEVE_WEIGHT='1\.5'; {WHOLE} 0 or more",
        ),
        (
            {"QP_LOADTEST_QUESTIONS": "lots"},
            rf"QP_LOADTEST_QUESTIONS='lots'; {WHOLE} 1 or more",
        ),
        (
            {"QP_LOADTEST_QUESTIONS": "0"},
            rf"QP_LOADTEST_QUESTIONS='0'; {WHOLE} 1 or more",
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
            {"QP_LOADTEST_SHAPE": "ramp"},
            r"QP_LOADTEST_SHAPE='ramp'; expected 'step' or unset",
        ),
        (
            {"QP_LOADTEST_SHAPE": "Step"},
            r"QP_LOADTEST_SHAPE='Step'; expected 'step' or unset",
        ),
        (
            {"QP_LOADTEST_QUERY_WEIGHT": "-1"},
            rf"QP_LOADTEST_QUERY_WEIGHT='-1'; {WHOLE} 0 or more",
        ),
        (
            {"QP_LOADTEST_RETRIEVE_WEIGHT": "-3"},
            rf"QP_LOADTEST_RETRIEVE_WEIGHT='-3'; {WHOLE} 0 or more",
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
            {"QP_LOADTEST_STEP_USERS": "5,ten"},
            r"QP_LOADTEST_STEP_USERS='5,ten'; expected comma-separated whole numbers",
        ),
        (
            {"QP_LOADTEST_STEP_USERS": "5,0"},
            r"QP_LOADTEST_STEP_USERS='5,0'; expected comma-separated whole numbers",
        ),
        ({"QP_LOADTEST_STEP_SECONDS": "0"}, r"step_seconds"),
        (
            {"QP_LOADTEST_WAIT_MIN_S": "2", "QP_LOADTEST_WAIT_MAX_S": "1"},
            r"QP_LOADTEST_WAIT_MAX_S=1\.0 is below QP_LOADTEST_WAIT_MIN_S=2\.0",
        ),
        (
            {"QP_TOP_K_RETRIEVE": "3", "QP_TOP_K_RERANK": "5"},
            r"QP_TOP_K_RERANK=5 exceeds QP_TOP_K_RETRIEVE=3",
        ),
    ],
)
def test_the_new_settings_refuse_what_cannot_run(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        settings_from(monkeypatch, variables)


def test_a_user_waits_60_seconds_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert settings_from(monkeypatch, {}).request_timeout_s == 60.0
    told = {"QP_LOADTEST_REQUEST_TIMEOUT_S": "180"}
    assert settings_from(monkeypatch, told).request_timeout_s == 180.0


def test_a_user_that_never_waits_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="request_timeout_s"):
        settings_from(monkeypatch, {"QP_LOADTEST_REQUEST_TIMEOUT_S": "0"})


def test_the_answer_records_settings_and_their_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = DefaultHttpSettings()
    defaults = settings_from(monkeypatch, {})
    told = settings_from(
        monkeypatch,
        {
            "QP_LOADTEST_RETRIES": "5",
            "QP_LOADTEST_BACKOFF_S": "0",
            "QP_LOADTEST_SCORE_TOLERANCE": "0.01",
            "QP_LOADTEST_ANSWER_ROUGE_L": "true",
            "QP_LOADTEST_ANSWER_ROUGE_L_THRESHOLD": "0.8",
        },
    )

    # The retries of a recording are the workflow file's shared http block's.
    assert (defaults.retries, defaults.backoff_s) == (shared.retries, shared.backoff_s)
    assert (defaults.score_tolerance, defaults.answer_rouge_l) == (1e-4, False)
    assert defaults.answer_rouge_l_threshold == 0.9
    assert (told.retries, told.backoff_s) == (5, 0.0)
    assert (told.score_tolerance, told.answer_rouge_l) == (0.01, True)
    assert told.answer_rouge_l_threshold == 0.8


@pytest.mark.parametrize(
    "variable",
    ["QP_LOADTEST_RETRIES", "QP_LOADTEST_BACKOFF_S", "QP_LOADTEST_SCORE_TOLERANCE"],
)
def test_a_negative_retry_count_backoff_or_tolerance_is_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    field = variable.removeprefix("QP_LOADTEST_").lower()
    with pytest.raises(ValueError, match=f"{field}\n.*greater than or equal to 0"):
        settings_from(monkeypatch, {variable: "-1"})


@pytest.mark.parametrize("value", ["-0.1", "1.5", "near"])
def test_a_rouge_l_threshold_outside_0_to_1_is_refused_naming_variable_and_value(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    variable = "QP_LOADTEST_ANSWER_ROUGE_L_THRESHOLD"
    with pytest.raises(ValueError, match=f"{variable}='{value}'; expected a number"):
        settings_from(monkeypatch, {variable: value})


def test_each_field_names_the_variable_it_is_read_from() -> None:
    assert LoadTestSettings.variable_of("target") == "QP_LOADTEST_TARGET"
    assert LoadTestSettings.variable_of("question_count") == "QP_LOADTEST_QUESTIONS"
    assert LoadTestSettings.variable_of("top_k_retrieve") == "QP_TOP_K_RETRIEVE"
    assert LoadTestSettings.variable_of("domino_api_key") == "DOMINO_API_KEY"
