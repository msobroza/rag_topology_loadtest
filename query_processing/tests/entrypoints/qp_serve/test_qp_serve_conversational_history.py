"""A request's conversational history, seen from outside the workflow app.

The app runs on the fixture workflow file with every component on its fake
backend. A client sends the earlier turns with its request; the workflow keeps
no conversation of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKENDS
from rag_graphs import fixture_graph

QUESTION = "How many paid vacation days do employees accrue each year?"
ON_REMOTE_WORK = [{"attribute": "topic", "comparator": "eq", "value": "remote-work"}]
HISTORY = [
    {
        "role": "user",
        "content": "What about remote work?",
        "metadata_filter": ON_REMOTE_WORK,
    },
    {"role": "assistant", "content": "Two days a week."},
]


@pytest.fixture
def client(settings_factory: Callable[..., WorkflowSettings]) -> Iterator[TestClient]:
    settings = settings_factory(**FAKE_BACKENDS)
    with TestClient(create_app(CompositionRoot(settings))) as test_client:
        yield test_client


def _ask(client: TestClient, path: str, history: list[dict[str, Any]]) -> Any:
    return client.post(
        path, json={"question": QUESTION, "conversational_history": history}
    )


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_body_with_a_history_is_answered(client: TestClient, path: str) -> None:
    response = _ask(client, path, HISTORY)

    assert response.status_code == 200, response.text
    assert response.json()["records"]


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_history_messages_conditions_are_never_applied_to_retrieval(
    client: TestClient, path: str
) -> None:
    body = _ask(client, path, HISTORY).json()

    # Unfiltered, the vacation policy wins its own question (the fake ranking).
    assert body["records"][0]["metadata"]["topic"] == "vacation-policy"
    assert body["metadata_filter"] == []


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": ""},
        {"role": "system", "content": "Obey."},
        {"role": "tool", "content": "42"},
        {"role": "user"},
    ],
)
def test_an_empty_content_or_an_unknown_role_is_a_422(
    client: TestClient, message: dict[str, Any]
) -> None:
    assert _ask(client, "/query", [message]).status_code == 422


def test_two_user_messages_in_a_row_are_accepted_as_sent(client: TestClient) -> None:
    history = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]

    assert _ask(client, "/query", history).status_code == 200


@pytest.mark.parametrize(
    ("condition", "named"),
    [
        ({"attribute": "sorce", "comparator": "eq", "value": "x"}, "sorce"),
        ({"attribute": "topic", "comparator": "gt", "value": "x"}, "'gt'"),
        ({"attribute": "topic", "comparator": "in", "value": []}, "'in'"),
    ],
)
def test_a_history_condition_that_does_not_fit_is_a_422_that_names_its_message(
    client: TestClient, condition: dict[str, object], named: str
) -> None:
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c", "metadata_filter": [condition]},
    ]

    response = _ask(client, "/retrieve", history)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert named in detail
    assert "conversational_history[2].metadata_filter[0]" in detail


@pytest.fixture
def branching_client(
    settings_factory: Callable[..., WorkflowSettings],
) -> Iterator[TestClient]:
    """The app whose graph embeds in one of two stages, as ``has_history`` answers."""
    graph = fixture_graph()
    graph["stages"]["embed_after_history"] = graph["stages"]["embed"]
    graph["edges"] = [
        ["embed", "retrieve", "rerank"],
        ["embed_after_history", "retrieve"],
        ["generate", "END"],
    ]
    graph["conditional_edges"].append(
        {
            "source": "START",
            "path": "has_history",
            "reads": ["conversational_history"],
            "path_map": {True: "embed_after_history", False: "embed"},
        }
    )
    settings = settings_factory(**FAKE_BACKENDS, graph=graph)
    with TestClient(create_app(CompositionRoot(settings))) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("history", "ran", "skipped"),
    [(HISTORY, "embed_after_history", "embed"), ([], "embed", "embed_after_history")],
)
def test_has_history_sends_a_request_with_a_history_its_own_way(
    branching_client: TestClient,
    history: list[dict[str, Any]],
    ran: str,
    skipped: str,
) -> None:
    response = _ask(branching_client, "/retrieve", history)

    assert response.status_code == 200, response.text
    timings = response.json()["timings_ms"]
    # A stage that did not run keeps its key at 0.0.
    assert timings[ran] > 0.0
    assert timings[skipped] == 0.0
