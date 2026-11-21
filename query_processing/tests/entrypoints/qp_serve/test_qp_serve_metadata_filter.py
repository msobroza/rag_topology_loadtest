"""A request's metadata filter, seen from outside the workflow app.

The app runs on the fixture workflow file with every component on its fake
backend: the in-memory store is seeded with the synthetic corpus, each record
of which has a ``topic``, the one field the fixture's vector store declares.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app, declared_metadata_fields
from rag_components import FAKE_BACKENDS
from rag_graphs import fixture_graph, rag_graph

QUESTION = "How many paid vacation days do employees accrue each year?"
ON_REMOTE_WORK = [{"attribute": "topic", "comparator": "eq", "value": "remote-work"}]


@pytest.fixture
def client(settings_factory: Callable[..., WorkflowSettings]) -> Iterator[TestClient]:
    settings = settings_factory(**FAKE_BACKENDS)
    with TestClient(create_app(CompositionRoot(settings))) as test_client:
        yield test_client


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_body_with_only_a_question_reports_no_condition(
    client: TestClient, path: str
) -> None:
    body = client.post(path, json={"question": QUESTION}).json()

    assert body["metadata_filter"] == []
    # Unfiltered, the vacation policy wins its own question (the fake ranking).
    assert body["records"][0]["metadata"]["topic"] == "vacation-policy"


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_filter_keeps_only_the_records_that_meet_it_and_is_reported(
    client: TestClient, path: str
) -> None:
    response = client.post(
        path, json={"question": QUESTION, "metadata_filter": ON_REMOTE_WORK}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metadata_filter"] == ON_REMOTE_WORK
    assert body["records"]
    assert {p["metadata"]["topic"] for p in body["records"]} == {"remote-work"}


def test_every_condition_must_hold(client: TestClient) -> None:
    conditions = [
        {
            "attribute": "topic",
            "comparator": "in",
            "value": ["onboarding", "remote-work"],
        },
        {"attribute": "topic", "comparator": "ne", "value": "onboarding"},
    ]

    body = client.post(
        "/retrieve", json={"question": QUESTION, "metadata_filter": conditions}
    ).json()

    assert body["records"]
    assert {p["metadata"]["topic"] for p in body["records"]} == {"remote-work"}


@pytest.mark.parametrize(
    ("condition", "named"),
    [
        ({"attribute": "sorce", "comparator": "eq", "value": "x"}, "sorce"),
        ({"attribute": "topic", "comparator": "gt", "value": "x"}, "'gt'"),
        ({"attribute": "topic", "comparator": "in", "value": "x"}, "'in'"),
        ({"attribute": "topic", "comparator": "eq", "value": 3}, "3 is not of type"),
        # Regression: an empty list was accepted, and Chroma then failed with a
        # ValueError mid-search.
        (
            {"attribute": "topic", "comparator": "in", "value": []},
            "'in' takes a list of at least one",
        ),
        (
            {"attribute": "topic", "comparator": "nin", "value": []},
            "'nin' takes a list of at least one",
        ),
    ],
)
@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_condition_that_does_not_fit_a_declared_field_is_a_422_that_names_it(
    client: TestClient, path: str, condition: dict[str, object], named: str
) -> None:
    response = client.post(
        path, json={"question": QUESTION, "metadata_filter": [condition]}
    )

    assert response.status_code == 422
    assert named in response.json()["detail"]


def test_a_comparator_that_does_not_exist_is_a_422(client: TestClient) -> None:
    condition = {"attribute": "topic", "comparator": "like", "value": "x"}

    response = client.post(
        "/query", json={"question": QUESTION, "metadata_filter": [condition]}
    )

    assert response.status_code == 422
    assert "like" in response.text


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_a_filter_restricts_each_retrieval_mode(
    settings_factory: Callable[..., WorkflowSettings], mode: str
) -> None:
    settings = settings_factory(**FAKE_BACKENDS, graph=rag_graph(mode))

    with TestClient(create_app(CompositionRoot(settings))) as client:
        body = client.post(
            "/retrieve", json={"question": QUESTION, "metadata_filter": ON_REMOTE_WORK}
        ).json()

    assert body["records"]
    assert {p["metadata"]["topic"] for p in body["records"]} == {"remote-work"}


def test_the_fields_declared_are_those_of_the_stores_retrieve_stages_call(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    def store(*fields: tuple[str, str]) -> dict[str, object]:
        declared = [{"name": n, "type": t, "description": "d"} for n, t in fields]
        return {"backend": "fake", "metadata_field_info": declared}

    graph = fixture_graph()
    graph["stages"]["retrieve_archive"] = {
        **graph["stages"]["retrieve"],
        "type": "retrieve",
        "call": "archive",
    }
    settings = settings_factory(
        components__archive={"type": "vector_store", **store(("topic", "string"))},
        components__unread={"type": "vector_store", **store(("owner", "string"))},
        components__vector_store=store(("topic", "string"), ("year", "integer")),
        graph=graph,
    )

    declared = declared_metadata_fields(settings)

    # Two stores declare topic alike: it is listed once, in the order of the
    # stages; the store no retrieve stage calls declares nothing a request can name.
    assert [(f.name, f.type) for f in declared] == [
        ("topic", "string"),
        ("year", "integer"),
    ]


def test_two_stores_that_declare_one_field_differently_are_refused_naming_both(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    topic_as = {
        "string": {"name": "topic", "type": "string", "description": "d"},
        "integer": {"name": "topic", "type": "integer", "description": "d"},
    }
    graph = fixture_graph()
    graph["stages"]["retrieve_archive"] = {
        **graph["stages"]["retrieve"],
        "type": "retrieve",
        "call": "archive",
    }
    archive = {"backend": "fake", "metadata_field_info": [topic_as["integer"]]}
    settings = settings_factory(
        components__archive={"type": "vector_store", **archive},
        components__vector_store__metadata_field_info=[topic_as["string"]],
        graph=graph,
    )

    with pytest.raises(ValueError, match="declare the metadata field") as refused:
        declared_metadata_fields(settings)

    message = str(refused.value)
    assert "metadata field 'topic'" in message
    assert "'vector_store'" in message
    assert "'archive'" in message
    assert "'string'" in message
    assert "'integer'" in message


# The synthetic corpus numbers its documents by ``seq``, a field no store declares.
LATER_DOCUMENTS = [{"attribute": "seq", "comparator": "gte", "value": 100}]


@pytest.fixture
def prefiltered_client(
    settings_factory: Callable[..., WorkflowSettings],
) -> Iterator[TestClient]:
    """The app whose store's prefilter keeps the later half of the documents."""
    settings = settings_factory(
        **FAKE_BACKENDS, components__vector_store__prefilter=LATER_DOCUMENTS
    )
    with TestClient(create_app(CompositionRoot(settings))) as test_client:
        yield test_client


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_the_prefilter_applies_with_the_requests_conditions_and_is_not_reported(
    prefiltered_client: TestClient, client: TestClient, path: str
) -> None:
    body = {"question": QUESTION, "metadata_filter": ON_REMOTE_WORK}
    unfiltered = client.post(path, json=body).json()

    answer = prefiltered_client.post(path, json=body).json()

    # Without the prefilter an earlier document is retrieved: the test can fail.
    assert min(p["metadata"]["seq"] for p in unfiltered["records"]) < 100
    assert answer["records"]
    assert {p["metadata"]["topic"] for p in answer["records"]} == {"remote-work"}
    assert min(p["metadata"]["seq"] for p in answer["records"]) >= 100
    assert answer["metadata_filter"] == ON_REMOTE_WORK


def test_a_client_cannot_name_a_field_of_the_prefilter(
    prefiltered_client: TestClient,
) -> None:
    response = prefiltered_client.post(
        "/retrieve", json={"question": QUESTION, "metadata_filter": LATER_DOCUMENTS}
    )

    assert response.status_code == 422
    assert "no field 'seq'" in response.json()["detail"]
