"""A generate stage that names an output schema, seen from outside the workflow app.

The test writes its own workflow and prompts files: the fixture's graph with a
``rewrite`` stage first, a generate stage that asks the LLM for an instance of
the tests' schema (``output_schema_fakes``) and writes its two fields, the
self-contained request and the metadata filter, which the stages after it read.
Every component is on its fake backend. The toolkit's own schema,
``standalone_query_with_filter``, narrowed to the fixture store's declared
fields by ``metadata_fields_of``, adds the LLM's conditions to the client's.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel

from output_schema_fakes import FIXTURE_SCHEMA, FixtureRewrite
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.prompts_file import PROMPTS_FILE_VAR
from query_processing.config.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints import qp_serve
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKEND_VARIABLES, FAKE_BACKENDS, shell_settings
from rag_graphs import FIXTURE_PROMPTS, fixture_graph, write_workflow_file

QUESTION = "And for a rented car?"
REWRITE_PROMPT = [{"role": "user", "content": "Rewrite: {{ conversational_query }}"}]
# A rewrite whose prompt lists the declared fields metadata_fields_of gives it.
LISTED = (
    "{% for f in metadata_field_info %}{{ f.name }}: {{ f.description }}{% endfor %}"
)
LISTED_PROMPT = [{"role": "user", "content": LISTED + "|{{ conversational_query }}"}]
# The client's condition, and the one the LLM adds: only remote work meets both.
EITHER = {
    "attribute": "topic",
    "comparator": "in",
    "value": ["onboarding", "remote-work"],
}
NOT_ONBOARDING = {"attribute": "topic", "comparator": "ne", "value": "onboarding"}
Graph = dict[str, Any]


class RecordingLLM(FakeLLM):
    """The fake LLM, recording the schema each call asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.schemas: list[type[BaseModel] | None] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
    ) -> str:
        self.schemas.append(output_schema)
        return await super().generate(messages, output_schema)


class RewritingLLM(RecordingLLM):
    """Under a schema, answers a rewrite with one condition; else the fake's text."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
    ) -> str:
        if output_schema is None:
            return await super().generate(messages, output_schema)
        self.schemas.append(output_schema)
        self.prompts.append(messages[-1]["content"])
        rewrite = {"standalone_query": QUESTION, "metadata_filter": [NOT_ONBOARDING]}
        return json.dumps(rewrite)


def rewrite_graph(**rewrite: object) -> Graph:
    """The fixture's graph with a rewrite stage first, changed by ``rewrite``."""
    graph = fixture_graph()
    graph["stages"]["rewrite"] = {
        "type": "generate",
        "call": "llm",
        "prompt": "rewrite",
        "output_schema": FIXTURE_SCHEMA,
        # A schema of conditions is limited to a store's declared fields.
        "metadata_fields_of": "vector_store",
        "reads": ["conversational_query"],
        "writes": ["standalone_query", "metadata_filter"],
        **rewrite,
    }
    graph["edges"][0] = ["START", "rewrite", "embed", "retrieve", "rerank"]
    return graph


@pytest.fixture
def files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A prompts file with the fixture's prompts and ``rewrite``, named in its variable."""
    prompts = yaml.safe_load(FIXTURE_PROMPTS.read_text(encoding="utf-8"))
    path = tmp_path / "prompts.yaml"
    rewrites = {"rewrite": REWRITE_PROMPT, "rewrite_with_fields": LISTED_PROMPT}
    path.write_text(yaml.safe_dump({**prompts, **rewrites}))
    monkeypatch.setenv(PROMPTS_FILE_VAR, str(path))
    return tmp_path


def test_a_stage_with_a_schema_writes_the_fields_the_stages_after_it_read(
    files: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    workflow = write_workflow_file(files / "workflow.yaml", rewrite_graph())
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))
    settings = settings_factory(**FAKE_BACKENDS)
    llm = RecordingLLM()

    with TestClient(create_app(CompositionRoot(settings, llm=llm))) as app:
        response = app.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    # The rewrite, its schema narrowed under its name, then the answer.
    assert [schema and schema.__name__ for schema in llm.schemas] == [
        FixtureRewrite.__name__,
        None,
    ]
    assert body["answer"].startswith("[fake-llm] ")
    assert body["passages"]
    assert body["metadata_filter"] == []  # the fake LLM's instance has none
    assert "rewrite" in body["timings_ms"]


def test_the_conditions_a_rewrite_writes_are_added_to_the_clients(
    files: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    graph = rewrite_graph(
        prompt="rewrite_with_fields",
        output_schema="standalone_query_with_filter",
        metadata_fields_of="vector_store",
    )
    workflow = write_workflow_file(files / "workflow.yaml", graph)
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))
    settings = settings_factory(**FAKE_BACKENDS)
    llm = RewritingLLM()

    with TestClient(create_app(CompositionRoot(settings, llm=llm))) as app:
        body = {"question": QUESTION, "metadata_filter": [EITHER]}
        response = app.post("/query", json=body)

    assert response.status_code == 200, response.text
    answer = response.json()
    # The prompt got the store's declared fields; the schema allows only them.
    assert llm.prompts == [f"topic: the handbook topic a passage belongs to|{QUESTION}"]
    (asked,) = llm.schemas[:1]
    assert asked is not None
    assert '"const": "topic"' in json.dumps(asked.model_json_schema())
    # Both conditions held in the search: onboarding met the client's alone.
    assert answer["metadata_filter"] == [EITHER, NOT_ONBOARDING]
    assert answer["passages"]
    assert {p["metadata"]["topic"] for p in answer["passages"]} == {"remote-work"}


@pytest.mark.parametrize(
    ("rewrite", "said"),
    [
        (
            {"output_schema": "fixture_rewrit"},
            "stage 'rewrite': names the output schema 'fixture_rewrit', which is "
            "not registered (expected one of: ",
        ),
        (
            {"writes": ["standalone_query"]},
            "stage 'rewrite': the output schema 'fixture_rewrite' has the fields "
            "['standalone_query', 'metadata_filter'], but the stage writes "
            "['standalone_query'] (expected writes listing each field of the "
            "schema, by name)",
        ),
        (
            {"metadata_fields_of": "vector_stor"},
            "stage 'rewrite': metadata_fields_of names 'vector_stor', which is no "
            "vector store (expected one of: 'vector_store')",
        ),
        (
            {"metadata_fields_of": None},
            "stage 'rewrite': the output schema 'fixture_rewrite' holds conditions "
            "in ['metadata_filter'], which no vector store's declared fields would "
            "limit (expected metadata_fields_of: <a vector store> (one of: "
            "'vector_store') on the stage)",
        ),
    ],
    ids=[
        "an-unregistered-schema",
        "writes-that-differ-from-its-fields",
        "metadata-fields-of-no-vector-store",
        "conditions-without-metadata-fields-of",
    ],
)
def test_a_schema_that_does_not_fit_its_stage_is_a_usage_error_at_start_up(
    files: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rewrite: dict[str, object],
    said: str,
) -> None:
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        if name != PROMPTS_FILE_VAR:
            monkeypatch.delenv(name)
    for name, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(name, value)
    served: list[object] = []
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda app, **_: served.append(app))
    workflow = write_workflow_file(files / "workflow.yaml", rewrite_graph(**rewrite))
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    with pytest.raises(SystemExit) as exited:
        qp_serve.main([])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert said in err
    assert "Traceback" not in err
    assert served == []  # the server never started
