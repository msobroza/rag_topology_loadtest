"""The deploy refuses what an App's start-up would refuse, before anything is published.

Through ``deploy`` over the fake Domino, against the tests' workflow file: an
orchestrator, a backend, an API format or a structured output method nobody
registered, and a graph the chosen orchestrator does not run, whether an
override or the workflow file says so. The refusal names the two files, then
says, to the character, what ``qp-serve``'s start-up checks say of the same
settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from domino_deploy_helpers import (
    ENV_1,
    LAUNCH,
    RunMain,
    make_client,
    make_settings,
    no_sleep,
)
from fake_domino.state import FakeDominoApps
from query_processing.composition.composition_root import check_workflow_graph
from query_processing.composition.registered_names import check_registered_names
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.publish import deploy
from rag_graphs import FIXTURE_WORKFLOW, fixture_graph

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main


def start_up_line(**overrides: object) -> str:
    """What qp-serve's two start-up checks raise for the workflow file with ``overrides``.

    Raises:
        AssertionError: Start-up accepts the settings: the test has no line to expect.
    """
    try:
        settings = WorkflowSettings(_env_file=None, **overrides)  # type: ignore[call-arg]
        check_registered_names(settings)  # in qp-serve's order
        check_workflow_graph(settings)
    except ValueError as refused:
        return str(refused)
    raise AssertionError(f"start-up accepts {overrides!r}")


def refusal(workflow_file: Path, line: str) -> str:
    return (
        f"the workflow file {str(workflow_file)!r} with the overrides of deploy.yaml "
        f"makes settings an App refuses at start-up: {line}"
    )


def refused_deploy(overrides: dict[str, object]) -> tuple[str, FakeDominoApps]:
    """Deploy every App with ``overrides``; the refusal and the fake Domino."""
    apps = FakeDominoApps(["Running"])
    settings = make_settings(overrides=overrides)
    with pytest.raises(DominoDeployError) as refused:
        deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    return str(refused.value), apps


def test_an_orchestrator_nobody_registered_is_refused_and_nothing_published() -> None:
    message, apps = refused_deploy({"orchestrator": "pydantic"})
    line = start_up_line(orchestrator="pydantic")
    assert message == refusal(FIXTURE_WORKFLOW, line)
    assert apps.received == []  # not even a lookup reached Domino


@pytest.mark.parametrize(
    "component",
    [
        {"reranker": {"backend": "bogus"}},
        {"vector_store": {"backend": "nope"}},
        {"llm": {"api": {"api_format": "nope"}}},
        {"embedder": {"api": {"api_format": "jina"}}},
        {"llm": {"api": {"structured_output": "telepathy"}}},
        {"llm": {"served": {"structured_output": "telepathy"}}},
    ],
    ids=[
        "reranker-backend",
        "vector-store-backend",
        "llm-api-format",
        "embedder-api-format",
        "api-structured-output",
        "served-structured-output",
    ],
)
def test_a_name_nobody_registered_is_refused_and_nothing_published(
    component: dict[str, object],
) -> None:
    message, apps = refused_deploy({"components": component})
    line = start_up_line(components=component)
    assert message == refusal(FIXTURE_WORKFLOW, line)
    assert apps.received == []  # not even a lookup reached Domino


def two_ways_out_of_rerank() -> list[list[str]]:
    """The fixture's edges and a second plain one out of rerank: pydantic-graph's refusal."""
    return [*fixture_graph()["edges"], ["rerank", "END"]]


def test_a_graph_the_orchestrator_an_override_chooses_does_not_run_is_refused() -> None:
    overrides = {
        "orchestrator": "pydantic_graph",
        "graph": {"edges": two_ways_out_of_rerank()},
    }
    message, apps = refused_deploy(overrides)
    line = start_up_line(**overrides)
    assert line.startswith("orchestrator 'pydantic_graph': ")
    assert message == refusal(FIXTURE_WORKFLOW, line)
    assert apps.received == []  # not even a lookup reached Domino


def test_a_workflow_file_the_app_refuses_is_refused_with_no_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
    data["orchestrator"] = "pydantic_graph"
    data["graph"]["edges"] = two_ways_out_of_rerank()
    workflow_file = tmp_path / "workflow.yaml"
    workflow_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow_file))
    message, apps = refused_deploy({})
    line = start_up_line()
    assert line.startswith("orchestrator 'pydantic_graph': ")
    assert message == refusal(workflow_file, line)
    assert apps.received == []  # not even a lookup reached Domino


def test_the_dry_run_exits_with_start_ups_line_and_prints_no_app(
    run_main: RunMain,
) -> None:
    Path("deploy.yaml").write_text(ENV_1 + "overrides: {orchestrator: pydantic}\n")
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    line = refusal(FIXTURE_WORKFLOW, start_up_line(orchestrator="pydantic"))
    assert (code, out) == (2, "")
    assert err.endswith(f"qp-deploy: error: {line}\n")
