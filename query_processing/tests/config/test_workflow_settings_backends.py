"""``WorkflowSettings``: defaults, the environment, backends, labels, the pinned field set.

Each component runs on the backend its own settings give, and nothing else
decides it: the topology alias and the run label are labels only.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from query_processing.config.workflow_settings import WorkflowSettings

EMBED_URL = "http://embedder/v3/embeddings"
RERANK_URL = "http://reranker/v3/rerank"
PACKAGED_LABEL = "embedder=local+reranker=local+llm=api+vector_store=elasticsearch"


def backends(settings: WorkflowSettings) -> tuple[str, str]:
    components = settings.components
    return components["embedder"].backend, components["reranker"].backend


# --- settings ----------------------------------------------------------------


def test_settings_defaults(settings_factory: Callable[..., WorkflowSettings]) -> None:
    s = settings_factory()

    assert s.topology_alias is None
    assert backends(s) == ("local", "local")
    assert s.run_label == PACKAGED_LABEL
    store = s.components["vector_store"]
    assert (store.backend, store.elasticsearch.url) == (
        "elasticsearch",
        "http://localhost:9200",
    )
    keys = [component.api.key for component in s.model_components().values()]
    assert keys == [None, None, None]  # no key block: no key sent
    assert store.elasticsearch.api_key == ""  # never in the file


def test_settings_read_qp_prefixed_env(
    settings_factory: Callable[..., WorkflowSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_TOPOLOGY_ALIAS", "split-reranker")
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__BACKEND", "served")
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__SERVED__URL", RERANK_URL)
    monkeypatch.setenv("QP_RUN_LABEL", "split-reranker-fastapi")
    monkeypatch.setenv("QP_DOMINO__TRACING", "true")

    s = settings_factory()
    assert s.topology_alias == "split-reranker"
    assert s.components["reranker"].backend == "served"
    assert s.components["reranker"].served.url == RERANK_URL
    assert s.run_label == "split-reranker-fastapi"
    assert s.domino.tracing is True


# --- nothing but a component's own backend places it ------------------------------


@pytest.mark.parametrize("alias", ["monolith", "split-reranker", "split-all"])
def test_a_topology_alias_moves_no_component(
    settings_factory: Callable[..., WorkflowSettings],
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    # The workflow file says local for both, and local they stay: no served URL
    # is needed, and the default run label names what really runs.
    monkeypatch.setenv("QP_TOPOLOGY_ALIAS", alias)

    s = settings_factory()

    assert backends(s) == ("local", "local")
    assert (s.topology_alias, s.run_label) == (alias, PACKAGED_LABEL)


def test_the_old_topology_variable_is_not_read(
    settings_factory: Callable[..., WorkflowSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_TOPOLOGY", "split-all")

    s = settings_factory()

    assert backends(s) == ("local", "local")
    assert s.topology_alias is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {
                "components__embedder__backend": "served",
                "components__embedder__served__url": EMBED_URL,
            },
            ("served", "local"),
        ),
        (
            {
                "components__reranker__backend": "served",
                "components__reranker__served__url": RERANK_URL,
            },
            ("local", "served"),
        ),
        (
            {
                "components__embedder__backend": "api",
                "components__embedder__api__url": EMBED_URL,
                "components__reranker__backend": "none",
            },
            ("api", "none"),
        ),
    ],
    ids=["served-embedder", "served-reranker", "api-and-none"],
)
def test_each_component_runs_on_the_backend_its_settings_give(
    settings_factory: Callable[..., WorkflowSettings],
    overrides: dict[str, str],
    expected: tuple[str, str],
) -> None:
    s = settings_factory(topology_alias="split-all", **overrides)

    assert backends(s) == expected


def test_a_component_of_any_name_keeps_its_own_backend(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    other = {"type": "reranker", "backend": "local", "source_id": "org/other"}

    s = settings_factory(topology_alias="split-all", components={"other": other})

    assert s.components["other"].backend == "local"


# --- the topology alias ------------------------------------------------------------


@pytest.mark.parametrize("unset", [{}, {"topology_alias": ""}], ids=["unset", "empty"])
def test_no_topology_alias_is_none(
    settings_factory: Callable[..., WorkflowSettings], unset: dict[str, str]
) -> None:
    assert settings_factory(**unset).topology_alias is None


def test_any_topology_alias_is_a_label(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # Not only the three of the example: the deploy settings define the aliases.
    assert settings_factory(topology_alias="split-embedder").topology_alias == (
        "split-embedder"
    )


@pytest.mark.parametrize("alias", ["split all", "x" * 129, "new\nline"])
def test_a_topology_alias_no_header_can_carry_is_refused(
    settings_factory: Callable[..., WorkflowSettings], alias: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_factory(topology_alias=alias)

    message = str(caught.value)
    assert f"QP_TOPOLOGY_ALIAS={alias!r} cannot be sent as the X-RAG-Topology" in (
        message
    )
    assert "letters, digits and ._+=:- only, 1 to 128 characters" in message


# --- the run label -----------------------------------------------------------------


def test_the_default_run_label_lists_each_component_with_its_backend(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    s = settings_factory(
        topology_alias="split-all",
        components__embedder__backend="served",
        components__embedder__served__url=EMBED_URL,
        components__reranker__backend="none",
        components__llm__backend="fake",
    )

    assert s.run_label == (
        "embedder=served+reranker=none+llm=fake+vector_store=elasticsearch"
    )


def test_the_default_run_label_follows_the_workflow_files_order(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    other = {"type": "reranker", "backend": "fake", "source_id": "org/fr"}

    s = settings_factory(components={"reranker_fr": other})

    assert s.run_label == PACKAGED_LABEL + "+reranker_fr=fake"


def test_a_default_run_label_no_header_can_carry_says_to_set_one(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    other = {"type": "reranker", "backend": "fake", "source_id": "org/fr"}

    with pytest.raises(ValidationError, match="set QP_RUN_LABEL"):
        settings_factory(components={"re ranker": other})


def test_settings_reject_a_bad_run_label(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    with pytest.raises(ValidationError, match="QP_RUN_LABEL='two words'"):
        settings_factory(run_label="two words")


def test_settings_field_set_is_pinned() -> None:
    # Experiments should vary the topology, not grow knobs: adding a field is a
    # decision. Every component, the vector store too, is the workflow file's
    # components block, each with its blocks of settings; the retrieval mode,
    # the sizes and the prompts are settings of the stages of its graph; what
    # the components share is four blocks named after what they are about. Only
    # the topology alias and the run label, two labels, and Domino's own run
    # variables stay flat. The extensions are the packages the file loads; the
    # prompts and the routes, the catalogue routers list from, files of their own.
    # The orchestrator is the library that runs the graph (ADR 0010).
    assert sorted(WorkflowSettings.model_fields) == [
        "calibration",
        "components",
        "domino",
        "domino_api_proxy",
        "domino_hardware_tier_id",
        "extensions",
        "graph",
        "http",
        "local_models",
        "orchestrator",
        "prompts",
        "routes",
        "run_label",
        "topology_alias",
    ]


def test_no_flat_variable_of_a_model_component_is_read(
    settings_factory: Callable[..., WorkflowSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "QP_EMBEDDING_BACKEND": "fake",
        "QP_RERANK_BACKEND": "none",
        "QP_GENERATION_BACKEND": "fake",
        "QP_RERANK_MODEL": "org/other",
        "QP_EMBEDDING_API_MODEL": "other",
        "QP_GENERATION_TIMEOUT_S": "1",
    }.items():
        monkeypatch.setenv(name, value)

    s = settings_factory()

    backends = [c.backend for c in s.components.values()]
    assert backends == ["local", "local", "api", "elasticsearch"]
    reranker = s.components["reranker"]
    assert reranker.source_id == "fixture-org/fixture-reranker"
    assert s.components["embedder"].alias == "fixture-embedder"
    assert s.components["llm"].http.timeout_s == 120.0
