"""``ovms_graph``: the plugin config of a tuning, and the graph's one-line rewrites."""

from __future__ import annotations

import json
import re

import pytest

from ovms_stand_ins import GRAPH_TEMPLATE, exported_graph
from rag_load_test.domain.calibration import StageTuning
from rag_load_test.infrastructure.openvino_cpu_properties import (
    openvino_cpu_properties,
)
from rag_load_test.infrastructure.ovms.ovms_graph import (
    graph_with_normalize_embeddings,
    graph_with_plugin_config,
    ovms_plugin_config,
)

GRAPH = GRAPH_TEMPLATE % {"alias": "rr", "calculator": "RerankCalculatorOV"}
EMBEDDER_GRAPH = exported_graph("embedder", "emb")


def plugin_config_of(graph: str) -> dict[str, str]:
    [found] = re.findall(r"plugin_config: '([^']*)'", graph)
    return dict(json.loads(found))


def test_the_plugin_config_holds_streams_and_the_thread_properties() -> None:
    config = ovms_plugin_config(StageTuning(threads=4, batch_size=20, streams=2))

    assert config == {
        "NUM_STREAMS": "2",
        "INFERENCE_NUM_THREADS": "4",
        "ENABLE_HYPER_THREADING": "YES",
    }
    assert "PERFORMANCE_HINT" not in config  # OVMS refuses it with NUM_STREAMS


def test_more_streams_than_threads_is_refused_naming_both() -> None:
    with pytest.raises(ValueError, match="streams=4, threads=2"):
        ovms_plugin_config(StageTuning(threads=2, batch_size=1, streams=4))


def test_the_thread_properties_allow_hyper_threads() -> None:
    assert openvino_cpu_properties(3) == {
        "INFERENCE_NUM_THREADS": "3",
        "ENABLE_HYPER_THREADING": "YES",
    }
    with pytest.raises(ValueError, match="threads=0"):
        openvino_cpu_properties(0)


def test_only_the_json_of_the_plugin_config_line_changes() -> None:
    config = {"NUM_STREAMS": "2", "INFERENCE_NUM_THREADS": "2"}
    rewritten = graph_with_plugin_config(GRAPH, config, source="rr/graph.pbtxt")

    assert plugin_config_of(rewritten) == config
    before, after = GRAPH.splitlines(), rewritten.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1
    assert after[changed[0]].endswith("',")  # the closing quote and comma stay
    assert 'models_path: "./",' in rewritten


def test_the_rewrite_is_valid_json_and_can_be_redone() -> None:
    once = graph_with_plugin_config(GRAPH, {"NUM_STREAMS": "4"}, source="g")
    twice = graph_with_plugin_config(once, {"NUM_STREAMS": "1"}, source="g")

    assert plugin_config_of(twice) == {"NUM_STREAMS": "1"}


@pytest.mark.parametrize(
    ("graph", "found"),
    [
        ('node { models_path: "./" }\n', 0),
        (GRAPH + GRAPH, 2),
    ],
)
def test_a_graph_without_exactly_one_line_is_refused(graph: str, found: int) -> None:
    with pytest.raises(ValueError, match=f"models/rr/graph.pbtxt: .*found {found}"):
        graph_with_plugin_config(graph, {}, source="models/rr/graph.pbtxt")


# --- normalize_embeddings ------------------------------------------------------------


def normalize_of(graph: str) -> list[str]:
    return re.findall(r"normalize_embeddings: (\w+)", graph)


def test_only_the_value_of_the_normalize_line_changes() -> None:
    rewritten = graph_with_normalize_embeddings(EMBEDDER_GRAPH, False, source="g")

    before, after = EMBEDDER_GRAPH.splitlines(), rewritten.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert [after[i] for i in changed] == ["      normalize_embeddings: false,"]


def test_the_normalize_rewrite_redone_changes_nothing() -> None:
    once = graph_with_normalize_embeddings(EMBEDDER_GRAPH, False, source="g")

    assert graph_with_normalize_embeddings(once, False, source="g") == once


def test_the_normalize_rewrite_goes_from_true_to_false_and_back() -> None:
    off = graph_with_normalize_embeddings(EMBEDDER_GRAPH, False, source="g")
    on = graph_with_normalize_embeddings(off, True, source="g")

    assert (normalize_of(off), normalize_of(on)) == (["false"], ["true"])
    assert on == EMBEDDER_GRAPH


@pytest.mark.parametrize(
    ("graph", "found"),
    [(GRAPH, 0), (EMBEDDER_GRAPH + EMBEDDER_GRAPH, 2)],
    ids=["reranker-graph", "two-lines"],
)
def test_a_graph_without_exactly_one_normalize_line_is_refused(
    graph: str, found: int
) -> None:
    with pytest.raises(ValueError, match=f"models/emb/graph.pbtxt: .*found {found}"):
        graph_with_normalize_embeddings(graph, True, source="models/emb/graph.pbtxt")
