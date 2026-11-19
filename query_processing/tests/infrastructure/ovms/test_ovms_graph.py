"""``ovms_graph``: the plugin config of a tuning, and the graph's one line rewritten."""

from __future__ import annotations

import json
import re

import pytest

from ovms_stand_ins import GRAPH_TEMPLATE
from query_processing.config.openvino_cpu_settings import OpenVinoCpuSettings
from query_processing.domain.runtime.calibration import StageTuning
from query_processing.infrastructure.ovms.ovms_graph import (
    graph_with_embeddings_options,
    graph_with_plugin_config,
    graph_with_queue_size,
    ovms_plugin_config,
)

GRAPH = GRAPH_TEMPLATE % {"alias": "rr", "calculator": "RerankCalculatorOV"}
EMBEDDINGS_GRAPH = GRAPH_TEMPLATE % {
    "alias": "emb",
    "calculator": "EmbeddingsCalculatorOV",
}
# An export written before the exporter wrote the graph-pool directive.
GRAPH_WITHOUT_DIRECTIVE = GRAPH.split("\n", 1)[1]
DEFAULTS = OpenVinoCpuSettings().chosen()


def plugin_config_of(graph: str) -> dict[str, str]:
    [found] = re.findall(r"plugin_config: '([^']*)'", graph)
    return dict(json.loads(found))


def test_the_plugin_config_holds_streams_and_the_thread_properties() -> None:
    config = ovms_plugin_config(
        StageTuning(threads=4, batch_size=20, streams=2), DEFAULTS
    )

    assert config == {
        "NUM_STREAMS": "2",
        "INFERENCE_NUM_THREADS": "4",
        "ENABLE_HYPER_THREADING": "YES",
    }
    assert "PERFORMANCE_HINT" not in config  # none chosen: NUM_STREAMS is written


def test_a_performance_hint_replaces_num_streams() -> None:
    # OVMS refuses NUM_STREAMS beside a hint: the hint then sets the streams.
    config = ovms_plugin_config(
        StageTuning(threads=4, batch_size=1, streams=1),
        {"PERFORMANCE_HINT": "THROUGHPUT", "ENABLE_HYPER_THREADING": "YES"},
    )

    assert "NUM_STREAMS" not in config
    assert config == {
        "INFERENCE_NUM_THREADS": "4",
        "PERFORMANCE_HINT": "THROUGHPUT",
        "ENABLE_HYPER_THREADING": "YES",
    }


def test_the_plugin_config_holds_the_apps_chosen_properties() -> None:
    chosen = OpenVinoCpuSettings(cpu_pinning=False).chosen()

    config = ovms_plugin_config(StageTuning(threads=2, batch_size=1, streams=1), chosen)

    assert config["ENABLE_CPU_PINNING"] == "NO"


def test_more_streams_than_threads_is_refused_naming_both() -> None:
    with pytest.raises(ValueError, match="streams=4, threads=2"):
        ovms_plugin_config(StageTuning(threads=2, batch_size=1, streams=4), DEFAULTS)


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


# --- the graph-pool directive ------------------------------------------------------


@pytest.mark.parametrize(
    "graph",
    [GRAPH, GRAPH_WITHOUT_DIRECTIVE],
    ids=["as-exported", "exported-before-the-directive"],
)
def test_the_queue_size_is_replaced_or_added_once(graph: str) -> None:
    rewritten = graph_with_queue_size(graph, "4", source="rr/graph.pbtxt")

    first, rest = rewritten.split("\n", 1)
    assert first == "# OVMS_GRAPH_QUEUE_MAX_SIZE: 4"
    assert rewritten.count("OVMS_GRAPH_QUEUE_MAX_SIZE") == 1
    assert rest == GRAPH_WITHOUT_DIRECTIVE  # nothing else moves


def test_the_queue_size_written_again_stays_one_line() -> None:
    once = graph_with_queue_size(GRAPH, "4", source="g")
    twice = graph_with_queue_size(once, "AUTO", source="g")

    assert twice == GRAPH


def test_the_queue_size_left_empty_keeps_the_exports_graph() -> None:
    assert graph_with_queue_size(GRAPH, "", source="g") == GRAPH
    assert (
        graph_with_queue_size(GRAPH_WITHOUT_DIRECTIVE, "", source="g")
        == GRAPH_WITHOUT_DIRECTIVE
    )


def test_a_graph_with_two_queue_size_lines_is_refused() -> None:
    graph = "# OVMS_GRAPH_QUEUE_MAX_SIZE: 2\n" + GRAPH

    with pytest.raises(ValueError, match=r"models/rr/graph.pbtxt: .*found 2"):
        graph_with_queue_size(graph, "4", source="models/rr/graph.pbtxt")


# --- the embeddings options ----------------------------------------------------------


def option_lines(graph: str, key: str) -> list[str]:
    return [line for line in graph.splitlines() if line.strip().startswith(key + ":")]


def test_the_embeddings_options_follow_the_plugin_config_line() -> None:
    rewritten = graph_with_embeddings_options(
        EMBEDDINGS_GRAPH, truncate=True, max_length=512, source="g"
    )

    lines = rewritten.splitlines()
    at = next(i for i, line in enumerate(lines) if "plugin_config:" in line)
    indent = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
    assert lines[at + 1 : at + 3] == [
        f"{indent}truncate: true,",
        f"{indent}max_length: 512,",
    ]
    assert len(lines) == len(EMBEDDINGS_GRAPH.splitlines()) + 2


def test_truncate_is_written_once_on_a_graph_that_already_has_it() -> None:
    once = graph_with_embeddings_options(
        EMBEDDINGS_GRAPH, truncate=True, max_length=512, source="g"
    )
    twice = graph_with_embeddings_options(
        once, truncate=True, max_length=256, source="g"
    )

    assert len(option_lines(twice, "truncate")) == 1
    assert [line.strip() for line in option_lines(twice, "max_length")] == [
        "max_length: 256,"
    ]


def test_truncate_false_removes_the_line_and_unset_keeps_it() -> None:
    truncated = graph_with_embeddings_options(
        EMBEDDINGS_GRAPH, truncate=True, max_length=0, source="g"
    )

    kept = graph_with_embeddings_options(
        truncated, truncate=None, max_length=0, source="g"
    )
    removed = graph_with_embeddings_options(
        truncated, truncate=False, max_length=0, source="g"
    )

    assert kept == truncated
    assert removed == EMBEDDINGS_GRAPH


def test_the_default_embeddings_options_leave_the_graph_unchanged() -> None:
    unchanged = graph_with_embeddings_options(
        EMBEDDINGS_GRAPH, truncate=None, max_length=0, source="g"
    )

    assert unchanged == EMBEDDINGS_GRAPH


def test_an_embeddings_option_on_a_graph_without_plugin_config_is_refused() -> None:
    with pytest.raises(ValueError, match=r"emb/graph.pbtxt: .*found 0"):
        graph_with_embeddings_options(
            'node { models_path: "./" }\n',
            truncate=True,
            max_length=0,
            source="emb/graph.pbtxt",
        )
