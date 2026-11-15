"""Each chart's table on the real registry: its columns, its rows, known values.

The values are the ones the tour quotes; a registry change that moves them fails
here first.
"""

from __future__ import annotations

from experiment_chart_support import chart_named, real_registry
from experiment_charts.chart_table import ChartTable


def table_of(slug: str) -> ChartTable:
    return chart_named(slug).table(real_registry())


def test_where_the_time_goes() -> None:
    table = table_of("01-where-the-time-goes")
    assert table.columns == (
        "panel",
        "experiment",
        "users",
        "stage",
        "label",
        "median_s",
    )
    assert len(table.rows) == 6
    assert table.column("label")[:3] == (
        "embed the question",
        "search the index",
        "rerank the 20 passages",
    )
    assert table.column("median_s")[4] == 0.008


def test_more_users_more_waiting() -> None:
    table = table_of("02-more-users-more-waiting")
    assert "expected_wait_s" in table.columns
    assert len(table.rows) == 10
    sixteen = table.rows_where("users", 16).rows_where("machine", "8 vCPUs")
    assert sixteen.column("expected_wait_s")[0] == 16 / 1.011


def test_bigger_machine_more_answers() -> None:
    table = table_of("03-bigger-machine-more-answers")
    assert {"vcpus", "weights", "requests_per_s", "vcpu_seconds_per_request"} <= set(
        table.columns
    )
    assert table.column("requests_per_s") == (0.282, 1.004, 1.808, 0.422, 1.553, 3.096)
    assert table.column("vcpu_seconds_per_request")[0] == 2 / 0.282


def test_same_weights_same_throughput() -> None:
    table = table_of("04-same-weights-same-throughput")
    assert table.columns == (
        "group",
        "server",
        "experiment",
        "users",
        "requests_per_s",
    )
    assert table.column("requests_per_s") == (
        0.235,
        0.238,
        0.362,
        0.379,
        1.004,
        0.969,
        1.553,
        1.567,
        1.808,
        2.071,
        3.096,
        3.055,
    )
    assert table.column("server")[:2] == ("FastAPI", "OVMS")


def test_who_waits_how_long() -> None:
    table = table_of("05-who-waits-how-long")
    assert len(table.rows) == 7
    eight = table.rows_where("panel", "reranker on 8 vCPUs")
    assert eight.column("median_s") == (15.0, 13.0, 12.0)
    assert eight.column("p95_s") == (16.0, 27.0, 37.0)


def test_one_request_fills_the_machine() -> None:
    table = table_of("06-one-request-fills-the-machine")
    assert len(table.rows) == 16
    streams = table.rows_where("machine", "16 vCPUs")
    assert streams.column("experiment")[0] == "v16-ovms-fp32-cal16-on-demand"
    assert streams.column("seconds") == (14.78, 14.51, 14.43, 18.25, 29.38)
    eight = table.rows_where("machine", "8 vCPUs")
    assert eight.column("seconds") == (31.83, 30.81, 38.38, 45.27)
    threads = table.rows_where("machine", "8-vCPU machine")
    assert threads.column("threads") == (1, 2, 4, 8)
    assert threads.column("seconds") == (4.93, 2.51, 1.34, 1.24)


def test_small_batches() -> None:
    table = table_of("07-small-batches")
    assert {"seconds", "percent_of_batch_20", "requests_per_s"} <= set(table.columns)
    assert len(table.rows) == 19
    two = table.rows_where("group", "2 vCPUs")
    assert two.column("batch_size") == (20, 10, 5, 2, 1)
    assert two.column("seconds")[0] == 7.132
    assert two.column("percent_of_batch_20")[0] == 100.0
    workflow = table.rows_where("users", 4)
    assert workflow.column("requests_per_s") == (0.934, 0.734, 1.701, 1.244)


def test_overload() -> None:
    table = table_of("08-overload")
    assert len(table.rows) == 12
    eight = table.rows_where("panel", "reranker on 8 vCPUs")
    ovms = eight.rows_where("server", "OVMS")
    assert ovms.column("failures") == (0, 31, 84)
    assert ovms.column("requests") == (87, 76, 113)
    fastapi_32 = eight.rows_where("server", "FastAPI").rows_where("users", 32)
    assert round(float(fastapi_32.column("failed_percent")[0]), 1) == 71.8
    sixteen = table.rows_where("panel", "reranker on 16 vCPUs")
    assert sixteen.column("failures") == (0, 0, 0, 0, 0, 47)


def test_topologies() -> None:
    table = table_of("09-topologies")
    assert table.column("requests_per_s") == (0.652, 0.676, 0.696, 0.649, 0.336, 0.360)
    assert table.column("reranker_vcpus_total") == (8, 8, 8, 8, 4, 4)
    assert table.column("requests_per_s_per_reranker_vcpu")[4] == 0.336 / 4


def test_larger_model() -> None:
    table = table_of("10-larger-model")
    assert len(table.rows) == 12
    sizes = table.rows_where("panel", "Requests a second, FastAPI model server")
    assert sorted(sizes.numbers("requests_per_s")) == [0.166, 0.326, 1.004, 1.808]
