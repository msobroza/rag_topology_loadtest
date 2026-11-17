"""The thirteen charts of ``experiments-explained/``, with the experiments they read.

This is the one place where experiment ids are written: a chart's module knows
how to select and draw, the catalog says from which runs.

Example:
    >>> [chart.slug for chart in experiment_charts()][:2]
    ['01-where-the-time-goes', '02-more-users-more-waiting']
"""

from __future__ import annotations

from .base_experiment_chart import BaseExperimentChart
from .bigger_machine_more_answers_chart import (
    BiggerMachineMoreAnswersChart,
    MachineRun,
    WeightSeries,
)
from .chart_selections import LabelledRun, RunPair
from .generation_under_load_chart import GenerationUnderLoadChart
from .larger_model_chart import LargerModelChart
from .more_users_more_waiting_chart import MoreUsersMoreWaitingChart
from .one_request_fills_the_machine_chart import (
    OneRequestFillsTheMachineChart,
    StreamTrials,
    ThreadTrials,
)
from .overload_chart import OverloadChart
from .reading_and_writing_chart import GeneratorProbe, ReadingAndWritingChart
from .same_weights_same_throughput_chart import SameWeightsSameThroughputChart
from .scale_out_or_scale_up_chart import ApproachSeries, ScaleOutOrScaleUpChart
from .small_batches_chart import BatchTrials, SmallBatchesChart
from .topologies_chart import TopologiesChart, TopologyBar, TopologyPanel
from .where_the_time_goes_chart import WhereTheTimeGoesChart
from .who_waits_how_long_chart import DumbbellPanel, ServerRun, WhoWaitsHowLongChart


def experiment_charts() -> tuple[BaseExperimentChart, ...]:
    """Build the thirteen charts, in the order the tour shows them.

    Returns:
        The charts, numbered 01 to 13 by their slugs.

    Example:
        >>> len(experiment_charts())
        13
    """
    return (
        _where_the_time_goes(),
        _more_users_more_waiting(),
        _bigger_machine_more_answers(),
        _scale_out_or_scale_up(),
        _same_weights_same_throughput(),
        _who_waits_how_long(),
        _one_request_fills_the_machine(),
        _small_batches(),
        _overload(),
        _topologies(),
        _larger_model(),
        _reading_and_writing(),
        _generation_under_load(),
    )


def _where_the_time_goes() -> WhereTheTimeGoesChart:
    panels = (
        LabelledRun("reranker on 8 vCPUs", "v8-e2e-fastapi-fp32"),
        LabelledRun("reranker on 16 vCPUs", "v16-e2e-fastapi-fp32"),
    )
    return WhereTheTimeGoesChart(panels, users=4)


def _more_users_more_waiting() -> MoreUsersMoreWaitingChart:
    machines = (
        LabelledRun("8 vCPUs", "v8-alone-fastapi-fp32"),
        LabelledRun("16 vCPUs", "v16-alone-fastapi-fp32"),
    )
    return MoreUsersMoreWaitingChart(machines, users=(1, 4, 8, 16, 32))


def _bigger_machine_more_answers() -> BiggerMachineMoreAnswersChart:
    full = WeightSeries(
        "32-bit weights",
        (
            MachineRun(2, "m7-calibrated-fastapi-rerank"),
            MachineRun(8, "v8-alone-fastapi-fp32"),
            MachineRun(16, "v16-alone-fastapi-fp32"),
        ),
    )
    compressed = WeightSeries(
        "8-bit weights",
        (
            MachineRun(2, "m8-fastapi-int8-calibrated-rerank"),
            MachineRun(8, "v8-alone-fastapi-int8"),
            MachineRun(16, "v16-alone-fastapi-int8"),
        ),
    )
    return BiggerMachineMoreAnswersChart((full, compressed), users=4)


def _scale_out_or_scale_up() -> ScaleOutOrScaleUpChart:
    out, up = "machines of 2 vCPUs", "one bigger machine"
    series = (
        ApproachSeries("4 machines of 2 vCPUs", out, "s8-out-split-fastapi", 4, 2),
        ApproachSeries("1 machine of 8 vCPUs", up, "s8-up-split-fastapi", 1, 8),
    )
    return ScaleOutOrScaleUpChart(series, (1, 2, 4, 8, 16), (out, up))


def _same_weights_same_throughput() -> SameWeightsSameThroughputChart:
    groups = (
        RunPair("2 vCPUs, 32-bit", "m1-fastapi-rerank", "m8-ovms-fp32-rerank"),
        RunPair("2 vCPUs, 8-bit", "m8-fastapi-int8-rerank", "m1-ovms-rerank"),
        RunPair("8 vCPUs, 32-bit", "v8-alone-fastapi-fp32", "v8-alone-ovms-fp32-cal4"),
        RunPair("8 vCPUs, 8-bit", "v8-alone-fastapi-int8", "v8-alone-ovms-int8-cal4"),
        RunPair(
            "16 vCPUs, 32-bit", "v16-alone-fastapi-fp32", "v16-alone-ovms-fp32-cal4"
        ),
        RunPair(
            "16 vCPUs, 8-bit", "v16-alone-fastapi-int8", "v16-alone-ovms-int8-cal4"
        ),
    )
    return SameWeightsSameThroughputChart(groups, users=4)


def _who_waits_how_long() -> WhoWaitsHowLongChart:
    eight = DumbbellPanel(
        "reranker on 8 vCPUs",
        (
            ServerRun("FastAPI", "FastAPI", "v8-alone-fastapi-fp32"),
            ServerRun("OVMS, 1 stream", "OVMS", "v8-alone-ovms-fp32-1stream"),
            ServerRun("OVMS, 2 streams", "OVMS", "v8-alone-ovms-fp32-cal4"),
        ),
    )
    sixteen = DumbbellPanel(
        "reranker on 16 vCPUs",
        (
            ServerRun("FastAPI", "FastAPI", "v16-alone-fastapi-fp32"),
            ServerRun("OVMS, 1 stream", "OVMS", "v16-alone-ovms-fp32-cal4"),
            ServerRun(
                "OVMS, 1 stream, second run", "OVMS", "v16-alone-ovms-fp32-1stream"
            ),
            ServerRun("OVMS, 4 streams", "OVMS", "v16-alone-ovms-fp32-cal16"),
        ),
    )
    return WhoWaitsHowLongChart((eight, sixteen), users=16)


def _one_request_fills_the_machine() -> OneRequestFillsTheMachineChart:
    threads = (
        ThreadTrials("8-vCPU machine", "v8-alone-fastapi-fp32", "trials_at_batch_2"),
        ThreadTrials("16-vCPU machine", "v16-alone-fastapi-fp32", "trials_at_batch_5"),
    )
    streams = (
        StreamTrials("8 vCPUs", "v8-ovms-fp32-cal16-on-demand", vcpus=8),
        StreamTrials("16 vCPUs", "v16-ovms-fp32-cal16-on-demand", vcpus=16),
    )
    return OneRequestFillsTheMachineChart(threads, streams)


def _small_batches() -> SmallBatchesChart:
    trials = (
        BatchTrials(
            "2 vCPUs",
            "m7-calibrated-fastapi-rerank",
            ("rerank_first_reranker", "trials"),
            threads=2,
        ),
        BatchTrials(
            "8 vCPUs", "v8-alone-fastapi-fp32", ("trials_on_8_threads",), threads=8
        ),
        BatchTrials(
            "16 vCPUs", "v16-alone-fastapi-fp32", ("trials_on_16_threads",), threads=16
        ),
    )
    pairs = (
        RunPair("8 vCPUs", "v8-e2e-fastapi-fp32", "v8-e2e-ovms-fp32-2streams"),
        RunPair("16 vCPUs", "v16-e2e-fastapi-fp32", "v16-e2e-ovms-fp32-cal16"),
    )
    return SmallBatchesChart(trials, (20, 10, 5, 2, 1), pairs, users=4)


def _overload() -> OverloadChart:
    panels = (
        RunPair(
            "reranker on 8 vCPUs", "v8-e2e-fastapi-fp32", "v8-e2e-ovms-fp32-2streams"
        ),
        RunPair(
            "reranker on 16 vCPUs", "v16-e2e-fastapi-fp32", "v16-e2e-ovms-fp32-cal16"
        ),
    )
    return OverloadChart(panels, users=(4, 16, 32))


def _topologies() -> TopologiesChart:
    big = "1 reranker, 8 vCPUs"
    small = "2 rerankers, 2 vCPUs each"
    eight = TopologyPanel(
        "Three machines of 8 vCPUs, 10 users at once",
        users=10,
        bars=(
            TopologyBar("monolith", "pr8-monolith", big, 8),
            TopologyBar("split-reranker", "pr8-split-reranker", big, 8),
            TopologyBar("split-all", "pr8-split-all", big, 8),
        ),
    )
    two = TopologyPanel(
        "Four machines of 2 vCPUs, 8 users at once",
        users=8,
        bars=(
            TopologyBar("monolith", "m2-monolith", "4 rerankers, 2 vCPUs each", 8),
            TopologyBar("split-reranker", "m2-split-reranker", small, 4),
            TopologyBar("split-all", "m2-split-all", small, 4),
        ),
    )
    return TopologiesChart((eight, two))


def _larger_model() -> LargerModelChart:
    sizes = (
        RunPair("8 vCPUs", "v8-alone-fastapi-fp32", "q8-fastapi-openvino-ir-uniform"),
        RunPair(
            "16 vCPUs", "v16-alone-fastapi-fp32", "q16-fastapi-openvino-ir-uniform"
        ),
    )
    names = ("small reranker (0.1 B parameters)", "large reranker (0.6 B parameters)")
    return LargerModelChart(sizes, names, _large_reranker_servers(), users=4)


def _large_reranker_servers() -> tuple[RunPair, ...]:
    return (
        RunPair(
            "8 vCPUs, uniform passages",
            "q8-fastapi-openvino-ir-uniform",
            "q8-ovms-openvino-ir-uniform",
        ),
        RunPair(
            "8 vCPUs, mixed passages",
            "q8-fastapi-openvino-ir-mixed",
            "q8-ovms-openvino-ir-mixed",
        ),
        RunPair(
            "16 vCPUs, uniform passages",
            "q16-fastapi-openvino-ir-uniform",
            "q16-ovms-openvino-ir-uniform",
        ),
        RunPair(
            "16 vCPUs, mixed passages",
            "q16-fastapi-openvino-ir-mixed",
            "q16-ovms-openvino-ir-mixed",
        ),
    )


def _reading_and_writing() -> ReadingAndWritingChart:
    reading, request = "reading_781_tokens_s", "request_922_tokens_s"
    probes = (
        GeneratorProbe(
            "OVMS",
            "g16-probe-ovms-settings",
            ("measurements", "as_shipped", reading),
            ("measurements", "as_shipped", request),
        ),
        GeneratorProbe(
            "PyTorch",
            "g16-probe-pytorch-threads",
            ("measurements", reading, "threads_8"),
            ("measurements", request, "threads_8"),
        ),
        GeneratorProbe(
            "vLLM",
            "g16-probe-vllm",
            ("measurements", reading),
            ("measurements", request),
        ),
    )
    return ReadingAndWritingChart(probes)


def _generation_under_load() -> GenerationUnderLoadChart:
    servers = (
        LabelledRun("OVMS, as it ships", "g16-check-ovms"),
        LabelledRun("OVMS, one pass", "g16-check-ovms-tuned"),
        LabelledRun("PyTorch", "g16-check-torch"),
        LabelledRun("vLLM", "g16-check-vllm"),
    )
    return GenerationUnderLoadChart(servers, users=(1, 4))
