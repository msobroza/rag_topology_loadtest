"""``OvmsSettings``: an OVMS App's own streams, threads and start timeout.

They are no value of the workflow file: one OVMS App serves one component, so
they are that App's flat ``QP_OVMS_*`` variables. The calibration users, which
every model server's calibration shares, are tested with the shared blocks
(``test_shared_settings.py``).
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from query_processing.config.apps.ovms_settings import OvmsSettings
from rag_components import shell_settings


@pytest.fixture(autouse=True)
def _no_ovms_variable_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in shell_settings(os.environ, ("QP_OVMS_",)):
        monkeypatch.delenv(name)


def ovms_settings(**values: object) -> OvmsSettings:
    """The settings without reading a stray ``.env``."""
    return OvmsSettings(_env_file=None, **values)  # type: ignore[arg-type]


def test_ovms_is_calibrated_by_default() -> None:
    s = ovms_settings()

    assert (s.num_streams, s.threads) == (0, 0)
    assert s.start_timeout_s == 120.0


def test_the_variables_are_named_as_qp_calibrate_prints_them() -> None:
    assert OvmsSettings.variable_of("num_streams") == "QP_OVMS_NUM_STREAMS"
    assert OvmsSettings.variable_of("threads") == "QP_OVMS_THREADS"
    assert OvmsSettings.variable_of("start_timeout_s") == "QP_OVMS_START_TIMEOUT_S"


def test_the_ovms_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_OVMS_NUM_STREAMS", "2")
    monkeypatch.setenv("QP_OVMS_THREADS", "4")
    monkeypatch.setenv("QP_OVMS_START_TIMEOUT_S", "30")

    s = ovms_settings()

    assert (s.num_streams, s.threads, s.start_timeout_s) == (2, 4, 30.0)


@pytest.mark.parametrize(
    "overrides",
    [{"num_streams": -1}, {"threads": -1}, {"start_timeout_s": 0}],
)
def test_out_of_range_values_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ovms_settings(**overrides)


def test_a_pinned_stream_count_above_the_pinned_threads_is_refused() -> None:
    with pytest.raises(ValidationError) as raised:
        ovms_settings(num_streams=4, threads=2)

    message = str(raised.value)
    assert "QP_OVMS_NUM_STREAMS=4" in message
    assert "QP_OVMS_THREADS=2" in message


@pytest.mark.parametrize(("streams", "threads"), [(4, 0), (0, 1), (2, 2), (1, 4)])
def test_streams_are_checked_against_threads_only_when_both_are_pinned(
    streams: int, threads: int
) -> None:
    s = ovms_settings(num_streams=streams, threads=threads)

    assert (s.num_streams, s.threads) == (streams, threads)


# --- the LLM's two batching options (OVMS's own flags) -------------------------------


def test_an_llms_default_command_line_reads_each_prompt_in_one_pass() -> None:
    s = ovms_settings()

    assert (s.dynamic_split_fuse, s.max_batched_tokens) == (False, 8192)
    assert s.llm_flags() == [
        "--dynamic_split_fuse",
        "false",
        "--max_num_batched_tokens",
        "8192",
    ]


def test_ovms_own_batching_is_one_setting_each_away() -> None:
    s = ovms_settings(dynamic_split_fuse=True, max_batched_tokens=0)

    assert s.llm_flags() == []  # at OVMS's own values, no flag is passed


def test_the_llms_batching_options_come_from_their_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_OVMS_DYNAMIC_SPLIT_FUSE", "true")
    monkeypatch.setenv("QP_OVMS_MAX_BATCHED_TOKENS", "4096")

    s = ovms_settings()

    assert s.llm_flags() == ["--max_num_batched_tokens", "4096"]


def test_a_negative_batched_token_count_is_refused() -> None:
    with pytest.raises(ValidationError):
        ovms_settings(max_batched_tokens=-1)


@pytest.mark.parametrize(
    ("variable", "value"),
    [("QP_OVMS_DYNAMIC_SPLIT_FUSE", "false"), ("QP_OVMS_MAX_BATCHED_TOKENS", "512")],
)
def test_an_llm_option_on_an_embedder_or_reranker_app_is_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match="applies to an LLM") as raised:
        ovms_settings().require_no_llm_option("reranker")

    message = str(raised.value)
    assert variable in message
    assert "applies to an LLM" in message
    assert "reranker" in message
    # Every place the option can have been set, for the deployer to unset it.
    for place in ("apps.reranker.settings", "defaults.settings", ".env"):
        assert place in message


def test_no_llm_option_set_lets_an_embedder_app_start() -> None:
    ovms_settings().require_no_llm_option("embedder")


@pytest.mark.parametrize("component", ["embedder", "reranker"])
def test_an_embedder_or_reranker_app_still_starts_at_the_llms_defaults(
    component: str,
) -> None:
    # The LLM's defaults write two flags, but only a set option is refused.
    settings = ovms_settings()

    assert settings.llm_flags()
    settings.require_no_llm_option(component)


@pytest.mark.parametrize("variable", ["QP_OVMS_NUM_STREAMS", "QP_OVMS_THREADS"])
def test_a_pinned_stream_or_thread_count_on_an_llm_app_is_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "2")

    with pytest.raises(ValueError, match="serves it uncalibrated") as raised:
        ovms_settings().require_no_stream_pin("llm")

    message = str(raised.value)
    assert variable in message
    assert "llm" in message


def test_no_stream_pin_lets_an_llm_app_start() -> None:
    ovms_settings().require_no_stream_pin("llm")


# --- the performance hint, beside the stream count ---------------------------------


def test_a_hint_beside_a_pinned_stream_count_is_refused() -> None:
    # OVMS refuses PERFORMANCE_HINT together with NUM_STREAMS.
    with pytest.raises(
        ValueError,
        match=r"QP_OVMS_OPENVINO__PERFORMANCE_HINT=THROUGHPUT.*QP_OVMS_NUM_STREAMS=2",
    ):
        ovms_settings(num_streams=2, openvino={"performance_hint": "THROUGHPUT"})


def test_a_hint_passed_through_beside_a_pinned_stream_count_is_refused() -> None:
    with pytest.raises(
        ValueError,
        match=r"QP_OVMS_OPENVINO__PROPERTIES.*PERFORMANCE_HINT=LATENCY"
        r".*QP_OVMS_NUM_STREAMS=1",
    ):
        ovms_settings(
            num_streams=1, openvino={"properties": {"PERFORMANCE_HINT": "LATENCY"}}
        )


def test_a_hint_with_calibrated_streams_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_OVMS_OPENVINO__PERFORMANCE_HINT", "THROUGHPUT")
    monkeypatch.setenv("QP_OVMS_THREADS", "4")

    s = ovms_settings()

    assert s.openvino.chosen()["PERFORMANCE_HINT"] == "THROUGHPUT"
    assert s.num_streams == 0


# --- the graph's options: the pool, the embeddings ---------------------------------


def test_the_graph_options_keep_the_exports_values_by_default() -> None:
    s = ovms_settings()

    assert s.graph_queue_max_size == ""
    assert s.embeddings_truncate is None
    assert s.embeddings_max_length == 0


def test_the_graph_options_come_from_their_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_OVMS_GRAPH_QUEUE_MAX_SIZE", "8")
    monkeypatch.setenv("QP_OVMS_EMBEDDINGS_TRUNCATE", "true")
    monkeypatch.setenv("QP_OVMS_EMBEDDINGS_MAX_LENGTH", "512")

    s = ovms_settings()

    assert (s.graph_queue_max_size, s.embeddings_truncate) == ("8", True)
    assert s.embeddings_max_length == 512


@pytest.mark.parametrize("size", ["", "AUTO", "0", "16"])
def test_a_queue_size_is_auto_a_count_or_empty(size: str) -> None:
    assert ovms_settings(graph_queue_max_size=size).graph_queue_max_size == size


@pytest.mark.parametrize("size", ["auto", "-1", "4 ", "big"])
def test_another_queue_size_is_refused_naming_the_variable(size: str) -> None:
    with pytest.raises(ValidationError, match="QP_OVMS_GRAPH_QUEUE_MAX_SIZE"):
        ovms_settings(graph_queue_max_size=size)


def test_a_negative_max_length_is_refused() -> None:
    with pytest.raises(ValidationError):
        ovms_settings(embeddings_max_length=-1)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("QP_OVMS_EMBEDDINGS_TRUNCATE", "false"),
        ("QP_OVMS_EMBEDDINGS_MAX_LENGTH", "512"),
    ],
)
def test_an_embeddings_option_is_refused_off_an_embedder(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match=f"{variable}={value}.*reranker"):
        ovms_settings().require_no_embeddings_option("reranker")


def test_no_embeddings_option_lets_a_reranker_app_start() -> None:
    ovms_settings(graph_queue_max_size="4").require_no_embeddings_option("reranker")


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("QP_OVMS_OPENVINO__HYPER_THREADING", "false"),
        ("QP_OVMS_OPENVINO__PERFORMANCE_HINT", "THROUGHPUT"),
        ("QP_OVMS_GRAPH_QUEUE_MAX_SIZE", "AUTO"),
        ("QP_OVMS_EMBEDDINGS_TRUNCATE", "true"),
        ("QP_OVMS_EMBEDDINGS_MAX_LENGTH", "512"),
    ],
)
def test_a_graph_option_on_an_llm_app_is_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match="writes that graph itself") as raised:
        ovms_settings().require_no_graph_option("llm")

    assert f"{variable}={value}" in str(raised.value)


def test_no_graph_option_lets_an_llm_app_start() -> None:
    ovms_settings().require_no_graph_option("llm")


# --- the server's flags and the LLM's remaining ones -------------------------------


def test_the_defaults_add_no_server_flag() -> None:
    assert ovms_settings().server_flags() == []


@pytest.mark.parametrize(
    ("named", "passed"),
    [
        ({"rest_workers": 4}, "--rest_workers 4"),
        ({"metrics": True}, "--metrics_enable"),
        ({"log_level": "DEBUG"}, "--log_level DEBUG"),
        ({"cache_dir": "/cache"}, "--cache_dir /cache"),
    ],
)
def test_a_server_flag_named_or_passed_through_gives_the_same_command_line(
    named: dict[str, object], passed: str
) -> None:
    assert ovms_settings(**named).server_flags() == (
        ovms_settings(args=passed).server_flags()
    )


@pytest.mark.parametrize(
    ("named", "passed"),
    [
        ({"rest_workers": 4}, "--rest-workers=8"),
        ({"log_level": "INFO"}, "--log_level DEBUG"),
        ({"metrics": True}, "--metrics_enable"),
        ({"max_num_seqs": 64}, "--max_num_seqs=32"),
        ({"dynamic_split_fuse": False}, "--dynamic_split_fuse true"),
        # At the defaults the named two are written, so the pass-through
        # cannot state them a second time.
        ({}, "--dynamic_split_fuse false"),
        ({}, "--max_num_batched_tokens=8192"),
        ({}, "--rest_port 9000"),
        ({}, "--config_path /x.json"),
        ({}, "--source-model=org/m"),
        ({}, "--task text_generation"),
    ],
)
def test_a_flag_given_both_ways_or_owned_by_the_script_is_refused(
    named: dict[str, object], passed: str
) -> None:
    with pytest.raises(ValueError, match="QP_OVMS_ARGS") as raised:
        ovms_settings(**named, args=passed)

    assert passed.split()[0] in str(raised.value)


def test_a_pass_through_with_an_open_quote_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="QP_OVMS_ARGS="):
        ovms_settings(args="--cache_dir '/a b")


@pytest.mark.parametrize(
    ("named", "flags"),
    [
        ({"max_num_seqs": 64}, ["--max_num_seqs", "64"]),
        ({"cache_size_gb": 8}, ["--cache_size", "8"]),
        ({"prefix_caching": False}, ["--enable_prefix_caching", "false"]),
        ({"prefix_caching": True}, ["--enable_prefix_caching", "true"]),
        ({"kv_cache_precision": "u8"}, ["--kv_cache_precision", "u8"]),
        (
            {"model_distribution_policy": "TENSOR_PARALLEL"},
            ["--model_distribution_policy", "TENSOR_PARALLEL"],
        ),
    ],
)
def test_an_llm_option_becomes_its_flag(
    named: dict[str, object], flags: list[str]
) -> None:
    settings = ovms_settings(dynamic_split_fuse=True, max_batched_tokens=0, **named)

    assert settings.llm_flags() == flags


def test_the_llms_flags_leave_the_server_flags_out() -> None:
    settings = ovms_settings(max_batched_tokens=512, rest_workers=2, args="--x 1")

    assert settings.llm_flags() == [
        *("--dynamic_split_fuse", "false"),
        *("--max_num_batched_tokens", "512"),
    ]
    assert settings.server_flags() == ["--rest_workers", "2", "--x", "1"]


def test_more_sequences_than_batched_tokens_is_refused() -> None:
    with pytest.raises(ValueError, match="QP_OVMS_MAX_NUM_SEQS") as raised:
        ovms_settings(max_num_seqs=512, max_batched_tokens=256)

    message = str(raised.value)
    assert "QP_OVMS_MAX_NUM_SEQS=512" in message
    assert "QP_OVMS_MAX_BATCHED_TOKENS=256" in message


def test_more_sequences_than_the_default_batched_tokens_is_refused() -> None:
    with pytest.raises(ValueError, match="QP_OVMS_MAX_BATCHED_TOKENS=8192"):
        ovms_settings(max_num_seqs=8193)


@pytest.mark.parametrize(
    "named",
    [
        {"max_num_seqs": 256},  # OVMS's own count, under the default 8192
        {"max_num_seqs": 9000, "max_batched_tokens": 0},
        {"max_num_seqs": 256, "max_batched_tokens": 256},
    ],
)
def test_sequences_are_checked_against_batched_tokens_only_when_both_are_set(
    named: dict[str, int],
) -> None:
    assert ovms_settings(**named).max_num_seqs == named["max_num_seqs"]


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("QP_OVMS_MAX_NUM_SEQS", "64"),
        ("QP_OVMS_CACHE_SIZE_GB", "8"),
        ("QP_OVMS_PREFIX_CACHING", "false"),
        ("QP_OVMS_KV_CACHE_PRECISION", "u8"),
        ("QP_OVMS_MODEL_DISTRIBUTION_POLICY", "TENSOR_PARALLEL"),
    ],
)
def test_the_new_llm_options_are_refused_off_an_llm(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match="applies to an LLM") as raised:
        ovms_settings().require_no_llm_option("reranker")

    assert f"{variable}={value}" in str(raised.value)


def test_a_server_flag_is_no_llm_option() -> None:
    settings = ovms_settings(rest_workers=4, args="--log_level DEBUG")

    settings.require_no_llm_option("reranker")
