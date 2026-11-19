"""``VllmSettings``: a vLLM App's flags, named or passed through ``QP_VLLM_ARGS``.

Either way gives the same command line. A named setting left at its default
yields to the pass-through; one that was set to a value that writes its flag,
or a flag the launch script gives, is refused beside it, naming both. The two
CPU variables vLLM reads are exported, and refused when the environment
already holds another value.
"""

from __future__ import annotations

import os
import shlex

import pytest

from query_processing.config.apps.vllm_settings import VllmSettings
from rag_components import shell_settings

EIGHT_GIB = str(8 * 2**30)


@pytest.fixture(autouse=True)
def _no_vllm_variable_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in shell_settings(os.environ, ("QP_VLLM_",)):
        monkeypatch.delenv(name)


def vllm_settings(**values: object) -> VllmSettings:
    """The settings without reading a stray ``.env``."""
    return VllmSettings(_env_file=None, **values)  # type: ignore[arg-type]


def as_vllm_reads(words: list[str]) -> dict[str, str]:
    """Each flag and the value vLLM runs on, in any order.

    A flag written twice fails: vLLM would read one of two values.
    """
    read: dict[str, str] = {}
    for index, word in enumerate(words):
        if not word.startswith("-"):
            continue
        following = words[index + 1] if index + 1 < len(words) else "-"
        assert word not in read, f"{word} is written twice in {words}"
        read[word] = "" if following.startswith("-") else following
    return read


def test_the_default_command_line_bounds_the_kv_cache_and_nothing_else() -> None:
    assert vllm_settings().serve_flags() == ["--kv-cache-memory-bytes", EIGHT_GIB]


def test_the_variables_are_flat_under_qp_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP_VLLM_MAX_MODEL_LEN", "4096")

    assert vllm_settings().serve_flags()[:2] == ["--max-model-len", "4096"]
    assert VllmSettings.variable_of("kv_cache_gib") == "QP_VLLM_KV_CACHE_GIB"


@pytest.mark.parametrize(
    ("named", "passed"),
    [
        ({"max_model_len": 4096}, "--max-model-len 4096"),
        ({"optimization_level": 0}, "--optimization-level 0"),
        ({"max_num_seqs": 32}, "--max-num-seqs 32"),
        ({"max_num_batched_tokens": 8192}, "--max-num-batched-tokens 8192"),
        ({"performance_mode": "throughput"}, "--performance-mode throughput"),
        ({"prefix_caching": True}, "--enable-prefix-caching"),
        ({"prefix_caching": False}, "--no-enable-prefix-caching"),
        ({"kv_cache_dtype": "fp8"}, "--kv-cache-dtype fp8"),
        ({"dtype": "bfloat16"}, "--dtype bfloat16"),
        ({"tensor_parallel_size": 2}, "--tensor-parallel-size 2"),
        ({"stream_interval": 10}, "--stream-interval 10"),
    ],
)
def test_a_flag_named_or_passed_through_gives_the_same_command_line(
    named: dict[str, object], passed: str
) -> None:
    # Compared as vLLM reads them: the named flags come before the cache's,
    # the pass-through after it.
    assert as_vllm_reads(vllm_settings(**named).serve_flags()) == (
        as_vllm_reads(vllm_settings(args=passed).serve_flags())
    )


def test_the_named_flags_come_first_then_the_pass_through() -> None:
    settings = vllm_settings(
        max_model_len=512, kv_cache_gib=0, args="--hf-overrides '{\"a\": 1}'"
    )

    assert settings.serve_flags() == [
        "--max-model-len",
        "512",
        "--hf-overrides",
        '{"a": 1}',
    ]


@pytest.mark.parametrize(
    ("named", "passed", "setting"),
    [
        ({"max_model_len": 4096}, "--max_model_len=8192", "QP_VLLM_MAX_MODEL_LEN=4096"),
        ({"optimization_level": 0}, "-O3", "QP_VLLM_OPTIMIZATION_LEVEL=0"),
        ({"optimization_level": 0}, "-O=3", "QP_VLLM_OPTIMIZATION_LEVEL=0"),
        ({"optimization_level": 0}, "-O", "QP_VLLM_OPTIMIZATION_LEVEL=0"),
        ({"optimization_level": 2}, "--enforce-eager", "QP_VLLM_OPTIMIZATION_LEVEL=2"),
        ({"tensor_parallel_size": 2}, "-tp 4", "QP_VLLM_TENSOR_PARALLEL_SIZE=2"),
        ({"prefix_caching": True}, "--no-enable-prefix-caching", "QP_VLLM_PREFIX"),
        ({"kv_cache_gib": 8}, "--kv-cache-memory-bytes 1", "QP_VLLM_KV_CACHE_GIB=8"),
        ({}, "--port 9000", "launch script"),
        ({}, "--served-model-name=x", "launch script"),
        ({}, "--host 0.0.0.0", "launch script"),
        ({}, "--root-path /p", "launch script"),
    ],
)
def test_a_flag_given_both_ways_is_refused_naming_both(
    named: dict[str, object], passed: str, setting: str
) -> None:
    with pytest.raises(ValueError, match="QP_VLLM_ARGS") as raised:
        vllm_settings(**named, args=passed)

    assert setting in str(raised.value)
    assert passed.split()[0] in str(raised.value)


def test_the_cache_left_at_its_default_yields_to_the_pass_through() -> None:
    settings = vllm_settings(args="--kv-cache-memory-bytes 1")

    assert settings.serve_flags() == ["--kv-cache-memory-bytes", "1"]


def test_the_cache_passed_through_needs_the_named_one_at_zero() -> None:
    settings = vllm_settings(kv_cache_gib=0, args="--kv-cache-memory-bytes 1")

    assert settings.serve_flags() == ["--kv-cache-memory-bytes", "1"]


@pytest.mark.parametrize(
    "args",
    ["--max-model-lens 8", "-Ofast", '--hf-overrides \'{"x": "-O3"}\''],
)
def test_a_word_that_only_looks_like_a_named_flag_is_kept(args: str) -> None:
    settings = vllm_settings(
        max_model_len=512, optimization_level=0, kv_cache_gib=0, args=args
    )

    assert settings.serve_flags() == [
        "--max-model-len",
        "512",
        "--optimization-level",
        "0",
        *shlex.split(args),
    ]


def test_a_pass_through_with_an_open_quote_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="QP_VLLM_ARGS="):
        vllm_settings(args="--dtype 'half")


def test_the_cpu_variables_are_exported_only_when_set() -> None:
    assert vllm_settings().exported_environment({}) == {}
    assert vllm_settings(
        cpu_omp_threads_bind="0-7", cpu_reserved_cpus=0
    ).exported_environment({"VLLM_CPU_OMP_THREADS_BIND": "0-7"}) == {
        "VLLM_CPU_OMP_THREADS_BIND": "0-7",
        "VLLM_CPU_NUM_OF_RESERVED_CPU": "0",
    }


@pytest.mark.parametrize(
    ("named", "environ", "named_in_message"),
    [
        (
            {"cpu_omp_threads_bind": "0-7"},
            {"VLLM_CPU_OMP_THREADS_BIND": "auto"},
            "QP_VLLM_CPU_OMP_THREADS_BIND=0-7",
        ),
        (
            {"cpu_reserved_cpus": 2},
            {"VLLM_CPU_NUM_OF_RESERVED_CPU": "1"},
            "QP_VLLM_CPU_RESERVED_CPUS=2",
        ),
        ({}, {"VLLM_CPU_KVCACHE_SPACE": "8"}, "QP_VLLM_KV_CACHE_GIB=8"),
        ({}, {"VLLM_CPU_KVCACHE_SPACE": "0"}, "QP_VLLM_KV_CACHE_GIB=8"),
        (
            {"kv_cache_gib": 0, "args": "--kv-cache-memory-bytes 1"},
            {"VLLM_CPU_KVCACHE_SPACE": "8"},
            "QP_VLLM_ARGS",
        ),
    ],
)
def test_a_vllm_variable_given_both_ways_is_refused(
    named: dict[str, object], environ: dict[str, str], named_in_message: str
) -> None:
    with pytest.raises(ValueError, match="state it once") as raised:
        vllm_settings(**named).exported_environment(environ)

    assert named_in_message in str(raised.value)
    [(variable, value)] = environ.items()
    assert f"{variable}={value}" in str(raised.value)


def test_the_legacy_cache_variable_alone_is_accepted_with_the_cache_at_zero() -> None:
    settings = vllm_settings(kv_cache_gib=0)

    assert settings.exported_environment({"VLLM_CPU_KVCACHE_SPACE": "8"}) == {}
    assert settings.serve_flags() == []


def test_a_fraction_of_a_gib_is_whole_bytes() -> None:
    assert vllm_settings(kv_cache_gib=0.5).serve_flags() == [
        "--kv-cache-memory-bytes",
        str(2**29),
    ]
