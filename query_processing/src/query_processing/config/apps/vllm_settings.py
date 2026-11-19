"""A vLLM App's own settings: what it passes to ``vllm serve``, and exports first.

A vLLM App serves one component, on that App's hardware, so these are the
App's, not the workflow's: flat ``QP_VLLM_*`` variables, set in the App's own
block of ``deploy.yaml`` (``apps.<component>.settings``) or in ``.env``, as the
``QP_OVMS_*`` ones are. ``qp-vllm-config`` reads them when a vLLM App starts.
Each named setting is left off the command line at vLLM's own value (0.32.0,
``vllm/engine/arg_utils.py``), but for the KV cache: left out, vLLM's CPU
worker reserves 0.92 of the NUMA node's memory, the host's and not the
container's, so 8 GiB are asked by default, what the one run on a machine
asked (experiments/2026-11-16-generator-ovms-pytorch-vllm-16-vcpus/).
``QP_VLLM_ARGS`` passes any other argument. A named setting left at its default
yields to the flag given there; one set to a value that writes its flag, or a
flag the launch script gives, is refused there, naming both.

Example:
    >>> VllmSettings().serve_flags()
    ['--kv-cache-memory-bytes', '8589934592']
    >>> VllmSettings.variable_of("max_model_len")
    'QP_VLLM_MAX_MODEL_LEN'
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from .base_app_settings import BaseAppSettings
from .flag_pass_through import names_flag, refuse_given_twice, split_pass_through

ENV_PREFIX = "QP_VLLM_"
_GIB = 2**30
# The variable vLLM's docs call legacy, which sets the KV cache as the flag does.
LEGACY_KV_CACHE_VARIABLE = "VLLM_CPU_KVCACHE_SPACE"


class VllmFlag(NamedTuple):
    """One named setting's ``vllm serve`` flag, and the other words that give it.

    Example:
        >>> VllmFlag("dtype", "--dtype", own="").flag
        '--dtype'
    """

    field: str  # the VllmSettings field, QP_VLLM_<FIELD>
    flag: str  # the flag written, in vLLM's spelling
    own: object = 0  # the field's value at which the flag is left out
    also: tuple[str, ...] = ()  # other long flags that set the same value
    short: str = ""  # a regular expression of its short spellings
    scale: int = 1  # the flag's unit in the field's: bytes a GiB


# vLLM's own value of each is the field's default (0, "", unset), but for the
# KV cache, whose 0 leaves it to vLLM. --enforce-eager is level 0 by another
# name; -O3, -O=3 and -O are the level's short spellings, -tp the parallel
# size's.
VLLM_FLAGS = (
    VllmFlag("max_model_len", "--max-model-len"),
    VllmFlag(
        "optimization_level",
        "--optimization-level",
        own=None,
        also=("--enforce-eager", "--no-enforce-eager"),
        short=r"-O(=?\d+)?",
    ),
    VllmFlag("max_num_seqs", "--max-num-seqs"),
    VllmFlag("max_num_batched_tokens", "--max-num-batched-tokens"),
    VllmFlag("performance_mode", "--performance-mode", own=""),
    VllmFlag(
        "prefix_caching",
        "--enable-prefix-caching",
        own=None,
        also=("--no-enable-prefix-caching",),
    ),
    VllmFlag("kv_cache_gib", "--kv-cache-memory-bytes", own=0.0, scale=_GIB),
    VllmFlag("kv_cache_dtype", "--kv-cache-dtype", own=""),
    VllmFlag("dtype", "--dtype", own=""),
    VllmFlag("tensor_parallel_size", "--tensor-parallel-size", short=r"-tp(=.*)?"),
    VllmFlag("stream_interval", "--stream-interval"),
)
# Written by domino/model_servers/vllm.sh: a pass-through giving one again
# would start vLLM on two values.
LAUNCH_FLAGS = ("--host", "--port", "--root-path", "--served-model-name")
# The CPU variables vLLM reads, each exported from a field when it is set.
CPU_VARIABLES = (
    ("cpu_omp_threads_bind", "VLLM_CPU_OMP_THREADS_BIND"),
    ("cpu_reserved_cpus", "VLLM_CPU_NUM_OF_RESERVED_CPU"),
)


class VllmSettings(BaseAppSettings):
    """Every ``QP_VLLM_*`` variable, one field each.

    Construction checks the flags, so a flag given twice stops a vLLM App
    before vLLM starts, with a message naming the variables and the values.

    Example:
        >>> VllmSettings(max_model_len=4096, kv_cache_gib=0).serve_flags()
        ['--max-model-len', '4096']
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        # The launch script's own variables may share the prefix.
        extra="ignore",
    )

    # The most tokens a request may hold; 0 = the model's own. A use case
    # states its LLM's in its deploy file.
    max_model_len: int = Field(default=0, ge=0)
    # Unset = vLLM's 2, which its docs call the production level: steady speed
    # over a fast start. 0 starts with no compilation, as --enforce-eager does.
    optimization_level: int | None = Field(default=None, ge=0)
    # Sequences a step holds; 0 = vLLM's 128 on a CPU. Its recipes set the
    # expected concurrency.
    max_num_seqs: int = Field(default=0, ge=0)
    # Tokens a step holds; 0 = vLLM's 2048 on a CPU. Larger gives a sooner
    # first token, by vLLM's docs.
    max_num_batched_tokens: int = Field(default=0, ge=0)
    # "" = vLLM's balanced; throughput only doubles the two above when unset.
    performance_mode: Literal["", "balanced", "interactivity", "throughput"] = ""
    # Unset = vLLM's own: on for an LLM, off for an encoder model.
    prefix_caching: bool | None = None
    # The KV cache, --kv-cache-memory-bytes in GiB; 0 = left to vLLM, which
    # sizes it from the host's memory, not the container's (the module says why
    # 8).
    kv_cache_gib: float = Field(default=8.0, ge=0)
    # "" = vLLM's auto. fp8 fits more tokens; its effect on answers is not
    # checked here.
    kv_cache_dtype: str = ""
    # "" = the model's; vLLM's docs advise bfloat16 only when float16 misbehaves.
    dtype: str = ""
    # Ranks, one a NUMA node; 0 = vLLM's 1, and machines here have one node.
    tensor_parallel_size: int = Field(default=0, ge=0)
    # Tokens between streamed chunks; 0 = vLLM's 1. The workflow app does not
    # stream.
    stream_interval: int = Field(default=0, ge=0)
    # Exported as VLLM_CPU_OMP_THREADS_BIND; "" = vLLM's auto, one thread a core.
    cpu_omp_threads_bind: str = ""
    # Exported as VLLM_CPU_NUM_OF_RESERVED_CPU; unset = vLLM's own, one CPU
    # kept for its API server.
    cpu_reserved_cpus: int | None = Field(default=None, ge=0)
    # Any other argument, as written on vLLM's command line, after the named ones.
    args: str = ""

    @model_validator(mode="after")
    def check_vllm_flags(self) -> VllmSettings:
        """Fail at start-up on a flag ``vllm serve`` would read twice.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If ``args`` is no list of shell words, or gives, in any
                spelling, a flag that the launch script gives or that a named
                setting set to a value away from vLLM's own gives too.
        """
        words = self._pass_through()
        given = [f for f in VLLM_FLAGS if f.field in self.model_fields_set]
        named = {
            spelling: self._as_written(f.field)
            for f in given
            if self._words_of(f)
            for spelling in (f.flag, *f.also)
        }
        launched = dict.fromkeys(LAUNCH_FLAGS, "the App's launch script")
        variable = self.variable_of("args")
        refuse_given_twice({**launched, **named}, words, pass_through=variable)
        for flag in (f for f in given if f.short and self._words_of(f)):
            _refuse_short_spelling(flag, self._as_written(flag.field), words, variable)
        return self

    def serve_flags(self) -> list[str]:
        """Return the arguments of ``vllm serve``: the named ones, then ``args``.

        Returns:
            Each named flag away from vLLM's own value and not given by
            ``args``, in the order of ``VLLM_FLAGS``, then ``args`` as a shell
            splits it.

        Example:
            >>> VllmSettings(kv_cache_gib=0, args="--dtype half").serve_flags()
            ['--dtype', 'half']
        """
        words = self._pass_through()
        named = [
            word
            for flag in VLLM_FLAGS
            if not _gives(flag, words)
            for word in self._words_of(flag)
        ]
        return named + words

    def exported_environment(self, environ: Mapping[str, str]) -> dict[str, str]:
        """Return the CPU variables vLLM reads, as the settings set them.

        Args:
            environ: The App's environment, such as ``os.environ``.

        Returns:
            ``VLLM_CPU_OMP_THREADS_BIND`` and ``VLLM_CPU_NUM_OF_RESERVED_CPU``,
            each only when its setting is set.

        Raises:
            ValueError: If ``environ`` holds either at another value, or holds
                ``VLLM_CPU_KVCACHE_SPACE`` (any value: vLLM reads 0 too) while
                the KV cache is set by its flag; it names both.

        Example:
            >>> VllmSettings(cpu_reserved_cpus=2).exported_environment({})
            {'VLLM_CPU_NUM_OF_RESERVED_CPU': '2'}
        """
        exported = {
            name: _spelled(getattr(self, field))
            for field, name in CPU_VARIABLES
            if getattr(self, field) not in ("", None)
        }
        for field, name in CPU_VARIABLES:
            given = environ.get(name)
            if name in exported and given is not None and given != exported[name]:
                raise ValueError(
                    f"{self._as_written(field)} exports {name}, and the App's "
                    f"environment holds {name}={given}: state it once"
                )
        self._refuse_the_legacy_cache(environ.get(LEGACY_KV_CACHE_VARIABLE, ""))
        return exported

    def _refuse_the_legacy_cache(self, space: str) -> None:
        if not space:
            return
        cache = next(f for f in VLLM_FLAGS if f.field == "kv_cache_gib")
        if _gives(cache, self._pass_through()):
            stated = f"{self.variable_of('args')} gives {cache.flag}"
        elif self.kv_cache_gib:
            stated = f"{self._as_written('kv_cache_gib')} gives {cache.flag}"
        else:
            return
        raise ValueError(
            f"{stated}, and the App's environment holds {LEGACY_KV_CACHE_VARIABLE}"
            f"={space}, which vLLM reads as the same cache: state it once (unset "
            f"{LEGACY_KV_CACHE_VARIABLE}, or set {self.variable_of('kv_cache_gib')}=0)"
        )

    def _words_of(self, flag: VllmFlag) -> list[str]:
        value = getattr(self, flag.field)
        # The types too: 0 is a level of optimization, None the level left out.
        if type(value) is type(flag.own) and value == flag.own:
            return []
        if isinstance(value, bool):
            return [flag.flag if value else f"--no-{flag.flag.removeprefix('--')}"]
        if flag.scale > 1:
            return [flag.flag, str(int(value * flag.scale))]
        return [flag.flag, str(value)]

    def _pass_through(self) -> list[str]:
        return split_pass_through(self.args, variable=self.variable_of("args"))

    def _as_written(self, field: str) -> str:
        return f"{self.variable_of(field)}={_spelled(getattr(self, field))}"


def _gives(flag: VllmFlag, words: Sequence[str]) -> bool:
    # Any word naming the flag, another long flag of the same value, or a short
    # spelling of it.
    spellings = (flag.flag, *flag.also)
    return any(
        any(names_flag(word, s) for s in spellings)
        or bool(flag.short and re.fullmatch(flag.short, word))
        for word in words
    )


def _refuse_short_spelling(
    flag: VllmFlag, setting: str, words: Sequence[str], variable: str
) -> None:
    short = next((w for w in words if re.fullmatch(flag.short, w)), None)
    if short is not None:
        raise ValueError(
            f"{setting} gives {flag.flag}, and {variable} gives it again as "
            f"{short!r}: state it once"
        )


def _spelled(value: object) -> str:
    # As a variable spells it: false, not False; 8, not 8.0.
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
