"""An OVMS model server App's own settings: streams, threads, CPU, LLM, server flags.

OpenVINO Model Server serves one component per App, on that App's hardware, so
its stream and thread counts are the App's, not the workflow's: they are no
value of the workflow file. They are flat ``QP_OVMS_*`` variables, beside the
ones the launch script reads (``QP_OVMS_MODELS_DIR``, ...), set in the App's own
block of ``deploy.yaml`` (``apps.<component>.settings``) or in ``.env``.
``qp-ovms-config`` reads them when an OVMS App starts, and ``qp-calibrate
--ovms-models`` prints the two that pin a calibration. The streams and threads,
the OpenVINO block and the graph's pool size are an embedder's or a reranker's,
the two embeddings options an embedder's; the batching, sequence and cache
options are an LLM's, which OVMS pulls and serves itself, uncalibrated, on a
graph of its own. An App refuses the other kinds'. The server's flags
(``QP_OVMS_REST_WORKERS``, ..., and the pass-through ``QP_OVMS_ARGS``) are
every App's; ``ovms_flags.py`` says which flag each setting becomes.

Example:
    >>> OvmsSettings().num_streams  # 0 = calibrated when the App starts
    0
    >>> OvmsSettings.variable_of("threads")
    'QP_OVMS_THREADS'
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from ..openvino_cpu_settings import OpenVinoCpuSettings
from .base_app_settings import BaseAppSettings
from .flag_pass_through import names_flag, refuse_given_twice, split_pass_through
from .ovms_flags import (
    LAUNCH_FLAGS,
    LLM_FLAGS,
    SERVER_FLAGS,
    OvmsFlag,
    flag_words,
    spelled,
)

ENV_PREFIX = "QP_OVMS_"
# The OpenVINO block's variables: QP_OVMS_OPENVINO__HYPER_THREADING, ...
_OPENVINO_PREFIX = f"{ENV_PREFIX}OPENVINO__"


class OvmsSettings(BaseAppSettings):
    """Every ``QP_OVMS_*`` variable Python reads, one field each.

    Construction checks them, so a bad pin stops an OVMS App before OVMS
    starts, with a message naming the variables and the values.

    Example:
        >>> OvmsSettings(num_streams=2, threads=8).threads
        8
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        # Between the block and its key: QP_OVMS_OPENVINO__HYPER_THREADING.
        env_nested_delimiter="__",
        # The launch script's own variables share the prefix (QP_OVMS_MODELS_DIR,
        # QP_OVMS_BUNDLE, QP_OVMS_PREFIX_PROXY) and are not read here.
        extra="ignore",
    )

    # Requests OVMS runs at once on its model (NUM_STREAMS); 0 = calibrated at
    # start-up, by qp-ovms-config.
    num_streams: int = Field(default=0, ge=0)
    # Threads its inference shares, all streams together (INFERENCE_NUM_THREADS);
    # 0 = calibrated at start-up, up to the CPUs.
    threads: int = Field(default=0, ge=0)
    # How long one OVMS start may take to answer before calibration fails.
    start_timeout_s: float = Field(default=120.0, gt=0)
    # OpenVINO's CPU properties of the graph's plugin_config, beside the two
    # counts above (QP_OVMS_OPENVINO__*): the block the in-process models read
    # as local_models.openvino.
    openvino: OpenVinoCpuSettings = Field(default_factory=OpenVinoCpuSettings)
    # The graph's "# OVMS_GRAPH_QUEUE_MAX_SIZE:" line: "" = as exported (AUTO,
    # which both exporters write and OVMS's docs/mediapipe.md calls beneficial),
    # else AUTO or a count of graphs.
    graph_queue_max_size: str = ""
    # An embedder's only: its graph's truncate and max_length. Unset = as
    # exported; OVMS's docs give no CPU effect of either.
    embeddings_truncate: bool | None = None
    embeddings_max_length: int = Field(default=0, ge=0)
    # An LLM's only: OVMS's own --dynamic_split_fuse and --max_num_batched_tokens.
    # By default OVMS reads each prompt in one pass: split fuse off, 8192 tokens
    # a step. Measured on 16 vCPUs with prompts of 750 to 920 tokens
    # (experiments/2026-11-16-generator-ovms-pytorch-vllm-16-vcpus/): 20.4
    # tokens a second against 11.6 at OVMS's own batching at 4 users, p50 23 s
    # against 42 s. 16 and 32 users are not measured. Each flag is passed only
    # away from OVMS's own value (split fuse on; 0 = OVMS picks the tokens per
    # step), so true and 0 give OVMS's batching back.
    dynamic_split_fuse: bool = False
    max_batched_tokens: int = Field(default=8192, ge=0)
    # An LLM's only, each left out at OVMS's own (ovms_flags.LLM_FLAGS):
    # --max_num_seqs (0 = OVMS's 256; at most max_batched_tokens), --cache_size
    # in GB (0 = dynamic, which OVMS's docs call the most efficient; a tier with
    # little memory sets one), --enable_prefix_caching (unset = as pull mode
    # sets it), --kv_cache_precision (u8, the one value pull mode takes; its
    # effect on answers is not checked here), --model_distribution_policy (for
    # several sockets; machines here have one).
    max_num_seqs: int = Field(default=0, ge=0)
    cache_size_gb: int = Field(default=0, ge=0)
    prefix_caching: bool | None = None
    kv_cache_precision: Literal["", "u8"] = ""
    model_distribution_policy: Literal["", "TENSOR_PARALLEL", "PIPELINE_PARALLEL"] = ""
    # The ovms server's flags, on every App, each left out at OVMS's own
    # (ovms_flags.SERVER_FLAGS): --rest_workers (0 = from the CPUs; 4 workers on
    # 4 streams gained nothing and lengthened the tail, docs/model-servers.md),
    # --metrics_enable (off: a share of CPU on light models), --log_level
    # ("" = INFO, as OVMS's docs advise), --cache_dir (shortens a start, not a
    # request).
    rest_workers: int = Field(default=0, ge=0)
    metrics: bool = False
    log_level: Literal["", "DEBUG", "INFO", "ERROR"] = ""
    cache_dir: str = ""
    # Any other ovms argument, as written on its command line, after the named
    # ones. A named setting left at its default yields to the flag given here,
    # as the OpenVINO block's do to its properties; a flag a named setting that
    # was set, or the launch script, gives is refused here.
    args: str = ""

    @field_validator("graph_queue_max_size")
    @classmethod
    def check_graph_queue_max_size(cls, size: str) -> str:
        """Refuse a graph-pool size OVMS would not read.

        Args:
            size: The value of ``QP_OVMS_GRAPH_QUEUE_MAX_SIZE``.

        Returns:
            ``size``, unchanged.

        Raises:
            ValueError: If it is neither empty, ``AUTO``, nor digits.
        """
        if size in ("", "AUTO") or (size.isascii() and size.isdigit()):
            return size
        raise ValueError(
            f"{cls.variable_of('graph_queue_max_size')}={size!r}: expected AUTO, "
            "a count of graphs such as 4, or empty to keep the export's line"
        )

    @model_validator(mode="after")
    def check_ovms_tuning(self) -> OvmsSettings:
        """Fail at start-up on a stream pin OVMS could not run.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If both are pinned and ``num_streams`` is above
                ``threads`` (a stream needs a thread), or if a performance
                hint is set beside a pinned ``num_streams`` (OVMS refuses
                ``PERFORMANCE_HINT`` with ``NUM_STREAMS``).
        """
        streams, threads = self.num_streams, self.threads
        if streams and threads and streams > threads:
            raise ValueError(
                f"{self.variable_of('num_streams')}={streams} exceeds "
                f"{self.variable_of('threads')}={threads}: a stream needs a "
                "thread; pin at most as many streams as threads"
            )
        self._refuse_a_hint_beside_pinned_streams()
        return self

    @model_validator(mode="after")
    def check_ovms_flags(self) -> OvmsSettings:
        """Fail at start-up on flags ovms would read twice or refuse.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If ``args`` is no list of shell words, or gives a flag
                a named setting that was set or the launch script gives too
                (ovms would read one of two values), or if ``max_num_seqs`` is
                above ``max_batched_tokens`` while both are written (a step
                could not hold one token of each sequence).
        """
        given = {f.flag: self._as_written(f.field)[0] for f in self._set_flags()}
        launched = dict.fromkeys(LAUNCH_FLAGS, "the App's launch script")
        refuse_given_twice(
            {**launched, **given},
            self._pass_through(),
            pass_through=self.variable_of("args"),
        )
        seqs, tokens = self.max_num_seqs, self.max_batched_tokens
        [batched] = [f for f in LLM_FLAGS if f.field == "max_batched_tokens"]
        if self._yields(batched):  # the pass-through gives the tokens
            tokens = 0
        if seqs and tokens and seqs > tokens:
            raise ValueError(
                f"{self.variable_of('max_num_seqs')}={seqs} exceeds "
                f"{self.variable_of('max_batched_tokens')}={tokens}: a step of "
                "that many tokens cannot hold one token of each sequence; lower "
                "the first or raise the second"
            )
        return self

    def _set_flags(self) -> list[OvmsFlag]:
        # Only a named flag that was set, even to OVMS's own value, is given
        # twice beside the pass-through; one left at its default yields to it.
        fields = self.model_fields_set
        return [f for f in (*SERVER_FLAGS, *LLM_FLAGS) if f.field in fields]

    def _yields(self, flag: OvmsFlag) -> bool:
        # Left at its default, and given by the pass-through: the pass-through
        # is the one statement of it (since ticket 6 the defaults write the
        # LLM's two batching flags, which nobody set).
        passed = self._pass_through()
        return flag.field not in self.model_fields_set and any(
            names_flag(word, flag.flag) for word in passed
        )

    def _words_of(self, flags: tuple[OvmsFlag, ...]) -> list[str]:
        return [
            word
            for f in flags
            if not self._yields(f)
            for word in flag_words(f, getattr(self, f.field))
        ]

    def _pass_through(self) -> list[str]:
        return split_pass_through(self.args, variable=self.variable_of("args"))

    def _refuse_a_hint_beside_pinned_streams(self) -> None:
        hint = self.openvino.chosen().get("PERFORMANCE_HINT")
        if not (hint and self.num_streams):
            return
        if "performance_hint" in self.openvino.model_fields_set:
            given = f"{_OPENVINO_PREFIX}PERFORMANCE_HINT={hint}"
        else:
            given = f"{_OPENVINO_PREFIX}PROPERTIES gives PERFORMANCE_HINT={hint}"
        raise ValueError(
            f"{given} is set beside {self.variable_of('num_streams')}="
            f"{self.num_streams}, and OVMS refuses a performance hint together "
            "with NUM_STREAMS: the hint sets the streams; unset one of the two"
        )

    def llm_flags(self) -> list[str]:
        """Return the ``ovms`` arguments of an LLM's options, without the server's.

        Returns:
            Each flag and its value, only for an option away from OVMS's own
            value and not left at its default beside ``args`` giving it, the
            two batching options first; ``[]`` at OVMS's own.

        Example:
            >>> OvmsSettings().llm_flags()  # each prompt read in one pass
            ['--dynamic_split_fuse', 'false', '--max_num_batched_tokens', '8192']
            >>> OvmsSettings(dynamic_split_fuse=True, max_batched_tokens=0).llm_flags()
            []
        """
        return self._words_of(LLM_FLAGS)

    def server_flags(self) -> list[str]:
        """Return the ``ovms`` server's arguments: the named ones, then ``args``.

        Every App passes them, and so does each start of a calibration, so a
        candidate is timed as it will be served.

        Returns:
            The words; ``[]`` at the defaults.

        Example:
            >>> OvmsSettings(rest_workers=4, args="--log_level DEBUG").server_flags()
            ['--rest_workers', '4', '--log_level', 'DEBUG']
        """
        return self._words_of(SERVER_FLAGS) + self._pass_through()

    def require_no_llm_option(self, component: str) -> None:
        """Refuse an LLM's option on an embedder's or a reranker's App.

        Args:
            component: The embedder or reranker the App serves.

        Raises:
            ValueError: If an option of ``ovms_flags.LLM_FLAGS`` was set,
                naming its variable.

        Example:
            >>> OvmsSettings().require_no_llm_option("reranker")
        """
        self._refuse_set(
            tuple(f.field for f in LLM_FLAGS),
            f"{component} is no LLM, and the option applies to an LLM served by "
            "OVMS only; unset it where it was set: this App's block of deploy.yaml "
            f"(apps.{component}.settings), its defaults.settings, or .env",
        )

    def require_no_stream_pin(self, component: str) -> None:
        """Refuse a pinned stream or thread count on an LLM's App.

        OVMS pulls and serves an LLM as it is: nothing is calibrated, and no
        stream or thread count would be applied.

        Args:
            component: The LLM the App serves.

        Raises:
            ValueError: If either count is pinned (above 0), naming its variable.

        Example:
            >>> OvmsSettings().require_no_stream_pin("llm")
        """
        pinned = [f for f in ("num_streams", "threads") if getattr(self, f)]
        self._refuse_set(
            tuple(pinned),
            f"{component} is an LLM, and OVMS serves it uncalibrated, so the count "
            "would be ignored; unset it in this App's block of deploy.yaml",
        )

    def require_no_embeddings_option(self, component: str) -> None:
        """Refuse an embeddings graph option on an App that serves no embedder.

        Args:
            component: The reranker the App serves.

        Raises:
            ValueError: If ``embeddings_truncate`` or ``embeddings_max_length``
                was set, naming its variable and value.

        Example:
            >>> OvmsSettings().require_no_embeddings_option("reranker")
        """
        self._refuse_set(
            ("embeddings_truncate", "embeddings_max_length"),
            f"{component} is no embedder, and the option applies to an "
            "embeddings graph only; unset it in this App's block of deploy.yaml "
            f"(apps.{component}.settings), its defaults.settings, or .env",
        )

    def require_no_graph_option(self, component: str) -> None:
        """Refuse a graph's option on an LLM's App, whose graph OVMS writes.

        Args:
            component: The LLM the App serves.

        Raises:
            ValueError: If the OpenVINO block, the graph-pool size or an
                embeddings option was set, naming each variable and value.

        Example:
            >>> OvmsSettings().require_no_graph_option("llm")
        """
        self._refuse_set(
            (
                "openvino",
                "graph_queue_max_size",
                "embeddings_truncate",
                "embeddings_max_length",
            ),
            f"{component} is an LLM, which OVMS pulls and writes that graph "
            "itself, so the option would be ignored; unset it in this App's "
            f"block of deploy.yaml (apps.{component}.settings), its "
            "defaults.settings, or .env",
        )

    def _refuse_set(self, fields: tuple[str, ...], reason: str) -> None:
        named = [
            written
            for field in fields
            if field in self.model_fields_set
            for written in self._as_written(field)
        ]
        if named:
            raise ValueError(f"{', '.join(named)} is set, but {reason}")

    def _as_written(self, field: str) -> list[str]:
        # The block is named by each of its variables that was set.
        if field == "openvino":
            block = self.openvino
            return [
                f"{_OPENVINO_PREFIX}{name.upper()}={spelled(getattr(block, name))}"
                for name in type(block).model_fields
                if name in block.model_fields_set
            ]
        return [f"{self.variable_of(field)}={spelled(getattr(self, field))}"]
