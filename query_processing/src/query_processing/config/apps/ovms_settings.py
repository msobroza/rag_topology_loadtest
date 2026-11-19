"""An OVMS model server App's own settings: streams, threads, start, CPU, batching.

OpenVINO Model Server serves one component per App, on that App's hardware, so
its stream and thread counts are the App's, not the workflow's: they are no
value of the workflow file. They are flat ``QP_OVMS_*`` variables, beside the
ones the launch script reads (``QP_OVMS_MODELS_DIR``, ...), set in the App's own
block of ``deploy.yaml`` (``apps.<component>.settings``) or in ``.env``.
``qp-ovms-config`` reads them when an OVMS App starts, and ``qp-calibrate
--ovms-models`` prints the two that pin a calibration. The streams and threads,
the OpenVINO block and the graph's pool size are an embedder's or a reranker's,
the two embeddings options an embedder's; the two batching options are an
LLM's, which OVMS pulls and serves itself, uncalibrated, on a graph of its own.
An App refuses the other kinds'.

Example:
    >>> OvmsSettings().num_streams  # 0 = calibrated when the App starts
    0
    >>> OvmsSettings.variable_of("threads")
    'QP_OVMS_THREADS'
"""

from __future__ import annotations

import json

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..openvino_cpu_settings import OpenVinoCpuSettings

ENV_PREFIX = "QP_OVMS_"
# The OpenVINO block's variables: QP_OVMS_OPENVINO__HYPER_THREADING, ...
_OPENVINO_PREFIX = f"{ENV_PREFIX}OPENVINO__"


def _spelled(value: object) -> str:
    # As the variable is written: a bool's value is "false", not "False", and a
    # mapping is JSON.
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


class OvmsSettings(BaseSettings):
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
    # Each flag is passed only away from OVMS's default (split fuse on; 0 = OVMS
    # picks the tokens per step), so the default command line is the one run on
    # a machine.
    dynamic_split_fuse: bool = True
    max_batched_tokens: int = Field(default=0, ge=0)

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
        """Return the ``ovms`` arguments of an LLM's two batching options.

        Returns:
            Each flag and its value, only for an option away from OVMS's own
            default; ``[]`` at both defaults.

        Example:
            >>> OvmsSettings(dynamic_split_fuse=False).llm_flags()
            ['--dynamic_split_fuse', 'false']
        """
        flags = []
        if not self.dynamic_split_fuse:
            flags += ["--dynamic_split_fuse", "false"]
        if self.max_batched_tokens:
            flags += ["--max_num_batched_tokens", str(self.max_batched_tokens)]
        return flags

    def require_no_llm_option(self, component: str) -> None:
        """Refuse an LLM's batching option on an embedder's or a reranker's App.

        Args:
            component: The embedder or reranker the App serves.

        Raises:
            ValueError: If either option was set, naming its variable.

        Example:
            >>> OvmsSettings().require_no_llm_option("reranker")
        """
        self._refuse_set(
            ("dynamic_split_fuse", "max_batched_tokens"),
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
                f"{_OPENVINO_PREFIX}{name.upper()}={_spelled(getattr(block, name))}"
                for name in type(block).model_fields
                if name in block.model_fields_set
            ]
        return [f"{self.variable_of(field)}={_spelled(getattr(self, field))}"]

    @classmethod
    def variable_of(cls, field: str) -> str:
        """Return the environment variable a field is read from.

        Args:
            field: The field name.

        Returns:
            ``QP_OVMS_<FIELD>``.

        Raises:
            ValueError: If ``field`` is not a field of these settings.

        Example:
            >>> OvmsSettings.variable_of("num_streams")
            'QP_OVMS_NUM_STREAMS'
        """
        if field not in cls.model_fields:
            raise ValueError(
                f"OvmsSettings has no field {field!r}; expected one of: "
                f"{', '.join(map(repr, cls.model_fields))}"
            )
        return ENV_PREFIX + field.upper()
