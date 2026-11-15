"""An in-process model: which files it loads from and what runs it.

``openvino`` loads the model's exported folder when there is one, else the
Hugging Face id (which sentence-transformers would convert in memory on every
load). ``LocalModelResources`` never lets it come to that: it exports a missing
folder first (``model_provisioning.py``). ``torch`` always loads the id.
Describing a model reads nothing but whether the folder exists.

The folder's name carries the weight format, so an export of another format is
never loaded by mistake and both can sit side by side: ``<model>`` for ``fp32``,
``<model>-int8`` for ``int8``.

Example:
    >>> model = local_model(
    ...     "org/name", runtime="openvino", models_dir="models/ov", weight_format="fp32"
    ... )
    >>> model.source  # "models/ov/name" once exported, else "org/name"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...domain.backends import LocalModelRuntime, OpenVinoWeightFormat

# Appended to a model's folder name, per weight format. fp32 adds nothing: every
# export made before the setting existed holds 32-bit weights and keeps its place.
_FOLDER_SUFFIXES: dict[OpenVinoWeightFormat, str] = {"fp32": "", "int8": "-int8"}


@dataclass(frozen=True)
class LocalModel:
    """One in-process model, as the sentence-transformers adapters load it."""

    name: str  # the id or path the settings name; what ``ready`` reports
    source: str  # what sentence-transformers loads: an exported folder, or the id
    runtime: LocalModelRuntime

    def load_arguments(self, threads: int) -> dict[str, Any]:
        """Return the keyword arguments that pick the runtime at load time.

        Args:
            threads: Threads one inference call may use. OpenVINO takes them
                here, at load; PyTorch is capped per call by a
                ``NativeThreadScope`` instead.

        Returns:
            Nothing for ``torch``; for ``openvino``, the backend, its inference
            thread count and leave to run them on hyper-threads.
        """
        if self.runtime == "torch":
            return {}
        # In its latency mode OpenVINO keeps to physical cores: asked for 2 threads
        # on one core with two hyper-threads, it reported INFERENCE_NUM_THREADS=1
        # and left the second CPU idle (GCP sitting of 2026-11-15). Allowed the
        # hyper-threads it ran 2, and a rerank request took 2.94 s instead of 3.38.
        ov_config = {
            "INFERENCE_NUM_THREADS": str(threads),
            "ENABLE_HYPER_THREADING": "YES",
        }
        return {"backend": "openvino", "model_kwargs": {"ov_config": ov_config}}


@dataclass(frozen=True)
class OpenVinoExports:
    """Where the OpenVINO exports are kept, and the weight format loaded and written.

    Example:
        >>> OpenVinoExports("models/openvino", "int8").folder("org/name")
        PosixPath('models/openvino/name-int8')
    """

    models_dir: str  # RAG_OPENVINO_MODELS_DIR
    weight_format: OpenVinoWeightFormat  # RAG_OPENVINO_WEIGHT_FORMAT

    def folder(self, model_id: str) -> Path:
        """Return where ``model_id``'s export of this weight format lives.

        Args:
            model_id: The model's Hugging Face id or path.

        Returns:
            The folder, as ``exported_model_dir`` names it.
        """
        return exported_model_dir(
            self.models_dir, model_id, weight_format=self.weight_format
        )


def exported_model_dir(
    models_dir: str, model_id: str, *, weight_format: OpenVinoWeightFormat
) -> Path:
    """Return where a model's OpenVINO export of ``weight_format`` lives.

    Args:
        models_dir: The folder holding every export (``RAG_OPENVINO_MODELS_DIR``).
        model_id: The model's Hugging Face id or path.
        weight_format: ``fp32`` or ``int8`` (``RAG_OPENVINO_WEIGHT_FORMAT``).

    Returns:
        ``models_dir`` joined with the last segment of ``model_id``, followed by
        ``-int8`` for ``int8``.

    Example:
        >>> exported_model_dir("models/openvino", "Snowflake/arctic-embed",
        ...                    weight_format="int8")
        PosixPath('models/openvino/arctic-embed-int8')
    """
    name = model_id.rstrip("/").rsplit("/", 1)[-1]
    return Path(models_dir) / f"{name}{_FOLDER_SUFFIXES[weight_format]}"


def local_model(
    model_id: str,
    *,
    runtime: LocalModelRuntime,
    models_dir: str,
    weight_format: OpenVinoWeightFormat,
) -> LocalModel:
    """Describe how ``model_id`` loads under ``runtime``.

    Args:
        model_id: The model's Hugging Face id or path.
        runtime: ``openvino`` or ``torch``.
        models_dir: The folder holding the OpenVINO exports.
        weight_format: Which export to load under ``openvino``; ``torch``
            ignores it.

    Returns:
        The model, loading from its exported folder when the runtime is
        ``openvino`` and the folder exists, else from ``model_id``.
    """
    exported = exported_model_dir(models_dir, model_id, weight_format=weight_format)
    use_export = runtime == "openvino" and exported.is_dir()
    return LocalModel(model_id, str(exported) if use_export else model_id, runtime)
