"""An in-process model: which files it loads from and what runs it.

``openvino`` loads the model's exported folder when there is one, else the
Hugging Face id (which sentence-transformers would convert in memory on every
load). ``RagPorts`` never lets it come to that: it exports a missing folder first
(``model_provisioning.py``). ``torch`` always loads the id. Describing a model
reads nothing but whether the folder exists.

Example:
    >>> model = local_model("org/name", runtime="openvino", models_dir="models/ov")
    >>> model.source  # "models/ov/name" once exported, else "org/name"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...domain.topology import LocalModelRuntime


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
            Nothing for ``torch``; for ``openvino``, the backend and its
            inference thread count.
        """
        if self.runtime == "torch":
            return {}
        ov_config = {"INFERENCE_NUM_THREADS": str(threads)}
        return {"backend": "openvino", "model_kwargs": {"ov_config": ov_config}}


def exported_model_dir(models_dir: str, model_id: str) -> Path:
    """Return where a model's OpenVINO export lives.

    Args:
        models_dir: The folder holding every export (``RAG_OPENVINO_MODELS_DIR``).
        model_id: The model's Hugging Face id or path.

    Returns:
        ``models_dir`` joined with the last segment of ``model_id``.

    Example:
        >>> exported_model_dir("models/openvino", "Snowflake/arctic-embed")
        PosixPath('models/openvino/arctic-embed')
    """
    return Path(models_dir) / model_id.rstrip("/").rsplit("/", 1)[-1]


def local_model(
    model_id: str, *, runtime: LocalModelRuntime, models_dir: str
) -> LocalModel:
    """Describe how ``model_id`` loads under ``runtime``.

    Args:
        model_id: The model's Hugging Face id or path.
        runtime: ``openvino`` or ``torch``.
        models_dir: The folder holding the OpenVINO exports.

    Returns:
        The model, loading from its exported folder when the runtime is
        ``openvino`` and the folder exists, else from ``model_id``.
    """
    exported = exported_model_dir(models_dir, model_id)
    use_export = runtime == "openvino" and exported.is_dir()
    return LocalModel(model_id, str(exported) if use_export else model_id, runtime)
