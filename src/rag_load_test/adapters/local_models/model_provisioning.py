"""Give an in-process model its OpenVINO export on the first start that lacks it.

Under the ``openvino`` runtime a model loads from its folder under the models
directory (``RAG_OPENVINO_MODELS_DIR``). When that folder is missing, the start
downloads the model, converts it and saves it there (``export_embedder`` /
``export_reranker``), then loads the folder; every later start loads the folder
directly. ``torch`` loads the id as before: the Hugging Face cache keeps its
download. ``rag-export-openvino`` still writes the same folders ahead of time.

The export blocks for minutes, so it runs on the executor given, never on the
event loop. No lock guards the folder: each App provisions on its own disk, and
within one process ``RagPorts`` opens each model once, so no two writers ever
share a folder. The export writes beside the folder and swaps it in at the end,
so a failed or interrupted run leaves nothing a later start would trust.

Example:
    >>> model = await provision_local_model(
    ...     "org/embedder",
    ...     runtime="openvino",
    ...     models_dir="models/openvino",
    ...     export=export_embedder,
    ...     executor=pool,
    ... )
    >>> model.source  # "models/openvino/embedder", exported now or earlier
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Executor
from pathlib import Path

from ...domain.errors import describe_error
from ...domain.topology import LocalModelRuntime
from ..json_log import log_json
from .local_model import LocalModel, exported_model_dir, local_model
from .sentence_transformer_models import run_blocking

# ``export_embedder`` or ``export_reranker``: (model id, target folder) -> folder.
Exporter = Callable[[str, Path], Path]


class ModelProvisioningError(RuntimeError):
    """A model's OpenVINO export could not be written where the runtime loads it."""


async def provision_local_model(
    model_id: str,
    *,
    runtime: LocalModelRuntime,
    models_dir: str,
    export: Exporter,
    executor: Executor,
) -> LocalModel:
    """Describe ``model_id``, exporting it first when ``openvino`` lacks its folder.

    Args:
        model_id: The model's Hugging Face id or path.
        runtime: ``openvino`` or ``torch``; only ``openvino`` ever exports.
        models_dir: The folder holding the OpenVINO exports.
        export: Downloads, converts and writes one model to a folder.
        executor: Where the blocking export runs, off the event loop.

    Returns:
        The model as ``local_model`` describes it once any export is done.

    Raises:
        ModelProvisioningError: The export failed; names the model, the folder
            and the cause. No folder is left behind.
    """
    folder = exported_model_dir(models_dir, model_id)
    if runtime == "openvino" and not folder.is_dir():
        try:
            await run_blocking(executor, _timed_export, export, model_id, folder)
        except Exception as exc:
            raise ModelProvisioningError(
                f"could not export {model_id!r} to OpenVINO in {folder}: "
                f"{describe_error(exc)} (a first start downloads it from the Hugging "
                "Face Hub, so it needs the network and no HF_HUB_OFFLINE=1; "
                "rag-export-openvino writes the folder ahead of time)"
            ) from exc
    return local_model(model_id, runtime=runtime, models_dir=models_dir)


# Runs on the executor; one JSON line once the folder is in place.
def _timed_export(export: Exporter, model_id: str, folder: Path) -> None:
    started = time.monotonic()
    export(model_id, folder)
    seconds = round(time.monotonic() - started, 1)
    log_json(
        event="local_model_provisioned",
        model=model_id,
        folder=str(folder),
        seconds=seconds,
    )
