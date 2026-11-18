"""Give an in-process model its OpenVINO export on the first start that lacks it.

Under the ``openvino`` runtime a model loads from its folder under the models
directory (``QP_LOCAL_MODELS__OPENVINO_DIR``), one folder per weight format
(``QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT``). When that folder is missing, the
start downloads the model, converts it at that weight format and saves it there
(``export_embedder`` / ``export_reranker``), then loads the folder; every later
start loads the folder directly. Changing the weight format therefore exports
again on the next start. ``torch`` loads the id as before: the Hugging Face cache
keeps its download. ``qp-export-openvino`` still writes the same folders ahead of time.

Under ``openvino`` the packages are checked first, whether or not the folder
exists: a missing one is then ``OpenVinoUnavailableError``, naming it and the pip
extra, instead of the bare ``Exception`` sentence-transformers raises while
loading (seen by qp-calibrate in the sitting of 15 November 2026).

The export blocks for minutes, so it runs on the executor given, never on the
event loop. No lock guards the folder: each App provisions on its own disk, and
within one process ``ResourceScope`` opens each model once, so no two writers ever
share a folder. The export writes beside the folder and swaps it in at the end,
so a failed or interrupted run leaves nothing a later start would trust.

Example:
    >>> model = await provision_local_model(
    ...     "org/embedder",
    ...     runtime="openvino",
    ...     exports=OpenVinoExports("models/openvino", "fp32"),
    ...     export=export_embedder,
    ...     require_packages=require_openvino,
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
from ...domain.runtime.backends import LocalModelRuntime, OpenVinoWeightFormat
from ..json_log import log_json
from .local_model import LocalModel, OpenVinoExports, local_model
from .model_pool import run_blocking

# ``export_embedder`` or ``export_reranker``:
# (model id, target folder, weight format) -> folder.
Exporter = Callable[[str, Path, OpenVinoWeightFormat], Path]
# ``require_openvino``: returns when the runtime's packages import, else raises.
PackageCheck = Callable[[], None]


class ModelProvisioningError(RuntimeError):
    """A model's OpenVINO export could not be written where the runtime loads it."""


async def provision_local_model(
    model_id: str,
    *,
    runtime: LocalModelRuntime,
    exports: OpenVinoExports,
    export: Exporter,
    require_packages: PackageCheck,
    executor: Executor,
) -> LocalModel:
    """Describe ``model_id``, exporting it first when ``openvino`` lacks its folder.

    Args:
        model_id: The model's Hugging Face id or path.
        runtime: ``openvino`` or ``torch``; only ``openvino`` ever exports.
        exports: The folder holding the OpenVINO exports and the weight format:
            which folder loads, and what a missing one is exported at. ``torch``
            ignores it.
        export: Downloads, converts and writes one model to a folder.
        require_packages: Raises when the ``openvino`` runtime's packages are
            missing; called under ``openvino`` before anything loads.
        executor: Where the blocking export runs, off the event loop.

    Returns:
        The model as ``local_model`` describes it once any export is done.

    Raises:
        OpenVinoUnavailableError: Under ``openvino``, a package is missing (as
            ``require_packages`` raises it); nothing is exported.
        ModelProvisioningError: The export failed; names the model, the folder
            and the cause. No folder is left behind.

    Example:
        >>> model = await provision_local_model(
        ...     "org/reranker", runtime="torch",
        ...     exports=OpenVinoExports("models/openvino", "fp32"),
        ...     export=export_reranker, require_packages=require_openvino,
        ...     executor=pool,
        ... )
        >>> model.source
        'org/reranker'
    """
    if runtime == "openvino":
        require_packages()  # before the folder test: loading needs them too
        folder = exports.folder(model_id)
        if not folder.is_dir():
            await _export_into(
                folder, export, model_id, exports.weight_format, executor
            )
    return local_model(
        model_id,
        runtime=runtime,
        models_dir=exports.models_dir,
        weight_format=exports.weight_format,
    )


async def _export_into(
    folder: Path,
    export: Exporter,
    model_id: str,
    weight_format: OpenVinoWeightFormat,
    executor: Executor,
) -> None:
    try:
        await run_blocking(
            executor, _timed_export, export, model_id, folder, weight_format
        )
    except Exception as exc:
        raise ModelProvisioningError(
            f"could not export {model_id!r} to OpenVINO ({weight_format}) in "
            f"{folder}: "
            f"{describe_error(exc)} (a first start downloads it from the Hugging "
            "Face Hub, so it needs the network and no HF_HUB_OFFLINE=1; "
            "qp-export-openvino writes the folder ahead of time)"
        ) from exc


# Runs on the executor; one JSON line once the folder is in place.
def _timed_export(
    export: Exporter, model_id: str, folder: Path, weight_format: OpenVinoWeightFormat
) -> None:
    started = time.monotonic()
    export(model_id, folder, weight_format)
    seconds = round(time.monotonic() - started, 1)
    log_json(
        event="local_model_provisioned",
        model=model_id,
        folder=str(folder),
        weight_format=weight_format,
        seconds=seconds,
    )
