"""Convert an in-process model to OpenVINO IR, in the folder layout it loads from.

Loading a model with ``backend="openvino"`` and ``export=True`` converts it in
memory (even when the Hub repo ships IR of its own);
``save_pretrained`` then writes a self-contained folder: the IR under
``openvino/``, the tokenizer, the config and the sentence-transformers module
files. The ``openvino`` runtime of ``LocalModel`` loads that folder as is, with
no conversion and no network. The export is written next to the target and
swapped in at the end, so a failed run leaves the previous export untouched.

The weight format (``QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT``) picks what the
weights are stored as: ``fp32`` keeps 32-bit floats, ``int8`` compresses the
weights to 8-bit integers after converting, as ``optimum-cli export openvino
--weight-format int8`` does (weights only; activations stay floats).

Example:
    >>> export_embedder("org/embedder", Path("models/openvino/embedder"), "fp32")
    PosixPath('models/openvino/embedder')
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any

from ...domain.backends import OpenVinoWeightFormat

# What the openvino backend of sentence-transformers imports; checked up front so
# a missing package is reported as such, not as the library's bare Exception.
_OPENVINO_MODULES = ("openvino", "optimum.intel.openvino")
_INSTALL_HINT = "pip install 'sentence-transformers[openvino]'"


class OpenVinoUnavailableError(ImportError):
    """The OpenVINO packages the ``openvino`` runtime needs are not installed."""


def export_embedder(
    model_id: str, target: Path, weight_format: OpenVinoWeightFormat
) -> Path:
    """Convert a ``SentenceTransformer`` to OpenVINO and write it to ``target``.

    Args:
        model_id: The model's Hugging Face id or local path.
        target: The folder to write; an existing one is replaced.
        weight_format: ``fp32`` or ``int8``, what the weights are stored as.

    Returns:
        ``target``, now holding the export.

    Example:
        >>> export_embedder("org/embedder", Path("models/openvino/embedder"), "int8")
        PosixPath('models/openvino/embedder')
    """
    require_openvino()
    from sentence_transformers import SentenceTransformer  # pulls in torch

    return _export(SentenceTransformer, model_id, target, weight_format)


def export_reranker(
    model_id: str, target: Path, weight_format: OpenVinoWeightFormat
) -> Path:
    """Convert a ``CrossEncoder`` to OpenVINO and write it to ``target``.

    Args:
        model_id: The model's Hugging Face id or local path.
        target: The folder to write; an existing one is replaced.
        weight_format: ``fp32`` or ``int8``, what the weights are stored as.

    Returns:
        ``target``, now holding the export.

    Example:
        >>> export_reranker("org/reranker", Path("models/openvino/reranker"), "int8")
        PosixPath('models/openvino/reranker')
    """
    require_openvino()
    from sentence_transformers import CrossEncoder  # pulls in torch

    return _export(CrossEncoder, model_id, target, weight_format)


def require_openvino() -> None:
    """Import the OpenVINO packages, or say which pip extra installs them.

    Raises:
        OpenVinoUnavailableError: A module is missing; names it and the extra.

    Example:
        >>> require_openvino()  # silent when openvino and optimum-intel import
    """
    for name in _OPENVINO_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            raise OpenVinoUnavailableError(
                f"the openvino runtime needs module {exc.name!r} (for {name!r}) "
                f"to export or load a model, and it is not installed; {_INSTALL_HINT}"
            ) from exc


def _export(
    model_class: Any, model_id: str, target: Path, weight_format: OpenVinoWeightFormat
) -> Path:
    # Written beside the target under a dotted name: local_model only looks for
    # the target itself, so a half-written export is never picked up.
    staging = target.with_name(f".{target.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = model_class(
            model_id,
            device="cpu",
            backend="openvino",
            model_kwargs=_export_arguments(weight_format),
        )
        model.save_pretrained(str(staging))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return target


def _export_arguments(weight_format: OpenVinoWeightFormat) -> dict[str, bool]:
    """Return the ``model_kwargs`` that make the load convert, at ``weight_format``.

    Args:
        weight_format: ``fp32`` or ``int8``.

    Returns:
        ``export`` and ``load_in_8bit``, both always set.

    Example:
        >>> _export_arguments("int8")
        {'export': True, 'load_in_8bit': True}
    """
    # export=True converts even when the repo ships its own IR: the files then
    # come from the installed converter, never from a Hub variant. load_in_8bit
    # is never left out: without it optimum-intel (2.2.0) picks by model size and
    # compresses large models to 8 bits, so the folder would not hold the format
    # its name says. True applies 8-bit weight compression after converting.
    return {"export": True, "load_in_8bit": weight_format == "int8"}
