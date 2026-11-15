"""Convert an in-process model to OpenVINO IR, in the folder layout it loads from.

Loading a model with ``backend="openvino"`` and ``export=True`` converts it in
memory (even when the Hub repo ships IR of its own);
``save_pretrained`` then writes a self-contained folder: the IR under
``openvino/``, the tokenizer, the config and the sentence-transformers module
files. The ``openvino`` runtime of ``LocalModel`` loads that folder as is, with
no conversion and no network. The export is written next to the target and
swapped in at the end, so a failed run leaves the previous export untouched.

Example:
    >>> export_embedder("org/embedder", Path("models/openvino/embedder"))
    PosixPath('models/openvino/embedder')
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any

# What the openvino backend of sentence-transformers imports; checked up front so
# a missing package is reported as such, not as the library's bare Exception.
_OPENVINO_MODULES = ("openvino", "optimum.intel.openvino")
_INSTALL_HINT = "pip install 'sentence-transformers[openvino]'"


class OpenVinoUnavailableError(ImportError):
    """The OpenVINO packages the export needs are not installed."""


def export_embedder(model_id: str, target: Path) -> Path:
    """Convert a ``SentenceTransformer`` to OpenVINO and write it to ``target``.

    Args:
        model_id: The model's Hugging Face id or local path.
        target: The folder to write; an existing one is replaced.

    Returns:
        ``target``, now holding the export.

    Example:
        >>> export_embedder("org/embedder", Path("models/openvino/embedder"))
        PosixPath('models/openvino/embedder')
    """
    require_openvino()
    from sentence_transformers import SentenceTransformer  # pulls in torch

    return _export(SentenceTransformer, model_id, target)


def export_reranker(model_id: str, target: Path) -> Path:
    """Convert a ``CrossEncoder`` to OpenVINO and write it to ``target``.

    Args:
        model_id: The model's Hugging Face id or local path.
        target: The folder to write; an existing one is replaced.

    Returns:
        ``target``, now holding the export.

    Example:
        >>> export_reranker("org/reranker", Path("models/openvino/reranker"))
        PosixPath('models/openvino/reranker')
    """
    require_openvino()
    from sentence_transformers import CrossEncoder  # pulls in torch

    return _export(CrossEncoder, model_id, target)


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
                f"OpenVINO export needs module {exc.name!r} (for {name!r}), "
                f"which is not installed; {_INSTALL_HINT}"
            ) from exc


def _export(model_class: Any, model_id: str, target: Path) -> Path:
    # Written beside the target under a dotted name: local_model only looks for
    # the target itself, so a half-written export is never picked up.
    staging = target.with_name(f".{target.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # export=True converts even when the repo ships its own IR: the files
        # then come from the installed converter, never from a Hub variant.
        model = model_class(
            model_id, device="cpu", backend="openvino", model_kwargs={"export": True}
        )
        model.save_pretrained(str(staging))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return target
