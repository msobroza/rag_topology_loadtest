"""The folder OVMS serves one model from: the export, with a graph of our own.

The export (``ovms/export_models.sh``) may live on a Dataset that several Apps
and replicas share, possibly read-only, so nothing is ever written into it. The
serving folder is the App's own and holds:

- a copy of the export's config file (``config_reranker.json`` or
  ``config_embedder.json``), whose ``base_path`` names ``<alias>`` relative to
  the config file, so it resolves here;
- ``<alias>/``, a symlink to every entry of the export's ``<alias>/`` but
  ``graph.pbtxt``, which is written with the plugin config and the graph
  options given (``ovms_graph.py``). The graph's
  ``models_path: "./"`` is relative to the graph's folder, so it finds the
  linked model files.

Building it again replaces what a previous build wrote.

The export also says what its weights are stored as: ``export_models.sh`` writes
the folder's ``weight-format`` record last (``recorded_weight_format``).

Example:
    >>> config = write_ovms_serving_folder(
    ...     OvmsExport(Path("/mnt/data/ovms-models"), "config_reranker.json", "rr"),
    ...     Path("/tmp/ovms-serving"),
    ...     {"NUM_STREAMS": "2", "INFERENCE_NUM_THREADS": "2"},
    ...     OvmsGraphOptions(),
    ... )
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from .ovms_graph import OvmsGraphOptions, rewritten_graph

GRAPH_FILE = "graph.pbtxt"  # the name OVMS's exporter gives every graph
WEIGHT_FORMAT_FILE = "weight-format"  # one line, written by ovms/export_models.sh
# What a folder without the record holds: it was exported before the record
# existed, when ovms/export_models.sh exported 8-bit weights only.
UNRECORDED_WEIGHT_FORMAT = "int8"


class OvmsExport(NamedTuple):
    """One model as OVMS's exporter wrote it.

    Attributes:
        models_dir: The folder the exporter wrote into.
        config_file: The model's config file in it, such as ``config_reranker.json``.
        alias: The model's folder in it, the name clients send.
    """

    models_dir: Path
    config_file: str
    alias: str


def write_ovms_serving_folder(
    export: OvmsExport,
    serving_dir: Path,
    plugin_config: Mapping[str, str],
    graph_options: OvmsGraphOptions,
) -> Path:
    """Build in ``serving_dir`` what OVMS serves ``export`` from, its graph rewritten.

    Args:
        export: Where the exporter wrote the model, read only.
        serving_dir: A folder of the App's own; created when missing.
        plugin_config: The properties the graph's ``plugin_config`` holds.
        graph_options: The graph's pool and embeddings options; at their
            defaults the export's lines are kept.

    Returns:
        The config file to start OVMS with (``--config_path``).

    Raises:
        FileNotFoundError: If the export's graph or config file is missing; the
            message names the path and ``ovms/export_models.sh``.
        ValueError: If the export's graph has not exactly one ``plugin_config``
            line, or holds the graph-pool directive twice.

    Example:
        >>> write_ovms_serving_folder(
        ...     export, Path("/tmp/s"), {"NUM_STREAMS": "1"}, OvmsGraphOptions()
        ... )
        PosixPath('/tmp/s/config_reranker.json')
    """
    source = export.models_dir / export.alias
    graph = _required(source / GRAPH_FILE)
    config = _required(export.models_dir / export.config_file)
    model_dir = serving_dir / export.alias
    model_dir.mkdir(parents=True, exist_ok=True)
    _link_model_files(source, model_dir)
    text = rewritten_graph(
        graph.read_text(encoding="utf-8"),
        plugin_config,
        graph_options,
        source=str(graph),
    )
    (model_dir / GRAPH_FILE).write_text(text, encoding="utf-8")
    served_config = serving_dir / export.config_file
    shutil.copyfile(config, served_config)
    return served_config


def recorded_weight_format(models_dir: Path) -> str:
    """Return the weight format the export in ``models_dir`` holds.

    Args:
        models_dir: The folder OVMS's exporter wrote into.

    Returns:
        What its ``weight-format`` record says; ``int8`` for a folder without
        the record.

    Example:
        >>> recorded_weight_format(Path("/mnt/data/ovms-models"))
        'fp32'
    """
    record = models_dir / WEIGHT_FORMAT_FILE
    if not record.is_file():
        return UNRECORDED_WEIGHT_FORMAT
    return record.read_text(encoding="utf-8").strip()


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"no OVMS export at {path}: export the models with "
            "ovms/export_models.sh into that folder first"
        )
    return path


def _link_model_files(source: Path, model_dir: Path) -> None:
    # What a previous build linked goes first, so a file the export dropped
    # does not stay; the graph is ours and is rewritten by the caller.
    for entry in model_dir.iterdir():
        if entry.is_symlink():
            entry.unlink()
    for entry in source.iterdir():
        if entry.name != GRAPH_FILE:
            (model_dir / entry.name).symlink_to(entry.absolute())
