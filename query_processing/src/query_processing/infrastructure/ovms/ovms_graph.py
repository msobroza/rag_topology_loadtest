"""An OVMS graph rewritten: its ``plugin_config``, its pool, its embeddings options.

OVMS's exporter (``ovms/export_models.sh`` runs it) writes one ``graph.pbtxt``
per model. At OVMS 2026.4.1 its first line is the graph-pool directive and its
calculator options hold exactly one ``plugin_config`` line::

    # OVMS_GRAPH_QUEUE_MAX_SIZE: AUTO
    ...
      plugin_config: '{"NUM_STREAMS": "1" }',

OVMS reads it once, when it starts. A malformed JSON there makes OVMS log a
parsing error and silently run its defaults, so what is written is always
``json.dumps`` of string values. ``NUM_STREAMS`` is written unless the App
chose a ``PERFORMANCE_HINT``: OVMS refuses the two together, and the hint then
sets the streams. Each rewrite adds or replaces a line, never writes one twice,
and at the settings' defaults leaves the export's line as it is. Pure
functions: no file is read here.

Example:
    >>> config = ovms_plugin_config(
    ...     StageTuning(threads=2, batch_size=20, streams=2), ovms.openvino.chosen()
    ... )
    >>> graph_with_plugin_config(graph_text, config, source="m/graph.pbtxt")
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from ...domain.runtime.calibration import StageTuning
from ..openvino_cpu_properties import openvino_cpu_properties

# The line's head up to the opening quote, the JSON, then the closing quote and
# whatever follows it (a comma), on one line.
_PLUGIN_CONFIG_LINE = re.compile(
    r"^(?P<head>[ \t]*plugin_config:[ \t]*')(?P<json>[^'\n]*)(?P<tail>'[^\n]*)$",
    re.MULTILINE,
)
# The graph-pool directive, a comment OVMS reads (docs/mediapipe.md).
_QUEUE_SIZE_LINE = re.compile(
    r"^#[ \t]*OVMS_GRAPH_QUEUE_MAX_SIZE:[^\n]*$", re.MULTILINE
)
_QUEUE_SIZE_DIRECTIVE = "# OVMS_GRAPH_QUEUE_MAX_SIZE: "
# OpenVINO's property that, set, takes the stream count from OVMS's hands.
_PERFORMANCE_HINT = "PERFORMANCE_HINT"


@dataclass(frozen=True)
class OvmsGraphOptions:
    """The graph's options beside its ``plugin_config``, each "as exported" unset.

    Example:
        >>> OvmsGraphOptions(queue_max_size="4").embeddings_truncate is None
        True
    """

    queue_max_size: str = ""  # "" = the export's directive (AUTO), else its value
    embeddings_truncate: bool | None = None  # None = as exported
    embeddings_max_length: int = 0  # 0 = as exported


def performance_hinted(chosen: Mapping[str, str]) -> bool:
    """Say whether the App's properties hold a performance hint.

    Args:
        chosen: The App's properties (``OvmsSettings.openvino.chosen()``).

    Returns:
        True when ``PERFORMANCE_HINT`` is among them: OVMS then takes no
        ``NUM_STREAMS``, and the streams are not calibrated.

    Example:
        >>> performance_hinted({"PERFORMANCE_HINT": "THROUGHPUT"})
        True
    """
    return _PERFORMANCE_HINT in chosen


def ovms_plugin_config(
    tuning: StageTuning, chosen: Mapping[str, str]
) -> dict[str, str]:
    """Return the CPU plugin properties OVMS runs a model with under ``tuning``.

    Args:
        tuning: Its ``streams`` become ``NUM_STREAMS``, its ``threads`` the
            threads all the streams share.
        chosen: The App's other properties, as ``OvmsSettings.openvino``'s
            ``chosen()`` returns them (``QP_OVMS_OPENVINO__*``).

    Returns:
        ``NUM_STREAMS``, ``INFERENCE_NUM_THREADS``, then ``chosen`` (at the
        block's defaults ``ENABLE_HYPER_THREADING``), as strings. Under a
        ``PERFORMANCE_HINT`` in ``chosen``, no ``NUM_STREAMS``: OVMS refuses
        the two together.

    Raises:
        ValueError: If ``tuning.streams`` exceeds ``tuning.threads``: a stream
            needs a thread.

    Example:
        >>> config = ovms_plugin_config(
        ...     StageTuning(threads=4, batch_size=1, streams=2), {}
        ... )
        >>> config["NUM_STREAMS"], config["INFERENCE_NUM_THREADS"]
        ('2', '4')
    """
    if tuning.streams > tuning.threads:
        raise ValueError(
            f"OVMS needs streams <= threads (a stream needs a thread), got "
            f"streams={tuning.streams}, threads={tuning.threads}"
        )
    properties = openvino_cpu_properties(tuning.threads, chosen)
    if performance_hinted(chosen):
        return properties
    return {"NUM_STREAMS": str(tuning.streams), **properties}


def graph_with_plugin_config(
    graph_text: str, plugin_config: Mapping[str, str], *, source: str
) -> str:
    """Return ``graph_text`` with ``plugin_config`` in its one ``plugin_config`` line.

    Args:
        graph_text: The exporter's ``graph.pbtxt``.
        plugin_config: The properties to write, string values.
        source: Where ``graph_text`` came from, for the error.

    Returns:
        The same text, only the JSON between the line's single quotes replaced.

    Raises:
        ValueError: If the graph has no ``plugin_config`` line, or several; the
            message says how many and names ``source``.

    Example:
        >>> line = "  plugin_config: '{}',"
        >>> print(graph_with_plugin_config(line, {"NUM_STREAMS": "2"}, source="g"))
          plugin_config: '{"NUM_STREAMS": "2"}',
    """
    _require_one_plugin_config(graph_text, source)
    written = json.dumps(dict(plugin_config))
    return _PLUGIN_CONFIG_LINE.sub(
        lambda line: line["head"] + written + line["tail"], graph_text
    )


def graph_with_queue_size(graph_text: str, size: str, *, source: str) -> str:
    r"""Return ``graph_text`` with its graph-pool directive set to ``size``.

    Args:
        graph_text: The exporter's ``graph.pbtxt``.
        size: ``AUTO`` or a count of graphs; ``""`` leaves the graph as it is.
        source: Where ``graph_text`` came from, for the error.

    Returns:
        The same text, its ``# OVMS_GRAPH_QUEUE_MAX_SIZE:`` line replaced, or
        put first when an older export has none.

    Raises:
        ValueError: If the graph holds the directive more than once; the
            message says how many and names ``source``.

    Example:
        >>> graph_with_queue_size("node {}\n", "4", source="g")
        '# OVMS_GRAPH_QUEUE_MAX_SIZE: 4\nnode {}\n'
    """
    if not size:
        return graph_text
    found = len(_QUEUE_SIZE_LINE.findall(graph_text))
    if found > 1:
        raise ValueError(
            f"{source}: expected at most one {_QUEUE_SIZE_DIRECTIVE.strip()} line, "
            f"as OVMS's exporter writes, found {found}"
        )
    directive = _QUEUE_SIZE_DIRECTIVE + size
    if found:
        return _QUEUE_SIZE_LINE.sub(lambda _: directive, graph_text)
    return f"{directive}\n{graph_text}"


def graph_with_embeddings_options(
    graph_text: str, *, truncate: bool | None, max_length: int, source: str
) -> str:
    r"""Return an embeddings graph with its ``truncate`` and ``max_length`` set.

    Each option set is written once, right after the ``plugin_config`` line
    and indented as it is, the way the exporter's template writes it; a line
    the graph already holds is replaced.

    Args:
        graph_text: The exporter's embeddings ``graph.pbtxt``.
        truncate: True writes ``truncate: true,``; False removes the line;
            None leaves the graph's own.
        max_length: Above 0 writes ``max_length: N,``; 0 leaves the graph's own.
        source: Where ``graph_text`` came from, for the error.

    Returns:
        The rewritten text; ``graph_text`` itself when nothing is set.

    Raises:
        ValueError: If an option is set and the graph has not exactly one
            ``plugin_config`` line; the message names ``source``.

    Example:
        >>> graph_with_embeddings_options(
        ...     "  plugin_config: '{}',", truncate=True, max_length=0, source="g"
        ... )
        "  plugin_config: '{}',\n  truncate: true,"
    """
    changed = _embeddings_option_values(truncate, max_length)
    if not changed:
        return graph_text
    _require_one_plugin_config(graph_text, source)
    for key in changed:
        option_line = re.compile(rf"^[ \t]*{key}:[^\n]*\n?", re.MULTILINE)
        graph_text = option_line.sub("", graph_text)
    written = {key: value for key, value in changed.items() if value is not None}
    return _after_plugin_config(graph_text, written)


def rewritten_graph(
    graph_text: str,
    plugin_config: Mapping[str, str],
    options: OvmsGraphOptions,
    *,
    source: str,
) -> str:
    """Return the export's graph with every rewrite an App's settings ask.

    Args:
        graph_text: The exporter's ``graph.pbtxt``.
        plugin_config: The properties its ``plugin_config`` line holds.
        options: The pool and the embeddings options.
        source: Where ``graph_text`` came from, for the errors.

    Returns:
        The text OVMS serves; at ``OvmsGraphOptions()`` it differs from the
        export's by its ``plugin_config`` line only.

    Example:
        >>> rewritten_graph(text, {"NUM_STREAMS": "1"}, OvmsGraphOptions(), source="g")
    """
    text = graph_with_plugin_config(graph_text, plugin_config, source=source)
    text = graph_with_queue_size(text, options.queue_max_size, source=source)
    return graph_with_embeddings_options(
        text,
        truncate=options.embeddings_truncate,
        max_length=options.embeddings_max_length,
        source=source,
    )


def _require_one_plugin_config(graph_text: str, source: str) -> None:
    found = len(_PLUGIN_CONFIG_LINE.findall(graph_text))
    if found != 1:
        raise ValueError(
            f"{source}: expected exactly one plugin_config: '<json>' line, as "
            f"OVMS's exporter writes, found {found}"
        )


def _embeddings_option_values(
    truncate: bool | None, max_length: int
) -> dict[str, str | None]:
    # Only the options to change; None as a value removes its line.
    changed: dict[str, str | None] = {}
    if truncate is not None:
        changed["truncate"] = "true" if truncate else None
    if max_length:
        changed["max_length"] = str(max_length)
    return changed


def _after_plugin_config(graph_text: str, written: Mapping[str, str]) -> str:
    line = _PLUGIN_CONFIG_LINE.search(graph_text)
    if line is None or not written:
        return graph_text
    indent = line["head"][: len(line["head"]) - len(line["head"].lstrip())]
    added = "".join(f"\n{indent}{key}: {value}," for key, value in written.items())
    return graph_text[: line.end()] + added + graph_text[line.end() :]
