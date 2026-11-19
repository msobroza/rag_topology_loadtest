"""An OVMS graph's ``plugin_config``: what it holds for a tuning, and rewriting it.

OVMS's exporter (``ovms/export_models.sh`` runs it) writes one ``graph.pbtxt``
per model, holding exactly one line such as::

    plugin_config: '{"NUM_STREAMS": "1" }',

OVMS reads it once, when it starts. A malformed JSON there makes OVMS log a
parsing error and silently run its defaults, so what is written is always
``json.dumps`` of string values. No ``PERFORMANCE_HINT`` is set: OVMS refuses
it together with ``NUM_STREAMS``. Pure functions: no file is read here.

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

from ...domain.runtime.calibration import StageTuning
from ..openvino_cpu_properties import openvino_cpu_properties

# The line's head up to the opening quote, the JSON, then the closing quote and
# whatever follows it (a comma), on one line.
_PLUGIN_CONFIG_LINE = re.compile(
    r"^(?P<head>[ \t]*plugin_config:[ \t]*')(?P<json>[^'\n]*)(?P<tail>'[^\n]*)$",
    re.MULTILINE,
)


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
        block's defaults ``ENABLE_HYPER_THREADING``), as strings.

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
    return {
        "NUM_STREAMS": str(tuning.streams),
        **openvino_cpu_properties(tuning.threads, chosen),
    }


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
    found = len(_PLUGIN_CONFIG_LINE.findall(graph_text))
    if found != 1:
        raise ValueError(
            f"{source}: expected exactly one plugin_config: '<json>' line, as "
            f"OVMS's exporter writes, found {found}"
        )
    written = json.dumps(dict(plugin_config))
    return _PLUGIN_CONFIG_LINE.sub(
        lambda line: line["head"] + written + line["tail"], graph_text
    )
