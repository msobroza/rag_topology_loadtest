"""The OpenVINO CPU plugin properties for a thread count: the one home of that rule.

Both runtimes of ours that run OpenVINO take them: an in-process model at load
(``local_models/local_model.py``) and OpenVINO Model Server in its graph's
``plugin_config`` (``ovms/ovms_graph.py``). The thread count is calibration's;
the other properties are the site's, chosen by an ``openvino`` settings block
(``config/openvino_cpu_settings.py``, whose ``hyper_threading`` comment holds
the measurement behind its one written default).

Example:
    >>> openvino_cpu_properties(2, {"ENABLE_HYPER_THREADING": "YES"})
    {'INFERENCE_NUM_THREADS': '2', 'ENABLE_HYPER_THREADING': 'YES'}
"""

from __future__ import annotations

from collections.abc import Mapping


def openvino_cpu_properties(threads: int, chosen: Mapping[str, str]) -> dict[str, str]:
    """Return the CPU plugin properties that give an inference ``threads`` threads.

    Args:
        threads: Threads the inference may use, all its streams together.
        chosen: The site's other properties, as an ``openvino`` block's
            ``chosen()`` returns them; it never holds the thread count.

    Returns:
        ``INFERENCE_NUM_THREADS`` first, then ``chosen``, as the strings
        OpenVINO takes: a fresh dict each call.

    Raises:
        ValueError: If ``threads`` is below 1.

    Example:
        >>> openvino_cpu_properties(4, {})["INFERENCE_NUM_THREADS"]
        '4'
    """
    if threads < 1:
        raise ValueError(f"OpenVINO needs threads >= 1, got {threads=}")
    return {"INFERENCE_NUM_THREADS": str(threads), **chosen}
