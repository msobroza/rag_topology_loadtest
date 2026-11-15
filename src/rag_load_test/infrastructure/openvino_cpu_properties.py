"""The OpenVINO CPU plugin properties for a thread count: the one home of that rule.

Both runtimes of ours that run OpenVINO take them: an in-process model at load
(``local_models/local_model.py``) and OpenVINO Model Server in its graph's
``plugin_config`` (``ovms/ovms_graph.py``).

Example:
    >>> openvino_cpu_properties(2)
    {'INFERENCE_NUM_THREADS': '2', 'ENABLE_HYPER_THREADING': 'YES'}
"""

from __future__ import annotations


def openvino_cpu_properties(threads: int) -> dict[str, str]:
    """Return the CPU plugin properties that give an inference ``threads`` threads.

    In its latency mode OpenVINO keeps to physical cores: asked for 2 threads on
    one core with two hyper-threads, it reported INFERENCE_NUM_THREADS=1 and left
    the second CPU idle (GCP sitting of 2026-11-15). Allowed the hyper-threads it
    ran 2, and a rerank request took 2.94 s instead of 3.38. So the hyper-threads
    are always allowed.

    Args:
        threads: Threads the inference may use, all its streams together.

    Returns:
        ``INFERENCE_NUM_THREADS`` and ``ENABLE_HYPER_THREADING``, as the strings
        OpenVINO takes.

    Raises:
        ValueError: If ``threads`` is below 1.

    Example:
        >>> openvino_cpu_properties(4)["INFERENCE_NUM_THREADS"]
        '4'
    """
    if threads < 1:
        raise ValueError(f"OpenVINO needs threads >= 1, got {threads=}")
    return {"INFERENCE_NUM_THREADS": str(threads), "ENABLE_HYPER_THREADING": "YES"}
