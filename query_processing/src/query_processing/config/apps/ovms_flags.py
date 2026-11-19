"""The ``ovms`` flags an OVMS App's named settings give, and the ones it may not.

``OvmsSettings`` (``ovms_settings.py``) reads the variables; this module says
which flag each named setting becomes. A flag is written only away from OVMS's
own value, so the defaults leave the command line a machine has run unchanged.
The server flags apply to every App; the LLM's to an LLM's App alone, which
passes them to ``ovms --pull``.

Example:
    >>> flag_words(SERVER_FLAGS[0], 4)
    ['--rest_workers', '4']
    >>> flag_words(SERVER_FLAGS[0], 0)  # OVMS's own: left out
    []
"""

from __future__ import annotations

import json
from typing import NamedTuple


class OvmsFlag(NamedTuple):
    """One named setting's ``ovms`` flag.

    Example:
        >>> OvmsFlag("rest_workers", "--rest_workers", 0).flag
        '--rest_workers'
    """

    field: str  # the OvmsSettings field, QP_OVMS_<FIELD>
    flag: str  # the ovms flag, in OVMS's spelling
    ovms_own: object  # OVMS's own value, at which the flag is left out
    switch: bool = False  # written alone when true, as --metrics_enable is


# The flags of the ovms server, on an App of any component type. OVMS's own
# values are its docs' (docs/parameters.md, 2026.4.1): REST workers from the
# CPUs, no metrics, log level INFO, no model cache.
SERVER_FLAGS = (
    OvmsFlag("rest_workers", "--rest_workers", 0),
    OvmsFlag("metrics", "--metrics_enable", False, switch=True),
    OvmsFlag("log_level", "--log_level", ""),
    OvmsFlag("cache_dir", "--cache_dir", ""),
)
# The flags of an LLM that ovms pulls, today's two first. OVMS's own values:
# split fuse on, 256 sequences, a dynamic KV cache, prefix caching as pull mode
# sets it, no KV cache precision, no distribution policy; 0 batched tokens
# leaves OVMS to pick them.
LLM_FLAGS = (
    OvmsFlag("dynamic_split_fuse", "--dynamic_split_fuse", True),
    OvmsFlag("max_batched_tokens", "--max_num_batched_tokens", 0),
    OvmsFlag("max_num_seqs", "--max_num_seqs", 0),
    OvmsFlag("cache_size_gb", "--cache_size", 0),
    OvmsFlag("prefix_caching", "--enable_prefix_caching", None),
    OvmsFlag("kv_cache_precision", "--kv_cache_precision", ""),
    OvmsFlag("model_distribution_policy", "--model_distribution_policy", ""),
)
# Written by domino/model_servers/ovms.sh, and by calibration's own starts
# (OvmsProcess): a pass-through giving one again would start ovms on two values.
LAUNCH_FLAGS = (
    "--rest_port",
    "--rest_bind_address",
    "--config_path",
    "--source_model",
    "--model_name",
    "--model_repository_path",
    "--task",
    "--target_device",
)


def spelled(value: object) -> str:
    """Write a setting's value as its variable and its flag spell it.

    Args:
        value: A field's value.

    Returns:
        ``"false"`` for False (not ``"False"``), JSON for a mapping, else ``str``.

    Example:
        >>> spelled(False), spelled({"A": "1"}), spelled(8)
        ('false', '{"A": "1"}', '8')
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def flag_words(flag: OvmsFlag, value: object) -> list[str]:
    """Return the ``ovms`` words one named setting gives.

    Args:
        flag: The setting's flag.
        value: The setting's value.

    Returns:
        ``[]`` at OVMS's own value; the flag alone for a switch; else the flag
        and its value.

    Example:
        >>> flag_words(OvmsFlag("metrics", "--metrics_enable", False, True), True)
        ['--metrics_enable']
    """
    # The types too: False == 0, and None is prefix caching left to OVMS.
    if type(value) is type(flag.ovms_own) and value == flag.ovms_own:
        return []
    if flag.switch:
        return [flag.flag]
    return [flag.flag, spelled(value)]
