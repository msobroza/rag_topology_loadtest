"""Where the emulation's Apps run: the placement file Terraform writes.

Emulation only (Domino has no such file): each hardware tier is a pool of
machines, and the fake reaches them over SSH. ``rag-fake-domino --placement
FILE`` loads it; without one the fake only records Apps, as before. Keys beyond
the documented shape are refused, so a misspelt one is reported, not ignored;
the optional ones (timeouts, the unit name) have the defaults below.

Example:
    >>> load_placement(Path("placement.json")).default_tier.id
    'cpu-8'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter, ValidationError

_STRICT = ConfigDict(extra="forbid", strict=True)
_LOWEST_PORT, _HIGHEST_PORT = 1, 65535


@dataclass(frozen=True)
class SshSettings:
    """How the fake reaches a machine over SSH.

    Attributes:
        user: The account the fake logs in as, which also runs the App.
        key: Path of the private key, on the fake's machine.
        connect_timeout_seconds: ssh's ``ConnectTimeout``.
        command_timeout_seconds: How long one ``ssh`` call may take in all.

    Example:
        >>> SshSettings(user="ubuntu", key="/home/ubuntu/.ssh/id_emulation")
    """

    __pydantic_config__ = _STRICT

    user: str
    key: str
    connect_timeout_seconds: float = 5.0
    command_timeout_seconds: float = 20.0


@dataclass(frozen=True)
class HardwareTierPool:
    """One hardware tier: its size, and the machines of that size.

    Attributes:
        id: The tier's id, as an App version's ``hardwareTierId`` names it.
        cores: CPU cores of each machine.
        memory_gib: Memory of each machine, in GiB.
        machines: Each machine's address, as ``ssh`` and the proxy reach it.
        default: Whether a version without a ``hardwareTierId`` runs here.

    Example:
        >>> HardwareTierPool("cpu-8", 8, 32, ("10.10.0.21",), default=True)
    """

    __pydantic_config__ = _STRICT

    id: str
    cores: float
    memory_gib: float
    machines: tuple[str, ...]
    default: bool = False


@dataclass(frozen=True)
class PlacementSettings:
    """The machines Apps are placed on, and how an App runs there.

    Attributes:
        ssh: How to reach a machine.
        app_port: The port an App listens on, on its machine.
        tiers: The hardware tiers; exactly one is the default.
        unit: The systemd unit an App runs as on its machine.
        stop_timeout_seconds: How long stopping the previous App may take.
        probe_timeout_seconds: Timeout of the TCP connect to ``app_port``.
        api_proxy_url: ``DOMINO_API_PROXY`` as an App sees it: each machine
            forwards that port to the fake, as Domino's sidecar does.

    Raises:
        ValueError: If no tier or several are the default, a tier id or a
            machine appears twice, or ``app_port`` is not a TCP port.

    Example:
        >>> PlacementSettings(SshSettings("ubuntu", "/k"), 8888, (tier,))
    """

    __pydantic_config__ = _STRICT

    ssh: SshSettings
    app_port: int
    tiers: tuple[HardwareTierPool, ...]
    unit: str = "rag-app"
    stop_timeout_seconds: float = 10.0
    probe_timeout_seconds: float = 1.0
    api_proxy_url: str = "http://localhost:8899"

    def __post_init__(self) -> None:
        if not _LOWEST_PORT <= self.app_port <= _HIGHEST_PORT:
            raise ValueError(f"app_port: {self.app_port} is not a TCP port")
        defaults = [tier.id for tier in self.tiers if tier.default]
        if len(defaults) != 1:
            raise ValueError(
                f'tiers: exactly one needs "default": true, found {defaults}'
            )
        _refuse_repeats("tiers: tier id", [tier.id for tier in self.tiers])
        machines = [machine for tier in self.tiers for machine in tier.machines]
        _refuse_repeats("tiers: machine", machines)

    @property
    def default_tier(self) -> HardwareTierPool:
        """The tier a version without a ``hardwareTierId`` runs on."""
        return next(tier for tier in self.tiers if tier.default)

    def tier(self, tier_id: str | None) -> HardwareTierPool | None:
        """Find a tier by id; an empty or missing id is the default tier.

        Args:
            tier_id: A version's ``hardwareTierId``.

        Returns:
            The tier, or None if no tier has that id.

        Example:
            >>> settings.tier("cpu-8").cores
            8
        """
        if not tier_id:
            return self.default_tier
        return next((tier for tier in self.tiers if tier.id == tier_id), None)


def _refuse_repeats(what: str, values: list[str]) -> None:
    # A machine in two pools could be handed to two Apps at once.
    repeated = sorted({value for value in values if values.count(value) > 1})
    if repeated:
        raise ValueError(f"{what} listed more than once: {repeated}")


def load_placement(path: Path) -> PlacementSettings:
    """Read and validate the placement file Terraform writes.

    Args:
        path: The JSON file, e.g. ``/etc/fake-domino/placement.json``.

    Returns:
        The validated placement.

    Raises:
        ValueError: Naming ``path`` and the offending key, if the file cannot
            be read or does not have the documented shape.

    Example:
        >>> load_placement(Path("placement.json")).app_port
        8888
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"placement file {path}: cannot be read: {exc}") from exc
    try:
        return TypeAdapter(PlacementSettings).validate_json(text)
    except ValidationError as exc:
        raise ValueError(f"placement file {path}: {_first_problem(exc)}") from exc


def _first_problem(error: ValidationError) -> str:
    # "tiers.0.cores: Input should be a valid number", without pydantic's banner.
    problem = error.errors()[0]
    where = ".".join(str(part) for part in problem["loc"])
    message = str(problem["msg"]).removeprefix("Value error, ")
    return f"{where}: {message}" if where else message
