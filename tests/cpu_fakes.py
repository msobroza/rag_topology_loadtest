"""A named fake for ``CpuCounter``, shared by the CPU-count and ``RagPorts`` tests."""

from __future__ import annotations

from rag_load_test.adapters.cpu_count import CpuCounter


class FakeCpuCounter(CpuCounter):
    """Answers ``value`` (``None`` = cannot say) and counts how often it was asked."""

    def __init__(self, value: float | None, name: str = "fake") -> None:
        self.value, self.name, self.asked = value, name, 0

    async def cpus(self) -> float | None:
        self.asked += 1
        return self.value
