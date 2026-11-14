"""A named fake for ``BaseCpuCounter``, shared by the CPU-count and ``CompositionRoot`` tests."""

from __future__ import annotations

from rag_load_test.infrastructure.local_models.cpu_count import BaseCpuCounter


class FakeCpuCounter(BaseCpuCounter):
    """Answers ``value`` (``None`` = cannot say) and counts how often it was asked."""

    def __init__(self, value: float | None, name: str = "fake") -> None:
        self.value, self.name, self.asked = value, name, 0

    async def cpus(self) -> float | None:
        self.asked += 1
        return self.value
