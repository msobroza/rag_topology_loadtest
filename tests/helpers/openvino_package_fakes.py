"""A named fake for the check that the OpenVINO packages are installed.

``provision_local_model`` takes the check as an argument (``require_openvino`` in
the App), so its tests never import the real ``openvino`` or ``optimum-intel``.

Example::

    check = FakeOpenVinoPackages(missing="openvino")
    check()  # raises OpenVinoUnavailableError naming "openvino"
"""

from __future__ import annotations

from rag_load_test.infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
)


class FakeOpenVinoPackages:
    """Stands in for ``require_openvino``: counts its calls, fails if told to.

    Args:
        missing: The module to report as not installed; ``None`` passes.
    """

    def __init__(self, missing: str | None = None) -> None:
        self.missing = missing
        self.checks = 0

    def __call__(self) -> None:
        self.checks += 1
        if self.missing is not None:
            raise OpenVinoUnavailableError(
                f"the openvino runtime needs module {self.missing!r}, "
                "which is not installed (fake)"
            )
