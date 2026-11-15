"""Where the repository is, for tests that read files outside ``tests/``."""

from __future__ import annotations

from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[2]  # query_processing/
REPO_ROOT = TOOLKIT_ROOT.parent  # README.md, emulation/
