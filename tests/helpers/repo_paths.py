"""Where the repository is, for tests that read files outside ``tests/``."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
