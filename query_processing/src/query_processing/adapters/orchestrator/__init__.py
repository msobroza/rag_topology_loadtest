"""The orchestrator: its port, and one directory per library that runs the graph.

Importing the package imports every module under it, so each orchestrator
registers itself and a new library is one new directory.
"""

from ...domain.registry import import_package_tree

import_package_tree(__name__)
