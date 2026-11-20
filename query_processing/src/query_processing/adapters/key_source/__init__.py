"""The key source: its port, and one directory per source an API key comes from.

Importing the package imports every module under it, so each source registers
itself and a new source is one new directory.
"""

from ...domain.registry import import_package_tree

import_package_tree(__name__)
