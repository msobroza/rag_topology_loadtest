"""The ``api`` embedder: an endpoint in an embeddings format.

Importing the package imports every module in it, so each embeddings API format
registers itself and a new format is one new module.
"""

from ....domain.registry import import_package_modules

import_package_modules(__name__, __path__)
