"""The ``api`` reranker: an endpoint in a rerank format.

Importing the package imports every module in it, so each rerank API format
registers itself and a new format is one new module.
"""

from ....domain.registry import import_package_modules

import_package_modules(__name__, __path__)
