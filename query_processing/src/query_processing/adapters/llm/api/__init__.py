"""The ``api`` LLM: an endpoint in a generation format.

Importing the package imports every module in it, so each generation API format
and each structured output method registers itself, and a new one is one new
module.
"""

from ....domain.registry import import_package_modules

import_package_modules(__name__, __path__)
