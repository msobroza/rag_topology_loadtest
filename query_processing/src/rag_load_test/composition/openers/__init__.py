"""What opens each component on each backend: one registered opener per module.

Importing the package imports every module in it, so each opener registers
itself and a new backend is one new module.
"""

from ...domain.registry import import_package_modules

import_package_modules(__name__, __path__)
