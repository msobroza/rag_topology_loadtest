"""The stage types, one module each; importing the package registers every one."""

from ...domain.registry import import_package_modules

import_package_modules(__name__, __path__)
