"""The model-server kinds, one module each; importing the package registers every one.

Each kind is also one launch script, ``domino/model_servers/<name>.sh``.
"""

from ...domain.registry import import_package_modules

import_package_modules(__name__, __path__)
