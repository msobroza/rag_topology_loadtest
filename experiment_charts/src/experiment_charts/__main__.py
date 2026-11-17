"""``python -m experiment_charts``: redraw ``experiments-explained/``.

Example:
    >>> python -m experiment_charts --image-format png --out /tmp/previews
"""

from __future__ import annotations

import sys

from .write_experiment_charts import main

sys.exit(main(sys.argv[1:]))
