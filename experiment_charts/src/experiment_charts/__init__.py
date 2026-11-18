"""Charts of the experiment registry: not the product, a reader's aid.

``registry_reading`` reads ``experiments/`` (every sitting's ``results.csv`` and
``experiments*.yaml``), each ``charts/*_chart`` module selects the numbers of one chart
(``table``) and draws them (``draw``), ``chart_catalog`` lists the thirteen charts with
the experiment ids they read, and ``write_experiment_charts`` writes their figures
and tables into ``experiments-explained/`` (``python -m experiment_charts``). It
imports nothing from ``query_processing``, ``fake_domino`` or a use case, a rule
checked by its own ``tests/test_experiment_charts_imports.py``.
"""
