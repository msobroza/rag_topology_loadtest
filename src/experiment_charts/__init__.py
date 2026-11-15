"""Charts of the experiment registry: not the product, a reader's aid.

``registry_reading`` reads ``experiments/`` (every sitting's ``results.csv`` and
``experiments*.yaml``), each ``*_chart`` module selects the numbers of one chart
(``table``) and draws them (``draw``), ``chart_catalog`` lists the ten charts with
the experiment ids they read, and ``write_experiment_charts`` writes their figures
and tables into ``experiments-explained/`` (``python -m experiment_charts``). It
imports nothing from ``rag_load_test`` or ``fake_domino``, a rule checked by
``tests/structure/test_layering.py``.
"""
