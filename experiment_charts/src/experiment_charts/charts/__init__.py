"""The charts, one ``<finding>_chart`` module each.

Each module implements ``BaseExperimentChart`` for one finding of
``experiments-explained/``: it selects the chart's numbers (``table``) and draws
them (``draw``) with what the modules at the package's root share.
``chart_catalog``, at the root, builds them with the experiment ids they read.
"""
