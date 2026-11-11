"""A stand-in for Domino 6.2's Apps API: no product code lives here.

``api_shapes`` holds the request/response shapes transcribed from the 6.2
reference, ``state`` the fake's Apps and scripted behaviours, ``instances`` the
seam that decides what an instance reports, ``apps_routes`` and ``stand_ins``
what it serves of the API and of a run. ``server`` builds it
(``create_fake_domino``, ``rag-fake-domino``); an extension may build on it
through ``create_fake_domino``'s keyword arguments and the ``instances`` seam,
and nothing here depends on one. It imports nothing from ``rag_load_test``;
inside the product only ``rag_load_test/deploy/cli.py`` (``--dry-run``) imports
it, both rules checked by ``tests/structure/test_layering.py``.
"""
