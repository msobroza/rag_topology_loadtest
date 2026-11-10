"""A stand-in for Domino 6.2's Apps API, and nothing else: no product code lives here.

``api_shapes`` holds the request/response shapes transcribed from the 6.2
reference, ``state`` the fake's Apps and scripted behaviours, ``apps_routes`` and
``stand_ins`` what it serves, and ``server`` builds it (``create_fake_domino``,
``rag-fake-domino``). It imports nothing from ``rag_load_test``; inside the
product only ``rag_load_test/deploy/cli.py`` (``--dry-run``) imports it, both
rules checked by ``tests/test_layering.py``.
"""
