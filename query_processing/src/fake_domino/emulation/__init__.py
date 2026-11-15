"""The Domino emulation: everything that exists only to emulate Domino at run time.

Removable as one folder when the project moves to a real Domino: nothing in the
core ``fake_domino`` package imports it. ``identity``, ``proxy_settings``,
``app_proxy`` and ``proxy_routes`` are the App proxy (credentials, Domino's
headers, forwarding through one pooled client); ``placement_settings``,
``project_env``, ``start_script``, ``launcher``, ``ssh_launcher``, ``placement``
and ``tiers`` place each App on a machine of its hardware tier and run it over
SSH, plugged into the core's ``BaseInstanceScheduler`` seam; ``about`` and ``flags``
are its Swagger text and command line; ``server`` builds it
(``create_domino_emulation``, ``rag-domino-emulation``). Only this package
imports ``jwt`` and ``cryptography``.
"""
