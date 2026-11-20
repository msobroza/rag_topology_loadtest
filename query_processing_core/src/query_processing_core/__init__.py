"""The core: what one toolkit writes and another reads (ADR 0014).

A distribution of its own, which every toolkit installs and which installs
none: data only, with no port, no settings and no I/O, importing the standard
library and Pydantic alone. Its modules are laid out by the glossary's areas:
``retrieval`` holds the metadata filter.
"""
