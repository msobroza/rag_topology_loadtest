"""The core: what one toolkit writes and another reads (ADR 0014).

A distribution of its own, which every toolkit installs and which installs
none: data only, with no port, no settings and no I/O, importing the standard
library, Pydantic and, in one module, doclang's core. What a client sends and
reads sits at the top, the conversational history and the body of a request and
of a response; ``retrieval`` holds the metadata filter and the vector-store
record with its scores.
"""
