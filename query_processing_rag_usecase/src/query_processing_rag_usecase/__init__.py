"""The RAG use case's own code: its kNN router and the command that trains it.

A workflow file loads it by naming it under ``extensions``: every module is
imported, and the router registers on its class line (ADR 0006). It imports the
toolkit through ``query_processing.extension_api`` alone.
"""
