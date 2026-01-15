"""
MCP Tools for Knowledge Graph

These tools provide the interface for the Deep Research Agent to query
the Knowledge Graph via Model Context Protocol (MCP).

Implements PRD Section 9: MCP Tools Specification.
"""

from .search_claims import search_claims
from .get_claim_support import get_claim_support
from .trace_provenance import trace_provenance
from .get_related_claims import get_related_claims
from .find_contradictions import find_contradictions
from .get_graph_statistics import get_graph_statistics

__all__ = [
    "search_claims",
    "get_claim_support",
    "trace_provenance",
    "get_related_claims",
    "find_contradictions",
    "get_graph_statistics",
]
