"""
External API clients for the Knowledge Graph system.
"""

from .arxiv_client import ArxivClient
from .semantic_scholar import SemanticScholarClient
from .embeddings import EmbeddingsClient

__all__ = ["ArxivClient", "SemanticScholarClient", "EmbeddingsClient"]
