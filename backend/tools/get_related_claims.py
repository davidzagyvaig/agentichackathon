"""
MCP Tool: get_related_claims

Get claims related by metadata (same paper, same author, similar topic).
Implements PRD Section 9.5.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain.tools import tool

from graph.graph_cache import get_graph_cache


class GetRelatedClaimsInput(BaseModel):
    """Input schema for get_related_claims tool."""
    claim_id: str = Field(..., description="ID of the source claim")
    relation_type: Literal["same_paper", "same_author", "similar_topic", "citing_same"] = Field(
        "same_paper",
        description="Type of relation to find"
    )
    limit: int = Field(10, ge=1, le=50, description="Maximum results")


class GetRelatedClaimsOutput(BaseModel):
    """Output schema for get_related_claims tool."""
    source_claim: Optional[dict]
    related_claims: list[dict]


@tool
async def get_related_claims(
    claim_id: str,
    relation_type: Literal["same_paper", "same_author", "similar_topic", "citing_same"] = "same_paper",
    limit: int = 10,
) -> dict:
    """Get claims related by metadata (same paper, same author, similar topic).
    
    Use this to find contextually related claims that share metadata relationships.
    Helps discover claims from the same paper, by the same authors, or on similar topics.
    
    Args:
        claim_id: ID of the source claim
        relation_type: Type of relationship:
            - same_paper: Claims from the same source paper (default)
            - same_author: Claims from papers by same authors
            - similar_topic: Claims with similar content (keyword-based)
            - citing_same: Claims that share graph connections
        limit: Maximum results (default: 10, max: 50)
        
    Returns:
        Dictionary with source_claim and related_claims list (each with relation and relevance_score).
    """
    graph_cache = get_graph_cache()
    
    source_claim = graph_cache.get_claim(claim_id)
    if not source_claim:
        return {
            "source_claim": None,
            "related_claims": [],
            "error": f"Claim {claim_id} not found"
        }
    
    related = []
    
    if relation_type == "same_paper":
        related = _find_same_paper_claims(source_claim, graph_cache, limit)
    elif relation_type == "same_author":
        related = _find_same_author_claims(source_claim, graph_cache, limit)
    elif relation_type == "similar_topic":
        related = _find_similar_topic_claims(source_claim, graph_cache, limit)
    elif relation_type == "citing_same":
        related = _find_citing_same_claims(source_claim, graph_cache, limit)
    
    return {
        "source_claim": source_claim,
        "related_claims": related,
    }


def _find_same_paper_claims(
    source_claim: dict,
    graph_cache,
    limit: int
) -> list[dict]:
    """Find claims from the same paper."""
    source_paper = source_claim.get("source_paper", {})
    arxiv_id = source_paper.get("arxiv_id")
    
    if not arxiv_id:
        return []
    
    paper_claims = graph_cache.get_claims_by_paper(arxiv_id)
    
    related = []
    for claim in paper_claims:
        if claim["id"] != source_claim["id"]:
            related.append({
                "claim": claim,
                "relation": "same_paper",
                "relevance_score": 1.0,
            })
    
    return related[:limit]


def _find_same_author_claims(
    source_claim: dict,
    graph_cache,
    limit: int
) -> list[dict]:
    """Find claims from papers by the same authors."""
    source_paper = source_claim.get("source_paper", {})
    source_authors = set(source_paper.get("authors", []))
    
    if not source_authors:
        return []
    
    related = []
    
    for node_id in graph_cache.G.nodes:
        if node_id == source_claim["id"]:
            continue
        
        node_data = graph_cache.G.nodes[node_id]
        node_paper = node_data.get("source_paper", {})
        node_authors = set(node_paper.get("authors", []))
        
        overlap = source_authors & node_authors
        if overlap:
            relevance = len(overlap) / len(source_authors)
            related.append({
                "claim": {"id": node_id, **node_data},
                "relation": "same_author",
                "relevance_score": relevance,
                "shared_authors": list(overlap),
            })
    
    related.sort(key=lambda x: x["relevance_score"], reverse=True)
    return related[:limit]


def _find_similar_topic_claims(
    source_claim: dict,
    graph_cache,
    limit: int
) -> list[dict]:
    """Find claims with similar content (keyword-based similarity)."""
    source_text = source_claim.get("text", "").lower()
    source_words = set(source_text.split())
    
    if not source_words:
        return []
    
    related = []
    
    for node_id in graph_cache.G.nodes:
        if node_id == source_claim["id"]:
            continue
        
        node_data = graph_cache.G.nodes[node_id]
        node_text = node_data.get("text", "").lower()
        node_words = set(node_text.split())
        
        if not node_words:
            continue
        
        intersection = source_words & node_words
        union = source_words | node_words
        jaccard = len(intersection) / len(union) if union else 0
        
        if jaccard > 0.1:
            related.append({
                "claim": {"id": node_id, **node_data},
                "relation": "similar_topic",
                "relevance_score": jaccard,
            })
    
    related.sort(key=lambda x: x["relevance_score"], reverse=True)
    return related[:limit]


def _find_citing_same_claims(
    source_claim: dict,
    graph_cache,
    limit: int
) -> list[dict]:
    """Find claims that share citation relationships."""
    source_id = source_claim["id"]
    G = graph_cache.G
    
    source_neighbors = set(G.predecessors(source_id)) | set(G.successors(source_id))
    
    if not source_neighbors:
        return []
    
    related = []
    
    for node_id in G.nodes:
        if node_id == source_id:
            continue
        
        node_neighbors = set(G.predecessors(node_id)) | set(G.successors(node_id))
        
        shared = source_neighbors & node_neighbors
        if shared:
            relevance = len(shared) / max(len(source_neighbors), 1)
            node_data = G.nodes[node_id]
            related.append({
                "claim": {"id": node_id, **node_data},
                "relation": "citing_same",
                "relevance_score": relevance,
                "shared_connections": len(shared),
            })
    
    related.sort(key=lambda x: x["relevance_score"], reverse=True)
    return related[:limit]


TOOL_SCHEMA = {
    "name": "get_related_claims",
    "description": "Get claims related by metadata (same paper, same author, similar topic)",
    "parameters": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "ID of the source claim"
            },
            "relation_type": {
                "type": "string",
                "enum": ["same_paper", "same_author", "similar_topic", "citing_same"],
                "default": "same_paper",
                "description": "Type of relationship to find"
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum number of results"
            }
        },
        "required": ["claim_id"]
    }
}
