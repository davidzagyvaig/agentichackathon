"""
MCP Tool: search_claims

Semantic and keyword search for claims in the knowledge graph.
Implements PRD Section 9.2.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

from graph.graph_cache import get_graph_cache
from clients.embeddings import EmbeddingsClient
from models.kg_schemas import ClaimSearchFilters, ClaimSearchResult


class SearchClaimsInput(BaseModel):
    """Input schema for search_claims tool."""
    query: str = Field(..., description="Natural language search query")
    search_type: Literal["semantic", "keyword", "hybrid"] = Field(
        "hybrid",
        description="Type of search to perform"
    )
    limit: int = Field(10, ge=1, le=100, description="Max results to return")
    filters: Optional[ClaimSearchFilters] = Field(
        None,
        description="Optional filters for results"
    )


class SearchClaimsOutput(BaseModel):
    """Output schema for search_claims tool."""
    claims: list[dict]
    total_results: int
    search_type: str


async def search_claims(
    query: str,
    search_type: Literal["semantic", "keyword", "hybrid"] = "hybrid",
    limit: int = 10,
    filters: Optional[dict] = None,
    supabase_client=None,
) -> dict:
    """
    Search for claims in the knowledge graph.
    
    Args:
        query: Natural language search query
        search_type: "semantic", "keyword", or "hybrid"
        limit: Maximum number of results
        filters: Optional filters (type, min_confidence, etc.)
        supabase_client: Supabase client for database queries
        
    Returns:
        Dictionary with claims and metadata.
    """
    graph_cache = get_graph_cache()
    filters = filters or {}
    
    results = []
    
    if search_type == "semantic" and supabase_client:
        results = await _semantic_search(query, limit, filters, supabase_client)
    elif search_type == "keyword":
        results = _keyword_search(query, limit, filters, graph_cache)
    else:
        if supabase_client:
            semantic_results = await _semantic_search(query, limit * 2, filters, supabase_client)
            keyword_results = _keyword_search(query, limit * 2, filters, graph_cache)
            results = _hybrid_rerank(semantic_results, keyword_results, limit)
        else:
            results = _keyword_search(query, limit, filters, graph_cache)
    
    return {
        "claims": results[:limit],
        "total_results": len(results),
        "search_type": search_type,
    }


async def _semantic_search(
    query: str,
    limit: int,
    filters: dict,
    supabase_client
) -> list[dict]:
    """Perform semantic search using embeddings."""
    try:
        embeddings_client = EmbeddingsClient()
        query_embedding = await embeddings_client.get_embedding(query)
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []
    
    try:
        params = {
            "query_embedding": query_embedding,
            "match_count": limit,
        }
        
        if filters.get("type"):
            params["filter_type"] = filters["type"]
        if filters.get("min_confidence"):
            params["min_confidence"] = filters["min_confidence"]
        
        response = supabase_client.rpc(
            "search_claims_by_embedding",
            params
        ).execute()
        
        results = []
        for row in response.data or []:
            results.append({
                "id": row["id"],
                "text": row["text"],
                "type": row["type"],
                "confidence": row["confidence"],
                "source_paper": row["source_paper"],
                "support_count": row.get("support_count", 0),
                "contradict_count": row.get("contradict_count", 0),
                "similarity_score": row.get("similarity", 0.0),
            })
        
        return results
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []


def _keyword_search(
    query: str,
    limit: int,
    filters: dict,
    graph_cache
) -> list[dict]:
    """Perform keyword search in the graph cache."""
    query_terms = query.lower().split()
    results = []
    
    for node_id in graph_cache.G.nodes:
        node_data = graph_cache.G.nodes[node_id]
        text = node_data.get("text", "").lower()
        
        if filters.get("type") and node_data.get("type") != filters["type"]:
            continue
        if filters.get("min_confidence") and node_data.get("confidence", 0) < filters["min_confidence"]:
            continue
        
        match_count = sum(1 for term in query_terms if term in text)
        
        if match_count > 0:
            score = match_count / len(query_terms)
            results.append({
                "id": node_id,
                "text": node_data.get("text", ""),
                "type": node_data.get("type", "empirical"),
                "confidence": node_data.get("confidence", 0.5),
                "source_paper": node_data.get("source_paper", {}),
                "support_count": node_data.get("support_count", 0),
                "contradict_count": node_data.get("contradict_count", 0),
                "similarity_score": score,
            })
    
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:limit]


def _hybrid_rerank(
    semantic_results: list[dict],
    keyword_results: list[dict],
    limit: int
) -> list[dict]:
    """Combine and rerank semantic and keyword results."""
    combined = {}
    
    for i, result in enumerate(semantic_results):
        claim_id = result["id"]
        semantic_score = 1.0 - (i / len(semantic_results)) if semantic_results else 0
        combined[claim_id] = {
            **result,
            "semantic_rank": semantic_score,
            "keyword_rank": 0.0,
        }
    
    for i, result in enumerate(keyword_results):
        claim_id = result["id"]
        keyword_score = 1.0 - (i / len(keyword_results)) if keyword_results else 0
        
        if claim_id in combined:
            combined[claim_id]["keyword_rank"] = keyword_score
        else:
            combined[claim_id] = {
                **result,
                "semantic_rank": 0.0,
                "keyword_rank": keyword_score,
            }
    
    for claim_id in combined:
        item = combined[claim_id]
        item["similarity_score"] = 0.6 * item["semantic_rank"] + 0.4 * item["keyword_rank"]
    
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["similarity_score"],
        reverse=True
    )
    
    for result in sorted_results:
        result.pop("semantic_rank", None)
        result.pop("keyword_rank", None)
    
    return sorted_results[:limit]


TOOL_SCHEMA = {
    "name": "search_claims",
    "description": "Semantic and keyword search for claims in the knowledge graph",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query"
            },
            "search_type": {
                "type": "string",
                "enum": ["semantic", "keyword", "hybrid"],
                "default": "hybrid",
                "description": "Type of search to perform"
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum number of results to return"
            },
            "filters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["empirical", "ground_truth", "unsupported"]
                    },
                    "min_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "has_support": {
                        "type": "boolean"
                    }
                }
            }
        },
        "required": ["query"]
    }
}
