"""
MCP Tool: get_graph_statistics

Get overview statistics about the knowledge graph.
Implements PRD Section 9.7.
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from graph.graph_cache import get_graph_cache


class GetGraphStatisticsOutput(BaseModel):
    """Output schema for get_graph_statistics tool."""
    total_claims: int
    claims_by_type: dict
    total_edges: int
    edges_by_type: dict
    avg_support_depth: float
    grounded_percentage: float
    papers_indexed: int
    last_updated: Optional[str]


async def get_graph_statistics() -> dict:
    """
    Get overview statistics about the knowledge graph.
    
    Returns:
        Dictionary with graph statistics including:
        - total_claims: Total number of claims in the graph
        - claims_by_type: Breakdown by empirical/ground_truth/unsupported
        - total_edges: Total number of edges
        - edges_by_type: Breakdown by supports/contradicts
        - avg_support_depth: Average depth to ground truth
        - grounded_percentage: Percentage of claims with path to ground truth
        - papers_indexed: Number of unique papers
        - last_updated: Timestamp of last update
    """
    graph_cache = get_graph_cache()
    
    if not graph_cache.is_loaded or graph_cache.node_count() == 0:
        return {
            "total_claims": 0,
            "claims_by_type": {
                "empirical": 0,
                "ground_truth": 0,
                "unsupported": 0
            },
            "total_edges": 0,
            "edges_by_type": {
                "supports": 0,
                "contradicts": 0
            },
            "avg_support_depth": 0.0,
            "grounded_percentage": 0.0,
            "papers_indexed": 0,
            "last_updated": None,
            "message": "Graph cache is empty or not loaded"
        }
    
    stats = graph_cache.get_statistics()
    
    avg_depth = _compute_average_depth(graph_cache)
    
    return {
        "total_claims": stats["total_claims"],
        "claims_by_type": stats["claims_by_type"],
        "total_edges": stats["total_edges"],
        "edges_by_type": stats["edges_by_type"],
        "avg_support_depth": avg_depth,
        "grounded_percentage": stats["grounded_percentage"],
        "papers_indexed": stats["papers_indexed"],
        "last_updated": stats.get("last_sync"),
    }


def _compute_average_depth(graph_cache) -> float:
    """Compute average depth to ground truth for grounded claims."""
    depths = []
    
    for node_id in graph_cache.G.nodes:
        node_data = graph_cache.G.nodes[node_id]
        
        if node_data.get("type") == "ground_truth":
            continue
        
        depth = graph_cache.compute_depth_to_ground_truth(node_id)
        if depth is not None:
            depths.append(depth)
    
    return sum(depths) / len(depths) if depths else 0.0


async def get_detailed_statistics() -> dict:
    """
    Get more detailed statistics about the knowledge graph.
    
    Includes additional metrics like:
    - Top papers by claim count
    - Claim type distribution over time
    - Edge density metrics
    - Cycle statistics
    """
    graph_cache = get_graph_cache()
    
    basic_stats = await get_graph_statistics()
    
    if basic_stats["total_claims"] == 0:
        return {
            **basic_stats,
            "top_papers": [],
            "connectivity_metrics": {},
            "cycle_count": 0,
        }
    
    paper_claim_counts = {}
    for node_id in graph_cache.G.nodes:
        node_data = graph_cache.G.nodes[node_id]
        source_paper = node_data.get("source_paper", {})
        arxiv_id = source_paper.get("arxiv_id")
        title = source_paper.get("title", "Unknown")
        
        if arxiv_id:
            key = arxiv_id
        else:
            key = title[:50]
        
        if key not in paper_claim_counts:
            paper_claim_counts[key] = {
                "arxiv_id": arxiv_id,
                "title": title,
                "claim_count": 0
            }
        paper_claim_counts[key]["claim_count"] += 1
    
    top_papers = sorted(
        paper_claim_counts.values(),
        key=lambda x: x["claim_count"],
        reverse=True
    )[:10]
    
    G = graph_cache.G
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    max_possible_edges = n_nodes * (n_nodes - 1)
    density = n_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    avg_in_degree = sum(d for _, d in G.in_degree()) / n_nodes if n_nodes > 0 else 0
    avg_out_degree = sum(d for _, d in G.out_degree()) / n_nodes if n_nodes > 0 else 0
    
    connectivity_metrics = {
        "density": round(density, 6),
        "avg_in_degree": round(avg_in_degree, 2),
        "avg_out_degree": round(avg_out_degree, 2),
    }
    
    cycles = graph_cache.detect_cycles()
    
    return {
        **basic_stats,
        "top_papers": top_papers,
        "connectivity_metrics": connectivity_metrics,
        "cycle_count": len(cycles) if cycles else 0,
    }


async def get_build_history(supabase_client=None, limit: int = 10) -> list[dict]:
    """
    Get history of graph build runs.
    
    Args:
        supabase_client: Supabase client
        limit: Maximum number of builds to return
        
    Returns:
        List of build run records.
    """
    if not supabase_client:
        return []
    
    try:
        response = supabase_client.table("kg_build_runs")\
            .select("*")\
            .order("started_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return response.data or []
    except Exception as e:
        print(f"Error fetching build history: {e}")
        return []


TOOL_SCHEMA = {
    "name": "get_graph_statistics",
    "description": "Get overview statistics about the knowledge graph",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}
