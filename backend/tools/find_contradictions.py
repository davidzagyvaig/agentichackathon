"""
MCP Tool: find_contradictions

Find claims that contradict a given claim or find all contradictions
within a topic.

Implements PRD Section 9.6.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool

from graph.graph_cache import get_graph_cache
from tools.search_claims import _search_claims_impl
from database import supabase


class FindContradictionsInput(BaseModel):
    """Input schema for find_contradictions tool."""
    claim_id: Optional[str] = Field(None, description="Specific claim to find contradictions for")
    topic_query: Optional[str] = Field(None, description="Find contradictions within a topic")
    min_weight: float = Field(0.5, ge=0, le=1, description="Minimum contradiction weight")


class ContradictionPair(BaseModel):
    """A pair of contradicting claims."""
    claim_a: dict
    claim_b: dict
    edge: dict
    resolution_hints: list[str]


class FindContradictionsOutput(BaseModel):
    """Output schema for find_contradictions tool."""
    contradictions: list[dict]


@tool
async def find_contradictions(
    claim_id: Optional[str] = None,
    topic_query: Optional[str] = None,
    min_weight: float = 0.5,
) -> dict:
    """Find claims that contradict a given claim or find all contradictions in a topic.
    
    Use this to identify conflicting evidence. Can search for contradictions of a specific
    claim, within a topic area, or across the entire graph.
    
    Args:
        claim_id: If provided, find claims contradicting this specific claim
        topic_query: If provided, find all contradictions within this topic (searches graph first)
        min_weight: Minimum contradiction edge weight to include (default: 0.5, range: 0-1)
        
    Note: Either claim_id OR topic_query should be provided, not both.
    
    Returns:
        Dictionary with contradictions list (each with claim_a, claim_b, edge, resolution_hints)
        and total_found count.
    """
    graph_cache = get_graph_cache()
    contradictions = []
    
    if claim_id:
        contradictions = _find_claim_contradictions(
            claim_id,
            min_weight,
            graph_cache
        )
    elif topic_query:
        contradictions = await _find_topic_contradictions(
            topic_query,
            min_weight,
            graph_cache,
            supabase
        )
    else:
        contradictions = _find_all_contradictions(min_weight, graph_cache)
    
    return {
        "contradictions": contradictions,
        "total_found": len(contradictions),
    }


def _find_claim_contradictions(
    claim_id: str,
    min_weight: float,
    graph_cache
) -> list[dict]:
    """Find contradictions for a specific claim."""
    G = graph_cache.G
    
    if claim_id not in G:
        return []
    
    claim_data = graph_cache.get_claim(claim_id)
    contradictions = []
    
    for pred in G.predecessors(claim_id):
        edge_data = G[pred][claim_id]
        if edge_data.get("type") == "contradicts" and edge_data.get("weight", 0) >= min_weight:
            pred_data = graph_cache.get_claim(pred)
            contradictions.append({
                "claim_a": pred_data,
                "claim_b": claim_data,
                "edge": {
                    "id": edge_data.get("id"),
                    "type": "contradicts",
                    "weight": edge_data.get("weight"),
                    "reasoning": edge_data.get("reasoning"),
                },
                "resolution_hints": _generate_resolution_hints(pred_data, claim_data),
            })
    
    for succ in G.successors(claim_id):
        edge_data = G[claim_id][succ]
        if edge_data.get("type") == "contradicts" and edge_data.get("weight", 0) >= min_weight:
            succ_data = graph_cache.get_claim(succ)
            contradictions.append({
                "claim_a": claim_data,
                "claim_b": succ_data,
                "edge": {
                    "id": edge_data.get("id"),
                    "type": "contradicts",
                    "weight": edge_data.get("weight"),
                    "reasoning": edge_data.get("reasoning"),
                },
                "resolution_hints": _generate_resolution_hints(claim_data, succ_data),
            })
    
    return contradictions


async def _find_topic_contradictions(
    topic_query: str,
    min_weight: float,
    graph_cache,
    supabase_client
) -> list[dict]:
    """Find all contradictions within a topic."""
    search_result = await _search_claims_impl(
        query=topic_query,
        search_type="hybrid",
        limit=50
    )
    
    topic_claim_ids = {c["id"] for c in search_result.get("claims", [])}
    
    if not topic_claim_ids:
        return []
    
    contradictions = []
    seen_pairs = set()
    G = graph_cache.G
    
    for claim_id in topic_claim_ids:
        if claim_id not in G:
            continue
        
        for other_id in G.successors(claim_id):
            edge_data = G[claim_id][other_id]
            if edge_data.get("type") != "contradicts":
                continue
            if edge_data.get("weight", 0) < min_weight:
                continue
            
            pair_key = tuple(sorted([claim_id, other_id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            claim_a = graph_cache.get_claim(claim_id)
            claim_b = graph_cache.get_claim(other_id)
            
            contradictions.append({
                "claim_a": claim_a,
                "claim_b": claim_b,
                "edge": {
                    "id": edge_data.get("id"),
                    "type": "contradicts",
                    "weight": edge_data.get("weight"),
                    "reasoning": edge_data.get("reasoning"),
                },
                "resolution_hints": _generate_resolution_hints(claim_a, claim_b),
            })
    
    contradictions.sort(key=lambda x: x["edge"]["weight"], reverse=True)
    return contradictions


def _find_all_contradictions(
    min_weight: float,
    graph_cache
) -> list[dict]:
    """Find all contradictions in the graph."""
    G = graph_cache.G
    contradictions = []
    seen_pairs = set()
    
    for source, target, data in G.edges(data=True):
        if data.get("type") != "contradicts":
            continue
        if data.get("weight", 0) < min_weight:
            continue
        
        pair_key = tuple(sorted([source, target]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        claim_a = graph_cache.get_claim(source)
        claim_b = graph_cache.get_claim(target)
        
        contradictions.append({
            "claim_a": claim_a,
            "claim_b": claim_b,
            "edge": {
                "id": data.get("id"),
                "type": "contradicts",
                "weight": data.get("weight"),
                "reasoning": data.get("reasoning"),
            },
            "resolution_hints": _generate_resolution_hints(claim_a, claim_b),
        })
    
    contradictions.sort(key=lambda x: x["edge"]["weight"], reverse=True)
    return contradictions[:100]


def _generate_resolution_hints(claim_a: dict, claim_b: dict) -> list[str]:
    """Generate hints for why claims might conflict."""
    hints = []
    
    paper_a = claim_a.get("source_paper", {})
    paper_b = claim_b.get("source_paper", {})
    
    year_a = paper_a.get("year")
    year_b = paper_b.get("year")
    if year_a and year_b and abs(year_a - year_b) > 5:
        newer = "A" if year_a > year_b else "B"
        hints.append(f"Claim {newer} is from a more recent study ({max(year_a, year_b)} vs {min(year_a, year_b)})")
    
    section_a = paper_a.get("section", "").lower()
    section_b = paper_b.get("section", "").lower()
    if section_a != section_b:
        if section_a == "results" or section_b == "results":
            hints.append("One claim is from Results section (primary findings)")
        if section_a == "discussion" or section_b == "discussion":
            hints.append("One claim is from Discussion (interpretation, not raw data)")
    
    conf_a = claim_a.get("confidence", 0.5)
    conf_b = claim_b.get("confidence", 0.5)
    if abs(conf_a - conf_b) > 0.3:
        higher = "A" if conf_a > conf_b else "B"
        hints.append(f"Claim {higher} has higher confidence score")
    
    type_a = claim_a.get("type")
    type_b = claim_b.get("type")
    if type_a == "empirical" and type_b == "unsupported":
        hints.append("Claim A is empirical (evidence-based), Claim B lacks cited evidence")
    elif type_b == "empirical" and type_a == "unsupported":
        hints.append("Claim B is empirical (evidence-based), Claim A lacks cited evidence")
    
    if not hints:
        hints.append("Review original papers for methodological differences")
        hints.append("Consider sample sizes, populations, and experimental conditions")
    
    return hints


TOOL_SCHEMA = {
    "name": "find_contradictions",
    "description": "Find claims that contradict a given claim or find all contradictions in a topic",
    "parameters": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "Specific claim to find contradictions for"
            },
            "topic_query": {
                "type": "string",
                "description": "Find contradictions within a topic"
            },
            "min_weight": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum contradiction weight to include"
            }
        }
    }
}
