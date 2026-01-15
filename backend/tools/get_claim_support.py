"""
MCP Tool: get_claim_support

Get all claims that support or contradict a specific claim.
Implements PRD Section 9.3.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from graph.graph_cache import get_graph_cache


class GetClaimSupportInput(BaseModel):
    """Input schema for get_claim_support tool."""
    claim_id: str = Field(..., description="ID of the claim to query")
    relationship_type: Literal["supports", "contradicts", "both"] = Field(
        "both",
        description="Type of relationships to retrieve"
    )
    direction: Literal["incoming", "outgoing", "both"] = Field(
        "incoming",
        description="Direction of relationships"
    )
    include_transitive: bool = Field(
        False,
        description="Follow chains up to max_depth"
    )
    max_depth: int = Field(1, ge=1, le=10, description="Maximum traversal depth")


class GetClaimSupportOutput(BaseModel):
    """Output schema for get_claim_support tool."""
    claim: Optional[dict]
    supporting_claims: list[dict]
    contradicting_claims: list[dict]


async def get_claim_support(
    claim_id: str,
    relationship_type: Literal["supports", "contradicts", "both"] = "both",
    direction: Literal["incoming", "outgoing", "both"] = "incoming",
    include_transitive: bool = False,
    max_depth: int = 1,
) -> dict:
    """
    Get claims that support or contradict a specific claim.
    
    Args:
        claim_id: ID of the claim to query
        relationship_type: "supports", "contradicts", or "both"
        direction: "incoming" (claims that support THIS claim),
                   "outgoing" (claims this claim supports), or "both"
        include_transitive: Whether to follow chains beyond direct connections
        max_depth: Maximum depth for transitive queries
        
    Returns:
        Dictionary with claim data and supporting/contradicting claims.
    """
    graph_cache = get_graph_cache()
    
    claim_data = graph_cache.get_claim(claim_id)
    if not claim_data:
        return {
            "claim": None,
            "supporting_claims": [],
            "contradicting_claims": [],
            "error": f"Claim {claim_id} not found"
        }
    
    supporting_claims = []
    contradicting_claims = []
    depth = max_depth if include_transitive else 1
    
    if direction in ["incoming", "both"]:
        if relationship_type in ["supports", "both"]:
            supporters = graph_cache.get_supporters(
                claim_id,
                depth=depth,
                edge_type="supports"
            )
            supporting_claims.extend([
                {
                    "claim": s["claim"],
                    "edge": s["edge"],
                    "depth": s["depth"]
                }
                for s in supporters
            ])
        
        if relationship_type in ["contradicts", "both"]:
            contradictors = graph_cache.get_supporters(
                claim_id,
                depth=depth,
                edge_type="contradicts"
            )
            contradicting_claims.extend([
                {
                    "claim": c["claim"],
                    "edge": c["edge"],
                    "depth": c["depth"]
                }
                for c in contradictors
            ])
    
    if direction in ["outgoing", "both"]:
        if relationship_type in ["supports", "both"]:
            supported = graph_cache.get_supported_claims(
                claim_id,
                depth=depth,
                edge_type="supports"
            )
            for s in supported:
                entry = {
                    "claim": s["claim"],
                    "edge": s["edge"],
                    "depth": s["depth"],
                    "direction": "outgoing"
                }
                if entry not in supporting_claims:
                    supporting_claims.append(entry)
        
        if relationship_type in ["contradicts", "both"]:
            contradicted = graph_cache.get_supported_claims(
                claim_id,
                depth=depth,
                edge_type="contradicts"
            )
            for c in contradicted:
                entry = {
                    "claim": c["claim"],
                    "edge": c["edge"],
                    "depth": c["depth"],
                    "direction": "outgoing"
                }
                if entry not in contradicting_claims:
                    contradicting_claims.append(entry)
    
    return {
        "claim": claim_data,
        "supporting_claims": supporting_claims,
        "contradicting_claims": contradicting_claims,
    }


TOOL_SCHEMA = {
    "name": "get_claim_support",
    "description": "Get all claims that support or contradict a specific claim",
    "parameters": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "ID of the claim to query"
            },
            "relationship_type": {
                "type": "string",
                "enum": ["supports", "contradicts", "both"],
                "default": "both",
                "description": "Type of relationships to retrieve"
            },
            "direction": {
                "type": "string",
                "enum": ["incoming", "outgoing", "both"],
                "default": "incoming",
                "description": "Direction of relationships"
            },
            "include_transitive": {
                "type": "boolean",
                "default": False,
                "description": "Follow chains up to max_depth"
            },
            "max_depth": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum traversal depth"
            }
        },
        "required": ["claim_id"]
    }
}
