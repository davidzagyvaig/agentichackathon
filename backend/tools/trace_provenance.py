"""
MCP Tool: trace_provenance

Trace a claim back to its foundational ground truths or identify
if ungrounded.

Implements PRD Section 9.4.
"""

from typing import Optional
from pydantic import BaseModel, Field

from graph.graph_cache import get_graph_cache
from graph.provenance import ProvenanceTracer


class TraceProvenanceInput(BaseModel):
    """Input schema for trace_provenance tool."""
    claim_id: str = Field(..., description="ID of the claim to trace")
    max_depth: int = Field(10, ge=1, le=20, description="Maximum search depth")


class TraceProvenanceOutput(BaseModel):
    """Output schema for trace_provenance tool."""
    claim: Optional[dict]
    is_grounded: bool
    ground_truths_found: list[dict]
    ungrounded_foundations: list[dict]
    cycles_detected: list[list[str]]
    provenance_chains: list[dict]


async def trace_provenance(
    claim_id: str,
    max_depth: int = 10,
) -> dict:
    """
    Trace a claim back to its foundational ground truths.
    
    Args:
        claim_id: ID of the claim to trace
        max_depth: Maximum depth to search for ground truths
        
    Returns:
        Dictionary with provenance analysis including:
        - is_grounded: Whether the claim has any path to ground truth
        - ground_truths_found: List of ground truth claims reached
        - ungrounded_foundations: Claims at end of chains with no support
        - cycles_detected: Any circular reasoning detected
        - provenance_chains: All paths from claim to ground truths
    """
    graph_cache = get_graph_cache()
    tracer = ProvenanceTracer(graph_cache)
    
    claim_data = graph_cache.get_claim(claim_id)
    if not claim_data:
        return {
            "claim": None,
            "is_grounded": False,
            "ground_truths_found": [],
            "ungrounded_foundations": [],
            "cycles_detected": [],
            "provenance_chains": [],
            "error": f"Claim {claim_id} not found"
        }
    
    result = tracer.trace_provenance(claim_id, max_depth)
    
    return {
        "claim": claim_data,
        "is_grounded": result.is_grounded,
        "ground_truths_found": [
            gt.model_dump() for gt in result.ground_truths_found
        ],
        "ungrounded_foundations": [
            uf.model_dump() for uf in result.ungrounded_foundations
        ],
        "cycles_detected": result.cycles_detected,
        "provenance_chains": [
            {
                "path": chain.path,
                "chain_confidence": chain.chain_confidence,
                "ground_truth_text": chain.ground_truth_text,
            }
            for chain in result.provenance_chains
        ],
    }


async def get_grounding_score(claim_id: str) -> dict:
    """
    Get an overall grounding score for a claim.
    
    The score considers:
    - Whether the claim is grounded at all
    - Number and quality of provenance chains
    - Presence of cycles (reduces score)
    
    Returns:
        Dictionary with score and explanation.
    """
    graph_cache = get_graph_cache()
    tracer = ProvenanceTracer(graph_cache)
    
    score = tracer.compute_grounding_score(claim_id)
    analysis = tracer.analyze_claim_foundation(claim_id)
    
    explanation = []
    if not analysis.is_grounded:
        explanation.append("Claim has no path to any ground truth")
    else:
        explanation.append(f"Found {len(analysis.chains)} provenance chains")
        if analysis.max_chain_confidence > 0:
            explanation.append(f"Strongest chain confidence: {analysis.max_chain_confidence:.2f}")
        if analysis.shortest_path_length:
            explanation.append(f"Shortest path to ground truth: {analysis.shortest_path_length} hops")
        if analysis.cycles:
            explanation.append(f"Warning: {len(analysis.cycles)} cycles detected (circular reasoning)")
    
    return {
        "claim_id": claim_id,
        "grounding_score": round(score, 3),
        "is_grounded": analysis.is_grounded,
        "explanation": explanation,
        "details": {
            "chain_count": len(analysis.chains),
            "max_confidence": analysis.max_chain_confidence,
            "avg_confidence": analysis.avg_chain_confidence,
            "shortest_path": analysis.shortest_path_length,
            "cycle_count": len(analysis.cycles),
        }
    }


TOOL_SCHEMA = {
    "name": "trace_provenance",
    "description": "Trace a claim back to its foundational ground truths or identify if ungrounded",
    "parameters": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "ID of the claim to trace"
            },
            "max_depth": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum search depth"
            }
        },
        "required": ["claim_id"]
    }
}
