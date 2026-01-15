"""
Knowledge Graph Builder
Builds a NetworkX graph from papers and claims, exports to JSON for frontend
"""

import networkx as nx
from typing import Optional
import json

from models.schemas import (
    Paper,
    Claim,
    ClaimValidation,
    ValidationStatus,
    GraphNode,
    GraphEdge,
    KnowledgeGraph,
)


# Color scheme for validation status
STATUS_COLORS = {
    ValidationStatus.VERIFIED: "#22c55e",    # Green
    ValidationStatus.SUSPICIOUS: "#ef4444",   # Red
    ValidationStatus.PENDING: "#6b7280",      # Gray
    ValidationStatus.UNKNOWN: "#9ca3af",      # Light gray
}


def create_knowledge_graph(
    papers: list[Paper],
    claims: list[Claim],
    validations: dict[str, ClaimValidation] = None
) -> KnowledgeGraph:
    """
    Build a knowledge graph from papers and claims.
    
    Args:
        papers: List of Paper objects
        claims: List of Claim objects
        validations: Optional dict mapping claim_id -> ClaimValidation
        
    Returns:
        KnowledgeGraph ready for frontend visualization
    """
    nodes = []
    edges = []
    validations = validations or {}
    
    # Create a mapping of paper_id to paper for quick lookup
    paper_map = {p.id: p for p in papers}
    claims_by_paper = {}
    for claim in claims:
        if claim.paper_id not in claims_by_paper:
            claims_by_paper[claim.paper_id] = []
        claims_by_paper[claim.paper_id].append(claim)
    
    # Create paper nodes
    for paper in papers:
        # Determine paper status based on its claims' validations
        paper_claims = claims_by_paper.get(paper.id, [])
        if paper_claims:
            claim_statuses = [
                validations.get(c.id, ClaimValidation(
                    claim_id=c.id,
                    overall_status=c.validation_status,
                    trust_score=0.7
                )).overall_status
                for c in paper_claims
            ]
            # If any claim is suspicious, paper is suspicious
            if ValidationStatus.SUSPICIOUS in claim_statuses:
                paper_status = ValidationStatus.SUSPICIOUS
            elif all(s == ValidationStatus.VERIFIED for s in claim_statuses):
                paper_status = ValidationStatus.VERIFIED
            else:
                paper_status = ValidationStatus.PENDING
        else:
            paper_status = ValidationStatus.PENDING
        
        # Override if paper itself is retracted
        if paper.is_retracted:
            paper_status = ValidationStatus.SUSPICIOUS
        
        node = GraphNode(
            id=f"paper-{paper.id}",
            type="paper",
            label=truncate_text(paper.title, 50),
            status=paper_status,
            color=STATUS_COLORS.get(paper_status, "#6b7280"),
            data={
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "doi": paper.doi,
                "cited_by_count": paper.cited_by_count,
                "is_retracted": paper.is_retracted,
                "abstract": truncate_text(paper.abstract or "", 300),
                "claim_count": len(paper_claims),
            }
        )
        nodes.append(node)
    
    # Create claim nodes
    for claim in claims:
        validation = validations.get(claim.id)
        
        if validation:
            claim_status = validation.overall_status
            trust_score = validation.trust_score
            bullshit_indicators = validation.bullshit_indicators
        else:
            claim_status = claim.validation_status
            trust_score = 0.7
            bullshit_indicators = []
        
        node = GraphNode(
            id=f"claim-{claim.id}",
            type="claim",
            label=truncate_text(claim.text, 60),
            status=claim_status,
            color=STATUS_COLORS.get(claim_status, "#6b7280"),
            data={
                "text": claim.text,
                "claim_type": claim.claim_type,
                "evidence_type": claim.evidence_type,
                "trust_score": trust_score,
                "paper_id": claim.paper_id,
                "bullshit_indicators": bullshit_indicators,
            }
        )
        nodes.append(node)
        
        # Create edge from paper to claim
        edge = GraphEdge(
            source=f"paper-{claim.paper_id}",
            target=f"claim-{claim.id}",
            type="contains_claim",
            label="contains"
        )
        edges.append(edge)
        
        # Add CITATION nodes if validation exists
        if validation and validation.citation_validations:
            for cv in validation.citation_validations:
                if cv.exists and cv.citation_title:
                    # Create a node for the cited paper
                    # Use DOI as ID if available, otherwise hash title
                    citation_node_id = f"citation-{cv.citation_doi}"
                    
                    # Determine status
                    citation_status = ValidationStatus.VERIFIED
                    if cv.is_retracted:
                        citation_status = ValidationStatus.SUSPICIOUS
                    elif not cv.is_relevant:
                        citation_status = ValidationStatus.PENDING # Or some other status for irrelevant
                    
                    citation_node = GraphNode(
                        id=citation_node_id,
                        type="citation", # Distinct type for styling
                        label=truncate_text(cv.citation_title, 40),
                        status=citation_status,
                        color=STATUS_COLORS.get(citation_status, "#6b7280"),
                        data={
                            "title": cv.citation_title,
                            "doi": cv.citation_doi,
                            "is_retracted": cv.is_retracted,
                            "is_relevant": cv.is_relevant,
                            "relevance_score": cv.relevance_score,
                            "validation_notes": cv.validation_notes
                        }
                    )
                    
                    # Avoid duplicates
                    if not any(n.id == citation_node_id for n in nodes):
                        nodes.append(citation_node)
                    
                    # Edge from Claim -> Citation
                    cite_edge = GraphEdge(
                        source=f"claim-{claim.id}",
                        target=citation_node_id,
                        type="cites",
                        label="cites"
                    )
                    edges.append(cite_edge)
    
    # Calculate summary statistics
    verified_count = sum(1 for n in nodes if n.status == ValidationStatus.VERIFIED)
    suspicious_count = sum(1 for n in nodes if n.status == ValidationStatus.SUSPICIOUS)
    
    metadata = {
        "total_papers": len(papers),
        "total_claims": len(claims),
        "verified_count": verified_count,
        "suspicious_count": suspicious_count,
        "pending_count": len(nodes) - verified_count - suspicious_count,
    }
    
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        metadata=metadata
    )


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def export_for_react_flow(graph: KnowledgeGraph) -> dict:
    """
    Export the knowledge graph in React Flow format.
    
    React Flow expects nodes with position and edges with specific structure.
    """
    rf_nodes = []
    rf_edges = []
    
    # Simple layout: papers in a row, claims below
    paper_nodes = [n for n in graph.nodes if n.type == "paper"]
    claim_nodes = [n for n in graph.nodes if n.type == "claim"]
    
    # Position paper nodes
    for i, node in enumerate(paper_nodes):
        rf_nodes.append({
            "id": node.id,
            "type": "paper",  # Custom node type
            "position": {"x": i * 300, "y": 0},
            "data": {
                "label": node.label,
                "status": node.status,
                "color": node.color,
                **node.data
            }
        })
    
    # Position claim nodes below their papers
    paper_claim_count = {}
    for node in claim_nodes:
        paper_id = node.data.get("paper_id", "")
        paper_node_id = f"paper-{paper_id}"
        
        # Find paper position
        paper_x = 0
        for pn in paper_nodes:
            if pn.id == paper_node_id:
                # get x from previous loop logic (simplified here)
                # paper nodes are at i * 300
                idx = paper_nodes.index(pn)
                paper_x = idx * 300
                break
        
        # Stack claims vertically
        claim_index = paper_claim_count.get(paper_id, 0)
        paper_claim_count[paper_id] = claim_index + 1
        
        rf_nodes.append({
            "id": node.id,
            "type": "claim",  # Custom node type
            "position": {"x": paper_x, "y": 150 + (claim_index * 120)},
            "data": {
                "label": node.label,
                "status": node.status,
                "color": node.color,
                **node.data
            }
        })

    # Position citation nodes below claims
    citation_nodes = [n for n in graph.nodes if n.type == "citation"]
    # We need to find which claims cite them to place them near
    # Build a map of claim -> citations
    claim_citations = {}
    for edge in graph.edges:
        if edge.type == "cites":
            if edge.source not in claim_citations:
                claim_citations[edge.source] = []
            claim_citations[edge.source].append(edge.target)
            
    # Place citations
    for node in citation_nodes:
        # Find a claim that cites this
        parent_claim_id = None
        for claim_id, cited_ids in claim_citations.items():
            if node.id in cited_ids:
                parent_claim_id = claim_id
                break
        
        # Default pos
        x, y = 0, 600
        
        if parent_claim_id:
            # Find parent claim position
            for n in rf_nodes:
                if n["id"] == parent_claim_id:
                    x = n["position"]["x"] + 50 # Shift slightly right
                    y = n["position"]["y"] + 200
                    break
                    
        rf_nodes.append({
            "id": node.id,
            "type": "citation",
            "position": {"x": x, "y": y},
            "data": {
                "label": node.label,
                "status": node.status,
                "color": node.color,
                **node.data
            }
        })
    
    # Convert edges
    for edge in graph.edges:
        rf_edges.append({
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "type": "smoothstep",
            "animated": edge.type == "cites",
            "label": edge.label if edge.label else None
        })
    
    return {
        "nodes": rf_nodes,
        "edges": rf_edges,
        "metadata": graph.metadata
    }


def export_for_cytoscape(graph: KnowledgeGraph) -> dict:
    """
    Export the knowledge graph in Cytoscape.js format.
    """
    elements = []
    
    # Add nodes
    for node in graph.nodes:
        elements.append({
            "data": {
                "id": node.id,
                "label": node.label,
                "type": node.type,
                "status": node.status,
                "color": node.color,
                **node.data
            }
        })
    
    # Add edges
    for edge in graph.edges:
        elements.append({
            "data": {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "label": edge.label
            }
        })
    
    return {
        "elements": elements,
        "metadata": graph.metadata
    }


# For testing
if __name__ == "__main__":
    from models.schemas import ClaimType, EvidenceType
    
    # Create test data
    papers = [
        Paper(
            id="p1",
            title="Climate Change and Machine Learning",
            authors=["Alice", "Bob"],
            year=2024,
            abstract="We study climate..."
        ),
        Paper(
            id="p2",
            title="Retracted Study on Fake Data",
            authors=["Evil Researcher"],
            year=2023,
            is_retracted=True
        )
    ]
    
    claims = [
        Claim(
            id="c1",
            text="ML improves climate predictions by 35%",
            claim_type=ClaimType.COMPARATIVE,
            evidence_type=EvidenceType.EXPERIMENT,
            paper_id="p1",
            validation_status=ValidationStatus.VERIFIED
        ),
        Claim(
            id="c2",
            text="Our method is revolutionary",
            claim_type=ClaimType.METHODOLOGICAL,
            evidence_type=EvidenceType.UNSUPPORTED,
            paper_id="p2",
            validation_status=ValidationStatus.SUSPICIOUS
        )
    ]
    
    graph = create_knowledge_graph(papers, claims)
    print(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"Metadata: {graph.metadata}")
    
    # Export for React Flow
    rf_export = export_for_react_flow(graph)
    print(f"\nReact Flow export has {len(rf_export['nodes'])} nodes")
