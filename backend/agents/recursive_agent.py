"""
Recursive Research Agent
Orchestrates the deep research pipeline: Search -> Extract -> Verify -> Recurse
"""

import asyncio
import logging
from typing import Optional, Set

from models.schemas import (
    Paper,
    Claim,
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    ValidationStatus
)
from database import Database
from agents.paper_search import search_papers, get_paper_by_doi
from agents.claim_extractor import extract_claims_from_paper
from agents.citation_validator import validate_claim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecursiveResearcher:
    def __init__(self, db: Database):
        self.db = db
        self.visited_dois: Set[str] = set()

    async def start_research(self, query: str, max_depth: int = 1, max_papers: int = 5) -> KnowledgeGraph:
        """
        Start the recursive research process.
        
        Args:
            query: User's research question/topic
            max_depth: How many layers of citations to follow (default 1 to save costs)
            max_papers: Max papers to process per layer
            
        Returns:
            KnowledgeGraph object with all found nodes and edges
        """
        logger.info(f"Starting deep research on: {query} (Depth: {max_depth})")
        
        # Step 1: Initial Search
        search_result = await search_papers(query, max_results=max_papers)
        logger.info(f"Found {len(search_result.papers)} initial papers")
        
        # Step 2: Process Queue
        # We process papers layer by layer
        current_layer_papers = search_result.papers
        
        # Track nodes and edges for the final graph
        nodes = {}
        edges = []
        
        for depth in range(max_depth + 1):
            logger.info(f"--- Processing Layer {depth} ({len(current_layer_papers)} papers) ---")
            next_layer_dois = set()
            
            for paper in current_layer_papers:
                if paper.doi and paper.doi in self.visited_dois:
                    continue
                if paper.doi:
                    self.visited_dois.add(paper.doi)
                
                # Store Paper in DB
                saved_paper = self.db.upsert_paper(paper.model_dump())
                
                # Add to Graph Nodes
                nodes[paper.id] = GraphNode(
                    id=paper.id,
                    label=paper.title,
                    type="paper",
                    data=paper.model_dump()
                )
                
                # Step 3: Extract Claims
                logger.info(f"Extracting claims for: {paper.title[:50]}...")
                extraction = await extract_claims_from_paper(paper)
                
                for claim in extraction.claims:
                    # Store Claim in DB
                    saved_claim = self.db.upsert_claim(claim.model_dump())
                    
                    # Add to Graph Nodes
                    nodes[claim.id] = GraphNode(
                        id=claim.id,
                        label=claim.text[:50] + "...",
                        type="claim",
                        data=claim.model_dump()
                    )
                    
                    # Edge: Paper -> Claim
                    edges.append(GraphEdge(
                        source=paper.id,
                        target=claim.id,
                        type="contains_claim",
                        label="asserts"
                    ))
                    
                    # Step 4: Verify Claim ("Bullshit Detector")
                    logger.info(f"Verifying claim: {claim.id}...")
                    validation = await validate_claim(claim, paper)
                    
                    # Store validation in DB (Conceptually - assume claim update or separate table)
                    # For now we update the claim status
                    claim.validation_status = validation.overall_status
                    claim.validation_notes = validation.validation_summary
                    self.db.upsert_claim(claim.model_dump())
                    
                    # Edge: Validation coloring (Visual feedback)
                    # We can add validation properties to the node data
                    nodes[claim.id].data["validation_status"] = validation.overall_status
                    nodes[claim.id].data["trust_score"] = validation.trust_score
                    
                    # Step 5: Process Citations (Recursion Prep)
                    if depth < max_depth:
                        for citation_val in validation.citation_validations:
                            if citation_val.exists and citation_val.citation_doi:
                                next_layer_dois.add(citation_val.citation_doi)
                                
                                # Edge: Claim -> Citation (if verifying)
                                # We need the ID of the cited paper. It might not exist in our DB yet.
                                # We'll fetch it in the next layer processing.
            
            # Prepare next layer
            if depth < max_depth and next_layer_dois:
                logger.info(f"Preparing next layer with {len(next_layer_dois)} citations...")
                current_layer_papers = []
                for doi in list(next_layer_dois)[:max_papers]: # Limit recursion width
                    paper = await get_paper_by_doi(doi)
                    if paper:
                        current_layer_papers.append(paper)
            else:
                current_layer_papers = []
                
        logger.info("Deep research complete.")
        
        return KnowledgeGraph(
            nodes=list(nodes.values()),
            edges=edges
        )

