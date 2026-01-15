"""
ClaimGraph Backend API
FastAPI server for scientific claim verification and knowledge graph generation
"""

import os
import sys
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from models.schemas import (
    Paper,
    Claim,
    SearchRequest,
    AnalyzeRequest,
    AnalyzeResponse,
    KnowledgeGraph,
    ValidationStatus,
    ClaimValidation,
)
from agents.paper_search import search_papers, get_paper_by_doi
from agents.claim_extractor import extract_claims_from_paper
from agents.citation_validator import validate_claim
from graph.knowledge_graph import create_knowledge_graph, export_for_react_flow


# In-memory storage for demo
papers_cache: dict[str, Paper] = {}
claims_cache: dict[str, Claim] = {}
validations_cache: dict[str, ClaimValidation] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 ClaimGraph Backend starting...")
    print(f"   OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-4o')}")
    yield
    print("👋 ClaimGraph Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="ClaimGraph API",
    description="Scientific claim verification and knowledge graph generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Health Check ============

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ClaimGraph API",
        "version": "1.0.0",
        "message": "We don't summarize. We VERIFY. 🔬"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "cached_papers": len(papers_cache),
        "cached_claims": len(claims_cache),
    }


# ============ Paper Search ============

@app.get("/api/search")
async def api_search(
    query: str = Query(..., description="Search query for papers"),
    max_results: int = Query(10, ge=1, le=25, description="Maximum number of results")
):
    """
    Search for scientific papers by query.
    
    Returns papers from OpenAlex with metadata and abstracts.
    """
    try:
        result = await search_papers(query, max_results)
        
        # Cache papers
        for paper in result.papers:
            papers_cache[paper.id] = paper
        
        return {
            "success": True,
            "query": query,
            "total": result.total_count,
            "papers": [paper.model_dump() for paper in result.papers]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/{doi:path}")
async def get_paper(doi: str):
    """Get a specific paper by DOI."""
    try:
        paper = await get_paper_by_doi(doi)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ Claim Extraction ============

@app.post("/api/extract-claims")
async def api_extract_claims(paper_id: str):
    """
    Extract claims from a cached paper.
    
    Uses GPT-4o to identify and classify scientific claims.
    """
    if paper_id not in papers_cache:
        raise HTTPException(status_code=404, detail="Paper not found in cache. Search first.")
    
    paper = papers_cache[paper_id]
    
    try:
        result = await extract_claims_from_paper(paper)
        
        # Cache claims
        for claim in result.claims:
            claims_cache[claim.id] = claim
        
        return {
            "success": True,
            "paper_id": paper_id,
            "extraction_time": result.extraction_time,
            "claims": [claim.model_dump() for claim in result.claims]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ Full Analysis Pipeline ============

@app.post("/api/analyze")
async def api_analyze(request: AnalyzeRequest):
    """
    Full analysis pipeline:
    1. Search for papers
    2. Extract claims from each paper
    3. Validate citations (optional)
    4. Build knowledge graph
    
    This is the main endpoint that powers the demo.
    """
    start_time = time.time()
    all_claims = []
    all_validations = {}
    
    try:
        # Step 1: Search for papers
        print(f"🔍 Searching for: {request.query}")
        search_result = await search_papers(request.query, request.max_papers)
        papers = search_result.papers
        
        if not papers:
            return AnalyzeResponse(
                query=request.query,
                papers=[],
                claims=[],
                graph=KnowledgeGraph(),
                summary={"error": "No papers found"}
            )
        
        # Cache papers
        for paper in papers:
            papers_cache[paper.id] = paper
        
        print(f"📚 Found {len(papers)} papers")
        
        # Step 2: Extract claims from each paper
        if request.extract_claims:
            for paper in papers:
                if paper.abstract:
                    print(f"   📝 Extracting claims from: {paper.title[:50]}...")
                    result = await extract_claims_from_paper(paper)
                    all_claims.extend(result.claims)
                    
                    # Cache claims
                    for claim in result.claims:
                        claims_cache[claim.id] = claim
            
            print(f"📋 Extracted {len(all_claims)} total claims")
        
        # Step 3: Validate citations (if enabled)
        if request.validate_citations and all_claims:
            print("🔬 Validating claims...")
            for claim in all_claims[:10]:  # Limit to first 10 for speed
                paper = papers_cache.get(claim.paper_id)
                if paper:
                    validation = await validate_claim(claim, paper)
                    all_validations[claim.id] = validation
                    validations_cache[claim.id] = validation
                    
                    # Update claim status
                    claim.validation_status = validation.overall_status
            
            print(f"✅ Validated {len(all_validations)} claims")
        
        # Step 4: Build knowledge graph
        print("🕸️ Building knowledge graph...")
        graph = create_knowledge_graph(papers, all_claims, all_validations)
        
        # Calculate summary
        verified_count = sum(1 for c in all_claims if c.validation_status == ValidationStatus.VERIFIED)
        suspicious_count = sum(1 for c in all_claims if c.validation_status == ValidationStatus.SUSPICIOUS)
        
        elapsed_time = time.time() - start_time
        
        summary = {
            "papers_analyzed": len(papers),
            "claims_extracted": len(all_claims),
            "claims_verified": verified_count,
            "claims_suspicious": suspicious_count,
            "elapsed_seconds": round(elapsed_time, 2),
        }
        
        print(f"✨ Analysis complete in {elapsed_time:.2f}s")
        
        return AnalyzeResponse(
            query=request.query,
            papers=papers,
            claims=all_claims,
            graph=graph,
            summary=summary
        )
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Knowledge Graph ============

@app.get("/api/graph")
async def get_graph(format: str = Query("reactflow", enum=["reactflow", "cytoscape", "raw"])):
    """
    Get the current knowledge graph in the specified format.
    
    Formats:
    - reactflow: For React Flow visualization
    - cytoscape: For Cytoscape.js visualization
    - raw: Raw graph structure
    """
    # Build graph from cached data
    papers = list(papers_cache.values())
    claims = list(claims_cache.values())
    validations = validations_cache
    
    if not papers:
        return {"nodes": [], "edges": [], "metadata": {}}
    
    graph = create_knowledge_graph(papers, claims, validations)
    
    if format == "reactflow":
        return export_for_react_flow(graph)
    elif format == "cytoscape":
        from graph.knowledge_graph import export_for_cytoscape
        return export_for_cytoscape(graph)
    else:
        return graph.model_dump()


# ============ Validation ============

@app.post("/api/validate/{claim_id}")
async def validate_single_claim(claim_id: str):
    """Validate a single claim by ID."""
    if claim_id not in claims_cache:
        raise HTTPException(status_code=404, detail="Claim not found")
    
    claim = claims_cache[claim_id]
    paper = papers_cache.get(claim.paper_id)
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found for claim")
    
    try:
        validation = await validate_claim(claim, paper)
        validations_cache[claim_id] = validation
        claim.validation_status = validation.overall_status
        
        return validation.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ Clear Cache ============

@app.post("/api/clear")
async def clear_cache():
    """Clear all cached data. Use for starting fresh."""
    papers_cache.clear()
    claims_cache.clear()
    validations_cache.clear()
    return {"success": True, "message": "Cache cleared"}


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🔬 ClaimGraph API Server                                ║
    ║   "We don't summarize. We VERIFY."                        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
