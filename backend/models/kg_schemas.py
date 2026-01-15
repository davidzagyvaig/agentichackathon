"""
Knowledge Graph Data Models

Pydantic schemas for Claims, Edges, and Graph structures
as specified in PRD_Knowledge_Graph.md Section 3.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime
import hashlib


def generate_claim_id(text: str) -> str:
    """
    Generate a claim ID from text using SHA256 hash.
    Format: "claim_{first 12 chars of hash}"
    """
    hash_value = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"claim_{hash_value}"


def generate_edge_id(source_id: str, target_id: str, edge_type: str) -> str:
    """
    Generate an edge ID from source, target, and type.
    Format: "edge_{source_id}_{target_id}_{type}"
    """
    return f"edge_{source_id}_{target_id}_{edge_type}"


# ============================================================================
# Source Paper Metadata (embedded in claims, NOT separate nodes)
# ============================================================================

class SourcePaperMetadata(BaseModel):
    """
    Metadata about the paper a claim was extracted from.
    Papers exist only as metadata on claims, not as separate nodes.
    """
    arxiv_id: Optional[str] = None
    title: str
    authors: list[str] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None


class ExternalSourceMetadata(BaseModel):
    """
    Metadata for claims from non-arXiv sources.
    """
    url: str
    retrieval_method: Literal["semantic_scholar", "web_search", "manual"]
    retrieval_date: str
    verified: bool = False


class ExtractionMetadata(BaseModel):
    """
    Metadata about how a claim was extracted.
    """
    model: str = "gpt-4o-mini"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    prompt_version: str = "1.0"
    context_snippet: Optional[str] = None


# ============================================================================
# Claim Node (Primary entity in the knowledge graph)
# ============================================================================

class ClaimNode(BaseModel):
    """
    A claim node in the knowledge graph.
    
    Claims are the ONLY first-class nodes. Papers exist only as metadata.
    """
    id: str = Field(default="", description="Format: claim_{hash}")
    
    text: str = Field(..., description="The claim itself, rephrased for clarity")
    original_text: Optional[str] = Field(None, description="Original text from paper")
    
    type: Literal["empirical", "ground_truth", "unsupported"] = Field(
        "empirical",
        description="Claim classification"
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="LLM-judged confidence in the claim"
    )
    
    source_paper: SourcePaperMetadata
    external_source: Optional[ExternalSourceMetadata] = None
    extraction: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    
    embedding: Optional[list[float]] = Field(
        None,
        description="1536-dim vector for text-embedding-3-small"
    )
    
    support_count: int = Field(0, description="How many claims support this one")
    contradict_count: int = Field(0, description="How many claims contradict this one")
    depth_to_ground_truth: Optional[int] = Field(
        None,
        description="Shortest path to a ground truth, null if ungrounded"
    )
    
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.id and self.text:
            self.id = generate_claim_id(self.text)
    
    class Config:
        use_enum_values = True
    
    def to_db_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "text": self.text,
            "original_text": self.original_text,
            "type": self.type,
            "confidence": self.confidence,
            "source_paper": self.source_paper.model_dump(),
            "external_source": self.external_source.model_dump() if self.external_source else None,
            "extraction": self.extraction.model_dump(),
            "embedding": self.embedding,
            "support_count": self.support_count,
            "contradict_count": self.contradict_count,
            "depth_to_ground_truth": self.depth_to_ground_truth,
        }
    
    @classmethod
    def from_db_row(cls, row: dict) -> "ClaimNode":
        """Create ClaimNode from database row."""
        return cls(
            id=row["id"],
            text=row["text"],
            original_text=row.get("original_text"),
            type=row["type"],
            confidence=row["confidence"],
            source_paper=SourcePaperMetadata(**row["source_paper"]),
            external_source=ExternalSourceMetadata(**row["external_source"]) if row.get("external_source") else None,
            extraction=ExtractionMetadata(**row["extraction"]),
            embedding=row.get("embedding"),
            support_count=row.get("support_count", 0),
            contradict_count=row.get("contradict_count", 0),
            depth_to_ground_truth=row.get("depth_to_ground_truth"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


# ============================================================================
# Claim Edge (Relationships between claims)
# ============================================================================

class ClaimEdge(BaseModel):
    """
    An edge between claims in the knowledge graph.
    
    Edge types:
    - supports: Source claim provides evidence that target claim is true
    - contradicts: Source claim provides evidence that target claim is false
    """
    id: str = Field(default="", description="Format: edge_{source}_{target}_{type}")
    
    source_id: str = Field(..., description="The claim providing evidence")
    target_id: str = Field(..., description="The claim being supported/contradicted")
    
    type: Literal["supports", "contradicts"]
    
    weight: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the relationship"
    )
    
    reasoning: Optional[str] = Field(
        None,
        description="LLM explanation for why this edge exists"
    )
    
    model: Optional[str] = Field(None, description="Which LLM created this edge")
    created_at: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.id and self.source_id and self.target_id and self.type:
            self.id = generate_edge_id(self.source_id, self.target_id, self.type)
    
    class Config:
        use_enum_values = True
    
    def to_db_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "reasoning": self.reasoning,
            "model": self.model,
        }
    
    @classmethod
    def from_db_row(cls, row: dict) -> "ClaimEdge":
        """Create ClaimEdge from database row."""
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            type=row["type"],
            weight=row["weight"],
            reasoning=row.get("reasoning"),
            model=row.get("model"),
            created_at=row.get("created_at"),
        )


# ============================================================================
# Build Run (Pipeline execution tracking)
# ============================================================================

class BuildRunConfig(BaseModel):
    """Configuration for a graph build run."""
    max_depth: int = 3
    max_claims: int = 500
    parallel_workers: int = 5
    relevance_threshold: float = 0.15


class BuildRunProgress(BaseModel):
    """Progress tracking for a build run."""
    papers_processed: int = 0
    claims_extracted: int = 0
    edges_created: int = 0
    current_depth: int = 0


class BuildRun(BaseModel):
    """
    Tracks a graph building pipeline execution.
    """
    id: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: Literal["running", "completed", "failed"] = "running"
    
    anchor_papers: list[str] = Field(default_factory=list)
    config: BuildRunConfig = Field(default_factory=BuildRunConfig)
    
    progress: BuildRunProgress = Field(default_factory=BuildRunProgress)
    errors: list[dict] = Field(default_factory=list)
    
    def to_db_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "status": self.status,
            "anchor_papers": self.anchor_papers,
            "config": self.config.model_dump(),
            "claims_extracted": self.progress.claims_extracted,
            "edges_created": self.progress.edges_created,
            "papers_processed": self.progress.papers_processed,
            "current_depth": self.progress.current_depth,
            "errors": self.errors,
        }


# ============================================================================
# Provenance and Analysis Results
# ============================================================================

class ProvenanceChain(BaseModel):
    """A single provenance chain from a claim to a ground truth."""
    path: list[str] = Field(..., description="Claim IDs from target to foundation")
    chain_confidence: float = Field(..., description="Product of edge weights")
    ground_truth_text: Optional[str] = None


class ProvenanceResult(BaseModel):
    """Result of tracing a claim's provenance."""
    claim_id: str
    claim_text: str
    is_grounded: bool
    ground_truths_found: list[ClaimNode] = []
    ungrounded_foundations: list[ClaimNode] = []
    cycles_detected: list[list[str]] = []
    provenance_chains: list[ProvenanceChain] = []


class ContradictionPair(BaseModel):
    """A pair of contradicting claims."""
    claim_a: ClaimNode
    claim_b: ClaimNode
    edge: ClaimEdge
    resolution_hints: list[str] = []


# ============================================================================
# Search and Query Models
# ============================================================================

class ClaimSearchFilters(BaseModel):
    """Filters for claim search."""
    type: Optional[Literal["empirical", "ground_truth", "unsupported"]] = None
    min_confidence: float = 0.0
    has_support: Optional[bool] = None
    source_paper_year_min: Optional[int] = None
    source_paper_year_max: Optional[int] = None


class ClaimSearchRequest(BaseModel):
    """Request for searching claims."""
    query: str
    search_type: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    limit: int = Field(10, ge=1, le=100)
    filters: ClaimSearchFilters = Field(default_factory=ClaimSearchFilters)


class ClaimSearchResult(BaseModel):
    """A single claim search result."""
    claim: ClaimNode
    similarity_score: Optional[float] = None
    relevance_rank: Optional[float] = None


class ClaimSearchResponse(BaseModel):
    """Response from claim search."""
    query: str
    search_type: str
    total_results: int
    results: list[ClaimSearchResult]


# ============================================================================
# Graph Statistics
# ============================================================================

class GraphStatistics(BaseModel):
    """Overview statistics about the knowledge graph."""
    total_claims: int = 0
    claims_by_type: dict = Field(default_factory=lambda: {
        "empirical": 0,
        "ground_truth": 0,
        "unsupported": 0
    })
    total_edges: int = 0
    edges_by_type: dict = Field(default_factory=lambda: {
        "supports": 0,
        "contradicts": 0
    })
    avg_support_depth: float = 0.0
    grounded_percentage: float = 0.0
    papers_indexed: int = 0
    last_updated: Optional[str] = None


# ============================================================================
# API Request/Response Models
# ============================================================================

class GraphBuildRequest(BaseModel):
    """Request to start a graph build."""
    anchor_papers: Optional[list[str]] = None
    max_depth: int = 3
    max_claims: int = 500
    parallel_workers: int = 5


class BuildStatusResponse(BaseModel):
    """Response with build status."""
    build_id: int
    status: str
    progress: BuildRunProgress
    estimated_time_minutes: Optional[int] = None


class ClaimDetailResponse(BaseModel):
    """Detailed response for a single claim."""
    claim: ClaimNode
    supporting_claims: list[ClaimNode] = []
    contradicting_claims: list[ClaimNode] = []
    is_grounded: bool = False


class SupportResponse(BaseModel):
    """Response for claim support query."""
    claim: ClaimNode
    supporting_claims: list[dict] = []
    contradicting_claims: list[dict] = []


class RefreshResponse(BaseModel):
    """Response from cache refresh."""
    success: bool
    nodes_loaded: int
    edges_loaded: int
    message: str = ""
