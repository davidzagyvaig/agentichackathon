"""
ClaimGraph Data Models
Pydantic schemas for Papers, Claims, Evidence, and Graph structures
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum
from datetime import datetime
import uuid


def generate_id() -> str:
    return str(uuid.uuid4())[:8]


class ClaimType(str, Enum):
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    COMPARATIVE = "comparative"
    METHODOLOGICAL = "methodological"
    EXISTENCE = "existence"


class EvidenceType(str, Enum):
    CITATION = "citation"
    FIGURE = "figure"
    EXPERIMENT = "experiment"
    OTHER_CLAIM = "other_claim"
    GENERAL_KNOWLEDGE = "general_knowledge"
    UNSUPPORTED = "unsupported"


class ValidationStatus(str, Enum):
    VERIFIED = "verified"        # Green node
    SUSPICIOUS = "suspicious"    # Red node
    PENDING = "pending"          # Gray node
    UNKNOWN = "unknown"


# ============ Paper Models ============

class Paper(BaseModel):
    """Represents a scientific paper"""
    id: str = Field(default_factory=generate_id)
    openalex_id: Optional[str] = None
    doi: Optional[str] = None
    title: str
    authors: list[str] = []
    year: Optional[int] = None
    abstract: Optional[str] = None
    source: str = "openalex"
    cited_by_count: int = 0
    references_count: int = 0
    is_retracted: bool = False
    trust_score: float = 1.0
    validation_status: ValidationStatus = ValidationStatus.PENDING
    
    class Config:
        use_enum_values = True


class PaperSearchResult(BaseModel):
    """Search results from paper APIs"""
    papers: list[Paper]
    total_count: int
    query: str


# ============ Claim Models ============

class Claim(BaseModel):
    """A scientific claim extracted from a paper"""
    id: str = Field(default_factory=generate_id)
    text: str
    claim_type: ClaimType
    evidence_type: EvidenceType
    paper_id: str
    confidence: float = 0.8
    validation_status: ValidationStatus = ValidationStatus.PENDING
    citation_refs: list[str] = []  # DOIs of cited papers
    validation_notes: str = ""
    
    class Config:
        use_enum_values = True


class ClaimExtractionResult(BaseModel):
    """Result of claim extraction from a paper"""
    paper_id: str
    claims: list[Claim]
    extraction_time: float = 0.0


# ============ Validation Models ============

class CitationValidation(BaseModel):
    """Validation result for a single citation"""
    citation_doi: Optional[str] = None
    citation_title: Optional[str] = None
    exists: bool = False
    is_retracted: bool = False
    is_relevant: Optional[bool] = None
    is_circular: bool = False
    relevance_score: float = 0.0
    validation_notes: str = ""


class ClaimValidation(BaseModel):
    """Full validation result for a claim"""
    claim_id: str
    overall_status: ValidationStatus
    trust_score: float
    citation_validations: list[CitationValidation] = []
    bullshit_indicators: list[str] = []
    validation_summary: str = ""


# ============ Graph Models ============

class GraphNode(BaseModel):
    """A node in the knowledge graph"""
    id: str
    type: Literal["paper", "claim", "evidence"]
    label: str
    status: ValidationStatus = ValidationStatus.PENDING
    color: str = "#808080"  # Gray default
    data: dict = {}
    
    class Config:
        use_enum_values = True


class GraphEdge(BaseModel):
    """An edge in the knowledge graph"""
    id: str = Field(default_factory=generate_id)
    source: str
    target: str
    type: Literal["contains_claim", "supported_by", "cites", "contradicts"]
    label: str = ""
    weight: float = 1.0


class KnowledgeGraph(BaseModel):
    """The complete knowledge graph"""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    metadata: dict = {}


# ============ API Request/Response Models ============

class SearchRequest(BaseModel):
    """Request to search for papers"""
    query: str
    max_results: int = 10




class AnalyzeRequest(BaseModel):
    """Request to analyze papers and build graph"""
    query: str
    max_papers: int = 5
    extract_claims: bool = True
    validate_citations: bool = True
    conversation_id: Optional[str] = None



class DeepVerifyRequest(BaseModel):
    """Request for recursive deep verification"""
    query: str
    max_depth: int = 1
    max_papers: int = 5



class AnalyzeResponse(BaseModel):
    """Full analysis response"""
    query: str
    papers: list[Paper]
    claims: list[Claim]
    graph: KnowledgeGraph
    summary: dict = {}


# ============ Deep Research Models ============

class ClarificationQuestion(BaseModel):
    """A clarification question for the user"""
    question: str
    options: list[str] = []
    required: bool = False


class ClarifyRequest(BaseModel):
    """Request to clarify a research query"""
    query: str


class ClarifyResponse(BaseModel):
    """Response from clarification step"""
    needs_clarification: bool
    questions: list[ClarificationQuestion] = []
    reasoning: str = ""


class DeepResearchRequest(BaseModel):
    """Request for deep research execution"""
    query: str
    clarifications: dict = {}
    max_tool_calls: int = 50
    use_background: bool = True


class ResearchCitation(BaseModel):
    """A citation from the research report"""
    node_id: str
    node_type: str
    label: str
    context: str = ""


class ContradictionFound(BaseModel):
    """A contradiction identified in the research"""
    paper_a_id: str
    paper_a_label: str
    paper_b_id: str
    paper_b_label: str
    summary: str
    edge_id: Optional[str] = None


class DeepResearchResponse(BaseModel):
    """Response from deep research execution"""
    query: str
    report: str
    research_goal: str
    citations: list[ResearchCitation] = []
    contradictions: list[ContradictionFound] = []
    confidence: float = 0.0
    reasoning_summary: str = ""
    status: str = "completed"
    execution_time_ms: int = 0
    graph: Optional[KnowledgeGraph] = None


# ============ Legacy Deep Research Request ============

class LegacyDeepResearchRequest(BaseModel):
    """Legacy request for deep research handoff (kept for compatibility)"""
    graph: KnowledgeGraph
    user_query: str
    verified_claims: list[Claim]
    suspicious_claims: list[Claim]
